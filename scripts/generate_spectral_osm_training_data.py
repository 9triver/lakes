#!/usr/bin/env python3
"""Cache-aware, resumable seed-first generation with batched Patch commits."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from lake_workbench.automatic_labels import generate_derived_label
from lake_workbench.catalog import SiteCatalog
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.runner import run_patch_export, run_dataset_build
from lake_workbench.utils import display_path, read_csv_records
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.workspaces.file_lock import process_lock

SOURCE = 'spectral_osm_intersection'


def _parser():
    parser = argparse.ArgumentParser(description='先补齐光谱∩OSM种子，再分批生成训练Patch；可断点恢复')
    parser.add_argument('--region', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sites', nargs='+')
    group.add_argument('--all-sites', action='store_true', help='扫描所有有本地影像的Site；已有匹配缓存将复用')
    parser.add_argument('--workspace', default='default')
    parser.add_argument('--stage', choices=['all','seeds','patches'], default='all', help='all先种子后Patch；seeds仅补种子；patches只使用已有种子')
    parser.add_argument('--max-dimension', type=int, default=2048)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--batch-sites', type=int, default=20, help='每批提交的Site数量，减少反复复制旧预览')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='检查匹配种子和选择，不生成标签/样本/Patch/状态文件')
    parser.add_argument('--state-file', type=Path, help='任务状态文件；相同选择和参数默认复用同一状态文件')
    parser.add_argument('--resume', type=Path, help='从任务状态恢复原始Site选择；保持region/workspace/尺寸参数一致')
    parser.add_argument('--build-dataset', action='store_true', help='Patch阶段成功后生成已纳入Patch的resize256_v1训练NPZ缓存')
    return parser


def _visible_layers():
    return {name: name=='spectral_osm_intersection' for name in [
        'basemap_osm','basemap_satellite','tile_grid','osm','hydrolakes','context_osm',
        'context_hydrolakes','esa','jrc','local_label','spectral_water',SOURCE,
        'spectral_osm_consensus','osm_spectral_consensus','model_prediction']}


def _selected_sites(catalog, site_ids, all_sites):
    if all_sites:
        return sorted([s for s in catalog.sites if catalog._active_local_imagery_row(s) is not None],key=lambda s:s.site_id)
    sites=[]
    for site_id in dict.fromkeys(site_ids or []):
        site=catalog.get_site(site_id)
        if site is None:raise ValueError(f'Site不存在：{site_id}')
        sites.append(site)
    return sites


def ensure_seed(catalog,store,workspace,site,max_dimension,*,cache_only=False):
    imagery=catalog._active_local_imagery_row(site)
    if not imagery or not imagery.get('tci_path') or not imagery['tci_path'].is_file():
        raise FileNotFoundError(f'{site.site_id}没有可读取的本地影像')
    payload=dict(asset_id=imagery.get('product_id') or imagery.get('product',''),
                 product=imagery.get('product',''),extent=list(site.bbox),max_dimension=max_dimension)
    output=store.workspace_derived_label_dir(workspace,catalog.region.key)
    result=generate_derived_label(catalog,site,SOURCE,payload,output,cache_only=True)
    hit=result is not None
    if result is None and not cache_only:
        result=generate_derived_label(catalog,site,SOURCE,payload,output)
    if result is None:return None
    return dict(site_id=site.site_id,label_id=str(result['label_id']),cache_hit=hit,
                imagery=display_path(imagery['tci_path']),asset_id=payload['asset_id'],product=payload['product'],
                tile=imagery.get('tile',''),extent=list(site.bbox))


def prepare_sample(catalog,store,workspace,site,seed):
    view=dict(region=catalog.region.key,site_id=site.site_id,
        map=dict(extent=seed['extent'],center=list(site.center),zoom=12),visible_layers=_visible_layers(),
        jrc_threshold=75,selected_local_label=None,selected_generated_labels={SOURCE:dict(id=seed['label_id'])},
        selected_tile=seed['tile'],selected_product=seed['product'],selected_imagery_asset_id=seed['asset_id'],
        model_prediction_excluded=True,model_validation=None)
    with store.transaction():
        sample=catalog.create_training_sample(site,dict(label_source='current_view',label_threshold='75',buffer_ratio=.8,view_state=view),
            samples_path=store.ensure_workspace_training_samples(workspace,catalog.region.key),
            label_dir=store.workspace_training_label_dir(workspace,catalog.region.key),
            derived_label_dir=store.workspace_derived_label_dir(workspace,catalog.region.key))
    return dict(seed,sample_id=sample['sample_id'])


def patch_index(store,workspace,region):
    by_sample=defaultdict(list)
    with store.transaction():
        for row in read_csv_records(store.workspace_logical_patch_manifest(workspace,region)):
            by_sample[row.get('sample_id','')].append(row)
    return by_sample


def patch_complete(rows,patch_size,stride):
    selected=[r for r in rows if int(r.get('logical_size') or 256)==patch_size and int(r.get('grid_stride') or r.get('logical_size') or 256)==stride]
    if not selected:return False
    for row in selected:
        for field in ['preview_path','preview_base_path']:
            value=row.get(field)
            if not value:return False
            path=Path(value);path=path if path.is_absolute() else PROJECT_ROOT/path
            if not path.is_file():return False
    return True


def atomic_state(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(state,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def commit_batch(store,args,batch):
    # build_transaction is cross-process and reentrant; only publication is batched.
    with store.build_transaction():
        result=run_patch_export(args.region,dict(workspace_id=args.workspace,
            sample_ids=[r['sample_id'] for r in batch],patch_size=args.patch_size,stride=args.stride),store)
        index=patch_index(store,args.workspace,args.region)
        for row in batch:
            if not patch_complete(index.get(row['sample_id'],[]),args.patch_size,args.stride):
                raise RuntimeError(f"提交后Patch不完整：{row['site_id']}")
    return result


def generate_for_site(catalog,store,workspace_id,site,*,max_dimension,patch_size,stride):
    """Compatibility API; CLI uses two phases and batches instead."""
    seed=ensure_seed(catalog,store,workspace_id,site,max_dimension)
    row=prepare_sample(catalog,store,workspace_id,site,seed)
    result=run_patch_export(catalog.region.key,dict(workspace_id=workspace_id,sample_id=row['sample_id'],patch_size=patch_size,stride=stride),store)
    return dict(row,patches=result.get('patches',0))


def main(argv=None):
    parser=_parser();args=parser.parse_args(argv)
    regions,default=load_region_configs()
    if args.region not in regions:parser.error('未知区域')
    if not 256<=args.max_dimension<=4096 or min(args.patch_size,args.stride,args.batch_sites)<=0:parser.error('尺寸/批次数量不合法')
    if args.stage=='seeds' and args.build_dataset:parser.error('--build-dataset需要Patch阶段')
    store=WorkspaceStore(regions);store.get(args.workspace,allow_archived=False)
    settings=dict(region=args.region,workspace=args.workspace,max_dimension=args.max_dimension,patch_size=args.patch_size,stride=args.stride)
    prior=json.loads(args.resume.read_text()) if args.resume else None
    if prior:
        if prior['settings']!=settings:parser.error('恢复参数与原任务不一致')
        if args.sites or args.all_sites:parser.error('--resume恢复原Site列表，不能同时指定--sites/--all-sites')
        args.sites=prior['site_ids']
    if not args.sites and not args.all_sites:parser.error('请指定--sites、--all-sites或--resume')
    catalog=SiteCatalog(regions[args.region],default)
    sites=_selected_sites(catalog,args.sites,args.all_sites)
    key=hashlib.sha256(json.dumps(dict(settings,sites=[s.site_id for s in sites]),sort_keys=True).encode()).hexdigest()[:20]
    path=args.resume or args.state_file or PROJECT_ROOT/'outputs/pipeline_jobs'/args.workspace/args.region/f'{key}.json'
    if args.dry_run:
        hits=missing=0
        for site in sites:
            seed=ensure_seed(catalog,store,args.workspace,site,args.max_dimension,cache_only=True)
            hits+=seed is not None;missing+=seed is None
        print(f'只读预览：Site={len(sites)}，匹配种子缓存={hits}，待补种子={missing}，阶段={args.stage}，每批{args.batch_sites}个Site',flush=True)
        print(f'任务状态路径（本次不写入）：{path}',flush=True)
        return 0
    try:
        guard=process_lock(store.root/'.locks'/f'spectral_osm_{args.workspace}_{args.region}.lock',blocking=False)
        with guard:
            state=prior or (json.loads(path.read_text()) if path.is_file() else dict(version=1,settings=settings,site_ids=[s.site_id for s in sites],sites={}))
            if state['settings']!=settings or state['site_ids']!=[s.site_id for s in sites]:raise ValueError('状态文件对应另一组参数/Site')
            state.update(status='running',stage=args.stage,pid=os.getpid(),updated_at=datetime.now().isoformat())
            def save():
                state['updated_at']=datetime.now().isoformat();atomic_state(path,state)
            def fail(site_id,error):
                record=state['sites'].setdefault(site_id,{})
                record.update(error=f'{type(error).__name__}: {error}',patches_done=False)
                print(f'[失败] {site_id}: {record["error"]}',file=sys.stderr,flush=True);save()
            save();print(f'任务状态：{path}',flush=True)
            try:
                prepared=[];failed=set()
                # Complete seed stage for the entire selection before generating any Patch.
                for i,site in enumerate(sites,1):
                    try:
                        seed=ensure_seed(catalog,store,args.workspace,site,args.max_dimension,cache_only=args.stage=='patches')
                        if seed is None:raise FileNotFoundError('缺少匹配种子；先执行--stage seeds或--stage all')
                        old=state['sites'].get(site.site_id,{})
                        if old.get('label_id')!=seed['label_id']:old={}
                        old.update(seed,error='');state['sites'][site.site_id]=old;prepared.append((site,seed));save()
                        print(f'[种子 {i}/{len(sites)}] {site.site_id} '+('复用缓存' if seed['cache_hit'] else '生成完成'),flush=True)
                    except Exception as error:
                        failed.add(site.site_id);fail(site.site_id,error)
                        if not args.continue_on_error:raise
                if args.stage=='seeds':
                    state.update(status='partial' if failed else 'seeds_completed',failed_sites=sorted(failed));save();return int(bool(failed))
                state['phase']='patches';save()
                index=patch_index(store,args.workspace,args.region);pending=[]
                for site,seed in prepared:
                    try:
                        current=ensure_seed(catalog,store,args.workspace,site,args.max_dimension,cache_only=True)
                        if not current or current['label_id']!=seed['label_id']:raise ValueError('活动影像/种子在两阶段之间发生变化，请重试')
                        row=prepare_sample(catalog,store,args.workspace,site,seed)
                        state['sites'][site.site_id].update(sample_id=row['sample_id'])
                        if patch_complete(index.get(row['sample_id'],[]),args.patch_size,args.stride):
                            state['sites'][site.site_id].update(patches_done=True,error='');print(f'[Patch复用] {site.site_id}',flush=True)
                        else:pending.append(row)
                        save()
                    except Exception as error:
                        failed.add(site.site_id);fail(site.site_id,error)
                        if not args.continue_on_error:raise
                for offset in range(0,len(pending),args.batch_sites):
                    batch=pending[offset:offset+args.batch_sites]
                    print(f'[Patch批次] {offset+1}–{offset+len(batch)}/{len(pending)}',flush=True)
                    try:
                        result=commit_batch(store,args,batch)
                        for row in batch:state['sites'][row['site_id']].update(patches_done=True,error='')
                        print(f'[批次完成] Site={len(batch)}，当前总Patch={result.get("patches",0)}，本批新生成={result.get("generated_patches",0)}',flush=True);save()
                    except Exception as error:
                        if not args.continue_on_error:
                            for row in batch:fail(row['site_id'],error)
                            raise
                        # Isolate a problematic sample without losing the entire successful seed phase.
                        print(f'[批次失败] 改为逐Site重试：{error}',file=sys.stderr,flush=True)
                        for row in batch:
                            try:
                                commit_batch(store,args,[row]);state['sites'][row['site_id']].update(patches_done=True,error='');save()
                            except Exception as individual:failed.add(row['site_id']);fail(row['site_id'],individual)
                if args.build_dataset and not failed:
                    state['phase']='dataset';save()
                    state['dataset']=run_dataset_build(args.region,dict(workspace_id=args.workspace,config_id='resize256_v1'),store)
                state.update(status='partial' if failed else 'completed',failed_sites=sorted(failed),completed_sites=sum(bool(r.get('patches_done')) for r in state['sites'].values()));save()
                print(f'任务结束：{state["status"]}，完成Site={state["completed_sites"]}，失败={len(failed)}',flush=True)
                return int(bool(failed))
            except BaseException as error:
                state.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}');save();raise
    except BlockingIOError:
        print('同一工作区/省份已有生成任务，拒绝重复启动。请查看其日志或等待完成。',file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
