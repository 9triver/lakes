#!/usr/bin/env python3
"""Resume terrain preparation for every configured site's available imagery."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from lake_workbench.paths import PROJECT_ROOT


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def worker(task):
    import numpy as np
    import rasterio
    from lake_workbench.models.features import aligned_slope, dem_tile
    from lake_workbench.models.preparation_progress import reporting
    def progress(**event):
        print(json.dumps(event, ensure_ascii=False), flush=True)
    with reporting(progress):
        if task['kind']=='dem':
            path=dem_tile(task['lat'],task['lon'])
            print(json.dumps(dict(source_ready=True,path=str(path))),flush=True)
        elif task['kind']=='image':
            with rasterio.open(task['path']) as src:
                valid=np.ones(src.shape, dtype=bool)
                nonzero=np.zeros(src.shape, dtype=bool)
                for index in range(1, src.count+1):
                    band=src.read(index)
                    valid &= np.isfinite(band) & (src.read_masks(index)>0)
                    nodata=src.nodatavals[index-1]
                    if nodata is not None:
                        valid &= (~np.isnan(band) if np.isnan(nodata) else band!=nodata)
                    nonzero |= band!=0
                valid &= nonzero
            slope,meta=aligned_slope(Path(task['path']),valid)
            print(json.dumps(dict(ready=True, valid_pixels=int(valid.sum()), metadata=meta)),flush=True)
        else:
            west,south,east,north=task['bounds']
            for lat in range(math.floor(south),math.floor(north)+1):
                for lon in range(math.floor(west),math.floor(east)+1):
                    dem_tile(lat,lon,bounds=(west,south,east,north))
            print(json.dumps(dict(source_ready=True, aligned=False, reason='no_available_imagery')),flush=True)


def inventory(region_key):
    from lake_workbench.catalog import SiteCatalog
    from lake_workbench.regions.config import load_region_configs
    from lake_workbench.models.comparison import _assets
    regions,default=load_region_configs()
    tasks={}; summary={}; errors=[]
    for key,region in regions.items():
        if key != region_key:continue
        print('INVENTORY '+key,flush=True)
        if not region.site_metadata.exists():
            summary[key]=dict(sites=0, error='site_metadata_missing');errors.append(dict(region=key,error='site_metadata_missing'));continue
        try:
            catalog=SiteCatalog(region,default)
            # Metadata-only site loading avoids loading every province's water polygons.
            sites=catalog._load_sites()
            summary[key]=dict(sites=len(sites),with_imagery=0,without_imagery=0)
            for site_index,site in enumerate(sites):
                if site_index % 100 == 0:print(f"SCAN {key} {site_index}/{len(sites)}",flush=True)
                ref=f'{key}/{site.site_id}'
                try:
                    rows=list(_assets(catalog,site))
                    active=catalog._active_local_imagery_row(site)
                    if active:rows.append(active)
                    rows.extend(catalog._model_validation_rows(site,5))
                    paths=sorted({str(Path(r['tci_path']).resolve()) for r in rows if r.get('tci_path') and Path(r['tci_path']).is_file()})
                    if paths:
                        summary[key]['with_imagery']+=1
                        for path in paths:
                            item=tasks.setdefault('image:'+path,dict(kind='image',path=path,sites=[]))
                            item['sites'].append(ref)
                    else:
                        summary[key]['without_imagery']+=1
                        w,s,e,n=site.bbox
                        tasks['site:'+ref]=dict(kind='bounds',bounds=[w-.002,s-.002,e+.002,n+.002],sites=[ref])
                except Exception as error:
                    errors.append(dict(site=ref,error=str(error)))
            del catalog
        except Exception as error:
            summary[key]=dict(error=str(error));errors.append(dict(region=key,error=str(error)))
    return dict(regions=summary,tasks=list(tasks.values()),inventory_errors=errors,created_at=time.time())


def group_grid_tasks(plan):
    import hashlib
    import rasterio
    grouped={}; errors=list(plan.get('inventory_errors',[]))
    original=len(plan['tasks'])
    for index,task in enumerate(plan['tasks']):
        if index%500==0:print(f'GROUP_GRIDS {index}/{original}',flush=True)
        try:
            if task['kind']=='image':
                with rasterio.open(task['path']) as src:
                    if not src.crs:raise ValueError('Imagery has no CRS')
                    identity=[src.crs.to_wkt(),list(src.transform),list(src.shape)]
            else:identity=['bounds',task['bounds']]
            key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
            if key not in grouped:
                grouped[key]=dict(task,grid_id=key,sites=[],source_image_count=0)
            item=grouped[key]
            item['sites']=sorted(set(item['sites'])|set(task['sites']))
            item['source_image_count']+=1
        except Exception as error:errors.append(dict(path=task.get('path'),sites=task['sites'],error=str(error)))
    return dict(plan,tasks=list(grouped.values()),inventory_errors=errors,
        schema_version=2,source_tasks=original,grouping='exact CRS + affine transform + raster shape')


def source_bounds(task):
    if task['kind']=='bounds':return tuple(task['bounds'])
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(task['path']) as src:
        if not src.crs or not src.crs.is_projected or src.crs.linear_units!='metre':
            raise ValueError('DEM alignment requires metre-projected imagery')
        l,b,r,t=src.bounds
        return transform_bounds(src.crs,'EPSG:4326',l-180,b-180,r+180,t+180,densify_pts=21)


def source_tiles(bounds):
    w,s,e,n=bounds
    return [(lat,lon) for lat in range(math.floor(s),math.floor(n)+1)
            for lon in range(math.floor(w),math.floor(e)+1)]


def cached_source(lat,lon,bounds):
    from lake_workbench.models.features import cached_dem_tile,dem_clip_path,_valid_dem
    full=cached_dem_tile(lat,lon)
    if full is not None:return full
    clip=dem_clip_path(lat,lon,bounds)
    return clip if clip.is_file() and _valid_dem(str(clip),clip.stat().st_mtime_ns,clip.stat().st_size) else None


def task_cached(task):
    if task['kind']=='image':
        from lake_workbench.models.preparation import _slope_ready
        return _slope_ready(Path(task['path']))
    bounds=source_bounds(task)
    return all(cached_source(lat,lon,bounds) is not None for lat,lon in source_tiles(bounds))


def plan_dem_downloads(tasks, *, skip_cached_grids=True):
    from lake_workbench.models.features import dem_name
    needed=set();already=set();cached_grids=0;errors=[]
    for index,task in enumerate(tasks):
        if index%100==0:print(f'DEM_CHECK {index}/{len(tasks)} grids',flush=True)
        try:
            if skip_cached_grids and task_cached(task):cached_grids+=1;continue
            bounds=source_bounds(task)
            for lat,lon in source_tiles(bounds):
                if cached_source(lat,lon,bounds) is not None:already.add((lat,lon))
                else:needed.add((lat,lon))
        except Exception as error:errors.append(dict(index=index,sites=task['sites'],error=str(error)))
    return dict(cached_grids=cached_grids,existing_source_tiles=len(already),
        required_source_tiles=len(already|needed),fully_cached_source_tiles=len(already-needed),errors=errors,
        downloads=[dict(kind='dem',lat=lat,lon=lon,tile=dem_name(lat,lon)) for lat,lon in sorted(needed)])


def prepare_dem_sources(args,root,tasks):
    plan=plan_dem_downloads(tasks,skip_cached_grids=not args.dem_only);save(root/'dem_plan.json',plan)
    print('DEM_PLAN '+json.dumps(dict(cached_grids=plan['cached_grids'],unique_missing_tiles=len(plan['downloads']),existing_source_tiles=plan['existing_source_tiles'],errors=len(plan['errors']))),flush=True)
    folder=root/'dem_sources';folder.mkdir(exist_ok=True)
    state=dict(status='running',total=len(plan['downloads']),ready=0,failed=0,items={},pid=os.getpid())
    try:
        for round_index in range(args.rounds):
            for task in plan['downloads']:
                key=task['tile']
                if state['items'].get(key,{}).get('status')=='ready':continue
                # A previous province or a concurrent reader may already have supplied it.
                from lake_workbench.models.features import cached_dem_tile
                if cached_dem_tile(task['lat'],task['lon']) is not None:code=0
                else:
                    if shutil.disk_usage(PROJECT_ROOT/'data').free/1024**3<args.min_free_gb:raise RuntimeError('Disk below reserved free space')
                    taskfile=folder/(key+'.json');save(taskfile,task)
                    state['current_tile']=key;save(folder/'status.json',state)
                    print(f'DEM_DOWNLOAD {key} round={round_index+1}',flush=True)
                    try:code=execute([sys.executable,'-u',str(Path(__file__).resolve()),'--worker-task',str(taskfile)],folder/(key+'.log'),timeout=args.timeout)
                    except subprocess.TimeoutExpired:code=124
                state['items'][key]=dict(status='ready' if code==0 else 'failed',code=code,round=round_index+1)
                state['ready']=sum(x['status']=='ready' for x in state['items'].values())
                state['failed']=sum(x['status']=='failed' for x in state['items'].values())
                save(folder/'status.json',state)
                print(f'DEM_PROGRESS ready={state["ready"]}/{state["total"]} failed={state["failed"]}',flush=True)
            if state['ready']==state['total']:break
        state['status']='complete' if state['ready']==state['total'] else 'partial'
    except BaseException as error:
        state.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:state['updated_at']=time.time();save(folder/'status.json',state)
    return state


def run_region(args):
    root=args.job_dir.resolve()/'regions'/args.region_worker
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'inventory.json';statuspath=root/'status.json'
    if manifest.exists():plan=json.loads(manifest.read_text())
    else:
        candidates=[PROJECT_ROOT/'outputs/all_site_slope_grids_20261030/regions'/args.region_worker/'inventory.json',
                    PROJECT_ROOT/'outputs/all_site_slopes_by_province_20261029/regions'/args.region_worker/'inventory.json']
        old=next((p for p in candidates if p.is_file()),None)
        source=json.loads(old.read_text()) if old is not None else inventory(args.region_worker)
        plan=source if source.get('schema_version')==2 else group_grid_tasks(source)
        save(manifest,plan)
    print('INVENTORY_DONE '+json.dumps(dict(regions=plan['regions'],tasks=len(plan['tasks']),errors=len(plan['inventory_errors'])),ensure_ascii=False),flush=True)
    if args.inventory_only:return
    state=json.loads(statuspath.read_text()) if statuspath.exists() else dict(items={})
    state.update(status='running',pid=os.getpid(),total=len(plan['tasks']),started_at=time.time())
    save(statuspath,state)
    try:
        os.environ['LAKES_DEM_REQUIRE_SHARED_SOURCE']='1'
        state['phase']='checking_and_downloading_unique_dem_tiles';save(statuspath,state)
        dem=prepare_dem_sources(args,root,plan['tasks'])
        if args.dem_only:
            source_plan=json.loads((root/'dem_plan.json').read_text())
            state.update(mode='dem_only',count_unit='DEM_source_tiles',phase='dem_sources_complete',
                status='complete' if dem['status']=='complete' and not source_plan['errors'] and not plan['inventory_errors'] else 'partial',
                total=source_plan['required_source_tiles'],ready=source_plan['fully_cached_source_tiles']+dem['ready'],
                failed=dem['failed'],source_grids=len(plan['tasks']),slope_cache_generation=False,
                dem_downloads=dict(total=dem['total'],ready=dem['ready'],failed=dem['failed']))
            return
        state.update(phase='aligning_unique_grids',dem_tiles=dict(total=dem['total'],ready=dem['ready'],failed=dem['failed']));save(statuspath,state)
        for round_index in range(args.rounds):
            for index,task in enumerate(plan['tasks']):
                key=str(index);old=state['items'].get(key,{})
                try:cached=task_cached(task)
                except Exception:cached=False
                if cached:
                    state['items'][key]=dict(status='ready',reused_cache=True,sites=task['sites'],kind=task['kind'],updated_at=time.time())
                    state['ready']=sum(x['status']=='ready' for x in state['items'].values())
                    state['failed']=sum(x['status']=='failed' for x in state['items'].values())
                    save(statuspath,state)
                    print(f'GRID_CACHE ready={state["ready"]}/{state["total"]}',flush=True)
                    continue
                free=shutil.disk_usage(PROJECT_ROOT/'data').free/1024**3
                if free<args.min_free_gb:raise RuntimeError(f'Disk below reserve: {free:.1f} GiB')
                taskfile=root/'tasks'/f'{index}.json';save(taskfile,task)
                log=root/'items'/f'{index}.log';log.parent.mkdir(exist_ok=True)
                state['current']=dict(index=index,sites=task['sites'],round=round_index+1)
                save(statuspath,state)
                print(f'ITEM {index+1}/{len(plan["tasks"])} round={round_index+1} {task["sites"][:3]}',flush=True)
                with log.open('a') as handle:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker-task',str(taskfile)],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=child.wait(timeout=args.timeout)
                    except BaseException as error:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                        if not isinstance(error,subprocess.TimeoutExpired):raise
                        code=124
                state['items'][key]=dict(status='ready' if code==0 else 'failed',code=code,attempts=old.get('attempts',0)+1,sites=task['sites'],kind=task['kind'],updated_at=time.time(),log=str(log))
                state['ready']=sum(x['status']=='ready' for x in state['items'].values())
                state['failed']=sum(x['status']=='failed' for x in state['items'].values())
                save(statuspath,state)
                print(f'PROGRESS ready={state["ready"]}/{state["total"]} failed={state["failed"]} code={code}',flush=True)
            if state.get('ready',0)==state['total']:break
        state['status']='complete' if state.get('ready',0)==state['total'] and not plan['inventory_errors'] else 'partial'
    except BaseException as error:
        state.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        state['updated_at']=time.time();save(statuspath,state)
    print('FINAL '+json.dumps({k:v for k,v in state.items() if k!='items'},ensure_ascii=False),flush=True)



def execute(command, log, timeout=None):
    with log.open('a') as handle:
        child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:return child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise


def main():
    parser=argparse.ArgumentParser(description='逐省扫描、逐省下载和计算坡度，保留可用缓存')
    parser.add_argument('--job-dir',type=Path,default=PROJECT_ROOT/'outputs/all_site_slope_grids_20261030')
    parser.add_argument('--worker-task',type=Path)
    parser.add_argument('--region-worker',help=argparse.SUPPRESS)
    parser.add_argument('--regions',nargs='+',help='默认处理全部配置省份；可指定省份及顺序')
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--dem-only',action='store_true',help='仅下载校验共享DEM源数据，不生成坡度缓存')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--min-free-gb',type=int,default=50)
    args=parser.parse_args()
    if args.worker_task:
        worker(json.loads(args.worker_task.read_text()));return
    if args.region_worker:
        run_region(args);return
    from lake_workbench.regions.config import load_region_configs
    from lake_workbench.workspaces.file_lock import process_lock
    regions,_=load_region_configs()
    selected=args.regions or list(regions)
    if any(key not in regions for key in selected):parser.error('未知省份')
    selected=list(dict.fromkeys(selected))
    root=args.job_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    with process_lock(PROJECT_ROOT/'data/terrain/.all_site_batch.lock',blocking=False):
        state=dict(status='running',mode='dem_only' if args.dem_only else 'slope_grids',pid=os.getpid(),province_order=selected,provinces={},started_at=time.time())
        try:
            for key in selected:
                if shutil.disk_usage(PROJECT_ROOT/'data').free/1024**3 < args.min_free_gb:
                    raise RuntimeError('Disk below reserved free space')
                state['current_province']=key;save(root/'status.json',state)
                print('PROVINCE_START '+key,flush=True)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job-dir',str(root),
                    '--region-worker',key,'--rounds',str(args.rounds),'--timeout',str(args.timeout),'--min-free-gb',str(args.min_free_gb)]
                if args.inventory_only:command.append('--inventory-only')
                if args.dem_only:command.append('--dem-only')
                log=root/(key+'.log')
                code=execute(command,log)
                statusfile=root/'regions'/key/'status.json'
                region_state=json.loads(statusfile.read_text()) if statusfile.exists() else {}
                state['provinces'][key]=dict(mode=region_state.get('mode','slope_grids'),count_unit=region_state.get('count_unit','slope_grids'),code=code,status=region_state.get('status','inventory_complete' if args.inventory_only and code==0 else 'failed'),
                    ready=region_state.get('ready',0),failed=region_state.get('failed',0),total=region_state.get('total',0),log=str(log))
                save(root/'status.json',state)
                print('PROVINCE_DONE '+key+' '+json.dumps(state['provinces'][key],ensure_ascii=False),flush=True)
            state['status']='complete' if all(p['status'] in {'complete','inventory_complete'} for p in state['provinces'].values()) else 'partial'
        except BaseException as error:
            state.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
        finally:
            state['updated_at']=time.time();save(root/'status.json',state)
        print('FINAL '+json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
