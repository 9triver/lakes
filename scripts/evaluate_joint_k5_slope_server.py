"""Frozen global versus K5 routed systems; original and geographic single-count holdouts."""
import fcntl
import json
import os
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree
from torch.utils.data import DataLoader

import run_joint_k5_slope_server as training
from lake_workbench.models.runtime import build_model
from lake_workbench.training.unet.data import Normalization

ROOT=training.ROOT
TRAIN=training.OUT
OUT=ROOT/'outputs/guizhou_hunan_joint_k5_evaluation_20261106'
FIELDS=['tp','fp','fn','tn','scored','valid','unknown','unknown_predicted']


def measure(pred,truth,valid):
    known=valid&(truth!=255);water=truth==1;unknown=valid&(truth==255)
    return dict(tp=int((pred&water&known).sum()),fp=int((pred&~water&known).sum()),
                fn=int((~pred&water&known).sum()),tn=int((~pred&~water&known).sum()),
                scored=int(known.sum()),valid=int(valid.sum()),unknown=int(unknown.sum()),
                unknown_predicted=int((pred&unknown).sum()))


def metric(c):
    ratio=lambda a,b:a/b if b else None
    return dict(c,iou=ratio(c['tp'],c['tp']+c['fp']+c['fn']),
                precision=ratio(c['tp'],c['tp']+c['fp']),recall=ratio(c['tp'],c['tp']+c['fn']),
                f1=ratio(2*c['tp'],2*c['tp']+c['fp']+c['fn']),
                unknown_km2=c['unknown']*.0001,unknown_predicted_km2=c['unknown_predicted']*.0001,
                unknown_predicted_fraction=ratio(c['unknown_predicted'],c['unknown']),
                total_predicted_km2=(c['tp']+c['fp']+c['unknown_predicted'])*.0001,
                valid_km2=c['valid']*.0001)


def ownership(split,rows):
    """Native 10m grids; latest valid acquisition first then ID, independent of labels."""
    packed={};region={};metadata=[];features=[]
    for province in ['guizhou','hunan']:
        rr=[r for r in rows if r['province']==province]
        crss={r['crs'] for r in rr};assert len(crss)==1
        crs=next(iter(crss))
        gs=[box(*[float(r['bounds_'+n]) for n in ['left','bottom','right','top']]) for r in rr]
        for g in gs:
            assert all(abs(v/10-round(v/10))<1e-7 for v in g.bounds)
            assert abs(g.bounds[2]-g.bounds[0]-2560)<1e-7 and abs(g.bounds[3]-g.bounds[1]-2560)<1e-7
        tree=STRtree(gs);parent=list(range(len(rr)))
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        overlap=touch=0
        for i,g in enumerate(gs):
            for jj in tree.query(g,predicate='intersects'):
                j=int(jj)
                if j<=i:continue
                if g.intersection(gs[j]).area>1e-6:overlap+=1
                else:touch+=1
                parent[find(j)]=find(i)
        components=defaultdict(list)
        for i in range(len(rr)):components[find(i)].append(i)
        groups=sorted(components.values(),key=lambda ids:min(rr[i]['patch_id'] for i in ids))
        original=unique=scored=unknown=water=0
        for number,ids in enumerate(groups,1):
            group=f'{province}_{split}_merged_{number:03d}'
            # Sparse blocks avoid allocating the bounding rectangle of large disconnected-looking groups.
            claimed={}
            order=sorted(ids,key=lambda i:rr[i]['patch_id'])
            order.sort(key=lambda i:rr[i].get('product_date',''),reverse=True)
            for i in order:
                r=rr[i]
                with np.load(training.mapped(r['npz_path']),allow_pickle=False) as d:
                    valid=d['valid'].astype(bool);truth=d['mask']
                x=round(float(r['bounds_left'])/10);y=round(-float(r['bounds_top'])/10)
                owned=np.zeros((256,256),bool)
                for bx in range(x//256,(x+255)//256+1):
                    for by in range(y//256,(y+255)//256+1):
                        block=claimed.setdefault((bx,by),np.zeros((256,256),bool))
                        x0=max(x,bx*256);x1=min(x+256,(bx+1)*256)
                        y0=max(y,by*256);y1=min(y+256,(by+1)*256)
                        s=np.s_[y0-y:y1-y,x0-x:x1-x]
                        b=np.s_[y0-by*256:y1-by*256,x0-bx*256:x1-bx*256]
                        owned[s]=valid[s]&~block[b]
                        block[b]|=owned[s]
                pid=r['patch_id'];packed[pid]=np.packbits(owned.ravel());region[pid]=group
                original+=int(valid.sum());unique+=int(owned.sum())
                scored+=int((owned&(truth!=255)).sum());unknown+=int((owned&(truth==255)).sum())
                water+=int((owned&(truth==1)).sum())
            features.append(dict(type='Feature',geometry=mapping(unary_union([gs[i] for i in ids])),
                                 properties=dict(group=group,patches=len(ids),crs=crs,sites=sorted({rr[i]['site_id'] for i in ids}))))
        assert unique<=original and scored+unknown==unique
        metadata.append(dict(split=split,province=province,source_patches=len(rr),merged_regions=len(groups),
                             positive_overlap_pairs=overlap,touching_pairs=touch,crs=crs,
                             original_valid_pixels=original,unique_valid_pixels=unique,
                             unique_scored_pixels=scored,unique_unknown_pixels=unknown,unique_water_pixels=water))
    np.savez_compressed(OUT/f'{split}_ownership.npz',**packed)
    training.dump(OUT/f'{split}_ownership.json',dict(metadata=metadata,patch_regions=region,
                  ownership='Latest product_date then lexical patch_id among valid pixels; no label preference; cross-date geographic union'))
    # Each feature explicitly states its native CRS; not a single-CRS GeoJSON layer.
    training.dump(OUT/f'{split}_merged_native_geometries.json',features)
    return packed,region,metadata


class EvaluationPatches(training.StreamPatches):
    def __init__(self,rows,entries,norm,owned):
        super().__init__(rows,entries,norm)
        self.owned=owned

    def __getitem__(self,i):
        result=super().__getitem__(i)
        result['owned']=torch.from_numpy(np.unpackbits(self.owned[self.rows[i]['patch_id']],count=65536).reshape(256,256).astype(bool))
        return result


def visual(split,province,ranked,byid,entries,predictions,owned):
    selected=[];sites=set()
    for item in sorted(ranked,key=lambda r:r['global_only']+r['route_only'],reverse=True):
        row=byid[item['patch_id']]
        if row['province']!=province or row['site_id'] in sites:continue
        if item['global_only']+item['route_only']==0:continue
        selected.append(item);sites.add(row['site_id'])
        if len(selected)==4:break
    if not selected:return
    cfg=json.loads((training.SOURCE/'display_config.json').read_text())
    low,high=np.array(cfg['low']),np.array(cfg['high'])
    canvas=Image.new('RGB',(1280,len(selected)*292),'white');draw=ImageDraw.Draw(canvas)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,13) if Path(fontpath).exists() else ImageFont.load_default()
    for number,item in enumerate(selected):
        pid=item['patch_id'];row=byid[pid]
        image,y,valid=training.load_patch(row,entries)
        rgb=(np.clip((image[:3].transpose(1,2,0)-low)/np.maximum(high-low,1),0,1)*255).astype('uint8')
        rgb[~valid]=235
        own=np.unpackbits(owned[pid],count=65536).reshape(256,256).astype(bool)
        a=np.unpackbits(predictions['global'][pid],count=65536).reshape(256,256).astype(bool)
        b=np.unpackbits(predictions['k5_routed'][pid],count=65536).reshape(256,256).astype(bool)
        u=own&(y==255)
        label=rgb.copy();label[valid&(y==255)]=(label[valid&(y==255)]*.4+np.array([180,70,220])*.6).astype('uint8')
        label[valid&(y==1)]=[0,180,255]
        overlays=[]
        for pred in [a,b]:
            overlay=rgb.copy();m=pred&valid
            overlay[m]=(overlay[m]*.4+np.array([0,255,255])*.6).astype('uint8');overlays.append(overlay)
        diff=rgb.copy();diff[u&a&~b]=[255,140,0];diff[u&~a&b]=[230,0,210];diff[u&a&b]=[0,190,245]
        draw.text((4,number*292),f'{row["site_id"]} cluster={row["route_cluster"]} | unique255: global-only={item["global_only"]} route-only={item["route_only"]}',fill='black',font=font)
        for col,(arr,title) in enumerate(zip([rgb,label,*overlays,diff],['RGB','Seed1 blue /255 purple','Global cyan','K5 routed cyan','255: orange global/pink K5'])):
            draw.text((col*256+3,number*292+21),title,fill='black',font=font)
            canvas.paste(Image.fromarray(arr).resize((244,244)),(col*256+3,number*292+43))
    canvas.save(OUT/f'{province}_{split}_255_comparison.png')


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((TRAIN/'queue_status.json').read_text())['status']=='training_completed'
    torch.set_num_threads(2);device=torch.device('cuda');assert torch.cuda.is_available()
    protocol=json.loads((TRAIN/'protocol.json').read_text())
    norm=Normalization(np.array(protocol['normalization']['mean'],dtype='float32'),np.array(protocol['normalization']['std'],dtype='float32'))
    router=joblib.load(protocol['router']['path']);assert training.sha(protocol['router']['path'])==protocol['router']['sha256']
    assert router['features']==training.NAMES
    entries={}
    for p in training.SLOPES.values():entries.update(json.loads((ROOT/p).read_text())['patches'])
    models={};weights=[]
    for job in protocol['jobs']:
        name=job['name'];folder=TRAIN/name/'rgb_nir_slope/runs/seed_42/joint_C_D_k5'
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        assert state['config']['normalization']==protocol['normalization'] and state['config']['pos_weight']==protocol['pos_weight']
        assert state['config']['input_bands']==training.BANDS
        model=build_model('unet',5,{'base_channels':32}).to(device)
        model.load_state_dict(state['model_state']);model.eval()
        key=-1 if name=='global_control' else int(name[-1]);models[key]=model
        weights.append(dict(job=name,epoch=state['epoch'],sha256=training.sha(folder/'best.pt')))
        del state
    training.dump(OUT/'protocol.json',dict(checkpoints=weights,threshold=.5,router=protocol['router'],training=False,
                  checkpoint_selection='Frozen original max full validation IoU; no reselection',
                  inference='Original full256 patch input; hard route by frozen features; valid ratio<0.8 falls back global. No sliding-window blending.',
                  ownership='Latest date first, stable patch ID; geographic union counts once across acquisitions, retains one original context',
                  caveat='Automatic 0/1 seeds only;255 unknown. Validation used for checkpoint selection; test sets historically explored. No threshold tuning.',
                  training_protocol_sha256=training.sha(TRAIN/'protocol.json')))
    cache={r['patch_id']:r for r in map(json.loads,(TRAIN/'preflight_features.jsonl').read_text().splitlines())}
    summaries=[];patchstats=[];routestats=[];differences=[];allmetadata=[]
    for split in ['validation','test']:
        rows=training.read(TRAIN/'splits'/f'{split}.csv');byid={r['patch_id']:r for r in rows}
        training.dump(OUT/'status.json',dict(status='features_and_ownership',split=split,pid=os.getpid()))
        if split=='test':
            with ThreadPoolExecutor(max_workers=4) as pool:
                for i,record in enumerate(pool.map(training.inspect,[(r,entries[r['patch_id']],'evaluation') for r in rows]),1):
                    cache[record['patch_id']]=record
                    if i%500==0:print('TEST FEATURES',i,'/',len(rows),flush=True)
            training.table(OUT/'test_features.csv',[cache[r['patch_id']] for r in rows])
        x=np.array([[cache[r['patch_id']][n] for n in training.NAMES] for r in rows])
        labels=router['kmeans'].predict(router['scaler'].transform(x))
        for r,label in zip(rows,labels):r['route_cluster']=int(label) if cache[r['patch_id']]['valid_ratio']>=.8 else -1
        owned,regions,meta=ownership(split,rows);allmetadata.extend(meta)
        print('OWNERSHIP',split,json.dumps(meta),flush=True)
        batches=DataLoader(EvaluationPatches(rows,entries,norm,owned),batch_size=8,num_workers=2,
                           pin_memory=True,shuffle=False)
        aggregates=defaultdict(lambda:Counter({f:0 for f in FIELDS}))
        regiontotals=defaultdict(lambda:Counter({f:0 for f in FIELDS}))
        routecounts=defaultdict(Counter);disagreements=defaultdict(Counter)
        predictions={s:{} for s in ['global','k5_routed']};ranked=[]
        with torch.inference_mode():
            for step,b in enumerate(batches,1):
                images=b['image'].to(device,non_blocking=True)
                probs=models[-1](images).squeeze(1).sigmoid()
                globalpred=(probs>=.5).cpu().numpy()
                routedpred=globalpred.copy()
                for c in range(5):
                    indices=torch.nonzero(b['cluster']==c).flatten()
                    if len(indices):
                        pred=(models[c](images[indices.to(device)]).squeeze(1).sigmoid()>=.5).cpu().numpy()
                        routedpred[indices.numpy()]=pred
                for i in range(len(globalpred)):
                    pid=b['patch_id'][i];r=byid[pid];province=r['province'];cluster=r['route_cluster']
                    truth=b['mask'][i].numpy();valid=b['valid'][i].numpy().astype(bool);own=b['owned'][i].numpy().astype(bool)
                    assert not (own&~valid).any()
                    a,bpred=globalpred[i],routedpred[i]
                    for system,pred in [('global',a),('k5_routed',bpred)]:
                        predictions[system][pid]=np.packbits(pred.ravel())
                        for counting,v in [('patch',valid),('unique',own)]:
                            values=measure(pred,truth,v)
                            for scope in [province,'combined']:
                                aggregates[system,counting,scope].update(values)
                            if counting=='unique':regiontotals[system,province,regions[pid]].update(values)
                            patchstats.append(dict(split=split,province=province,patch_id=pid,site_id=r['site_id'],
                                                   merged_region=regions[pid],cluster=cluster,system=system,counting=counting,**values))
                    u=own&(truth==255)
                    item=dict(patch_id=pid,global_only=int((u&a&~bpred).sum()),route_only=int((u&~a&bpred).sum()),
                              both_water=int((u&a&bpred).sum()),both_nonwater=int((u&~a&~bpred).sum()))
                    ranked.append(item)
                    for scope in [province,'combined']:
                        disagreements[scope].update({k:v for k,v in item.items() if k!='patch_id'})
                    routecounts[province,cluster].update(patches=1,owned_valid_pixels=int(own.sum()),
                        owned_scored_pixels=int((own&(truth!=255)).sum()),owned_unknown_pixels=int(u.sum()))
                if step==1 or step%50==0 or step==len(batches):
                    print('INFERENCE',split,step,'/',len(batches),'seconds',round(time.time()-started,1),flush=True)
                    training.dump(OUT/'status.json',dict(status='inference',split=split,batch=step,batches=len(batches),pid=os.getpid()))
        for (system,counting,scope),values in aggregates.items():
            if counting=='unique':
                mm=[m for m in meta if scope=='combined' or m['province']==scope]
                assert values['valid']==sum(m['unique_valid_pixels'] for m in mm)
                assert values['scored']==sum(m['unique_scored_pixels'] for m in mm)
                assert values['unknown']==sum(m['unique_unknown_pixels'] for m in mm)
                assert values['tp']+values['fn']==sum(m['unique_water_pixels'] for m in mm)
            summaries.append(dict(split=split,system=system,counting=counting,scope=scope,**metric(dict(values))))
        for (province,c),values in routecounts.items():
            routestats.append(dict(split=split,province=province,cluster=c,role='fallback_global' if c<0 else f'expert_C{c}',
                                  unique_influence_km2=values['owned_valid_pixels']*.0001,**dict(values)))
        for scope,values in disagreements.items():differences.append(dict(split=split,scope=scope,**dict(values)))
        for system,pp in predictions.items():np.savez_compressed(OUT/f'{split}_{system}_binary.npz',**pp)
        for province in ['guizhou','hunan']:visual(split,province,ranked,byid,entries,predictions,owned)
        regionrows=[dict(split=split,system=s,province=p,merged_region=g,**metric(dict(v))) for (s,p,g),v in regiontotals.items()]
        training.table(OUT/f'{split}_per_merged_region.csv',regionrows)
        training.table(OUT/f'{split}_unknown_disagreements.csv',[dict(split=split,province=byid[r['patch_id']]['province'],**r) for r in ranked])
        # Sanity check: frozen global validation prediction must reproduce training's pooled counts.
        if split=='validation':
            history=json.loads((TRAIN/'global_control/rgb_nir_slope/runs/seed_42/joint_C_D_k5/history.json').read_text())
            best=max(history,key=lambda h:h['validation']['iou'])['validation']
            actual=aggregates['global','patch','combined']
            assert all(actual[k]==best[k] for k in ['tp','fp','fn']), 'Global validation counts mismatch training'
    training.dump(OUT/'comparison.json',summaries)
    training.table(OUT/'summary.csv',summaries);training.table(OUT/'per_patch.csv',patchstats)
    training.table(OUT/'routing_distribution.csv',routestats);training.dump(OUT/'unknown_disagreements.json',differences)
    training.dump(OUT/'ownership_metadata.json',allmetadata)
    lines=['# 贵州C＋湖南D联合K5：全局基准与自动路由评测','',
           '冻结六个最高原验证IoU权重；R/G/B/NIR/SLOPE；阈值0.5。验证集曾用于权重选择。',
           'unique为每个原生10m地理像元固定选择一个有效来源：日期新者优先，同日期按patch ID。跨日期不重复计数。',
           '完整原patch输入推理，固定归属像元计分；不是整site滑窗概率融合。255没有独立真值。','',
           '| 集合 | 省份 | 系统 | 去重IoU | Precision | Recall | 255预测km² | 255预测比例 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['counting']=='unique':
            lines.append(f'| {r["split"]} | {r["scope"]} | {r["system"]} | {r["iou"]:.4%} | {r["precision"]:.4%} | {r["recall"]:.4%} | {r["unknown_predicted_km2"]:.4f} | {r["unknown_predicted_fraction"]:.4%} |')
    lines+=['','## 路由分布','', '| 集合 | 省份 | 类别 | patch | 去重影响km² |','|---|---|---|---:|---:|']
    for r in routestats:lines.append(f'| {r["split"]} | {r["province"]} | {r["role"]} | {r["patches"]} | {r["unique_influence_km2"]:.4f} |')
    lines+=['','## 255分歧图','', '按去重255分歧面积选择不同site窗口，属于定向诊断，不能代表所有样本。青色为预测水体；分歧图橙色仅全局判水、粉色仅路由判水。面积减少不等于准确率提高。']
    for split in ['validation','test']:
        for province in ['guizhou','hunan']:
            p=OUT/f'{province}_{split}_255_comparison.png'
            if p.exists():lines+=['',f'![{province} {split}]({p.name})']
    (OUT/'report.md').write_text('\n'.join(lines))
    training.dump(OUT/'status.json',dict(status='completed',seconds=time.time()-started,checkpoints=6,
                                       global_validation_reproduced=True,training=False,threshold=.5))
    print('COMPLETE',round(time.time()-started,1),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        training.dump(OUT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        raise
