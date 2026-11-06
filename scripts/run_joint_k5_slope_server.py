"""Isolated CUDA queue: joint Guizhou C/Hunan D global control + frozen K5 experts."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from torch.utils.data import Dataset, DataLoader

from lake_workbench.models.runtime import build_model
from lake_workbench.training.unet.data import Normalization, augment_patch
from lake_workbench.training.unet.engine import run_epoch, save_checkpoint, seed_everything, metrics_from_totals

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/guizhou_hunan_train_slope_kmeans_20261106'
OUT = ROOT/'outputs/guizhou_hunan_joint_k5_rgb_nir_slope_seed42_lr001_batch8_20261106'
SOURCES = {
    'guizhou': 'outputs/guizhou_spatial_C_75_13_12_20261026',
    'hunan': 'outputs/hunan_spatial_D_75_13_12_buffer2560_seed42_20261102',
}
SLOPES = {
    'guizhou': 'outputs/guizhou_C_slope_features_20261026/index.json',
    'hunan': 'outputs/hunan_C_encoder_rgb_nir_slope_seed42_lr001_batch8_20261030/slope/index.json',
}
BANDS = ['red', 'green', 'blue', 'nir', 'slope_degrees']
NAMES = [f'{b}_{s}' for b in BANDS for s in ['mean', 'std', 'q25', 'q50', 'q75']]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def mapped(value):
    value = str(value)
    for prefix in ['/Users/cm/projects/yan01/lakes/', '/home/lake/lakes/']:
        if value.startswith(prefix):
            return ROOT/value[len(prefix):]
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def geometry(row):
    l, b, r, t = [float(row['bounds_'+n]) for n in ['left', 'bottom', 'right', 'top']]
    if row['crs'] == 'EPSG:32649':
        return box(l, b, r, t)
    tr = Transformer.from_crs(row['crs'], 'EPSG:32649', always_xy=True)
    xx = np.r_[np.linspace(l, r, 21), np.full(21, r), np.linspace(r, l, 21), np.full(21, l)]
    yy = np.r_[np.full(21, b), np.linspace(b, t, 21), np.full(21, t), np.linspace(t, b, 21)]
    x, y = tr.transform(xx, yy)
    result = Polygon(zip(x, y))
    assert result.is_valid
    return result


def spatial_audit(splits):
    gs = {s: [geometry(r) for r in rr] for s, rr in splits.items()}
    counts = {}
    examples = []
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        ids_a = {r['patch_id'] for r in splits[a]}
        ids_b = {r['patch_id'] for r in splits[b]}
        sites_a = {r['site_id'] for r in splits[a]}
        sites_b = {r['site_id'] for r in splits[b]}
        assert ids_a.isdisjoint(ids_b), f'{a}/{b} patch identity leakage'
        assert sites_a.isdisjoint(sites_b), f'{a}/{b} site identity leakage'
        tree = STRtree(gs[b])
        count = 0
        for i, g in enumerate(gs[a]):
            for j in tree.query(g, predicate='intersects'):
                area = g.intersection(gs[b][int(j)]).area
                if area > 1e-6:
                    count += 1
                    if len(examples) < 30:
                        examples.append(dict(a=a, b=b, patch_a=splits[a][i]['patch_id'],
                                             patch_b=splits[b][int(j)]['patch_id'], area_m2=area))
        counts[a+'__'+b] = count
    audit = dict(common_crs='EPSG:32649', boundary_edge_samples=21,
                 positive_intersection_tolerance_m2=1e-6, counts=counts, examples=examples)
    dump(OUT/'spatial_audit.json', audit)
    assert not any(counts.values()), 'Cross-split spatial leakage; see spatial_audit.json'


def load_patch(row, entries):
    with np.load(mapped(row['npz_path']), allow_pickle=False) as d:
        raw, mask, valid = d['image'], d['mask'], d['valid'].astype(bool)
    slope = np.load(mapped(entries[row['patch_id']]['path']), allow_pickle=False)
    image = np.concatenate([raw[[2, 1, 0, 3]].astype('float32'), slope[None].astype('float32')])
    return image, mask, valid


def inspect(pair):
    row, entry, split = pair
    pid = row['patch_id']
    pp, sp = mapped(row['npz_path']), mapped(entry['path'])
    assert pp.is_file() and sp.is_file(), pid+' missing file'
    assert entry['crs'] == row['crs'], pid+' CRS mismatch'
    assert np.allclose(entry['bounds'], [float(row['bounds_'+n]) for n in ['left','bottom','right','top']])
    assert sha(sp) == entry['sha256'], pid+' slope checksum'
    slope = np.load(sp, allow_pickle=False)
    assert slope.shape == (256,256) and np.isfinite(slope).all()
    assert slope.min() >= 0 and slope.max() <= 90
    with np.load(pp, allow_pickle=False) as d:
        assert {'image','mask','valid'}.issubset(d.files)
        raw = d['image']
        assert raw.shape == (5,256,256) and np.isfinite(raw).all()
        if split == 'test':
            # Only image/integrity metadata: no test mask/valid arrays or statistics read.
            return dict(patch_id=pid, province=row['province'], split=split, integrity='ok')
        mask, valid = d['mask'], d['valid'].astype(bool)
    assert mask.shape == valid.shape == (256,256)
    assert set(np.unique(mask)).issubset({0,1,255})
    assert int(valid.sum()) == int(row['valid_pixels'])
    assert int((mask==1).sum()) == int(row['water_pixels'])
    assert hashlib.sha256(mask.tobytes()).hexdigest() == entry['mask_sha256']
    assert hashlib.sha256(valid.tobytes()).hexdigest() == entry['valid_sha256']
    assert valid.any()
    image = np.concatenate([raw[[2,1,0,3]][:,valid].astype('float64'), slope[valid][None].astype('float64')])
    means, stds = image.mean(1), image.std(1)
    q = np.quantile(image, [.25,.5,.75], axis=1, method='linear')
    result = dict(patch_id=pid, province=row['province'], split=split,
                  valid_pixels=int(valid.sum()), valid_ratio=float(valid.mean()),
                  count_0=int((valid&(mask==0)).sum()), count_1=int((valid&(mask==1)).sum()),
                  count_255=int((valid&(mask==255)).sum()), npz_sha256=sha(pp), slope_sha256=sha(sp))
    result.update(zip(NAMES, np.column_stack([means,stds,q.T]).ravel().tolist()))
    return result


def prepare():
    splits = {s: [] for s in ['train','validation','test']}
    manifests = {}
    entries = {}
    for province, base in SOURCES.items():
        indexfile = ROOT/SLOPES[province]
        index = json.loads(indexfile.read_text())['patches']
        for split in splits:
            path = SOURCE/'inputs'/f'{province}_train.csv' if split=='train' else ROOT/base/'splits'/f'{split}.csv'
            if not path.exists():
                path = OUT/'inputs'/f'{province}_{split}.csv'
            manifests[province+'_'+split] = dict(path=str(path),sha256=sha(path))
            rr = read(path)
            for row in rr:
                assert row['spatial_split'] == split
                row['province'] = province
                entries[row['patch_id']] = index[row['patch_id']]
            splits[split].extend(rr)
    assignments = {r['patch_id']:int(r['cluster']) for r in read(SOURCE/'cluster_assignments.csv') if int(r['k'])==5}
    assert len(assignments)==18516
    splits['train'] = [r for r in splits['train'] if r['patch_id'] in assignments]
    assert {r['patch_id'] for r in splits['train']} == set(assignments)
    assert [len(splits[s]) for s in ['train','validation','test']] == [18516,3692,3460]
    print('SPATIAL AUDIT', flush=True)
    spatial_audit(splits)
    routerpath = SOURCE/'rgb_nir_slope_joint_balanced_k5.joblib'
    bundle = joblib.load(routerpath)
    assert bundle['features'] == NAMES and bundle['kmeans'].n_clusters == 5
    featurepath = OUT/'preflight_features.jsonl'
    previous = {}
    if featurepath.exists():
        for line in featurepath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); previous[r['patch_id']] = r
    # Partial cached checks are reused only if the source provenance is unchanged.
    provenance = dict(manifests=manifests, router_sha256=sha(routerpath),
                      assignments_sha256=sha(SOURCE/'cluster_assignments.csv'),
                      slope_indexes={p:sha(ROOT/v) for p,v in SLOPES.items()}, script_sha256=sha(Path(__file__)))
    if (OUT/'provenance.json').exists():
        assert json.loads((OUT/'provenance.json').read_text()) == provenance
    dump(OUT/'provenance.json', provenance)
    pending = [(r,entries[r['patch_id']],s) for s,rr in splits.items() for r in rr if r['patch_id'] not in previous]
    print(f'PREFLIGHT total={sum(map(len,splits.values()))} cached={len(previous)} pending={len(pending)}', flush=True)
    with featurepath.open('a') as handle, ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(inspect,pending),1):
            previous[record['patch_id']] = record
            handle.write(json.dumps(record)+'\n')
            if i%500==0 or i==len(pending):
                handle.flush()
                dump(OUT/'queue_status.json', dict(status='preflight',checked=len(previous),total=sum(map(len,splits.values())),pid=os.getpid()))
                print(f'PREFLIGHT {len(previous)}/{sum(map(len,splits.values()))}',flush=True)
    for split in ['train','validation']:
        rr = splits[split]
        features = [previous[r['patch_id']] for r in rr]
        x = np.array([[f[n] for n in NAMES] for f in features])
        labels = bundle['kmeans'].predict(bundle['scaler'].transform(x))
        for row,f,label in zip(rr,features,labels):
            row['route_cluster'] = int(label) if f['valid_ratio']>=.8 else -1
            if split=='train':
                assert f['valid_ratio']>=.8 and int(label)==assignments[row['patch_id']]
    trainfeatures = [previous[r['patch_id']] for r in splits['train']]
    weights = np.array([f['valid_pixels'] for f in trainfeatures],dtype='float64')
    means = np.array([[f[b+'_mean'] for b in BANDS] for f in trainfeatures])
    stds = np.array([[f[b+'_std'] for b in BANDS] for f in trainfeatures])
    mean = np.average(means,axis=0,weights=weights)
    variance = np.maximum(np.average(stds**2+means**2,axis=0,weights=weights)-mean**2,1e-6)
    norm = Normalization(mean.astype('float32'),np.sqrt(variance).astype('float32'))
    neg = sum(f['count_0'] for f in trainfeatures)
    pos = sum(f['count_1'] for f in trainfeatures)
    positive = float(min(max(neg/pos,1.),50.)) if pos else 1.
    jobs = [('global_control',splits['train'])]
    for c in [3,4,1,0,2]:
        jobs.append((f'k5_expert_cluster{c}',[r for r in splits['train'] if r['route_cluster']==c]))
    assert [len(rr) for _,rr in jobs] == [18516,14046,1675,1561,1138,96]
    for name,rr in jobs:
        table(OUT/'splits'/name/'train.csv',rr)
    for s in ['validation','test']:
        table(OUT/'splits'/f'{s}.csv',splits[s])
    valdist = Counter((r['province'],r['route_cluster']) for r in splits['validation'])
    validation_profile = []
    for province in SOURCES:
        for c in [-1,0,1,2,3,4]:
            ff = [previous[r['patch_id']] for r in splits['validation'] if r['province']==province and r['route_cluster']==c]
            validation_profile.append(dict(province=province,cluster=c,patches=valdist[province,c],
                                           count_0=sum(f['count_0'] for f in ff),count_1=sum(f['count_1'] for f in ff),
                                           count_255=sum(f['count_255'] for f in ff)))
    dump(OUT/'validation_cluster_profile.json',validation_profile)
    protocol = dict(experiment='贵州C＋湖南D · 联合K=5 · 原U-Net · 四光谱＋坡度',
                    region='guizhou_hunan',workspace_id='default',arm='joint_C_D_k5',
                    model='project U-Net',model_type='unet',model_options={'base_channels':32},base_channels=32,
                    input_bands=BANDS,input_indices=[2,1,0,3,5],in_channels=5,epochs=12,batch_size=8,
                    lr=.001,weight_decay=.0001,seed=42,seeds=[42],device='cuda',threshold=.5,
                    optimizer='AdamW betas=(0.9,0.999)',loss='weighted BCE; valid and mask!=255',
                    normalization=norm.to_json(),pos_weight=positive,
                    augmentation='flips and90degree rotations',scheduler='validation IoU max factor0.5 patience4',
                    checkpoint_selection=['max_validation_iou'],
                    validation_selection='same full pooled3692 patches for every model; perprovince/percluster diagnostics',
                    initialization='same seed42 optical baseline remapped RGB/NIR; slope input kernel zero; scratch',
                    router=dict(path=str(routerpath),sha256=sha(routerpath),features=NAMES,min_valid_ratio=.8),
                    shared_statistics=dict(train_only=True,train_patch_count=18516,negative_pixels=neg,positive_pixels=pos),
                    counts={s:len(rr) for s,rr in splits.items()},provenance=provenance,
                    jobs=[dict(name=n,train_count=len(rr),provinces=dict(Counter(r['province'] for r in rr))) for n,rr in jobs],
                    test_labels_read=False,streaming=True,num_workers=2,precision='float32')
    dump(OUT/'protocol.json',protocol)
    print('PREPARED '+json.dumps(protocol['jobs'],ensure_ascii=False),flush=True)
    return splits,entries,norm,positive,jobs,protocol


class StreamPatches(Dataset):
    def __init__(self,rows,entries,norm,augment=False):
        self.rows,self.entries,self.norm,self.augment=rows,entries,norm,augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        image,mask,valid=load_patch(row,self.entries)
        image=(image-self.norm.mean[:,None,None])/self.norm.std[:,None,None]
        if self.augment:
            image,mask,valid=augment_patch(image,mask,valid)
        return dict(image=torch.from_numpy(np.ascontiguousarray(image)),
                    mask=torch.from_numpy(np.ascontiguousarray(mask)).long(),
                    valid=torch.from_numpy(np.ascontiguousarray(valid)),
                    province=row['province'],cluster=int(row['route_cluster']),patch_id=row['patch_id'])


def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


def batches(rows,entries,norm,augment=False):
    return DataLoader(StreamPatches(rows,entries,norm,augment),batch_size=8,shuffle=augment,
                      num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,
                      worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(42))


def network():
    baseline=build_model('unet',5,{'base_channels':32})
    model=build_model('unet',5,{'base_channels':32})
    state=baseline.state_dict()
    key='down1.block.0.weight'
    original=state[key].clone()
    state[key].zero_()
    for target,source in enumerate([2,1,0,3]):
        state[key][:,target]=original[:,source]
    model.load_state_dict(state)
    return model


def total():
    return dict(loss=0.,valid=0,tp=0,fp=0,fn=0,unknown_pixels=0,unknown_water_pixels=0)


def validation(model,loader,device,weight):
    groups={'pooled':total()}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image=batch['image'].to(device,non_blocking=True)
            mask=batch['mask'].to(device,non_blocking=True)
            valid=batch['valid'].to(device,non_blocking=True)
            logits=model(image).squeeze(1)
            loss=F.binary_cross_entropy_with_logits(logits,mask.clamp(0,1).float(),pos_weight=weight,reduction='none')
            pred=torch.sigmoid(logits)>=.5
            known=valid&(mask!=255)
            unknown=valid&(mask==255)
            data=torch.stack([(loss*known).sum((1,2)),known.sum((1,2)),
                              (pred&(mask==1)&known).sum((1,2)),(pred&(mask==0)&known).sum((1,2)),
                              (~pred&(mask==1)&known).sum((1,2)),unknown.sum((1,2)),
                              (pred&unknown).sum((1,2))],1).cpu().numpy()
            for i,values in enumerate(data):
                keys=['pooled','province_'+batch['province'][i],f'cluster_{int(batch["cluster"][i])}']
                for key in keys:
                    dest=groups.setdefault(key,total())
                    for field,value in zip(dest,values):
                        dest[field]+=float(value) if field=='loss' else int(value)
    return {k:dict(metrics_from_totals(v),unknown_pixels=v['unknown_pixels'],
                   unknown_water_pixels=v['unknown_water_pixels'],
                   unknown_predicted_water_km2=v['unknown_water_pixels']*.0001) for k,v in groups.items()}


class Progress:
    def __init__(self,loader,name,epoch,completed):
        self.loader,self.name,self.epoch,self.completed=loader,name,epoch,completed

    def __iter__(self):
        for i,batch in enumerate(self.loader,1):
            if i==1 or i%100==0:
                dump(OUT/'queue_status.json',dict(status='training',job=self.name,epoch=self.epoch,
                     completed_epochs=self.epoch-1,batch=i,total_batches=len(self.loader),completed_jobs=self.completed,pid=os.getpid()))
            yield batch


def fit(name,rows,splits,entries,norm,positive,protocol,completed):
    folder=OUT/name/'rgb_nir_slope/runs/seed_42/joint_C_D_k5'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_done.json').exists():
        return
    seed_everything(42)
    model=network().to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
    device=torch.device('cuda')
    weight=torch.tensor([positive],dtype=torch.float32,device=device)
    trainloader=batches(rows,entries,norm,True)
    valloader=batches(splits['validation'],entries,norm)
    config=dict(protocol,job=name,training_sites=len({r['site_id'] for r in rows}),training_patch_count=len(rows),
                manifests={'train':str(OUT/'splits'/name/'train.csv'),'validation':str(OUT/'splits/validation.csv')})
    phash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['protocol_hash']=phash
    if (folder/'config.json').exists():
        assert json.loads((folder/'config.json').read_text())==config
    dump(folder/'config.json',config)
    history=[];best=-1.;elapsed=0.
    seed_everything(1042)
    resume=folder/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config']['protocol_hash']==phash
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state']);history=state['history'];elapsed=state['seconds']
        best=max(h['validation']['iou'] for h in history)
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        trainloader.generator.set_state(state['train_generator']);valloader.generator.set_state(state['val_generator'])
        del state
    else:
        # Independent smoke step; restore all numerical training state before the real fit.
        original=copy.deepcopy(model.state_dict())
        batch=next(iter(DataLoader(StreamPatches(rows[:8],entries,norm),batch_size=8,num_workers=0)))
        model.train();optimizer.zero_grad(set_to_none=True)
        image=batch['image'].to(device);mask=batch['mask'].to(device);valid=batch['valid'].to(device)&(mask!=255)
        logits=model(image).squeeze(1)
        raw=F.binary_cross_entropy_with_logits(logits,mask.clamp(0,1).float(),pos_weight=weight,reduction='none')
        loss=raw[valid].mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        model.load_state_dict(original)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999))
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
        dump(folder/'smoke.json',dict(loss=float(loss.detach()),cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                                     finite_gradients=True,restored_before_training=True))
        del original,batch,image,mask,valid,logits,raw,loss
        optimizer.zero_grad(set_to_none=True);seed_everything(1042)
    started=time.monotonic()
    for epoch in range(len(history)+1,13):
        epoch_start=time.monotonic()
        train=run_epoch(model,Progress(trainloader,name,epoch,completed),device,optimizer,weight,.5)
        dump(OUT/'queue_status.json',dict(status='validating',job=name,epoch=epoch,completed_jobs=completed,pid=os.getpid()))
        diag=validation(model,valloader,device,weight)
        val=diag['pooled']
        assert np.isfinite(train['loss']) and np.isfinite(val['loss'])
        scheduler.step(val['iou'])
        history.append(dict(epoch=epoch,train=train,validation=val,validation_diagnostics=diag,
                            lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-epoch_start))
        if val['iou']>best:
            save_checkpoint(folder/'best.pt.tmp',model,optimizer,epoch,config,norm,history)
            (folder/'best.pt.tmp').replace(folder/'best.pt');best=val['iou']
        state=dict(epoch=epoch,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                   config=config,normalization=norm.to_json(),history=history,scheduler_state=scheduler.state_dict(),
                   seconds=elapsed+time.monotonic()-started,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                   train_generator=trainloader.generator.get_state(),val_generator=valloader.generator.get_state(),
                   resume_note='worker augmentation RNG resets on process restart; resumed trajectory is not bitwise identical')
        torch.save(state,folder/'resume.pt.tmp');(folder/'resume.pt.tmp').replace(resume);del state
        dump(folder/'history.json',history)
        dump(OUT/'queue_status.json',dict(status='epoch_completed',job=name,completed_epochs=epoch,epochs=12,
             best_validation_iou=best,latest_validation=val,epoch_seconds=history[-1]['seconds'],
             completed_jobs=completed,pid=os.getpid()))
        print(f'{name} epoch={epoch}/12 train_iou={train["iou"]:.6f} val_iou={val["iou"]:.6f} seconds={history[-1]["seconds"]:.1f}',flush=True)
    dump(folder/'training_done.json',dict(protocol_hash=phash,epochs=12,seconds=elapsed+time.monotonic()-started,
                                         best_validation_iou=best,best_sha256=sha(folder/'best.pt')))
    resume.unlink(missing_ok=True)
    del model,optimizer,scheduler,trainloader,valloader
    import gc
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert torch.cuda.is_available(),'CUDA unavailable'
        torch.set_num_threads(2)
        dump(OUT/'queue_status.json',dict(status='preparing',pid=os.getpid()))
        splits,entries,norm,positive,jobs,protocol=prepare()
        if args.prepare_only:
            dump(OUT/'queue_status.json',dict(status='prepared',pid=os.getpid()))
            return
        completed=[]
        for name,rows in jobs:
            print(f'START {name} train={len(rows)} validation={len(splits["validation"])}',flush=True)
            fit(name,rows,splits,entries,norm,positive,protocol,completed)
            completed.append(name)
        dump(OUT/'queue_status.json',dict(status='training_completed',completed_jobs=completed,pid=os.getpid(),
             next='frozen routed system validation/test and255 diagnostics; training alone does not establish improvement'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        detail=traceback.format_exc()
        print(detail,flush=True)
        dump(OUT/'queue_status.json',dict(status='failed',error=str(error),traceback=detail,pid=os.getpid()))
        raise
