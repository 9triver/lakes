"""Compare existing Guizhou K2/K4 and Guizhou+Hunan K5 systems; inference only."""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lake_workbench.models.runtime import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/guizhou_vs_joint_frozen_evaluation_20261106'
JOINT = ROOT / 'outputs/guizhou_hunan_joint_k5_rgb_nir_slope_seed42_lr001_batch8_20261106'
CACHE = ROOT / 'outputs/guizhou_hunan_joint_k5_evaluation_20261106'
GZ4 = ROOT / 'outputs/guizhou_C_slope_k4_experts_server_seed42_lr001_batch8_20261105'
BANDS = ['red', 'green', 'blue', 'nir', 'slope_degrees']
NAMES = [f'{b}_{s}' for b in BANDS for s in ['mean', 'std', 'q25', 'q50', 'q75']]
FIELDS = ['tp', 'fp', 'fn', 'tn', 'scored', 'valid', 'unknown', 'unknown_predicted']
SYSTEMS = ['gz_global', 'gz_k2', 'gz_k4', 'joint_global', 'joint_k5']
LABELS = ['GZ global', 'GZ K2', 'GZ K4', 'GZ+HN global', 'GZ+HN K5']
STARTED = time.time()
AUDIT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped(path):
    value = str(path)
    for prefix in ['/Users/cm/projects/yan01/lakes/', '/home/lake/lakes/']:
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    return Path(value) if Path(value).is_absolute() else ROOT / value


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def status(**values):
    values.update(seconds=round(time.time() - STARTED, 1), pid=os.getpid(), training=False)
    dump(OUT / 'status.json', values)
    print(json.dumps(values), flush=True)


def measure(pred, truth, valid):
    known = valid & (truth != 255)
    water = truth == 1
    unknown = valid & (truth == 255)
    return dict(tp=int((pred & water & known).sum()), fp=int((pred & ~water & known).sum()),
                fn=int((~pred & water & known).sum()), tn=int((~pred & ~water & known).sum()),
                scored=int(known.sum()), valid=int(valid.sum()), unknown=int(unknown.sum()),
                unknown_predicted=int((pred & unknown).sum()))


def metrics(counts):
    c = {key: int(counts[key]) for key in FIELDS}
    ratio = lambda a, b: a / b if b else None
    return dict(c, iou=ratio(c['tp'], c['tp']+c['fp']+c['fn']),
                precision=ratio(c['tp'], c['tp']+c['fp']), recall=ratio(c['tp'], c['tp']+c['fn']),
                unknown_predicted_km2=c['unknown_predicted']*.0001,
                unknown_km2=c['unknown']*.0001,
                total_predicted_km2=(c['tp']+c['fp']+c['unknown_predicted'])*.0001)


def vectors(image, valid):
    p = image[:, valid].astype('float64')
    if not p.shape[1]:
        return np.zeros(25)
    q = np.quantile(p, [.25, .5, .75], axis=1, method='linear')
    return np.column_stack([p.mean(1), p.std(1), q.T]).ravel()


def routes(routers, x, valid_ratios):
    return {k: np.where(np.asarray(valid_ratios) >= .8,
                        bundle['kmeans'].predict(bundle['scaler'].transform(x)), -1)
            for k, bundle in routers.items()}


def infer(images, assigned, models, mean, std):
    normalized = (np.stack(images).astype('float32') - mean) / std
    x = torch.from_numpy(np.ascontiguousarray(normalized)).to('cuda')
    with torch.inference_mode():
        base = models[-1](x).squeeze(1).sigmoid().cpu().numpy()
        result = {'gz_global': base}
        for k, clusters in assigned.items():
            pred = base.copy()
            for c in range(k):
                ids = np.flatnonzero(clusters == c)
                if len(ids):
                    idx = torch.as_tensor(ids, device='cuda')
                    pred[ids] = models[k, c](x[idx]).squeeze(1).sigmoid().cpu().numpy()
            result[f'gz_k{k}'] = pred
    return result


def panel(path, rgb, truth, valid, predictions, title):
    # RGB and all overlays use one shared stretch/extent, no per-model adjustment.
    rgb = np.asarray(rgb).astype('uint8')
    base = Image.fromarray(rgb).convert('RGBA')
    h, w = valid.shape
    def overlay(mask, color):
        arr = np.zeros((h, w, 4), 'uint8')
        arr[mask] = color
        return Image.fromarray(arr).resize(base.size, Image.Resampling.NEAREST)
    reference = base.copy()
    reference.alpha_composite(overlay(valid & (truth == 255), [180, 70, 220, 135]))
    reference.alpha_composite(overlay(valid & (truth == 1), [0, 180, 255, 200]))
    panels = [('RGB', base), ('Seed1 blue /255 purple', reference)]
    for key, label in zip(SYSTEMS, LABELS):
        pic = base.copy()
        pic.alpha_composite(overlay(predictions[key] & valid, [0, 235, 255, 180]))
        score = metrics(measure(predictions[key], truth, valid))['iou']
        panels.append((label + (f' IoU {score:.3%}' if score is not None else ''), pic))
    canvas = Image.new('RGB', (7*320, 410), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    draw.text((5, 4), title, font=font, fill='black')
    for col, (name, pic) in enumerate(panels):
        draw.text((col*320+3, 27), name, font=font, fill='black')
        pic = pic.convert('RGB')
        pic.thumbnail((314, 350))
        canvas.paste(pic, (col*320+3+(314-pic.width)//2, 52+(350-pic.height)//2))
    canvas.save(path)


def load_models():
    protocol = json.loads((OUT/'inputs/gz_training_protocol.json').read_text())
    records = json.loads((OUT/'inputs/weight_sources.json').read_text())
    weights = []
    models = {}
    for r in records:
        path = OUT/'inputs'/r['file']
        assert sha(path) == r['sha256']
        slot = -1 if r['key'] == 'gz_global' else (2, int(r['key'][-1]))
        weights.append(dict(system=r['key'], path=str(path), sha256=r['sha256'],
                            original_sha256=r['original_sha256'], original_path=r['original_path']))
        models[slot] = path
    for c in range(4):
        path = GZ4/f'k4_expert_cluster{c}/rgb_nir_slope/runs/seed_42/C_overlap_grouped/best.pt'
        assert (path.parent/'training_done.json').exists()
        models[4, c] = path
        weights.append(dict(system=f'gz_k4_c{c}', path=str(path), sha256=sha(path)))
    for key, path in list(models.items()):
        state = torch.load(path, map_location='cpu', weights_only=True)
        cfg = state['config']
        assert cfg['input_bands'] == BANDS
        norm = state.get('normalization') or cfg['normalization']
        assert norm == protocol['normalization']
        model = build_model('unet', 5, {'base_channels': 32}).to('cuda')
        model.load_state_dict(state['model_state'])
        model.eval()
        models[key] = model
        next(r for r in weights if r['path'] == str(path))['epoch'] = int(state['epoch'])
        del state
    routers = {}
    for k in [2, 4]:
        path = OUT/'inputs'/f'gz_k{k}_router.joblib'
        assert sha(path) == protocol['cluster_models'][str(k)]['sha256']
        routers[k] = joblib.load(path)
        assert routers[k]['features'] == NAMES
    norm = protocol['normalization']
    mean = np.array(norm['mean'], 'float32')[None, :, None, None]
    std = np.array(norm['std'], 'float32')[None, :, None, None]
    joint_protocol = json.loads((JOINT/'protocol.json').read_text())
    cached_protocol = json.loads((CACHE/'protocol.json').read_text())
    assert sha(JOINT/'protocol.json') == cached_protocol['training_protocol_sha256']
    assert sha(mapped(joint_protocol['router']['path'])) == joint_protocol['router']['sha256']
    joint_hashes = {}
    for r in cached_protocol['checkpoints']:
        path = JOINT/r['job']/'rgb_nir_slope/runs/seed_42/joint_C_D_k5/best.pt'
        assert sha(path) == r['sha256']
        joint_hashes[r['job']] = r['sha256']
    AUDIT.update(training=False, threshold=.5, input_bands=BANDS, weights=weights,
                 guizhou_training_protocol_sha256=sha(OUT/'inputs/gz_training_protocol.json'),
                 joint_checkpoint_hashes=joint_hashes, routers={str(k):sha(OUT/'inputs'/f'gz_k{k}_router.joblib') for k in routers},
                 frozen_joint_router=joint_protocol['router'], source_files={},
                 confounders=['GZ K2/K4 vs joint K5; different K', '2311 vs18516 eligible train patches',
                              'GZ experts selected on routed validation subsets; joint experts on full mixed validation',
                              'GZ global/K2 trained MPS, K4 and joint CUDA; single seed',
                              'Different normalization/positive weight and training exposure'],
                 inference='GZ probabilities freshly evaluated on CUDA; frozen joint predictions reused after checkpoint/input/count verification')
    dump(OUT/'protocol.json', AUDIT)
    return models, routers, mean, std


def holdouts(models, routers, mean, std):
    feature_cache = {r['patch_id']: r for r in map(json.loads, (JOINT/'preflight_features.jsonl').read_text().splitlines())}
    feature_cache.update({r['patch_id']:r for r in read(CACHE/'test_features.csv')})
    slope_entries = {}
    for path in ['outputs/guizhou_C_slope_features_20261026/index.json',
                 'outputs/hunan_C_encoder_rgb_nir_slope_seed42_lr001_batch8_20261030/slope/index.json']:
        slope_entries.update(json.loads((ROOT/path).read_text())['patches'])
    previous = json.loads((CACHE/'comparison.json').read_text())
    all_summary, all_patch, all_route, differences = [], [], [], []
    for split in ['validation', 'test']:
        rows = read(JOINT/'splits'/f'{split}.csv')
        owned_file = CACHE/f'{split}_ownership.npz'
        with np.load(owned_file) as d:
            owned = {k:d[k] for k in d.files}
        cache_files = {'joint_global':CACHE/f'{split}_global_binary.npz', 'joint_k5':CACHE/f'{split}_k5_routed_binary.npz'}
        cached = {}
        for system, path in cache_files.items():
            with np.load(path) as d:
                cached[system] = {k:d[k] for k in d.files}
            assert set(cached[system]) == {r['patch_id'] for r in rows}
        AUDIT['source_files'].update({str(p):sha(p) for p in [owned_file, *cache_files.values(), JOINT/'splits'/f'{split}.csv']})
        x = np.array([[float(feature_cache[r['patch_id']][n]) for n in NAMES] for r in rows])
        assigned = routes(routers, x, [float(feature_cache[r['patch_id']]['valid_ratio']) for r in rows])
        aggregates = defaultdict(Counter)
        rank = defaultdict(list)
        packed = {s:{} for s in SYSTEMS[:3]}
        def load(row):
            pid = row['patch_id']
            raw_bytes = mapped(row['npz_path']).read_bytes()
            assert hashlib.sha256(raw_bytes).hexdigest() == feature_cache[pid]['npz_sha256']
            with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as d:
                raw, truth, valid = d['image'], d['mask'], d['valid'].astype(bool)
            entry = slope_entries[pid]
            slope_bytes = mapped(entry['path']).read_bytes()
            assert hashlib.sha256(slope_bytes).hexdigest() == feature_cache[pid]['slope_sha256']
            slope = np.load(io.BytesIO(slope_bytes), allow_pickle=False)
            image = np.concatenate([raw[[2,1,0,3]].astype('float32'), slope[None].astype('float32')])
            return image, truth, valid
        with ThreadPoolExecutor(max_workers=2) as pool:
            for offset in range(0, len(rows), 8):
                rr = rows[offset:offset+8]
                loaded = list(pool.map(load, rr))
                probabilities = infer([r[0] for r in loaded], {k:v[offset:offset+len(rr)] for k,v in assigned.items()}, models, mean, std)
                for i, (row, (image, truth, valid)) in enumerate(zip(rr, loaded)):
                    pid = row['patch_id']
                    own = np.unpackbits(owned[pid], count=65536).reshape(256,256).astype(bool)
                    assert not (own & ~valid).any()
                    preds = {s:(probabilities[s][i] >= .5) for s in SYSTEMS[:3]}
                    preds.update({s:np.unpackbits(cached[s][pid], count=65536).reshape(256,256).astype(bool) for s in cache_files})
                    for system, pred in preds.items():
                        count = measure(pred, truth, own)
                        all_patch.append(dict(split=split, province=row['province'], site_id=row['site_id'], patch_id=pid, system=system, **count))
                        for scope in [row['province'], 'combined']:
                            aggregates[system, scope].update(count)
                        if system in packed:
                            packed[system][pid] = np.packbits(pred.ravel())
                    u = own & (truth == 255)
                    a, b = preds['gz_k4'], preds['joint_k5']
                    diff = dict(split=split, province=row['province'], site_id=row['site_id'], patch_id=pid,
                                all_disagreement=int((own & (a != b)).sum()),
                                gz_only=int((u & a & ~b).sum()), joint_only=int((u & ~a & b).sum()))
                    differences.append(diff)
                    rank[row['province']].append((diff['all_disagreement'], pid, row))
                    for k in [2,4]:
                        all_route.append(dict(split=split, province=row['province'], site_id=row['site_id'], patch_id=pid, system=f'gz_k{k}', cluster=int(assigned[k][offset+i]), owned_pixels=int(own.sum())))
                if offset == 0 or (offset//8+1)%50 == 0 or offset+8 >= len(rows):
                    status(status='holdout_inference', split=split, done=min(offset+8,len(rows)), total=len(rows))
        for (system, scope), counter in aggregates.items():
            item = dict(split=split, scope=scope, system=system, **metrics(counter))
            if system in cache_files:
                old = next(r for r in previous if r['split']==split and r['scope']==scope and r['counting']=='unique' and r['system']==('global' if system=='joint_global' else 'k5_routed'))
                assert all(item[key] == old[key] for key in FIELDS), (split, scope, system, 'cached count mismatch')
            all_summary.append(item)
        for system, pp in packed.items():
            np.savez_compressed(OUT/f'{split}_{system}_binary.npz', **pp)
        for province, candidates in rank.items():
            chosen, sites = [], set()
            for score,pid,row in sorted(candidates, reverse=True):
                if row['site_id'] in sites:
                    continue
                chosen.append((pid,row))
                sites.add(row['site_id'])
                if len(chosen) == 4:
                    break
            for number,(pid,row) in enumerate(chosen,1):
                image, truth, valid = load(row)
                pixels = image[:3,valid].T
                lo,hi = np.quantile(pixels,[.02,.98],axis=0)
                rgb = (np.clip((image[:3].transpose(1,2,0)-lo)/np.maximum(hi-lo,1),0,1)*255).astype('uint8')
                rgb[~valid] = 235
                preds = {s:np.unpackbits((packed if s in packed else cached)[s][pid],count=65536).reshape(256,256).astype(bool) for s in SYSTEMS}
                panel(OUT/f'{province}_{split}_rank{number:02d}.png',rgb,truth,valid,preds,f'{province} {split} {row["site_id"]}; selected by GZ K4 vs joint K5 owned-pixel disagreement; image uses full patch')
        table(OUT/'holdout_summary.csv', all_summary)
        dump(OUT/'holdout_summary.json', all_summary)
    table(OUT/'holdout_per_patch.csv', all_patch)
    table(OUT/'holdout_routes.csv', all_route)
    table(OUT/'holdout_disagreements.csv', differences)
    AUDIT['holdout_patch_input_checksums_verified'] = len({r['patch_id'] for r in all_patch})
    AUDIT['cached_joint_holdout_counts_reproduced'] = True
    dump(OUT/'protocol.json', AUDIT)
    return all_summary


def starts(size):
    values = list(range(0, max(1,size-256+1),128))
    last = max(0,size-256)
    if values[-1] != last:
        values.append(last)
    return values


def site_evaluation(models, routers, mean, std):
    plans = [
        ('guizhou_hunan_joint_k5_gansu_20261106',['gansu_21666','gansu_21763']),
        ('guizhou_hunan_joint_k5_shaanxi24000_20261106',['shaanxi_24000']),
        ('guizhou_hunan_joint_k5_six_diagnostic_sites_20261106',['hunan_25700','hunan_25550','hunan_25323','guizhou_22803','guizhou_22759','guizhou_22774']),
        ('guizhou_hunan_joint_k5_chongqing_diagnostic_sites_20261106',['chongqing_24107','chongqing_24096','chongqing_23285','chongqing_23025','chongqing_23375','chongqing_23374']),
    ]
    membership = defaultdict(set)
    for split in ['train','validation','test']:
        manifest = JOINT/'splits'/('global_control/train.csv' if split == 'train' else f'{split}.csv')
        for r in read(manifest):
            membership[r['site_id']].add(split)
    results, route_rows, diffs, contexts = [], [], [], []
    taper = np.maximum(np.outer(np.hanning(256),np.hanning(256)),1e-3).astype('float32')
    for source_name, sites in plans:
        source = ROOT/'outputs'/source_name
        source_protocol = json.loads((source/'protocol.json').read_text())
        for r in source_protocol['checkpoints']:
            assert r['sha256'] == AUDIT['joint_checkpoint_hashes'][r['job']]
        previous = json.loads((source/'comparison.json').read_text())
        for site in sites:
            folder = OUT/site
            folder.mkdir(exist_ok=True)
            inputfile = source/'inputs'/f'{site}.npz'
            meta = json.loads((source/'inputs'/f'{site}.json').read_text())
            assert sha(inputfile) == meta['input_sha256']
            contexts.append(dict(site_id=site, source_folder=str(source), input_sha256=meta['input_sha256'],
                                 context_id=meta['context_id'], imagery=meta['imagery'], reference=meta['reference'], joint_membership=sorted(membership[site])))
            with np.load(inputfile, allow_pickle=False) as d:
                image = np.concatenate([d['optical'][[2,1,0,3]].astype('float32'), d['slope'][None].astype('float32')])
                truth, valid, preview, transform = d['truth'], d['valid'].astype(bool), d['preview'], d['transform']
            assert int(valid.sum()) == meta['valid_pixels']
            for value,key in [(0,'background_pixels'),(1,'water_pixels'),(255,'ignored_pixels')]:
                assert int((valid & (truth==value)).sum()) == meta['reference'][key]
            h,w = valid.shape
            windows, feature = [], []
            for y in starts(h):
                for x in starts(w):
                    vh,vw = min(256,h-y),min(256,w-x)
                    v = valid[y:y+vh,x:x+vw]
                    if not v.any():
                        continue
                    feature.append(vectors(image[:,y:y+vh,x:x+vw],v))
                    windows.append(dict(window=len(windows),x=x,y=y,height=vh,width=vw,valid_ratio=float(v.mean())))
            assigned = routes(routers,np.array(feature),[r['valid_ratio'] for r in windows])
            sums = {s:np.zeros((h,w),'float32') for s in SYSTEMS[:3]}
            denom = np.zeros((h,w),'float32')
            for offset in range(0,len(windows),4):
                ww = windows[offset:offset+4]
                images = []
                for r in ww:
                    patch = np.zeros((5,256,256),'float32')
                    y,x,vh,vw = r['y'],r['x'],r['height'],r['width']
                    patch[:,:vh,:vw] = image[:,y:y+vh,x:x+vw]
                    images.append(patch)
                probabilities = infer(images,{k:v[offset:offset+len(ww)] for k,v in assigned.items()},models,mean,std)
                for i,r in enumerate(ww):
                    y,x,vh,vw = r['y'],r['x'],r['height'],r['width']
                    tap = taper[:vh,:vw]
                    denom[y:y+vh,x:x+vw] += tap
                    for system in SYSTEMS[:3]:
                        sums[system][y:y+vh,x:x+vw] += probabilities[system][i,:vh,:vw]*tap
            assert (denom[valid] > 0).all()
            preds = {}
            for system in SYSTEMS[:3]:
                prob = np.divide(sums[system],denom,out=np.zeros_like(denom),where=denom>0)
                prob[~valid] = 0
                preds[system] = (prob >= .5) & valid
                np.savez_compressed(folder/f'{system}_native_prediction.npz',prediction=np.packbits(preds[system].ravel()),shape=valid.shape,transform=transform)
            for system, cached_name in [('joint_global','global'),('joint_k5','k5_routed')]:
                path = source/site/f'{cached_name}_native_prediction.npz'
                AUDIT['source_files'][str(path)] = sha(path)
                with np.load(path,allow_pickle=False) as d:
                    assert tuple(d['shape']) == valid.shape
                    assert np.allclose(d['transform'],transform)
                    preds[system] = np.unpackbits(d['prediction'],count=h*w).reshape(h,w).astype(bool)
                    assert np.array_equal(preds[system],(d['probability'] >= .5) & valid)
                old = next(r for r in previous if r['site']==site and r['system']==cached_name)
                assert all(measure(preds[system],truth,valid)[key] == old[key] for key in FIELDS)
            for system in SYSTEMS:
                results.append(dict(site_id=site,province=site.split('_')[0],system=system,context_id=meta['context_id'],**metrics(measure(preds[system],truth,valid))))
            unknown = valid & (truth == 255)
            a,b = preds['gz_k4'],preds['joint_k5']
            diffs.append(dict(site_id=site,gz_only_255_km2=int((unknown&a&~b).sum())*.0001,
                              joint_only_255_km2=int((unknown&~a&b).sum())*.0001,
                              disagreement_km2=int((valid&(a!=b)).sum())*.0001))
            for k,lab in assigned.items():
                for r,c in zip(windows,lab):
                    route_rows.append(dict(site_id=site,system=f'gz_k{k}',cluster=int(c),**r))
            table(folder/'window_routes.csv',[r for r in route_rows if r['site_id']==site])
            panel(folder/'comparison.png',preview,truth,valid,preds,site+'; identical frozen date/grid/reference, window256 stride128, cosine probability blend')
            dump(OUT/'site_summary.json',results)
            table(OUT/'site_summary.csv',results)
            status(status='site_inference',completed_sites=len(contexts),total_sites=15,last_site=site)
            del image,truth,valid,sums,preds
            gc.collect()
    dump(OUT/'site_contexts.json',contexts)
    table(OUT/'site_disagreements.csv',diffs)
    table(OUT/'site_routes.csv',route_rows)
    AUDIT['cached_joint_site_counts_reproduced'] = True
    dump(OUT/'protocol.json', AUDIT)
    return results


def report(holdout, sites):
    pct = lambda x: '-' if x is None else f'{x:.4%}'
    lines = ['# 贵州内部聚类与贵州＋湖南联合聚类：冻结系统对比','',
             '未训练、未调整阈值、未重选权重。GZ global/K2/K4使用原最高验证IoU权重，joint global/K5复用原冻结预测并复核模型哈希、输入和像元计数。',
             '两组输入均为R/G/B/NIR/坡度，但K、训练量、归一化、正类权重、选权重规则和部分训练设备不同，因此不能把差异单独归因于聚类是否加入湖南。',
             '验证集用于历史权重选择；测试区域曾探索。0/1为自动高置信种子而非独立人工真值；255预测面积不是误判面积。','',
             '## 同范围验证／测试','',
             '按此前联合评测的固定像元来源去重：新日期优先，同日按patch ID。所有系统用同一像元、参考和上下文；不是整site滑窗评测。','',
             '| 集合 | 范围 | 系统 | IoU | Precision | Recall | FP | FN | 255预测km² |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in holdout:
        if r['scope']=='combined':
            continue
        lines.append(f'| {r["split"]} | {r["scope"]} | {r["system"]} | {pct(r["iou"])} | {pct(r["precision"])} | {pct(r["recall"])} | {r["fp"]} | {r["fn"]} | {r["unknown_predicted_km2"]:.4f} |')
    lines += ['','## 15个历史诊断site','',
              '使用已有冻结日期、影像和光谱∩OSM参考。256窗口/128步长/余弦概率融合后阈值0.5；有效影像比例不足80%的窗口回退各自全局模型，不增加距离回退规则。',
              '贵州三个site在联合训练中，湖南三个在联合验证中；重庆六个按旧模型差异挑选，甘肃与陕西也反复查看。因此这里是诊断而非新盲测。','',
              '| site | 系统 | IoU | Precision | Recall | 255预测km² | 总预测km² |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in sites:
        lines.append(f'| {r["site_id"]} | {r["system"]} | {pct(r["iou"])} | {pct(r["precision"])} | {pct(r["recall"])} | {r["unknown_predicted_km2"]:.4f} | {r["total_predicted_km2"]:.4f} |')
    lines += ['','## 同范围对照图','', '青色为预测水体；种子1为蓝色、255为紫色。所有系统使用同一原图显示。']
    for site in dict.fromkeys(r['site_id'] for r in sites):
        lines += ['',f'### {site}','',f'![{site}]({site}/comparison.png)']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-sites', action='store_true')
    args = parser.parse_args()
    previous = json.loads((OUT/'protocol.json').read_text()) if args.resume_sites else None
    status(status='loading_frozen_models')
    models,routers,mean,std=load_models()
    if args.resume_sites:
        assert previous['weights'] == AUDIT['weights']
        assert previous['cached_joint_holdout_counts_reproduced']
        assert previous['holdout_patch_input_checksums_verified'] == 7152
        AUDIT.update(previous)
        holdout = json.loads((OUT/'holdout_summary.json').read_text())
        assert len(holdout) == 30
        dump(OUT/'protocol.json', AUDIT)
    else:
        holdout=holdouts(models,routers,mean,std)
    sites=site_evaluation(models,routers,mean,std)
    report(holdout,sites)
    status(status='completed',holdout_patches=AUDIT['holdout_patch_input_checksums_verified'],sites=15,systems=5)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        status(status='failed',error=str(error),traceback=traceback.format_exc())
        raise
