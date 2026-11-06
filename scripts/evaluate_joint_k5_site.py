"""Joint K5/global full-site probability blending on two frozen local Gansu contexts."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
import torch
from PIL import Image,ImageDraw,ImageFont
import run_joint_k5_slope_server as train
from evaluate_joint_k5_slope_server import measure,metric
from lake_workbench.models.runtime import build_model

ROOT=train.ROOT
OUT=ROOT/'outputs/guizhou_hunan_joint_k5_gansu_20261106'


def starts(size):
    values=list(range(0,max(1,size-256+1),128))
    last=max(0,size-256)
    if values[-1]!=last:values.append(last)
    return values


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--sites',nargs='+',default=['gansu_21666','gansu_21763'])
    parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args()
    sites=args.sites
    OUT=Path(args.output)
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time();torch.set_num_threads(2);assert torch.cuda.is_available()
    protocol=json.loads((train.OUT/'protocol.json').read_text())
    bundle=joblib.load(protocol['router']['path']);assert train.sha(protocol['router']['path'])==protocol['router']['sha256']
    mean=np.array(protocol['normalization']['mean'],dtype='float32')[:,None,None]
    std=np.array(protocol['normalization']['std'],dtype='float32')[:,None,None]
    models={};weights=[]
    for job in protocol['jobs']:
        name=job['name'];path=train.OUT/name/'rgb_nir_slope/runs/seed_42/joint_C_D_k5/best.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model('unet',5,{'base_channels':32}).to('cuda');model.load_state_dict(state['model_state']);model.eval()
        c=-1 if name=='global_control' else int(name[-1]);models[c]=model
        weights.append(dict(job=name,path=str(path),sha256=train.sha(path),epoch=state['epoch']));del state
    feature_rows=[r for r in train.read(train.SOURCE/'patch_features.csv') if r['status']=='ok']
    X=np.array([[float(r[n]) for n in train.NAMES] for r in feature_rows])
    Z=bundle['scaler'].transform(X);lab=bundle['kmeans'].predict(Z);dist=bundle['kmeans'].transform(Z)
    q95={c:float(np.quantile(dist[lab==c,c],.95)) for c in range(5)}
    records=[];route_stats=[];contexts=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for site in sites:
        folder=OUT/site;folder.mkdir(exist_ok=True)
        meta=json.loads((OUT/'inputs'/f'{site}.json').read_text());inputfile=OUT/'inputs'/f'{site}.npz'
        assert train.sha(inputfile)==meta['input_sha256']
        with np.load(inputfile,allow_pickle=False) as d:
            image=np.concatenate([d['optical'][[2,1,0,3]].astype('float32'),d['slope'][None]],axis=0)
            valid=d['valid'].astype(bool);truth=d['truth'];preview=d['preview'];transform=d['transform']
        assert valid.sum()==meta['valid_pixels']
        for value,key in [(0,'background_pixels'),(1,'water_pixels'),(255,'ignored_pixels')]:
            assert int((valid&(truth==value)).sum())==meta['reference'][key]
        contexts.append(meta)
        h,w=valid.shape;taper=np.maximum(np.outer(np.hanning(256),np.hanning(256)),1e-3).astype('float32')
        windows=[];vectors=[]
        for y in starts(h):
            for x in starts(w):
                vh,vw=min(256,h-y),min(256,w-x);v=valid[y:y+vh,x:x+vw]
                if not v.any():continue
                p=image[:,y:y+vh,x:x+vw][:,v].astype('float64');q=np.quantile(p,[.25,.5,.75],axis=1,method='linear')
                vectors.append(np.column_stack([p.mean(1),p.std(1),q.T]).ravel())
                windows.append(dict(window=len(windows),x=x,y=y,width=vw,height=vh,valid_ratio=float(v.mean())))
        zz=bundle['scaler'].transform(np.array(vectors));labels=bundle['kmeans'].predict(zz);distances=bundle['kmeans'].transform(zz)
        for i,r in enumerate(windows):
            c=int(labels[i]);r.update(cluster=c if r['valid_ratio']>=.8 else -1,nearest_cluster=c,
                                    center_distance=float(distances[i,c]),train_cluster_q95=q95[c],
                                    beyond_train_q95=bool(distances[i,c]>q95[c]))
        train.table(folder/'window_routes.csv',windows)
        sums=np.zeros((2,h,w),'float32');mass=np.zeros((6,h,w),'float32');denom=np.zeros((h,w),'float32')
        with torch.inference_mode():
            for offset in range(0,len(windows),4):
                ww=windows[offset:offset+4];patches=[]
                for r in ww:
                    patch=np.zeros((5,256,256),'float32');y,x=r['y'],r['x'];vh,vw=r['height'],r['width']
                    patch[:,:vh,:vw]=image[:,y:y+vh,x:x+vw];patches.append((patch-mean)/std)
                xinput=torch.from_numpy(np.stack(patches)).to('cuda')
                globalprob=models[-1](xinput).squeeze(1).sigmoid().cpu().numpy();routeprob=globalprob.copy()
                clusters=np.array([r['cluster'] for r in ww])
                for c in range(5):
                    indices=np.flatnonzero(clusters==c)
                    if len(indices):routeprob[indices]=models[c](xinput[torch.as_tensor(indices,device='cuda')]).squeeze(1).sigmoid().cpu().numpy()
                for i,r in enumerate(ww):
                    y,x=r['y'],r['x'];vh,vw=r['height'],r['width'];tap=taper[:vh,:vw]
                    sums[0,y:y+vh,x:x+vw]+=globalprob[i,:vh,:vw]*tap
                    sums[1,y:y+vh,x:x+vw]+=routeprob[i,:vh,:vw]*tap
                    denom[y:y+vh,x:x+vw]+=tap;mass[r['cluster']+1,y:y+vh,x:x+vw]+=tap
        assert (denom[valid]>0).all()
        probabilities=np.divide(sums,denom[None],out=np.zeros_like(sums),where=denom[None]>0)
        probabilities[:,~valid]=0;preds=(probabilities>=.5)&valid[None]
        unknown=valid&(truth==255)
        for system,pred,prob in zip(['global','k5_routed'],preds,probabilities):
            rec=dict(site=site,context_id=meta['context_id'],system=system,**metric(measure(pred,truth,valid)))
            records.append(rec)
            np.savez_compressed(folder/f'{system}_native_prediction.npz',probability=prob,
                                prediction=np.packbits(pred.ravel()),shape=valid.shape,transform=transform)
        for c in [-1,0,1,2,3,4]:
            ww=[r for r in windows if r['cluster']==c]
            influence=np.divide(mass[c+1],denom,out=np.zeros_like(denom),where=denom>0)
            route_stats.append(dict(site=site,cluster=c,windows=len(ww),window_fraction=len(ww)/len(windows),
                                    effective_valid_area_km2=float(influence[valid].sum(dtype='float64'))*.0001,
                                    beyond_train_q95=sum(r['beyond_train_q95'] for r in ww) if c>=0 else None))
        assert abs(sum(r['effective_valid_area_km2'] for r in route_stats if r['site']==site)-valid.sum()*.0001)<.001
        a,b=preds
        diff=dict(global_only=int((unknown&a&~b).sum()),route_only=int((unknown&~a&b).sum()),
                  both_water=int((unknown&a&b).sum()),both_nonwater=int((unknown&~a&~b).sum()))
        assert sum(diff.values())==int(unknown.sum())
        train.dump(folder/'unknown_disagreement.json',diff)
        # All overlays share the exact frozen local preview extent.
        base=Image.fromarray(preview).convert('RGBA');size=base.size
        rgb=base.convert('RGB')
        def layer(mask,color):
            rgba=np.zeros((h,w,4),'uint8');rgba[mask]=color
            return Image.fromarray(rgba).resize(size,Image.Resampling.NEAREST)
        label=base.copy();label.alpha_composite(layer(unknown,[180,70,220,150]));label.alpha_composite(layer(valid&(truth==1),[0,180,255,200]))
        panels=[('RGB',rgb),('Seed1 blue /255 purple',label)]
        for sys,pred in zip(['global','k5_routed'],preds):
            pic=base.copy();pic.alpha_composite(layer(pred,[0,235,255,180]));pic.save(folder/f'{sys}_overlay.png');panels.append((sys,pic))
        dpic=base.copy();dpic.alpha_composite(layer(unknown&a&~b,[255,140,0,220]));dpic.alpha_composite(layer(unknown&~a&b,[230,0,210,220]));panels.append(('255: orange global /pink K5',dpic))
        canvas=Image.new('RGB',(2500,760),'white');draw=ImageDraw.Draw(canvas)
        for col,(title,pic) in enumerate(panels):
            draw.text((col*500+5,5),site+' '+title,font=font,fill='black')
            pic=pic.convert('RGB');pic.thumbnail((490,680));canvas.paste(pic,(col*500+5+(490-pic.width)//2,45+(680-pic.height)//2))
        canvas.save(folder/'comparison.png')
        codes=np.arange(-1,5);dominant=codes[np.argmax(mass,axis=0)]
        colors=np.array([[140,140,140],[230,90,40],[70,180,50],[220,70,210],[40,140,230],[230,190,20]],dtype='uint8')
        arr=colors[dominant+1];arr[~valid]=245
        mapimage=Image.fromarray(arr);mapimage.thumbnail((1100,900))
        routing=Image.new('RGB',(1200,1000),'white');d=ImageDraw.Draw(routing)
        routing.paste(mapimage,((1200-mapimage.width)//2,70))
        d.text((5,5),site+' dominant taper-weight expert influence (overlapping windows)',font=font,fill='black')
        for i,c in enumerate([-1,0,1,2,3,4]):
            d.rectangle((i*200+5,34,i*200+23,52),fill=tuple(colors[i]))
            count=sum(r['cluster']==c for r in windows)
            d.text((i*200+28,34),f'{"fallback" if c<0 else "C"+str(c)}: {count}',font=font,fill='black')
        routing.save(folder/'routing_map.png')
        print('SITE',site,json.dumps([r for r in records if r['site']==site]),flush=True)
        train.dump(OUT/'status.json',dict(status='evaluating',last_site=site,seconds=time.time()-start))
    train.dump(OUT/'comparison.json',records);train.table(OUT/'metrics.csv',records);train.table(OUT/'routing_distribution.csv',route_stats)
    train.dump(OUT/'protocol.json',dict(contexts=contexts,checkpoints=weights,router=protocol['router'],threshold=.5,
               window=256,stride=128,blending='max(hanning outer product,1e-3), probability average then0.5',
               fallback='valid ratio<0.8 uses joint global model',device='cuda',training=False,
               reference='Exact frozen local18768 spectral intersection OSM seeds from prior evaluations; not server regenerated references',
               caveat='255 unlabelled. Sites repeatedly examined, same acquisition, no independent final accuracy claim.'))
    lines=['# 联合全局与K5路由：固定site','',
           '沿用此前本地18768冻结影像与光谱∩OSM参考；256窗口、128步长、余弦概率融合、阈值0.5；最高原验证IoU权重。',
           '与验证/测试patch固定来源去重口径不同，这里是完整site滑窗部署方式。255预测面积不是误判面积。','',
           '| site | 系统 | IoU | Precision | Recall | 255预测km² | 总预测km² |','|---|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f'| {r["site"]} | {r["system"]} | {r["iou"]:.4%} | {r["precision"]:.4%} | {r["recall"]:.4%} | {r["unknown_predicted_km2"]:.4f} | {r["total_predicted_km2"]:.4f} |')
    for site in sites:lines+=['',f'![{site}]({site}/comparison.png)','',f'![routing]({site}/routing_map.png)']
    (OUT/'report.md').write_text('\n'.join(lines))
    train.dump(OUT/'status.json',dict(status='completed',seconds=time.time()-start,sites=len(sites),systems=2))


if __name__=='__main__':main()
