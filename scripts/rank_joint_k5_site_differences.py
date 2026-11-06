"""Rank held-out sites by within-site, single-count global/K5 prediction disagreement."""
from collections import defaultdict, Counter
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import run_joint_k5_slope_server as training
from evaluate_joint_k5_slope_server import measure,metric

ROOT=training.ROOT
SOURCE=ROOT/'outputs/guizhou_hunan_joint_k5_evaluation_20261106'
OUT=SOURCE/'site_differences'


def owned_masks(rows):
    claimed={}
    ordered=sorted(rows,key=lambda r:r['patch_id'])
    ordered.sort(key=lambda r:r.get('product_date',''),reverse=True)
    assert len({r['crs'] for r in rows})==1
    for row in ordered:
        with np.load(training.mapped(row['npz_path']),allow_pickle=False) as data:
            raw=data['image'];truth=data['mask'];valid=data['valid'].astype(bool)
        x=round(float(row['bounds_left'])/10);y=round(-float(row['bounds_top'])/10)
        own=np.zeros((256,256),bool)
        for bx in range(x//256,(x+255)//256+1):
            for by in range(y//256,(y+255)//256+1):
                block=claimed.setdefault((bx,by),np.zeros((256,256),bool))
                x0=max(x,bx*256);x1=min(x+256,(bx+1)*256)
                y0=max(y,by*256);y1=min(y+256,(by+1)*256)
                s=np.s_[y0-y:y1-y,x0-x:x1-x]
                b=np.s_[y0-by*256:y1-by*256,x0-bx*256:x1-bx*256]
                own[s]=valid[s]&~block[b];block[b]|=own[s]
        yield row,raw,truth,own


def main():
    OUT.mkdir(exist_ok=True)
    groups={};ranking=[]
    for split in ['validation','test']:
        rows=training.read(training.OUT/'splits'/f'{split}.csv')
        sites=defaultdict(list)
        for r in rows:sites[r['site_id']].append(r)
        with np.load(SOURCE/f'{split}_global_binary.npz') as aa,np.load(SOURCE/f'{split}_k5_routed_binary.npz') as bb:
            for site,rr in sites.items():
                gcount=Counter();rcount=Counter();diff=Counter()
                for row,raw,y,v in owned_masks(rr):
                    pid=row['patch_id']
                    a=np.unpackbits(aa[pid],count=65536).reshape(256,256).astype(bool)
                    b=np.unpackbits(bb[pid],count=65536).reshape(256,256).astype(bool)
                    gcount.update(measure(a,y,v));rcount.update(measure(b,y,v))
                    u=v&(y==255)
                    diff.update(disagreement_pixels=int((v&(a^b)).sum()),
                                unknown_disagreement_pixels=int((u&(a^b)).sum()),
                                global_only=int((v&a&~b).sum()),route_only=int((v&~a&b).sum()),
                                unknown_global_only=int((u&a&~b).sum()),unknown_route_only=int((u&~a&b).sum()))
                gm,rm=metric(dict(gcount)),metric(dict(rcount))
                record=dict(site_id=site,province=rr[0]['province'],split=split,patches=len(rr),
                            valid_km2=gm['valid_km2'],disagreement_km2=diff['disagreement_pixels']*.0001,
                            disagreement_fraction=diff['disagreement_pixels']/max(1,gcount['valid']),
                            unknown_disagreement_km2=diff['unknown_disagreement_pixels']*.0001,
                            global_only_km2=diff['global_only']*.0001,route_only_km2=diff['route_only']*.0001,
                            global_predicted_km2=gm['total_predicted_km2'],route_predicted_km2=rm['total_predicted_km2'],
                            predicted_net_change_km2=rm['total_predicted_km2']-gm['total_predicted_km2'],
                            global_unknown_predicted_km2=gm['unknown_predicted_km2'],route_unknown_predicted_km2=rm['unknown_predicted_km2'],
                            global_iou=gm['iou'],route_iou=rm['iou'])
                ranking.append(record);groups[site]=rr
        print('SITES',split,len(sites),flush=True)
    ranking.sort(key=lambda r:(-r['disagreement_km2'],r['site_id']))
    for n,r in enumerate(ranking,1):r['rank']=n
    training.table(OUT/'all_sites_ranked.csv',ranking)
    training.dump(OUT/'top10.json',ranking[:10])
    display=json.loads((training.SOURCE/'display_config.json').read_text())
    low,high=np.array(display['low']),np.array(display['high'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(1500,10*340),'white');draw=ImageDraw.Draw(sheet)
    for n,result in enumerate(ranking[:10]):
        site=result['site_id'];rr=groups[site];split=result['split']
        left=min(float(r['bounds_left']) for r in rr);top=max(float(r['bounds_top']) for r in rr)
        right=max(float(r['bounds_right']) for r in rr);bottom=min(float(r['bounds_bottom']) for r in rr)
        w,h=round((right-left)/10),round((top-bottom)/10)
        # Render at <=1200px side without allocating a very large native mosaic.
        scale=min(1.,1200/max(w,h));dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
        panels=[np.full((dh,dw,3),235,dtype='uint8') for _ in range(5)]
        with np.load(SOURCE/f'{split}_global_binary.npz') as aa,np.load(SOURCE/f'{split}_k5_routed_binary.npz') as bb:
            for row,raw,y,v in owned_masks(rr):
                if not v.any():continue
                a=np.unpackbits(aa[row['patch_id']],count=65536).reshape(256,256).astype(bool)
                b=np.unpackbits(bb[row['patch_id']],count=65536).reshape(256,256).astype(bool)
                rgb=(np.clip((raw[[2,1,0]].transpose(1,2,0)-low)/np.maximum(high-low,1),0,1)*255).astype('uint8')
                label=rgb.copy();label[y==255]=(label[y==255]*.4+np.array([180,70,220])*.6).astype('uint8');label[y==1]=[0,180,255]
                overlays=[]
                for p in [a,b]:
                    overlay=rgb.copy();overlay[p]=(overlay[p]*.4+np.array([0,255,255])*.6).astype('uint8');overlays.append(overlay)
                diff=rgb.copy();diff[a&~b]=[255,140,0];diff[~a&b]=[230,0,210];diff[a&b]=[0,190,245]
                x=round((float(row['bounds_left'])-left)/10);yy=round((top-float(row['bounds_top']))/10)
                x0,y0=round(x*scale),round(yy*scale);x1,y1=round((x+256)*scale),round((yy+256)*scale)
                size=(x1-x0,y1-y0)
                if min(size)<=0:continue
                vv=np.array(Image.fromarray(v).resize(size,Image.Resampling.NEAREST)).astype(bool)
                for dest,src in zip(panels,[rgb,label,*overlays,diff]):
                    arr=np.array(Image.fromarray(src).resize(size,Image.Resampling.NEAREST))
                    tile=dest[y0:y1,x0:x1];tile[vv]=arr[vv]
        rowimage=Image.new('RGB',(1500,340),'white');d=ImageDraw.Draw(rowimage)
        d.text((4,3),f'#{n+1} {site} [{split}] | XOR={result["disagreement_km2"]:.4f} km2 / 255 XOR={result["unknown_disagreement_km2"]:.4f} km2',font=font,fill='black')
        for col,(panel,title) in enumerate(zip(panels,['RGB','Seed1 blue /255 purple','Global cyan','K5 routed cyan','Orange global /pink K5'])):
            d.text((col*300+4,26),title,font=font,fill='black')
            im=Image.fromarray(panel);im.thumbnail((292,286))
            rowimage.paste(im,(col*300+4+(292-im.width)//2,49+(286-im.height)//2))
        rowimage.save(OUT/f'{n+1:02d}_{site}.png');sheet.paste(rowimage,(0,n*340))
    sheet.save(OUT/'top10_comparison.png')
    lines=['# 全局与K5路由预测分歧最大的10个site','',
           '范围：贵州C＋湖南D验证集和测试集，共同冻结权重、阈值0.5。只包括本轮评测清单覆盖区域，不是完整原始site影像。',
           '按site内部去重后，两个系统水/非水预测不一致（XOR）的面积降序；同时统计255内分歧。',
           '同一site内日期新者优先，同日期固定patch ID，不依据标签挑选；不同site之间可覆盖同一地点，因此本表面积不能直接相加与全局去重结果比较。',
           '差异不是错误真值，预测面积差也不等同位置分歧面积。','',
           '| 排名 | site | 集合 | 分歧km² | 255内分歧km² | 全局预测km² | K5预测km² | 全局IoU | K5 IoU |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in ranking[:10]:
        gi='—' if r['global_iou'] is None else f'{r["global_iou"]:.4%}'
        ri='—' if r['route_iou'] is None else f'{r["route_iou"]:.4%}'
        lines.append(f'| {r["rank"]} | {r["site_id"]} | {r["split"]} | {r["disagreement_km2"]:.4f} | {r["unknown_disagreement_km2"]:.4f} | {r["global_predicted_km2"]:.4f} | {r["route_predicted_km2"]:.4f} | {gi} | {ri} |')
    lines+=['','![Top10](top10_comparison.png)']
    (OUT/'report.md').write_text('\n'.join(lines))
    training.dump(OUT/'status.json',dict(status='completed',sites=len(ranking),rank='all valid disagreement km2',training=False))
    print(json.dumps(ranking[:10],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
