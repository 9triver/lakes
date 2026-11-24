"""Frozen K-means expert combinations exposed as selectable inference systems."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.models.thresholds import _digest

CONFIG=PROJECT_ROOT/'config/guizhou_D_k4_routing.json'
JOINT_K3_CONFIG=PROJECT_ROOT/'config/guizhouD_chongqing_K3_routing.json'
CHONGQING_K3_CONFIG=PROJECT_ROOT/'config/chongqing_only_K3_routing.json'
HUNAN_E_K5_CONFIG=PROJECT_ROOT/'config/hunan_E_K5_routing.json'
GUIZHOU_ADAPTIVE_EFFB0_K4_CONFIG=PROJECT_ROOT/'config/guizhou_D_adaptive_effb0_K4_routing.json'
GUIZHOU_CONTINUED_K4_CONFIG=PROJECT_ROOT/'config/guizhou_D_continued_global_K4_routing.json'
COLORS=[(140,140,140),(230,90,40),(70,180,50),(220,70,210),(40,140,230),(20,175,175)]

@lru_cache(maxsize=8)
def _read(path,mtime,size):return json.loads(Path(path).read_text())

def profile_for(path):
    path=Path(path).resolve()
    if not any(path==p.resolve() for p in [CONFIG,JOINT_K3_CONFIG,CHONGQING_K3_CONFIG,HUNAN_E_K5_CONFIG,GUIZHOU_ADAPTIVE_EFFB0_K4_CONFIG,GUIZHOU_CONTINUED_K4_CONFIG]) or not path.exists():return None
    stat=path.stat();return _read(str(path),stat.st_mtime_ns,stat.st_size)

def expert_count(profile):
    return int(profile.get('k',len([e for e in profile['components'] if e['cluster']>=0])))

def architecture_label(profile):
    k=expert_count(profile)
    return f'K{k}自动路由（{k}个专家＋Global回退）'

def dependencies(path):
    p=profile_for(path)
    if p is None:return []
    result=[]
    for e in [*p['components'],p['router']]:
        f=PROJECT_ROOT/e['path'];s=f.stat();result.append([str(f.resolve()),s.st_mtime_ns,s.st_size])
    return result

def routed_models(workspace_id):
    from lake_workbench.models.thresholds import calibration_profile
    items=[]
    for path in [CONFIG,JOINT_K3_CONFIG,CHONGQING_K3_CONFIG,HUNAN_E_K5_CONFIG,GUIZHOU_ADAPTIVE_EFFB0_K4_CONFIG,GUIZHOU_CONTINUED_K4_CONFIG]:
        p=profile_for(path)
        if not p or p['workspace_id']!=workspace_id:continue
        item=dict(key=f"routing/{workspace_id}/{p['id']}",label=p['label'],name=p['label'],path=str(path.relative_to(PROJECT_ROOT)),
                  workspace_id=workspace_id,scope='all',region=p.get('region','guizhou'),weight='组合推理',experimental=True,completed=True,
                  routing_model=True,in_channels=5,base_channels=32,model_type='unet',model_options={'base_channels':32},architecture_label=architecture_label(p),
                  config={'input_bands':p['input_bands'],'routing_profile':p['id']},selection='各组件最高验证IoU')
        calibration=calibration_profile(path)
        if calibration:item['threshold_calibration']={'id':calibration['id'],'label':calibration['label']}
        try:dependencies(path)
        except OSError as exc:item['error']=f'路由组件缺失：{exc}'
        items.append(item)
    return items

def load_routed_model(path,device):
    path=Path(path).resolve();p=profile_for(path)
    if p is None:raise ValueError('未知自动路由配置')
    s=path.stat();signature=tuple(tuple(x) for x in dependencies(path))
    return _load(str(path),s.st_mtime_ns,s.st_size,signature,str(device))

@lru_cache(maxsize=2)
def _load(path,mtime,size,signature,device):
    import joblib
    from lake_workbench.models.runtime import load_model_checkpoint,LoadedWaterModel
    p=profile_for(Path(path));networks={}
    for e in [*p['components'],p['router']]:
        f=PROJECT_ROOT/e['path'];s=f.stat()
        if _digest(str(f),s.st_mtime_ns,s.st_size)!=e['sha256']:raise ValueError(f"路由组件已变化，请重新确认配置：{e['path']}")
    for e in p['components']:
        model=load_model_checkpoint(PROJECT_ROOT/e['path'],device)
        if model.config['input_bands']!=p['input_bands'] or model.config['normalization']!=p['normalization']:raise ValueError('路由组件的通道或归一化不一致')
        networks[e['cluster']]=model
    k=expert_count(p)
    if not 1<=k<len(COLORS) or set(networks)!={-1,*range(k)}:raise ValueError('路由专家编号或Global组件不完整')
    bundle=joblib.load(PROJECT_ROOT/p['router']['path']);global_model=networks[-1]
    expected=[f'{b}_{stat}' for b in p['input_bands'] for stat in ['mean','std','q25','q50','q75']]
    if bundle['features']!=expected:raise ValueError('路由特征顺序不一致')
    return LoadedWaterModel(path=Path(path),device=global_model.device,model=SimpleNamespace(networks=networks,bundle=bundle,profile=p),
        model_type='unet',model_options={'base_channels':32},architecture_label=architecture_label(p),in_channels=5,base_channels=32,
        mean=global_model.mean,std=global_model.std,epoch=global_model.epoch,config=dict(input_bands=p['input_bands'],routing_profile=p['id']))

def predict_routed_tiled(model,image,valid,batch_size=4,*,calibrated=False,return_components=False):
    import torch
    from lake_workbench.models.comparison import _starts,overlay
    p=model.model.profile;bundle=model.model.bundle;networks=model.model.networks;h,w=valid.shape;tile=p['tile_size'];stride=p['stride']
    if image.shape!=(5,h,w) or not valid.any():raise ValueError('自动路由需要有效的五通道影像')
    calibration=p.get('per_route_calibration')
    if calibrated and calibration is None:raise ValueError('该组合缺少网页推理专属分路由校准参数')
    if calibrated:
        for c in networks:
            fit=calibration[str(c)]
            if not np.isfinite([fit['a'],fit['b']]).all() or fit['a']<=0:raise ValueError('分路由校准参数无效')
    windows=[];vectors=[]
    for y in _starts(h,tile,stride):
        for x in _starts(w,tile,stride):
            vh,vw=min(tile,h-y),min(tile,w-x);v=valid[y:y+vh,x:x+vw]
            if not v.any():continue
            values=image[:,y:y+vh,x:x+vw][:,v].astype('float64')
            if not np.isfinite(values).all():raise ValueError('自动路由特征存在无效值')
            q=np.quantile(values,[.25,.5,.75],axis=1,method='linear');vectors.append(np.column_stack([values.mean(1),values.std(1),q.T]).ravel())
            windows.append(dict(x=x,y=y,h=vh,w=vw,valid_ratio=float(v.mean())))
    z=bundle['scaler'].transform(np.array(vectors));nearest=bundle['kmeans'].predict(z)
    rejected=np.zeros(len(windows),bool)
    if bundle.get('require_rejection_gate'):
        source=bundle['rejection_kmeans'].predict(z);rejected=source==int(bundle['rejected_cluster'])
        if np.any(nearest[~rejected]!=source[~rejected]):raise ValueError('保留簇与原拒绝门控编号不一致')
    for i,(r,c) in enumerate(zip(windows,nearest)):
        r['fallback_reason']='low_valid' if r['valid_ratio']<p['minimum_valid_ratio'] else 'rejected_original_cluster' if rejected[i] else ''
        r['cluster']=-1 if r['fallback_reason'] else int(c)
    clusters_available=sorted(networks);k=len(clusters_available)-1;colors=COLORS[:k+1]
    if clusters_available!=[-1,*range(k)]:raise ValueError('专家编号必须连续')
    taper=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),1e-3).astype('float32');total=np.zeros((h,w),'float32');denom=np.zeros((h,w),'float32');mass=np.zeros((k+1,h,w),'float32')
    parts=np.zeros_like(mass) if calibrated or return_components else None
    with torch.inference_mode():
        for offset in range(0,len(windows),batch_size):
            ww=windows[offset:offset+batch_size];patches=[]
            for r in ww:
                patch=np.zeros((5,tile,tile),'float32');patch[:,:r['h'],:r['w']]=image[:,r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]
                patches.append((patch-model.mean[:,None,None])/model.std[:,None,None])
            tensor=torch.from_numpy(np.ascontiguousarray(np.stack(patches))).to(model.device);clusters=np.array([r['cluster'] for r in ww]);prob=np.zeros((len(ww),tile,tile),'float32')
            for c in np.unique(clusters):
                ids=np.flatnonzero(clusters==c);network=networks[int(c)].model
                network.eval();prob[ids]=network(tensor[torch.as_tensor(ids,device=model.device)]).squeeze(1).sigmoid().cpu().numpy()
            for i,r in enumerate(ww):
                win=np.s_[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']];tap=taper[:r['h'],:r['w']]
                total[win]+=prob[i,:r['h'],:r['w']]*tap;denom[win]+=tap;mass[r['cluster']+1,win[0],win[1]]+=tap
                if parts is not None:parts[r['cluster']+1,win[0],win[1]]+=prob[i,:r['h'],:r['w']]*tap
    if not (denom[valid]>0).all():raise ValueError('路由拼接遗漏有效像元')
    probability=np.divide(total,denom,out=np.zeros_like(total),where=denom>0);probability[~valid]=0
    component_output={}
    if parts is not None:
        combined=np.zeros_like(total)
        for c in clusters_available:
            ownmass=mass[c+1];conditional=np.divide(parts[c+1],ownmass,out=np.zeros_like(total),where=ownmass>0);weight=np.divide(ownmass,denom,out=np.zeros_like(total),where=denom>0)
            if return_components:component_output[str(c)]=dict(probability=conditional,weight=weight)
            if calibrated:
                from scipy.special import expit,logit
                fit=calibration[str(c)];q=expit(fit['a']*logit(np.clip(conditional.astype('float64'),1e-6,1-1e-6))+fit['b']);combined+=q*weight
        if calibrated:probability=combined;probability[~valid]=0
    rows=[]
    for c in clusters_available:
        influence=np.divide(mass[c+1],denom,out=np.zeros_like(denom),where=denom>0);coverage=float(influence[valid].sum(dtype='float64')/valid.sum())
        rows.append(dict(cluster=c,label='Global回退' if c<0 else f'专家C{c}',windows=sum(r['cluster']==c for r in windows),coverage_fraction=coverage,color='#%02x%02x%02x'%colors[c+1]))
    if not np.isclose(sum(r['coverage_fraction'] for r in rows),1.,atol=1e-6):raise ValueError('路由覆盖统计不完整')
    dominant=np.argmax(mass,axis=0);rgba=np.zeros((h,w,4),'uint8')
    for i,color in enumerate(colors):rgba[valid&(dominant==i)]=[*color,170]
    rule=('逐窗口原K4判别；原C3或有效像元比例不足80%回退Global，其余使用保留的C0/C1/C2；概率融合后取阈值' if bundle.get('require_rejection_gate') else '逐窗口最近中心；有效影像比例不足80%回退Global；概率融合后取阈值')
    info=dict(profile_id=p['id'],k=k,rejected_windows=sum(r['fallback_reason']=='rejected_original_cluster' for r in windows),low_valid_windows=sum(r['fallback_reason']=='low_valid' for r in windows),windows=len(windows),components=rows,fallback_fraction=rows[0]['coverage_fraction'],minimum_valid_ratio=p['minimum_valid_ratio'],overlay=overlay_from_rgba(rgba),rule=rule)
    info['probability_space']='per_route_calibrated' if calibrated else 'raw'
    if return_components:return probability,info,component_output
    return probability,info

def overlay_from_rgba(rgba):
    from lake_workbench.models.comparison import png_url
    return png_url(rgba,nearest=True)
