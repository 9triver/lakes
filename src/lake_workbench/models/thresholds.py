"""Frozen, checkpoint-bound calibration and label-free site threshold selection."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from lake_workbench.paths import PROJECT_ROOT

STRATEGIES={'manual','calibrated_adaptive'}
CONFIG=PROJECT_ROOT/'config/model_threshold_calibration.json'

@lru_cache(maxsize=8)
def _read_profile(path:str,mtime:int,size:int):
    return json.loads(Path(path).read_text())

@lru_cache(maxsize=128)
def _digest(path:str,mtime:int,size:int):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def calibration_profile(model_path:Path, *, verify:bool=False) -> dict | None:
    if not CONFIG.exists():return None
    stat=CONFIG.stat();document=_read_profile(str(CONFIG),stat.st_mtime_ns,stat.st_size)
    profiles=document.get('profiles',[document])
    profile=next((p for p in profiles if model_path.resolve()==(PROJECT_ROOT/p['model_path']).resolve()),None)
    if profile is None:return None
    if not math.isfinite(profile['a']) or profile['a']<=0 or not math.isfinite(profile['b']):raise ValueError('校准参数无效')
    if verify:
        stat=model_path.stat()
        if _digest(str(model_path),stat.st_mtime_ns,stat.st_size)!=profile['model_sha256']:
            raise ValueError('模型权重已变化，不能使用原校准参数')
        for dependency in profile.get('dependencies',[]):
            path=PROJECT_ROOT/dependency['path'];stat=path.stat()
            if _digest(str(path),stat.st_mtime_ns,stat.st_size)!=dependency['sha256']:
                raise ValueError('路由组件已变化，不能使用原校准参数')
    return profile

def strategy_identity(model_path:Path,strategy:str='manual') -> list:
    if strategy not in STRATEGIES:raise ValueError('未知阈值策略')
    if strategy=='manual':return []
    p=calibration_profile(model_path,verify=True)
    if p is None:raise ValueError('该权重尚无专属校准参数，请使用手动阈值')
    return [strategy,p['id'],p['model_sha256'],p['a'],p['b'],p['rule_version']]

def calibrated_rule(probabilities:np.ndarray,a:float,b:float) -> dict:
    """Same soft-IoU proxy as the frozen experiment; no labels are accepted."""
    p=np.asarray(probabilities,dtype='float64').reshape(-1)
    if not p.size:raise ValueError('没有有效像元，无法选择自适应阈值')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('模型概率无效')
    if not math.isfinite(a) or a<=0 or not math.isfinite(b):raise ValueError('校准参数无效')
    raw=np.sort(p)[::-1];clipped=np.clip(raw,1e-6,1-1e-6);z=a*(np.log(clipped)-np.log1p(-clipped))+b
    q=np.exp(-np.logaddexp(0.,-z));cumulative=np.cumsum(q,dtype='float64')
    n=np.arange(1,len(q)+1,dtype='float64');score=cumulative/np.maximum(n+cumulative[-1]-cumulative,1e-15)
    ends=np.r_[raw[:-1]!=raw[1:],True];score[~ends]=-np.inf;i=int(np.argmax(score))
    return dict(threshold=float(raw[i]),calibrated_threshold=float(q[i]),soft_iou_proxy=float(score[i]),selection_pixels=int(p.size),selected_positive_pixels=i+1)

def select_threshold(probability:np.ndarray,valid:np.ndarray,model_path:Path,manual:float=.5,strategy:str='manual') -> dict:
    strategy_identity(model_path,strategy)
    if strategy=='manual':
        if not math.isfinite(manual) or not 0<manual<1:raise ValueError('阈值必须在0和1之间')
        return dict(strategy='manual',threshold=float(manual))
    p=calibration_profile(model_path,verify=True)
    return dict(strategy=strategy,profile_id=p['id'],calibration_source=p['source'],**calibrated_rule(probability[valid],p['a'],p['b']))
