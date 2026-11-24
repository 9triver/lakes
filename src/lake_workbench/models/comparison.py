"""Compare models on one fixed native raster grid and one explicit reference."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import array_bounds
from affine import Affine
from shapely.geometry import mapping

from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.imagery.display import DISPLAY_STYLES, hunan_display_rgb, hunan_rgb_enabled
from lake_workbench.models.features import aligned_slope, assemble_features, feature_names
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.models.thresholds import select_threshold, strategy_identity, calibration_profile
from lake_workbench.paths import PROJECT_ROOT

ENGINE_VERSION='native256-overlap128-cosine-v1'
SEED_REFERENCE='spectral_osm_intersection'
_REFERENCE_LOCK=threading.Lock()


@lru_cache(maxsize=4)
def _seed_reference(catalog,site_id:str,asset:str,image_signature:str) -> Path:
    """Use the same persisted three-state labels as training, never water polygons alone."""
    from lake_workbench.automatic_labels import generate_derived_label
    site=catalog.get_site(site_id)
    root=PROJECT_ROOT/'data/model_comparison_references'/catalog.region.key
    with _REFERENCE_LOCK:
        result=generate_derived_label(catalog,site,SEED_REFERENCE,
            dict(asset_id=asset,product=asset,extent=list(site.bbox),max_dimension=2048),root)
    return root/SEED_REFERENCE/f"{result['label_id']}.npz"


def aligned_seed_labels(path:Path,shape:tuple,transform,crs) -> np.ndarray:
    with np.load(path,allow_pickle=False) as data:
        source=np.asarray(data['labels'],dtype='uint8').copy()
        source[~np.asarray(data['valid'],dtype=bool)]=255
        target=np.full(shape,255,dtype='uint8')
        reproject(source,target,src_transform=Affine(*data['transform']),src_crs=str(data['crs'].item()),
            dst_transform=transform,dst_crs=crs,src_nodata=255,dst_nodata=255,resampling=Resampling.nearest)
    return target


def png_url(array: np.ndarray, limit: int=768, *, nearest: bool=False) -> str:
    picture=Image.fromarray(array)
    if max(picture.size)>limit:
        ratio=limit/max(picture.size)
        picture=picture.resize(tuple(max(1,round(v*ratio)) for v in picture.size),
                               Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    stream=io.BytesIO();picture.save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def rgb_preview(image: np.ndarray, valid: np.ndarray, *, transparent: bool=False, style: str='rgb', hunan_rgb: bool=False) -> str:
    if style not in DISPLAY_STYLES:raise ValueError('未知影像配色')
    rgb=np.zeros((*valid.shape,3),dtype='uint8')
    for channel,index in enumerate(DISPLAY_STYLES[style]):
        values=image[index,valid]
        low,high=np.percentile(values,[2,98]) if values.size else (0,1)
        rgb[:,:,channel]=(np.clip((image[index]-low)/max(high-low,1),0,1)*255).astype('uint8')
    if hunan_rgb and style == "rgb":
        rgb = np.moveaxis(hunan_display_rgb(image[DISPLAY_STYLES["rgb"]]), 0, -1)
    rgb[~valid]=[24,30,38]
    if transparent:
        rgba=np.dstack((rgb,np.where(valid,255,0).astype('uint8')))
        rgba[~valid]=0
        return png_url(rgba)
    return png_url(rgb)


def overlay(mask: np.ndarray,color: tuple[int,int,int,int]) -> str:
    rgba=np.zeros((*mask.shape,4),dtype='uint8');rgba[mask]=color
    return png_url(rgba,nearest=True)


def reference_mask_preview(truth: np.ndarray,valid: np.ndarray, *, transparent: bool=False) -> str:
    """Display the actual three-state evaluation mask without the optical background."""
    rgb=np.full((*valid.shape,3),[24,30,38],dtype='uint8')
    rgb[(truth==0)&valid]=[8,12,17]
    rgb[(truth==1)&valid]=[61,211,130]
    rgb[(truth==255)&valid]=[130,130,130]
    if transparent:
        rgba=np.dstack((rgb,np.where(valid,255,0).astype('uint8')))
        rgba[~valid]=0
        return png_url(rgba,nearest=True)
    return png_url(rgb,nearest=True)


def ignored_mask_preview(truth: np.ndarray,valid: np.ndarray) -> str:
    """Highlight only label 255 inside the readable imagery footprint."""
    return overlay((truth==255)&valid,(168,85,247,180))


def scored_metrics(prediction: np.ndarray,truth: np.ndarray,valid: np.ndarray) -> dict:
    scored=valid & (truth!=255)
    target=truth==1
    tp=int((prediction & target & scored).sum());fp=int((prediction & ~target & scored).sum())
    fn=int((~prediction & target & scored).sum());tn=int((~prediction & ~target & scored).sum())
    # Empty reference/prediction unions have no water IoU; do not report a spurious perfect score.
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,scored_pixels=int(scored.sum()),
        iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def _starts(size:int,tile:int,stride:int) -> list[int]:
    positions=list(range(0,max(1,size-tile+1),stride))
    last=max(0,size-tile)
    if positions[-1]!=last:positions.append(last)
    return positions


def predict_tiled(model, image: np.ndarray, valid: np.ndarray, batch_size:int=4, *, return_routing:bool=False, calibrated_routing:bool=False):
    config = getattr(model, "config", {})
    if config.get("routing_profile"):
        from lake_workbench.models.routing import predict_routed_tiled
        probability,routing=predict_routed_tiled(model,image,valid,batch_size,calibrated=calibrated_routing)
        return (probability,routing) if return_routing else probability
    import torch
    height,width=valid.shape
    tile=256;stride=128
    taper=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),1e-3).astype('float32')
    total=np.zeros((height,width),dtype='float32');weights=np.zeros_like(total)
    patches=[];locations=[]
    def flush():
        if not patches:return
        tensor=torch.from_numpy(np.stack(patches)).to(model.device)
        with torch.inference_mode():
            probabilities=torch.sigmoid(model.model(tensor).squeeze(1)).cpu().numpy()
        for probability,(y,x,h,w) in zip(probabilities,locations,strict=True):
            total[y:y+h,x:x+w]+=probability[:h,:w]*taper[:h,:w]
            weights[y:y+h,x:x+w]+=taper[:h,:w]
        patches.clear();locations.clear()
    model.model.eval()
    for y in _starts(height,tile,stride):
        for x in _starts(width,tile,stride):
            h,w=min(tile,height-y),min(tile,width-x)
            if not valid[y:y+h,x:x+w].any():continue
            patch=np.zeros((image.shape[0],tile,tile),dtype='float32')
            patch[:,:h,:w]=image[:,y:y+h,x:x+w]
            if config.get("zscore_training_recipe"):
                patch_valid = np.zeros((tile, tile), dtype=bool)
                patch_valid[:h, :w] = valid[y:y+h, x:x+w]
                values = patch[:4, patch_valid]
                normalized = (patch - model.mean[:, None, None]) / model.std[:, None, None]
                normalized[:4] = 0
                normalized[:4, patch_valid] = (
                    (values - values.mean(axis=1, keepdims=True))
                    / np.maximum(values.std(axis=1, keepdims=True), 1)
                )
                patch = normalized
            elif config.get("normalization_mode") == "patch_zscore":
                channels = config["dynamic_channels"]
                window_valid = valid[y:y+h, x:x+w]
                values = patch[:, :h, :w][:, window_valid].astype(np.float64)
                mean = model.mean.astype(np.float64)
                std = model.std.astype(np.float64)
                mean[channels] = values[channels].mean(axis=1)
                std[channels] = np.maximum(values[channels].std(axis=1), 1e-6)
                patch.fill(0)
                patch[:, :h, :w][:, window_valid] = (values - mean[:, None]) / std[:, None]
            else:
                patch=(patch-model.mean[:,None,None])/model.std[:,None,None]
            patches.append(np.ascontiguousarray(patch));locations.append((y,x,h,w))
            if len(patches)==batch_size:flush()
    flush()
    if np.any(valid & (weights<=0)):raise ValueError('推理拼接遗漏有效像素')
    probability=np.divide(total,weights,out=np.zeros_like(total),where=weights>0)
    return (probability,None) if return_routing else probability


def _assets(catalog,site) -> list[dict]:
    local=[r for r in catalog._local_imagery_rows_for_site(site) if r.get('tci_path') and Path(r['tci_path']).exists()]
    return local or catalog._model_validation_rows(site,5)


def _source(catalog,site,asset:str='') -> tuple[dict,list[dict]]:
    rows=_assets(catalog,site)
    if not rows:raise FileNotFoundError('该区域没有可读的五波段影像')
    chosen=next((row for row in rows if asset and asset in {str(row.get('product','')),str(row.get('product_id',''))}),None)
    if asset and chosen is None:raise ValueError('指定影像不可用，请重新选择')
    if chosen is None:chosen=catalog._active_local_imagery_row(site) or rows[0]
    return chosen,rows


def _label_path(image_path:Path) -> Path | None:
    # Only an annotation with the exact same image stem can be used as reference.
    for suffix in ['_Swater.shp','_water.shp','_Swater.gpkg']:
        candidate=image_path.with_name(image_path.stem+suffix)
        if candidate.exists():return candidate
    return None


def _signature(path:Path) -> list:
    from lake_workbench.models.routing import dependencies
    return [str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size,*dependencies(path)]


@lru_cache(maxsize=2)
def _read_grid(path_string:str,signature:str,label_string:str,label_signature:str):
    path=Path(path_string)
    with rasterio.open(path) as src:
        if src.count!=5:raise ValueError(f'需要已核对波段顺序的五波段影像，当前为{src.count}波段')
        image=src.read().astype('float32')
        valid=valid_pixel_mask(image,src.nodatavals,src.read_masks())
        transform,crs=src.transform,src.crs
        if crs is None:raise ValueError('影像缺少坐标系')
    if not valid.any():raise ValueError('影像没有有效像素')
    truth=None
    if label_string:
        if Path(label_string).suffix=='.npz':
            truth=aligned_seed_labels(Path(label_string),valid.shape,transform,crs)
        else:
            frame=pyogrio.read_dataframe(label_string)
            if frame.crs is None:raise ValueError('本地参考标注缺少坐标系')
            frame=frame.to_crs(crs)
            geometries=[(geom,1) for geom in frame.geometry if geom is not None and not geom.is_empty]
            if geometries:
                truth=rasterize(geometries,out_shape=valid.shape,transform=transform,fill=0,dtype='uint8')
            else:truth=np.zeros(valid.shape,dtype='uint8')
    return image,valid,truth,transform,crs


def _grid(catalog,site,asset:str='',reference:str=SEED_REFERENCE):
    if reference not in {'local','none',SEED_REFERENCE}:raise ValueError('不支持的参考标签类型')
    row,rows=_source(catalog,site,asset)
    path=Path(row['tci_path']);label=_label_path(path) if reference=='local' else None
    signature=_signature(path)
    if reference==SEED_REFERENCE:
        selected_asset=str(row.get('product') or row.get('product_id') or '')
        label=_seed_reference(catalog,site.site_id,selected_asset,json.dumps(signature))
    labels=[_signature(p) for p in sorted(label.parent.glob(label.stem+'.*'))] if label else []
    image,valid,truth,transform,crs=_read_grid(str(path),json.dumps(signature),str(label) if label else '',json.dumps(labels))
    identity=dict(image=signature,label=labels,reference=reference,grid='native',version=ENGINE_VERSION)
    context_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    return row,rows,path,image,valid,truth,transform,crs,label,context_id


def comparison_preview(catalog,site,asset:str='',style:str='rgb') -> bytes:
    if style not in DISPLAY_STYLES:raise ValueError('未知影像配色')
    _,_,_,image,valid,*_=_grid(catalog,site,asset,'none')
    return base64.b64decode(rgb_preview(image,valid,transparent=True,style=style,hunan_rgb=hunan_rgb_enabled(catalog.region.key)).split(',',1)[1])


def comparison_context(catalog,site,asset:str='',reference:str=SEED_REFERENCE) -> dict:
    row,rows,path,image,valid,truth,transform,crs,label,identity=_grid(catalog,site,asset,reference)
    description='没有匹配的参考标签，仅展示预测统计，不计算IoU。'
    counts={}
    reference_overlay=None
    if truth is not None:
        counts=dict(water_pixels=int(((truth==1)&valid).sum()),background_pixels=int(((truth==0)&valid).sum()),
                    ignored_pixels=int(((truth==255)&valid).sum()),scored_pixels=int(((truth!=255)&valid).sum()))
        description=('沿用训练数据生成规则：光谱水体∩OSM水体为正类；光谱高置信背景且非OSM水体为负类；分歧、不确定、无覆盖及区域外像素忽略。属于伪标签一致性评测，不代表人工真值精度。'
            if reference==SEED_REFERENCE else '同日期本地水体矢量标注；完整性及人工核验程度未确认，指标仅表示与该参考的一致性。')
        rgba=np.zeros((*valid.shape,4),dtype='uint8')
        rgba[(truth==1)&valid]=[61,211,130,200]
        rgba[(truth==255)&valid]=[130,130,130,110]
        reference_overlay=png_url(rgba,nearest=True)
    return dict(context_id=identity,region=catalog.region.key,site_id=site.site_id,site_bounds_wgs84=list(site.bbox),site_geometry=mapping(site.geometry),
        imagery=dict(asset=str(row.get('product') or row.get('product_id') or ''),name=path.name,date=str(row.get('date') or ''),
                     width=valid.shape[1],height=valid.shape[0],crs=str(crs),bounds_native=list(array_bounds(valid.shape[0],valid.shape[1],transform)),bounds_wgs84=list(transform_bounds(crs,'EPSG:4326',*array_bounds(valid.shape[0],valid.shape[1],transform),densify_pts=21)),resolution=[abs(transform.a),abs(transform.e)]),
        assets=[dict(key=str(r.get('product') or r.get('product_id') or ''),name=Path(r['tci_path']).name,date=str(r.get('date') or '')) for r in rows],
        valid_pixels=int(valid.sum()),reference=dict(available=truth is not None,name=label.name if label else '',
            kind=reference if reference==SEED_REFERENCE else 'local_vector_unverified' if label else 'none',
            description=description,**counts),
        preview=rgb_preview(image,valid,transparent=True,hunan_rgb=hunan_rgb_enabled(catalog.region.key)),reference_overlay=reference_overlay,
        reference_mask=reference_mask_preview(truth,valid,transparent=True) if truth is not None else None,
        ignored_overlay=ignored_mask_preview(truth,valid) if truth is not None else None,
        inference=dict(tile_size=256,stride=128,blending='tapered cosine',counting='native grid, each valid pixel scored once'))


def _cache_path(region: str, site_id: str, identity: str, model_path: Path, threshold: float, threshold_strategy: str='manual') -> Path:
    identity_parts=[ENGINE_VERSION,identity,_signature(model_path),threshold]
    if threshold_strategy!='manual':identity_parts=[ENGINE_VERSION,identity,_signature(model_path),strategy_identity(model_path,threshold_strategy)]
    key=hashlib.sha256(json.dumps(identity_parts).encode()).hexdigest()
    return PROJECT_ROOT/'data/model_comparisons'/region/site_id/f'{key}.json'


def iter_saved_comparisons(catalog,site,models,asset='',reference=SEED_REFERENCE,threshold=.5,threshold_strategy="manual"):
    """Read current cached results incrementally without inference or model loading."""
    if not math.isfinite(threshold) or not 0<threshold<1:
        raise ValueError('阈值必须在0和1之间')
    *_,identity=_grid(catalog,site,asset,reference)
    total=len(models);loaded=0
    yield dict(type='context',context_id=identity,total=total,completed=0,loaded=0)
    for completed,(model_key,model_path) in enumerate(models,1):
        if model_path.exists() and (threshold_strategy != 'calibrated_adaptive' or calibration_profile(model_path) is not None):
            cache=_cache_path(catalog.region.key,site.site_id,identity,model_path,threshold,threshold_strategy)
            if cache.exists():
                result=json.loads(cache.read_text())
                result['cached']=True;result['model']['key']=model_key
                loaded+=1
                yield dict(type='result',context_id=identity,item=dict(key=model_key,result=result),
                           completed=completed,total=total,loaded=loaded)
        if completed%10==0 or completed==total:
            yield dict(type='progress',stage='results',message='正在检查并读取已保存结果',
                       completed=completed,total=total,loaded=loaded)
    yield dict(type='complete',context_id=identity,completed=total,total=total,loaded=loaded)


def saved_comparisons(catalog,site,models: list[tuple[str,Path]],asset:str='',reference:str=SEED_REFERENCE,threshold:float=.5,threshold_strategy:str="manual") -> dict:
    results=[];identity=''
    for event in iter_saved_comparisons(catalog,site,models,asset,reference,threshold,threshold_strategy):
        identity=event.get('context_id',identity)
        if event['type']=='result':results.append(event['item'])
    return dict(context_id=identity,items=results)


def compare_one(catalog,site,model_path:Path,model_key:str,asset:str='',reference:str=SEED_REFERENCE,threshold:float=.5,threshold_strategy:str='manual') -> dict:
    strategy_identity(model_path,threshold_strategy)
    threshold=float(threshold)
    if not math.isfinite(threshold) or not 0<threshold<1:raise ValueError('阈值必须在0和1之间')
    row,_,path,image,valid,truth,transform,crs,label,identity=_grid(catalog,site,asset,reference)
    cache=_cache_path(catalog.region.key,site.site_id,identity,model_path,threshold,threshold_strategy)
    if cache.exists():
        result=json.loads(cache.read_text());result['cached']=True;result['model']['key']=model_key
        if reference != 'none' or 'map_overlay' in result:
            return result
    start=time.monotonic()
    model=load_model_checkpoint(model_path)
    names=feature_names(model.config,model.in_channels)
    slope,terrain=(None,None)
    if 'slope_degrees' in names:slope,terrain=aligned_slope(path,valid)
    features=assemble_features(image,names,slope)
    from lake_workbench.models.routing import profile_for
    route_profile=profile_for(model_path)
    calibrated_routing=bool(threshold_strategy=='calibrated_adaptive' and route_profile and route_profile.get('per_route_calibration'))
    # Calibrated route probabilities require their own cache; raw manual inference stays unchanged.
    probability_identity=[ENGINE_VERSION,identity,_signature(model_path),'probability']
    if calibrated_routing:probability_identity.append('per_route_calibrated')
    probability_key=hashlib.sha256(json.dumps(probability_identity).encode()).hexdigest()
    probability_file=PROJECT_ROOT/'data/model_comparison_probabilities'/catalog.region.key/site.site_id/f'{probability_key}.npz'
    if probability_file.exists():
        with np.load(probability_file,allow_pickle=False) as data:
            probability=data['probability'];routing=json.loads(str(data['routing'].item())) if 'routing' in data else None
        if probability.shape!=valid.shape or not np.isfinite(probability[valid]).all():raise ValueError('概率缓存无效')
    else:
        probability,routing=predict_tiled(model,features,valid,return_routing=True,calibrated_routing=calibrated_routing)
        probability_file.parent.mkdir(parents=True,exist_ok=True)
        temporary=probability_file.with_suffix('.tmp.npz');np.savez_compressed(temporary,probability=probability,routing=np.asarray(json.dumps(routing)));temporary.replace(probability_file)
    threshold_info=select_threshold(probability,valid,model_path,threshold,threshold_strategy)
    threshold=threshold_info['threshold']
    prediction=(probability>=threshold)&valid
    metrics=scored_metrics(prediction,truth,valid) if truth is not None else None
    errors=None
    if truth is not None:
        scored=valid&(truth!=255);target=truth==1
        rgba=np.zeros((*valid.shape,4),dtype='uint8')
        rgba[valid&(truth==255)]=[130,130,130,110]
        rgba[prediction&target&scored]=[46,153,255,170]
        rgba[prediction&~target&scored]=[237,72,153,230]
        rgba[~prediction&target&scored]=[255,175,55,230]
        errors=png_url(rgba,nearest=True)
    if not crs.is_projected or crs.linear_units!='metre':pixel_area=None
    else:pixel_area=abs(transform.a*transform.e-transform.b*transform.d)
    result=dict(context_id=identity,site_id=site.site_id,region=catalog.region.key,cached=False,
        model=dict(key=model_key,name=model_path.parent.name,epoch=model.epoch,architecture=model.architecture_label,
                   input_bands=names,threshold=threshold,threshold_strategy=threshold_strategy,threshold_selection=threshold_info),
        metrics=metrics,terrain=terrain,routing=routing,reference_kind=reference if reference==SEED_REFERENCE else 'local_vector_unverified' if truth is not None else 'none',
        stats=dict(valid_pixels=int(valid.sum()),predicted_pixels=int(prediction.sum()),predicted_ratio=float(prediction.sum()/valid.sum()),
                   area_km2=float(prediction.sum()*pixel_area/1e6) if pixel_area is not None else None,
                   mean_probability=float(probability[valid].mean()),seconds=time.monotonic()-start),
        prediction_overlay=overlay(prediction,(46,153,255,180)),error_overlay=errors)
    if reference == 'none':
        height, width = prediction.shape
        map_transform, cols, rows = calculate_default_transform(
            crs, 'EPSG:3857', width, height, *array_bounds(height, width, transform),
        )
        scale = min(1, 768 / max(cols, rows))
        map_cols, map_rows = max(1, round(cols * scale)), max(1, round(rows * scale))
        map_transform *= Affine.scale(cols / map_cols, rows / map_rows)
        projected = np.zeros((map_rows, map_cols), dtype=np.uint8)
        reproject(
            prediction.astype(np.uint8), projected,
            src_transform=transform, src_crs=crs,
            dst_transform=map_transform, dst_crs='EPSG:3857',
            resampling=Resampling.nearest,
        )
        result['map_overlay'] = overlay(projected.astype(bool), (46, 153, 255, 255))
        result['overlay_extent'] = list(array_bounds(map_rows, map_cols, map_transform))
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False));temporary.replace(cache)
    return result
