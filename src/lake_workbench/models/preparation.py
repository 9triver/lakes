"""Background preparation and cache-aware site discovery for model comparisons."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np
import pyogrio
import rasterio
from rasterio.warp import transform_bounds

from lake_workbench.automatic_labels.service import generate_derived_label
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.models.comparison import _source, comparison_context, SEED_REFERENCE, _REFERENCE_LOCK
from lake_workbench.models.features import aligned_slope, dem_name, _valid_dem
from lake_workbench.models.slope_cache import slope_cache_paths
from lake_workbench.models.random_site_index import remember_slope_ready
from lake_workbench.models.preparation_progress import report, reporting
from lake_workbench.paths import PROJECT_ROOT

_LOCK=threading.RLock()
_SLOTS=threading.BoundedSemaphore(2)
_JOBS={}
_SITE_CACHE={}


def _seed(catalog,site,row,workspace_dir=None):
    asset=str(row.get('product') or row.get('product_id') or '')
    payload=dict(asset_id=asset,product=asset,extent=list(site.bbox),max_dimension=2048)
    roots=[PROJECT_ROOT/'data/model_comparison_references'/catalog.region.key]
    if workspace_dir is not None:roots.append(Path(workspace_dir))
    for root in roots:
        result=generate_derived_label(catalog,site,SEED_REFERENCE,payload,root,cache_only=True)
        if result is not None:
            return root/SEED_REFERENCE/f"{result['label_id']}.npz"
    return None


def _slope_ready(path):
    # Reconstruct the existing cache identity; never download while listing sites.
    with rasterio.open(path) as src:
        if not src.crs or not src.crs.is_projected:return False
        left,bottom,right,top=src.bounds
        bounds=transform_bounds(src.crs,'EPSG:4326',left-180,bottom-180,right+180,top+180,densify_pts=21)
        west,south,east,north=bounds
        tiles=[]
        for lat in range(math.floor(south),math.floor(north)+1):
            for lon in range(math.floor(west),math.floor(east)+1):
                name=dem_name(lat,lon)
                candidates=[PROJECT_ROOT/'data/terrain/copernicus_glo30'/f'{name}.tif',
                            *sorted((PROJECT_ROOT/'outputs').glob(f'*/dem/{name}.tif')),
                            PROJECT_ROOT/f'outputs/glo30_gansu_21981/dem_N{lat}_E{lon}.tif']
                tile=next((p for p in candidates if p.exists() and _valid_dem(str(p),p.stat().st_mtime_ns,p.stat().st_size)),None)
                if tile is None:
                    digest=hashlib.sha256(json.dumps([name,[round(v,7) for v in bounds]]).encode()).hexdigest()[:20]
                    tile=PROJECT_ROOT/'data/terrain/copernicus_clips'/f'{name}_{digest}.tif'
                    if not tile.is_file():return False
                tiles.append(tile)
        cache,legacy=slope_cache_paths(src,path,tiles)
        for candidate in [cache,legacy]:
            if candidate.is_file():
                slope=np.load(candidate,allow_pickle=False,mmap_mode='r')
                if slope.shape==src.shape:return True
        return False


def _comparison_site(catalog,site,workspace_dir):
    item=dict(site_id=site.site_id,display_name=site.display_name,region=catalog.region.key,
              seed_cached=False,slope_cached=False,imagery_available=False,rank=2,cache_checked=True)
    try:
        row,_=_source(catalog,site)
        item['imagery_available']=True
        item['asset']=str(row.get('product') or row.get('product_id') or '')
        item['seed_cached']=_seed(catalog,site,row,workspace_dir) is not None
        item['slope_cached']=_slope_ready(Path(row['tci_path']))
    except Exception as error:item['note']=str(error)
    item['rank']=0 if item['seed_cached'] and item['slope_cached'] else 1 if item['seed_cached'] else 2
    return item


def _comparison_directory(catalog):
    # Names do not need polygons, external-water layers or imagery cache checks.
    metadata=getattr(catalog.region,'site_metadata',None)
    if metadata is not None and Path(metadata).is_file() and not getattr(catalog,'_loaded',False):
        frame=pyogrio.read_dataframe(metadata,layer='sites',columns=['site_id','display_name'],read_geometry=False)
        return [dict(site_id=str(row.site_id),display_name=str(row.display_name or row.site_id)) for row in frame.itertuples(index=False)]
    return [dict(site_id=site.site_id,display_name=site.display_name) for site in catalog.sites]


def iter_comparison_sites(catalog,workspace_id,workspace_dir=None):
    """Send selectable names before checking terrain/seed caches. Never download."""
    yield dict(type='progress',message='正在读取观测区域目录',completed=0,total=None)
    cache_key=(workspace_id,catalog.region.key)
    with _LOCK:cached=_SITE_CACHE.get(cache_key)
    if cached and time.monotonic()-cached[0]<30:
        result=cached[1]
        yield dict(type='sites',items=result['items'],completed=result['checked'],total=result['total'])
        yield dict(type='complete',completed=result['checked'],total=result['total'])
        return
    directory=_comparison_directory(catalog);total=len(directory)
    initial=[dict(**item,region=catalog.region.key,seed_cached=False,slope_cached=False,imagery_available=False,rank=2,cache_checked=False) for item in directory]
    yield dict(type='sites',items=initial,completed=0,total=total)
    # Keep every site selectable, but bound automatic readiness I/O to ten sites.
    items_by_id={item['site_id']:item for item in initial}
    checked=0
    for entry in directory[:10]:
        site=catalog.get_site(entry['site_id'])
        item=_comparison_site(catalog,site,workspace_dir) if site is not None else dict(items_by_id[entry['site_id']],cache_checked=True)
        items_by_id[entry['site_id']]=item
        checked+=1
        yield dict(type='batch',items=[item],completed=checked,total=total)
    items=sorted(items_by_id.values(),key=lambda item:(item['rank'],item['display_name'] or item['site_id'],item['site_id']))
    result=dict(items=items,total=total,checked=checked,cache_basis='current_default_imagery')
    with _LOCK:_SITE_CACHE[cache_key]=(time.monotonic(),result)
    yield dict(type='complete',completed=checked,total=total,message=f'已自动检查前{checked}个区域，其余区域未自动检查')


def comparison_sites(catalog,workspace_id,workspace_dir=None):
    items={}
    for event in iter_comparison_sites(catalog,workspace_id,workspace_dir):
        for item in event.get('items',[]):items[item['site_id']]=item
    ordered=sorted(items.values(),key=lambda item:(item['rank'],item['display_name'] or item['site_id'],item['site_id']))
    return dict(items=ordered,total=len(ordered),cache_basis='current_default_imagery')


def _public(job):
    return {key:deepcopy(value) for key,value in job.items() if key!='context'} | ({'context':job['context']} if job.get('context') else {})


def prepare(catalog,site,workspace_id,workspace_dir=None,asset='',reference=SEED_REFERENCE,retry='0'):
    try:
        if reference=='single_model_terrain':
            # Match single-model inference's highest-valid-coverage imagery exactly.
            rows=catalog._model_validation_rows(site,5)
            if not rows:raise FileNotFoundError('当前区域没有可用于模型预测的五波段影像')
            row=sorted(rows,key=lambda item:float(item.get('valid_ratio',0) or 0),reverse=True)[0]
        else:
            row,_=_source(catalog,site,asset)
        path=Path(row['tci_path'])
        signature=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
    except Exception as error:
        return dict(id='',status='failed',error=str(error),progress={},context=None)
    key=(workspace_id,catalog.region.key,site.site_id,asset,reference,str(retry),signature)
    with _LOCK:
        if key in _JOBS:return _public(_JOBS[key])
        if len(_JOBS)>64:
            for old in list(_JOBS):
                if _JOBS[old]['status'] in {'ready','failed'}:del _JOBS[old]
                if len(_JOBS)<=32:break
        job=dict(id=uuid.uuid4().hex,status='queued',progress={},error='',warning='',context=None,started_at=time.time())
        _JOBS[key]=job
        threading.Thread(target=_run,args=(job,catalog,site,workspace_id,workspace_dir,asset,reference,row,path),daemon=True,name=f'prepare-{site.site_id}').start()
        return _public(job)


def _run(job,catalog,site,workspace_id,workspace_dir,asset,reference,row,path):
    def update(**event):
        category='terrain' if event['stage'].startswith('terrain') else 'seed' if event['stage'].startswith('seed') else 'context'
        with _LOCK:job['progress'][category]={**event,'updated_at':time.time()}
    with _SLOTS,reporting(update):
        with _LOCK:job['status']='running'
        try:
            if reference=='single_model_terrain':
                report('terrain_read','正在检查单模型预测影像的坡度缓存')
                with rasterio.open(path) as src:
                    image=src.read();valid=valid_pixel_mask(image,src.nodatavals,src.read_masks())
                del image
                aligned_slope(path,valid)
                remember_slope_ready(catalog.region.key, site.site_id)
                with _LOCK:
                    job['status']='ready'
                    _SITE_CACHE.pop((workspace_id,catalog.region.key),None)
                return
            if reference==SEED_REFERENCE:
                report('seed_check','正在检查同影像高置信种子缓存')
                cached=_seed(catalog,site,row,workspace_dir)
                if cached is not None:
                    target=PROJECT_ROOT/'data/model_comparison_references'/catalog.region.key/SEED_REFERENCE/cached.name
                    if cached!=target:
                        with _REFERENCE_LOCK:
                            target.parent.mkdir(parents=True,exist_ok=True)
                            for source in [cached,cached.with_suffix('.geojson')]:
                                destination=target.with_suffix(source.suffix)
                                temporary=destination.with_name(destination.name+'.'+job['id']+'.tmp')
                                shutil.copy2(source,temporary);temporary.replace(destination)
                else:
                    report('seed_prepare','正在生成光谱 ∩ OSM 高置信种子')
                    selected=str(row.get('product') or row.get('product_id') or '')
                    with _REFERENCE_LOCK:
                        generate_derived_label(catalog,site,SEED_REFERENCE,
                            dict(asset_id=selected,product=selected,extent=list(site.bbox),max_dimension=2048),
                            PROJECT_ROOT/'data/model_comparison_references'/catalog.region.key)
                report('seed_prepare','正在生成高置信种子共同预览')
            else:report('seed_skipped','当前参考类型无需生成高置信种子',1,1)
            context=comparison_context(catalog,site,asset,reference)
            if reference==SEED_REFERENCE:report('seed_done','光谱 ∩ OSM 高置信种子已就绪',1,1)
            report('terrain_read','正在读取影像有效范围并检查坡度缓存')
            try:
                with rasterio.open(path) as src:
                    image=src.read();valid=valid_pixel_mask(image,src.nodatavals,src.read_masks())
                del image
                aligned_slope(path,valid)
            except Exception as error:
                logging.getLogger(__name__).exception('Terrain preparation failed for %s',site.site_id)
                with _LOCK:job['warning']=f'坡度准备失败：{error}；五光谱模型仍可运行，可重试准备坡度。'
                report('terrain_failed',str(error))
            report('context_done','共同影像与参考标签已就绪',1,1)
            with _LOCK:
                job['context']=context;job['status']='ready'
                _SITE_CACHE.pop((workspace_id,catalog.region.key),None)
        except Exception as error:
            logging.getLogger(__name__).exception('Comparison preparation failed for %s',site.site_id)
            with _LOCK:job['error']=str(error);job['status']='failed'
        finally:
            with _LOCK:job['finished_at']=time.time()
