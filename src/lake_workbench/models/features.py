"""Explicit optical-channel assembly and metric DEM slope for site inference."""
from __future__ import annotations

import hashlib
import json
import math
import logging
import time
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from lake_workbench.models.preparation_progress import report
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy import ndimage

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.models.slope_cache import METHOD, slope_cache_paths, publish_slope, promote_legacy
from lake_workbench.workspaces.file_lock import process_lock

OPTICAL = ['blue','green','red','nir','swir1']
_DEM_LOCK = threading.Lock()
_TILE_LOCKS: dict[str,threading.Lock] = {}


@lru_cache(maxsize=128)
def _valid_dem(path:str,mtime:int,size:int) -> bool:
    try:
        with rasterio.open(path) as src:
            src.read(1)
        return True
    except (OSError,rasterio.errors.RasterioError):
        return False


def feature_names(config: dict, channels: int) -> list[str]:
    names = config.get('input_bands') or config.get('input_features')
    if names:
        names = list(names)
    elif channels == 5:
        names = OPTICAL.copy()
    elif channels == 6 and config.get('slope_features'):
        names = OPTICAL + ['slope_degrees']
    else:
        raise ValueError(f'缺少明确的{channels}通道定义，无法安全推理')
    if len(names) != channels or any(name not in OPTICAL + ['slope_degrees'] for name in names):
        raise ValueError(f'不支持的模型输入通道：{names}')
    return names


def assemble_features(optical: np.ndarray, names: list[str], slope: np.ndarray | None = None) -> np.ndarray:
    if optical.shape[0] != 5:
        raise ValueError('比较输入必须是蓝、绿、红、NIR、SWIR1五波段影像')
    bands=[]
    for name in names:
        if name == 'slope_degrees':
            if slope is None or slope.shape != optical.shape[1:]:
                raise ValueError('缺少与影像对齐的坡度')
            bands.append(slope)
        else:
            bands.append(optical[OPTICAL.index(name)])
    return np.stack(bands).astype('float32')


def dem_name(lat: int, lon: int) -> str:
    return f"Copernicus_DSM_COG_10_{'N' if lat>=0 else 'S'}{abs(lat):02d}_00_{'E' if lon>=0 else 'W'}{abs(lon):03d}_00_DEM"


def cached_dem_tile(lat: int, lon: int) -> Path | None:
    name=dem_name(lat,lon)
    target=PROJECT_ROOT/'data/terrain/copernicus_glo30'/f'{name}.tif'
    candidates=[target,*sorted((PROJECT_ROOT/'outputs').glob(f'*/dem/{name}.tif')),
                PROJECT_ROOT/f'outputs/glo30_gansu_21981/dem_N{lat}_E{lon}.tif']
    return next((path for path in candidates if path.exists() and _valid_dem(str(path),path.stat().st_mtime_ns,path.stat().st_size)),None)


def dem_clip_path(lat: int, lon: int, bounds: tuple) -> Path:
    name=dem_name(lat,lon)
    identity=hashlib.sha256(json.dumps([name,[round(v,7) for v in bounds]]).encode()).hexdigest()[:20]
    return PROJECT_ROOT/'data/terrain/copernicus_clips'/f'{name}_{identity}.tif'


def dem_tile(lat: int, lon: int, bounds: tuple | None = None) -> Path:
    name=dem_name(lat,lon)
    target=PROJECT_ROOT/'data/terrain/copernicus_glo30'/f'{name}.tif'
    existing=cached_dem_tile(lat,lon)
    if existing:return existing
    if bounds is not None:
        return dem_clip(lat,lon,bounds)
    with _DEM_LOCK:
        lock=_TILE_LOCKS.setdefault(str(target),threading.Lock())
    with lock, process_lock(PROJECT_ROOT/'data/terrain/.locks'/f'{name}.download.lock'):
        existing=cached_dem_tile(lat,lon)
        if existing:return existing
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix('.partial')
        if not partial.exists():
            previous=sorted(target.parent.glob(target.stem+'.*.partial'),key=lambda p:p.stat().st_size,reverse=True)
            if previous:previous[0].replace(partial)
        url=f'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif'
        command=['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
                 '--max-time','180','--continue-at','-','--speed-time','30','--speed-limit','1024']
        proxy=os.environ.get('LAKES_DEM_PROXY','')
        if proxy:
            command+=['--proxy',proxy]
        for attempt in range(3):
            try:
                host_url=url if attempt%2==0 else url.replace('s3.eu-central-1.amazonaws.com','s3.amazonaws.com')
                subprocess.run([*command,'--output',str(partial),host_url],check=True)
                break
            except subprocess.CalledProcessError:
                if attempt==2:raise
        with rasterio.open(partial) as dem:
            if dem.crs is None:
                raise ValueError('DEM文件缺少坐标系')
            dem.read(1)
        partial.replace(target)
    return target


def _read_dem_window(url: str, bounds: tuple, options: dict):
    # Discard the URL's GDAL block cache when closing, including after a failed read.
    with rasterio.Env(**options, CPL_VSIL_CURL_NON_CACHED=url), rasterio.open(url) as src:
        west,south,east,north=bounds
        clipped=(max(west,src.bounds.left),max(south,src.bounds.bottom),min(east,src.bounds.right),min(north,src.bounds.top))
        window=from_bounds(*clipped,transform=src.transform)
        x0=max(0,math.floor(window.col_off)-3);y0=max(0,math.floor(window.row_off)-3)
        x1=min(src.width,math.ceil(window.col_off+window.width)+3)
        y1=min(src.height,math.ceil(window.row_off+window.height)+3)
        window=Window(x0,y0,x1-x0,y1-y0)
        data=src.read(1,window=window)
        profile=src.profile.copy();profile.update(width=data.shape[1],height=data.shape[0],
            transform=window_transform(window,src.transform),count=1,compress='deflate')
    return data,profile,window


def dem_clip(lat:int,lon:int,bounds:tuple) -> Path:
    """Reuse validated source data; batch jobs never download overlapping per-site clips."""
    name=dem_name(lat,lon)
    target=dem_clip_path(lat,lon,bounds)
    def usable():
        return target.exists() and _valid_dem(str(target),target.stat().st_mtime_ns,target.stat().st_size)
    if usable():return target
    with _DEM_LOCK:
        lock=_TILE_LOCKS.setdefault(str(target),threading.Lock())
    with lock, process_lock(PROJECT_ROOT/'data/terrain/.locks'/f'{name}.download.lock'):
        existing=cached_dem_tile(lat,lon)
        if existing:return existing
        if usable():return target
        if os.environ.get('LAKES_DEM_REQUIRE_SHARED_SOURCE')=='1':
            raise FileNotFoundError(f'共享 DEM {name} 尚未就绪，批量任务禁止回退到逐 Site 下载裁剪')
        if target.exists():
            invalid=target.with_suffix(f'.{time.time_ns()}.invalid')
            target.replace(invalid)
            logging.getLogger(__name__).warning('Unreadable DEM cache preserved at %s',invalid)
        options=dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                     GDAL_HTTP_CONNECTTIMEOUT=15,GDAL_HTTP_TIMEOUT=60,GDAL_HTTP_MAX_RETRY=2,
                     GDAL_HTTP_RETRY_DELAY=1,GDAL_HTTP_MULTIRANGE='SERIAL')
        proxy=os.environ.get('LAKES_DEM_PROXY','')
        if proxy:options['GDAL_HTTP_PROXY']=proxy.removeprefix('http://').removeprefix('https://')
        url=f'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif'
        for attempt in range(3):
            try:
                report('terrain_download', f'正在下载 DEM {name}，第 {attempt + 1}/3 次尝试')
                data,profile,window=_read_dem_window(url,bounds,options)
                report('terrain_download', f'DEM {name} 已读取 {data.shape[1]}×{data.shape[0]} 栅格')
                break
            except (OSError,rasterio.errors.RasterioError) as error:
                logging.getLogger(__name__).warning(
                    'DEM read failed: tile=%s bounds=%s attempt=%s/3 url=%s',
                    name,bounds,attempt+1,url,exc_info=True)
                if attempt==2:
                    raise RuntimeError(f'坡度源 DEM 读取失败：{name}，已尝试 3 次；'
                                       f'影像范围 {bounds}；原因：{error}') from error
                time.sleep(attempt+1)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with rasterio.open(temporary,'w',**profile) as dst:
                dst.write(data,1);dst.update_tags(source_url=url,source_window=str(window))
            if not _valid_dem(str(temporary),temporary.stat().st_mtime_ns,temporary.stat().st_size):
                raise ValueError(f'DEM 裁剪缓存写入后校验失败：{name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def aligned_slope(path: Path, valid: np.ndarray) -> tuple[np.ndarray,dict]:
    with rasterio.open(path) as src:
        if not src.crs or not src.crs.is_projected or src.crs.linear_units != 'metre':
            raise ValueError('坡度比较目前需要米制投影影像')
        left,bottom,right,top=src.bounds
        west,south,east,north=transform_bounds(src.crs,'EPSG:4326',left-180,bottom-180,right+180,top+180,densify_pts=21)
        coordinates=[(lat,lon) for lat in range(math.floor(south),math.floor(north)+1)
                     for lon in range(math.floor(west),math.floor(east)+1)]
        report('terrain_tiles', f'需要 {len(coordinates)} 块 DEM，正在检查缓存和下载', 0, len(coordinates))
        work=[(copy_context(),pair) for pair in coordinates]
        tiles=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for tile in pool.map(lambda item:item[0].run(dem_tile,*item[1],bounds=(west,south,east,north)),work):
                tiles.append(tile)
                report('terrain_tiles', f'DEM 瓦片已就绪 {len(tiles)}/{len(coordinates)}', len(tiles), len(coordinates))
        cache,legacy=slope_cache_paths(src,path,tiles)
        with process_lock(PROJECT_ROOT/'data/terrain/.locks'/f'{cache.stem}.lock'):
            if cache.exists():
                report('terrain_cache', '正在复用同一地理网格的坡度缓存')
                slope=np.load(cache,allow_pickle=False)
            elif legacy.exists():
                report('terrain_cache', '正在复用已有坡度并登记为地理网格共享缓存')
                slope=np.load(legacy,allow_pickle=False)
                if slope.shape != valid.shape or not np.isfinite(slope[valid]).all():
                    raise ValueError('旧坡度缓存未覆盖全部有效影像像素')
                promote_legacy(cache,legacy)
            else:
                report('terrain_slope', '正在重投影 DEM、计算 30 米坡度并对齐影像')
                step=30.
                x0,y0=math.floor(left/step)*step-90,math.ceil(top/step)*step+90
                w,h=math.ceil((right-x0)/step)+3,math.ceil((y0-bottom)/step)+3
                coarse_transform=from_origin(x0,y0,step,step)
                heights=np.full((h,w),np.nan,dtype='float32')
                for tile in tiles:
                    with rasterio.open(tile) as dem:
                        reproject(rasterio.band(dem,1),heights,src_transform=dem.transform,src_crs=dem.crs,
                                  src_nodata=dem.nodata,dst_transform=coarse_transform,dst_crs=src.crs,
                                  dst_nodata=np.nan,init_dest_nodata=False,resampling=Resampling.bilinear)
                kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype='float32')
                dx=ndimage.correlate(heights,kernel,mode='nearest')/(8*step)
                dy=ndimage.correlate(heights,kernel.T,mode='nearest')/(8*step)
                coarse=np.degrees(np.arctan(np.hypot(dx,dy))).astype('float32')
                coarse[~ndimage.binary_erosion(np.isfinite(heights),structure=np.ones((3,3)))]=np.nan
                slope=np.full(src.shape,np.nan,dtype='float32')
                reproject(coarse,slope,src_transform=coarse_transform,src_crs=src.crs,src_nodata=np.nan,
                          dst_transform=src.transform,dst_crs=src.crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
                if slope.shape != valid.shape or not np.isfinite(slope[valid]).all():
                    raise ValueError('坡度未覆盖全部有效影像像素，不能缩小评分范围来比较')
                publish_slope(cache,slope)
        if slope.shape != valid.shape or not np.isfinite(slope[valid]).all():
            raise ValueError('坡度未覆盖全部有效影像像素，不能缩小评分范围来比较')
        report('terrain_done', '坡度已就绪，覆盖全部有效影像像素', 1, 1)
        return np.nan_to_num(slope,nan=0.),dict(method=METHOD,source_tiles=[p.name for p in tiles],valid_coverage=1.)
