"""Terrain cache identities depend on spatial grids, never optical acquisition dates."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from lake_workbench.paths import PROJECT_ROOT

METHOD='Horn slope on globally aligned metric30m grid, bilinear to native image grid, v1'

def slope_cache_paths(src,path,tiles):
    old=dict(method=METHOD,image=[str(path.resolve()),path.stat().st_mtime_ns],
        tiles=[[str(p),p.stat().st_mtime_ns] for p in tiles],transform=list(src.transform),shape=list(src.shape))
    grid=dict(method=METHOD,cache_version='spatial-grid-v2',crs=src.crs.to_wkt(),
        tiles=[[str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns] for p in tiles],
        transform=list(src.transform),shape=list(src.shape))
    folder=PROJECT_ROOT/'data/terrain/aligned_slope'
    def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]
    return folder/f'grid_{key(grid)}.npy',folder/f'{key(old)}.npy'

def publish_slope(cache,slope):
    cache.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+cache.stem+'.',suffix='.tmp',dir=cache.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            np.save(handle,slope,allow_pickle=False);handle.flush();os.fsync(handle.fileno())
        os.replace(name,cache)
    finally:Path(name).unlink(missing_ok=True)

def promote_legacy(cache,legacy):
    cache.parent.mkdir(parents=True,exist_ok=True)
    # Existing grid entry wins. Hard linking adds no array storage and keeps old paths.
    try:os.link(legacy,cache)
    except FileExistsError:pass
