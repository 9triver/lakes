#!/usr/bin/env python3
"""Rebuild Jiangsu first-5 patches with spectral∩OSM high-confidence seed water labels.

Water label per scene = spectral water (threshold 0.62) ∩ OSM blue water,
following lake_workbench.automatic_labels.service SPECTRAL_OSM_INTERSECTION.
Tile dedup: keep one representative region per tile, others excluded.
"""
from __future__ import annotations

import argparse, concurrent.futures, csv, json, math, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path('/home/lake/lakes')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'dataset_builds'))

import rebuild_hubei_first5_fullgrid as fullgrid
from build_shaanxi_first5 import (
    REGION, SOURCE_ROOT, WORKSPACE, OBSERVATION_FIELDS, TILE_FIELDS,
    write_rows, read_rows, normalized_tile, inventory, add_tile_column,
    candidate_metrics, choose_representatives,
)
from lake_workbench.automatic_labels.evidence_cache import cached_spectral_evidence, cached_osm_mask
from lake_workbench.automatic_labels.basemap import aligned_osm_rgb, osm_tile_settings
from lake_workbench.automatic_labels.spectral import (
    BACKGROUND_LABEL, WATER_LABEL, IGNORE_LABEL,
    STABLE_LABEL_MAX_DIMENSION, DEFAULT_WATER_THRESHOLD,
)
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.training.patch_processing import image_fingerprint, label_geometries, training_label_fingerprint, grid_cell_bounds

REGION = 'shaanxi'
SOURCE_ROOT = Path('/mnt/data/lakes-data/regions/shaanxi')
WORKSPACE = Path('/home/lake/lakes/data/workspaces/default/logical_patches/shaanxi')
SIZE, STRIDE = 256, 128
PIXELS = SIZE * SIZE
THRESHOLD = DEFAULT_WATER_THRESHOLD  # 0.62
OSM_ZOOM = 14
OSM_TILE_URL = osm_tile_settings()[0]
LABEL_SOURCE = 'spectral_osm_consensus_seed_connected'

FIELDS = [*fullgrid.FIELDS, 'tile']


def scene_water_mask(image_path: Path, region) -> tuple[np.ndarray, np.ndarray, object]:
    """Full-grid spectral∩OSM seed water mask in the image CRS grid."""
    with rasterio.open(image_path) as src:
        bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
    cache_dir = region.cache_dir
    spectral, spectral_key, _ = cached_spectral_evidence(
        cache_dir, image_path, bounds,
        max_dimension=STABLE_LABEL_MAX_DIMENSION, threshold=THRESHOLD, quality_path=None,
    )
    osm, _ = cached_osm_mask(
        cache_dir, spectral_key, bounds, OSM_ZOOM, OSM_TILE_URL,
        lambda: aligned_osm_rgb(bounds, OSM_ZOOM, spectral.crs, spectral.transform, spectral.shape, cache_dir),
    )
    valid = spectral.valid & osm.valid
    spectral_water = spectral.labels == WATER_LABEL
    spectral_water_valid = valid & spectral_water
    osm_water_valid = valid & osm.water
    consensus_seed = spectral_water_valid & osm.water
    # Seed + connected expansion (site spectral_osm_consensus): keep spectral
    # water components that contain a cross-source high-confidence seed.
    from lake_workbench.automatic_labels.service import _seeded_connected_mask
    connected = _seeded_connected_mask(spectral_water_valid, consensus_seed)
    water = connected & valid
    return water, valid, spectral


def process_scene(args):
    item, output_text, region = args
    output = Path(output_text)
    scene_key = item['sample_id']
    scene_csv = output / 'scenes' / f'{scene_key}.csv'
    scene_json = output / 'scenes' / f'{scene_key}.json'
    if scene_csv.is_file() and scene_json.is_file():
        return json.loads(scene_json.read_text())
    image_path = Path(item['image_path'])

    water_full, valid_full, spectral = scene_water_mask(image_path, region)
    label_path = output / 'labels' / f'{scene_key}.geojson'
    # Persist the seed-water mask as a GeoJSON snapshot (rasterized polygons) for the website.
    from rasterio.features import shapes
    geoms = [
        geom for geom, value in shapes(water_full.astype('uint8'), mask=water_full, transform=spectral.transform)
        if value == 1
    ]
    payload = {'type': 'FeatureCollection',
               'properties': {'source': LABEL_SOURCE, 'product_date': item['date'],
                              'method': LABEL_SOURCE, 'label_available': bool(geoms),
                              'threshold': THRESHOLD, 'osm_zoom': OSM_ZOOM},
               'features': [{'type': 'Feature', 'properties': {'label_value': 1, 'label_class': 'water'},
                             'geometry': g} for g in geoms]}
    fullgrid.atomic_json(label_path, payload)
    label_signature = training_label_fingerprint(label_path, payload)

    with rasterio.Env(GDAL_CACHEMAX=32*1024*1024):
        with rasterio.open(image_path) as src:
            signature = image_fingerprint(image_path, src)
            height, width = src.height, src.width
            # Seed water was computed on the spectral grid; resample back to the source grid.
            from rasterio.warp import reproject, Resampling
            water_src = np.zeros((src.height, src.width), dtype='uint8')
            reproject(water_full.astype('uint8'), water_src,
                      src_transform=spectral.transform, src_crs=spectral.crs,
                      dst_transform=src.transform, dst_crs=src.crs,
                      resampling=Resampling.nearest)
            strip_height = max(SIZE, min(height, max(SIZE, (4_000_000 // width) // STRIDE * STRIDE)))
            rows = []
            included = 0
            reasons = Counter()
            for base_row in range(0, height, strip_height):
                read_height = min(strip_height + SIZE - STRIDE, height - base_row)
                win = Window(0, base_row, width, read_height)
                image = src.read(window=win)
                masks = src.read_masks(window=win)
                valid = valid_pixel_mask(image, src.nodatavals, masks)
                water = water_src[base_row:base_row + read_height].astype(bool) & valid
                valid_sums = fullgrid.integral(valid)
                water_sums = fullgrid.integral(water)
                del image, masks, valid, water
                for row in range(base_row, min(base_row + strip_height, height), STRIDE):
                    for col in range(0, width, STRIDE):
                        valid_count = fullgrid.window_sum(valid_sums, row - base_row, col, read_height, width)
                        water_count = fullgrid.window_sum(water_sums, row - base_row, col, read_height, width)
                        low_water = valid_count == 0 or water_count * 20 < valid_count
                        low_valid = valid_count * 5 < PIXELS * 4
                        include = not (low_water or low_valid)
                        included += int(include)
                        reason = ';'.join(s for s, b in [('water_below_5_percent', low_water), ('valid_below_80_percent', low_valid)] if b)
                        reasons[reason or 'included'] += 1
                        bounds = grid_cell_bounds(src.transform, row, col, SIZE)
                        rows.append({'logical_patch_id': fullgrid.logical_patch_id(scene_key, 0, row, col, SIZE, signature, label_signature),
                                     'sample_id': scene_key, 'site_id': item['site_id'], 'region': REGION, 'image_index': 0,
                                     'image_path': str(image_path), 'label_path': str(label_path),
                                     'row_off': row, 'col_off': col, 'logical_size': SIZE, 'grid_stride': STRIDE,
                                     'window_width': SIZE, 'window_height': SIZE,
                                     'image_fingerprint': signature, 'label_fingerprint': label_signature,
                                     'valid_ratio': f'{valid_count/PIXELS:.9f}', 'valid_pixels': valid_count,
                                     'water_pixels': water_count,
                                     'water_ratio_valid': f'{water_count/valid_count if valid_count else 0:.9f}',
                                     'ignore_pixels': PIXELS - valid_count,
                                     'bounds_left': bounds[0], 'bounds_bottom': bounds[1],
                                     'bounds_right': bounds[2], 'bounds_top': bounds[3],
                                     'crs': str(src.crs), 'product_name': item['product_name'],
                                     'product_date': item['date'], 'selection_rank': item['selection_rank'],
                                     'label_source': LABEL_SOURCE,
                                     'include': 'true' if include else 'false',
                                     'review_status': 'included' if include else 'excluded',
                                     'exclude_reason': reason})
            expected = math.ceil(height / STRIDE) * math.ceil(width / STRIDE)
            if len(rows) != expected:
                raise AssertionError(f'{scene_key}: {len(rows)} != {expected}')
    fullgrid.write_csv(scene_csv, rows, FIELDS)
    summary = {**item, 'patches': len(rows), 'included': included, 'excluded': len(rows) - included,
               'width': width, 'height': height, 'expected_grid_windows': expected, 'reasons': dict(reasons)}
    fullgrid.atomic_json(scene_json, summary)
    return summary


def validate_publish(output, representative_items, representative_stats, publish):
    if len(representative_stats) != len(representative_items):
        raise ValueError('Not every representative scene completed')
    path = output / 'manifest.csv'
    temp = path.with_suffix('.csv.tmp')
    totals = Counter(); sites = Counter(); ids = set(); errors = []
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
        for item in representative_items:
            source = output / 'scenes' / f"{item['sample_id']}.csv"
            with source.open(encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    pid = row['logical_patch_id']
                    if pid in ids: errors.append(f'duplicate {pid}')
                    ids.add(pid)
                    v, w = int(row['valid_pixels']), int(row['water_pixels'])
                    expected = v > 0 and w * 20 >= v and v * 5 >= PIXELS * 4
                    if (row['review_status'] == 'included') != expected or (row['include'] == 'true') != expected:
                        errors.append(f'classification mismatch {pid}')
                    totals['patches'] += 1; totals['included' if expected else 'excluded'] += 1
                    sites[row['site_id']] += 1
                    writer.writerow(row)
    if errors: raise ValueError(str(errors[:10]))
    if totals['patches'] != sum(s['expected_grid_windows'] for s in representative_stats):
        raise AssertionError('Grid windows were lost during merge')
    os.replace(temp, path)
    validation = {'status': 'passed', 'errors': errors, 'sites': len(sites), 'observations': len(representative_stats), **totals,
                  'all_grid_windows_retained': True, 'source_files_modified': False,
                  'rules': {'min_water_ratio_valid': .05, 'min_valid_ratio': .8, 'patch_size': SIZE, 'stride': STRIDE,
                            'label_source': LABEL_SOURCE, 'spectral_threshold': THRESHOLD, 'osm_zoom': OSM_ZOOM}}
    fullgrid.atomic_json(output / 'validation.json', validation)
    if publish:
        import shutil
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive = WORKSPACE / 'archives' / f'before_seed_{stamp}'
        archive.mkdir(parents=True, exist_ok=False)
        for name in ('manifest.csv', 'source_dataset.json'):
            old = WORKSPACE / name
            if old.exists(): shutil.copy2(old, archive / name)
        published = WORKSPACE / 'manifest.csv.seed.tmp'
        shutil.copyfile(path, published)
        os.replace(published, WORKSPACE / 'manifest.csv')
        validation['published_manifest'] = str(WORKSPACE / 'manifest.csv')
        fullgrid.atomic_json(WORKSPACE / 'source_dataset.json', {
            'dataset_id': output.name, 'label_source': LABEL_SOURCE,
            'spectral_threshold': THRESHOLD, 'osm_zoom': OSM_ZOOM,
            'tile_representatives': str(output / 'tile_representatives.csv'),
            'duplicate_regions': str(output / 'duplicate_regions.csv'),
            'dataset_summary': str(output / 'dataset.json'),
            'validation': validation,
        }, )
        fullgrid.atomic_json(output / 'validation.json', validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--expected-regions', type=int, default=80)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()

    output = args.output.resolve()
    source = SOURCE_ROOT.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('output must be outside the lakes source data tree')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('scenes', 'labels'):
        (output / name).mkdir(exist_ok=True)

    if args.publish:
        occupied = [WORKSPACE / name for name in ('manifest.csv', 'source_dataset.json') if (WORKSPACE / name).exists()]
        if occupied:
            raise FileExistsError('Refusing to replace existing shaanxi workspace data: ' + ', '.join(map(str, occupied)))

    observations, plan, unavailable, missing_sources, tile_conflicts, missing_labels = inventory()
    plan.update({'dataset_id': output.name, 'expected_regions_from_request': args.expected_regions,
                 'difference_from_expected_regions': plan['metadata_regions'] - args.expected_regions,
                 'source_root': str(source), 'output': str(output),
                 'label_source': LABEL_SOURCE, 'spectral_threshold': THRESHOLD, 'osm_zoom': OSM_ZOOM})
    write_rows(output / 'observations.csv', observations, OBSERVATION_FIELDS)
    write_rows(output / 'unavailable_regions.csv', unavailable, ['site_id', 'tile', 'image_count', 'reason'])
    write_rows(output / 'missing_sources.csv', missing_sources, ['site_id', 'tile', 'product_id', 'date', 'image_path'])
    write_rows(output / 'tile_conflicts.csv', tile_conflicts, ['site_id', 'tile', 'tile_counts'])
    fullgrid.atomic_json(output / 'plan.json', plan)
    if args.plan_only:
        print(json.dumps({'status': 'planned', **plan}, ensure_ascii=False), flush=True)
        return 0

    region = REGION
    from lake_workbench.regions.config import load_region_configs
    region_obj = load_region_configs()[0][REGION]
    fullgrid.REGION = REGION
    fullgrid.ROOT = source
    fullgrid.WORKSPACE = WORKSPACE
    selected_stats = []
    failures = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        tasks = {pool.submit(process_scene, (item, str(output), region_obj)): item for item in observations}
        for future in concurrent.futures.as_completed(tasks):
            item = tasks[future]
            try:
                selected_stats.append(future.result())
            except Exception as error:
                failures.append({**item, 'error': repr(error)})
                print(json.dumps({'event': 'error', 'sample_id': item.get('sample_id'), 'error': repr(error)}, ensure_ascii=False), flush=True)
            completed = len(selected_stats) + len(failures)
            if completed % 25 == 0 or completed == len(observations):
                status = {'status': 'building', 'completed_observations': completed, 'total_observations': len(observations),
                          'completed_regions': len({row['site_id'] for row in selected_stats}),
                          'candidate_patches': sum(int(row['patches']) for row in selected_stats),
                          'candidate_included': sum(int(row['included']) for row in selected_stats),
                          'candidate_excluded': sum(int(row['excluded']) for row in selected_stats),
                          'errors': len(failures), 'elapsed_seconds': round(time.monotonic() - started)}
                fullgrid.atomic_json(output / 'status.json', status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
    if failures:
        fullgrid.atomic_json(output / 'failures.json', failures)
        raise SystemExit(f'{len(failures)} observations failed; no workspace manifest was published')

    metrics = candidate_metrics(observations, selected_stats, output)
    representatives, duplicates = choose_representatives(metrics)
    representative_ids = {row['site_id'] for row in representatives}
    representative_items = [row for row in observations if row['site_id'] in representative_ids]
    representative_stats = [row for row in selected_stats if row['site_id'] in representative_ids]
    write_rows(output / 'tile_representatives.csv', representatives, TILE_FIELDS)
    write_rows(output / 'duplicate_regions.csv', duplicates,
               ['tile', 'site_id', 'retained_site_id', 'patches', 'included', 'excluded', 'included_water_pixels', 'reason'])

    for item in observations:
        add_tile_column(output / 'scenes' / f"{item['sample_id']}.csv", item['tile'])
    fullgrid.FIELDS = FIELDS
    validation = validate_publish(output, representative_items, representative_stats, args.publish)
    validation.update({
        'candidate_regions': len(metrics),
        'candidate_observations': len(observations),
        'candidate_patches': sum(row['patches'] for row in metrics.values()),
        'candidate_included': sum(row['included'] for row in metrics.values()),
        'candidate_excluded': sum(row['excluded'] for row in metrics.values()),
        'unique_tiles': len(representatives),
        'duplicate_regions_excluded_by_tile': len(duplicates),
        'tile_representatives': str(output / 'tile_representatives.csv'),
        'duplicate_regions': str(output / 'duplicate_regions.csv'),
        'tile_column_present': True,
        'source_files_modified': False,
    })
    fullgrid.atomic_json(output / 'validation.json', validation)
    dataset = {'dataset_id': output.name, 'status': 'complete',
               'requested_regions': args.expected_regions,
               'catalog_regions': plan['metadata_regions'],
               'catalog_regions_minus_requested': plan['difference_from_expected_regions'],
               'candidate_regions': len(metrics), 'unique_tiles': len(representatives),
               'retained_regions': len(representatives),
               'duplicate_regions_excluded_by_tile': len(duplicates),
               'candidate_observations': len(observations),
               'retained_observations': len(representative_items),
               'retained_patches': validation['patches'],
               'included': validation['included'],
               'excluded_by_quality_threshold': validation['excluded'],
               'thresholds': {'min_water_ratio_valid': 0.05, 'min_valid_ratio': 0.80},
               'label_source': LABEL_SOURCE, 'spectral_threshold': THRESHOLD, 'osm_zoom': OSM_ZOOM,
               'selection_rule': plan['selection_rule'],
               'representative_rule': plan['representative_rule'],
               'patch_size': SIZE, 'stride': STRIDE,
               'source_root': str(source), 'output': str(output),
               'workspace_manifest': str(WORKSPACE / 'manifest.csv') if args.publish else '',
               'source_data_policy': 'Source imagery and annotations are read-only; only generated workspace files are published.'}
    fullgrid.atomic_json(output / 'dataset.json', dataset)
    status = {**validation, 'status': 'complete', 'elapsed_seconds': round(time.monotonic() - started), 'output': str(output)}
    fullgrid.atomic_json(output / 'status.json', status)
    print(json.dumps({**dataset, 'validation': validation}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
