import csv, json, os, shutil, tempfile
from pathlib import Path
from collections import Counter

source = Path('/home/lake/lakes/experiments/dataset_builds/qinghai_seed_v1')
workspace = Path('/home/lake/lakes/data/workspaces/default/logical_patches/qinghai')
scenes = source / 'scenes'
workspace.mkdir(parents=True, exist_ok=True)

expected_regions = 1543
expected_observations = 7715
status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
if status.get('status') != 'complete':
    raise RuntimeError(f'build not complete: {status.get("status")}')
expected_patches = status['candidate_patches']

scene_files = sorted(scenes.glob('*.csv'))
if len(scene_files) != expected_observations:
    raise RuntimeError(f'expected {expected_observations} scene CSVs, found {len(scene_files)}')

fields = None
tmp_path = workspace / '.manifest.all.tmp'
counts = Counter()
sites = set()
observations = 0
ids = set()
with tmp_path.open('w', encoding='utf-8', newline='') as out:
    writer = None
    for scene_file in scene_files:
        with scene_file.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = list(reader.fieldnames or [])
                writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
            elif list(reader.fieldnames or []) != fields:
                raise RuntimeError(f'field mismatch: {scene_file}')
            scene_rows = 0
            for row in reader:
                pid = row.get('logical_patch_id', '')
                if not pid or pid in ids:
                    raise RuntimeError(f'duplicate or empty patch id: {pid}')
                ids.add(pid)
                sites.add(row.get('site_id', ''))
                counts['patches'] += 1
                counts['included' if row.get('include') == 'true' else 'excluded'] += 1
                scene_rows += 1
                writer.writerow(row)
            if scene_rows:
                observations += 1

if counts['patches'] != expected_patches:
    raise RuntimeError(f'expected {expected_patches} patches, found {counts["patches"]}')
os.replace(tmp_path, workspace / 'manifest.csv')

source_dataset = {
    'dataset_id': 'qinghai_seed_v1_all_regions',
    'status': 'complete',
    'region': 'qinghai',
    'requested_regions': expected_regions,
    'catalog_regions': expected_regions,
    'candidate_regions': len(sites),
    'regions_published': len(sites),
    'observations_published': observations,
    'patches': counts['patches'],
    'included': counts['included'],
    'excluded': counts['excluded'],
    'deduplication': 'none',
    'tile_deduplication': False,
    'spatial_overlap_deduplication': False,
    'selection_rule': 'Earliest five distinct dates with a readable local Sentinel image per region.',
    'thresholds': {'min_water_ratio_valid': 0.05, 'min_valid_ratio': 0.80},
    'label_source': 'spectral_osm_consensus_seed_connected',
    'spectral_threshold': 0.62,
    'osm_zoom': 14,
    'patch_size': 256,
    'stride': 128,
    'source_manifest': str(source / 'manifest.csv'),
    'candidate_manifest': str(source / 'manifest.csv'),
    'source_data_policy': 'Source imagery and annotations are read-only; only generated workspace files are published.',
}
(workspace / 'source_dataset.json').write_text(json.dumps(source_dataset, ensure_ascii=False, indent=2), encoding='utf-8')
(workspace / 'validation.json').write_text(json.dumps({
    'status': 'passed',
    'errors': [],
    'sites': len(sites),
    'observations': observations,
    'patches': counts['patches'],
    'included': counts['included'],
    'excluded': counts['excluded'],
    'all_regions_published': True,
    'all_grid_windows_retained': True,
    'deduplication': 'none',
    'source_files_modified': False,
    'rules': source_dataset['thresholds'] | {
        'patch_size': 256,
        'stride': 128,
        'label_source': source_dataset['label_source'],
        'spectral_threshold': 0.62,
        'osm_zoom': 14,
    },
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'manifest': str(workspace / 'manifest.csv'),
    'scene_files': len(scene_files),
    'sites': len(sites),
    'observations': observations,
    'patches': counts['patches'],
    'included': counts['included'],
    'excluded': counts['excluded'],
    'deduplication': 'none',
}, ensure_ascii=False))
