#!/bin/bash
# 七省批量：备份旧workspace -> 发布 -> tile去重
set -u
cd /home/lake/lakes
PROVS="gansu shaanxi shandong sichuan jiangxi"
OLD_WS="gansu shaanxi"
TS=$(date +%Y%m%d_%H%M%S)

echo "=== STEP 1: backup old manifests ==="
for R in $OLD_WS; do
  WS=/home/lake/lakes/data/workspaces/default/logical_patches/$R
  ARC=$WS/archives/pre_seed_rebuild_$TS
  if [ -f $WS/manifest.csv ]; then
    mkdir -p $ARC
    cp -a $WS/manifest.csv $ARC/
    [ -d $WS/preview ] && cp -a $WS/preview $ARC/
    echo "backed up $R -> $ARC"
  fi
done

echo "=== STEP 2: publish ==="
for R in $PROVS; do
  echo "== publish $R"
  python3 scripts/publish_region_all.py --region $R 2>&1 | tail -1
done

echo "=== STEP 3: dedup ==="
for R in $PROVS; do
  echo "== dedup $R"
  python3 scripts/apply_tile_dedup_status.py --region $R 2>&1 | tail -12
done
echo "=== ALL_BATCH_STEPS_DONE ==="
