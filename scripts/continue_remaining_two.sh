#!/usr/bin/env bash
set -u
cd /home/lake/lakes
mkdir -p reports/chongqing_five_site_batches
exec >> reports/chongqing_five_site_batches/remaining_two_worker.log 2>&1
echo "CONTINUATION_START $(date -Is)"
while pgrep -f '[c]hongqing_serial_site_batch.py.*run' >/dev/null; do
  echo "WAITING_FOR_CURRENT_WORKER $(date -Is)"
  sleep 30
done
for n in 7 8 9 10; do
  id="batch_00\${n}"
  if [ ! -f "reports/chongqing_five_site_batches/\${id}_plan.json" ]; then
    echo "PREPARE \${id} $(date -Is)"
    CQ_BATCH_ID="\${id}" PYTHONPATH=src .venv/bin/python scripts/chongqing_serial_site_batch.py prepare || exit $?
  fi
  if [ ! -f "reports/chongqing_five_site_batches/\${id}_report.json" ] || ! grep -q '"status": "awaiting_human_review"' "reports/chongqing_five_site_batches/\${id}_report.json"; then
    echo "RUN \${id} $(date -Is)"
    CQ_BATCH_ID="\${id}" PYTHONPATH=src .venv/bin/python scripts/chongqing_serial_site_batch.py run || exit $?
  fi
done
echo "CONTINUATION_DONE $(date -Is)"
