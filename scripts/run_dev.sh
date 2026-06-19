#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m lakes_browser.server --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
