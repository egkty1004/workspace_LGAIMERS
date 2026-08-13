#!/bin/bash
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="${LGAIMERS_REPRO_ROOT:-$SCRIPT_DIR}"
cd "$REPRO_ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=0
PY="${AIMERS9_PYTHON:-python3}"
echo "=== START gen $(date) ==="
"$PY" gen_lgb_f3_adopted.py 2>&1 | tee experiments/gen_adopted_run.log
echo "GEN_EXIT=${PIPESTATUS[0]}"
echo "=== START blend $(date) ==="
"$PY" e6c_blend_adopted.py 2>&1 | tee experiments/e6c_blend_adopted_run.log
echo "BLEND_EXIT=${PIPESTATUS[0]}"
echo "=== DONE $(date) ==="
