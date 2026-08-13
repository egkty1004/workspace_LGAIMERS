#!/usr/bin/env bash
# run_deploy.sh — 전체 데이터 재학습 (LGB 10시드 → MLP 10시드 순차 실행)
set -e
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="${LGAIMERS_REPRO_ROOT:-$SCRIPT_DIR}"
cd "$REPRO_ROOT"
PY="${AIMERS9_PYTHON:-python3}"
mkdir -p logs

echo "=== [1/2] LGB 10시드 full data ==="
"$PY" deploy_train_lgb.py 2>&1 | tee logs/deploy_lgb.log
echo "LGB exit: $?"

echo "=== [2/2] MLP 10시드 full data (CUDA_VISIBLE_DEVICES=0) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" deploy_train_mlp.py 2>&1 | tee logs/deploy_mlp.log
echo "MLP exit: $?"

echo "=== deploy train 완료 ==="
ls -la submit/model/
