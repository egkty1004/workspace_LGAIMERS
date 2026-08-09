#!/usr/bin/env bash
# run_deploy.sh — 전체 데이터 재학습 (LGB 10시드 → MLP 10시드 순차 실행)
set -e
cd "$(dirname "$0")"
PY=/home/gpu_01/.conda/envs/aimers9/bin/python
mkdir -p logs

echo "=== [1/2] LGB 10시드 full data ==="
$PY deploy_train_lgb.py 2>&1 | tee logs/deploy_lgb.log
echo "LGB exit: $?"

echo "=== [2/2] MLP 10시드 full data (CUDA_VISIBLE_DEVICES=0) ==="
CUDA_VISIBLE_DEVICES=0 $PY deploy_train_mlp.py 2>&1 | tee logs/deploy_mlp.log
echo "MLP exit: $?"

echo "=== deploy train 완료 ==="
ls -la submit/model/
