#!/bin/bash
cd /home/gpu_01/workspace_LGAIMERS/repro_979
export CUDA_VISIBLE_DEVICES=0
PY=/home/gpu_01/.conda/envs/aimers9/bin/python
echo "=== START gen $(date) ==="
$PY gen_lgb_f3_adopted.py 2>&1 | tee experiments/gen_adopted_run.log
echo "GEN_EXIT=${PIPESTATUS[0]}"
echo "=== START blend $(date) ==="
$PY e6c_blend_adopted.py 2>&1 | tee experiments/e6c_blend_adopted_run.log
echo "BLEND_EXIT=${PIPESTATUS[0]}"
echo "=== DONE $(date) ==="
