#!/usr/bin/env python3
"""deploy_train_lgb.py — 채택 피처 포함 F3 전체 데이터 10시드 학습 → submit/model/f3_s{seed}.txt.

Wave D 채택 피처(asof_n_bucket, score_diff_binary)를 포함한 F3 구성을 **전체 데이터
(2019~2024, 1,475,092행)**로 10시드(42~51) 재학습해 제출용 LightGBM 모델을 저장한다.

- feats = base + platoon + count_state + asof_n_bucket + score_diff_binary (채택 2피처만)
- cats  = CAT_COLS + ["platoon", "count_state"] — 신규 int8/float32 피처는 범주형 아님
          (screen_all.py §핵심, gen_lgb_f3_adopted.py와 동일)
- ⚠️ early_stopping 없음 (full data) — num_boost_round 고정.
  GIHO 원본 제출(f3_s42..51)은 fold best_iter_primary(95) 기반 전체 학습 num_boost_round=109
  사용(train_meta.json). 채택 피처 fold(gen_adopted_run.log) primary best_iter median≈102,
  전체 데이터 기준 109 유지 (동일 방식).
- 각 시드 → submit/model/f3_s{seed}.txt (lgb.Booster.save_model)
- meta → submit/model/train_meta.json (features/cats/num_boost_round/params)

공통 전처리는 repro_979/common.py(학습/추론 단일 모듈)를 그대로 사용.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common

SEEDS = list(range(42, 52))  # 10시드
NUM_BOOST_ROUND = 109  # 전체 학습 고정 라운드 (GIHO 원본 deploy와 동일 방식)
CATS = common.CAT_COLS + ["platoon", "count_state"]  # ⚠️ 신규 int8/float32는 numeric
ADOPTED_EXTRA = ["platoon", "count_state", "asof_n_bucket", "score_diff_binary"]
MODEL_DIR = "submit/model"


def main():
    t0 = time.time()
    print("[deploy_lgb] 채택 피처 포함 F3 전체 데이터 10시드 학습", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns
    base_feats = [c for c in test_cols if c != common.ID]
    feats = base_feats + ADOPTED_EXTRA
    X = train[feats]
    y = train[common.TARGET]
    print(f"train: {train.shape} | feats: {len(feats)} (base {len(base_feats)} "
          f"+ {len(ADOPTED_EXTRA)}) | cats: {CATS} | rounds: {NUM_BOOST_ROUND}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    for seed in SEEDS:
        params = dict(common.PARAMS)
        params["seed"] = seed
        dtr = lgb.Dataset(X, y, categorical_feature=CATS)
        m = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND)
        path = os.path.join(MODEL_DIR, f"f3_s{seed}.txt")
        m.save_model(path)
        print(f"  seed={seed} trees={m.num_trees()} 저장 {path} "
              f"({time.time()-t0:.0f}s)", flush=True)

    meta = dict(
        seeds=SEEDS,
        num_boost_round=NUM_BOOST_ROUND,
        features=feats,
        cats=CATS,
        adopted_extra=ADOPTED_EXTRA,
        params={k: common.PARAMS[k] for k in [
            "objective", "metric", "learning_rate", "num_leaves",
            "min_data_in_leaf", "feature_fraction", "bagging_fraction",
            "bagging_freq", "num_threads", "verbosity", "deterministic"]},
        total_time=time.time() - t0,
    )
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n완료: submit/model/f3_s*.txt (10개) + train_meta.json "
          f"(총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
