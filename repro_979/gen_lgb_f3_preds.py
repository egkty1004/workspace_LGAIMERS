#!/usr/bin/env python3
"""gen_lgb_f3_preds.py — LGB F3 4폴드 로짓 예측 생성 (e6c_blend_folds 전제 조건).

F3 구성(원본 47 + platoon + count_state, 10시드 42~51 로짓 평균)으로
각 폴드 홀드아웃의 로짓 예측을 cache/ 에 npy로 저장한다.
  cache/preds_primary_lgb_f3.npy  (≤2023→2024 전체)
  cache/h2b_r2022_lgb_f3.npy      (≤2021→2022 R-only)
  cache/h2b_r2023_lgb_f3.npy
  cache/h2b_r2024_lgb_f3.npy
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common

SEEDS = list(range(42, 52))
OUT = {
    "primary": "cache/preds_primary_lgb_f3.npy",
    "r2022": "cache/h2b_r2022_lgb_f3.npy",
    "r2023": "cache/h2b_r2023_lgb_f3.npy",
    "r2024": "cache/h2b_r2024_lgb_f3.npy",
}


def main():
    t0 = time.time()
    print("[gen] LGB F3 4폴드 로짓 예측 생성", flush=True)
    train, base_feats = common.load_train()
    common.preprocess_for_submission(train)
    feats = common.get_feature_cols(
        pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = common.CAT_COLS + common.F3_EXTRA
    print(f"train: {train.shape} | features: {len(feats)}", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        zs = []
        for seed in SEEDS:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            zs.append(common.logit(m.predict(X_va, num_iteration=m.best_iteration)))
            print(f"  [{fn}] seed={seed} best_iter={m.best_iteration} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        z = np.mean(zs, axis=0)
        np.save(OUT[fn], z)
        p = common.sigmoid(z)
        print(f"  -> {fn}: BSS={common.score(p, yv):.1f} pred_mean={p.mean():.4f} "
              f"저장 {OUT[fn]}", flush=True)

    print(f"\n완료 (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
