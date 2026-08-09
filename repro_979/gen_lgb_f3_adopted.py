#!/usr/bin/env python3
"""gen_lgb_f3_adopted.py — Wave D 채택 피처 LGB 10시드 캐시 생성 + 10시드 기준선 재계산.

채택 피처(asof_n_bucket, score_diff_binary) 포함 F3 4폴드 10시드(42~51) 로짓 예측을
cache/ 에 npy로 저장한다. 동시에 "채택 피처 없는 F3 10시드 기준선"(deterministic=True,
cats 제한)을 재계산해 gain 비교용 캐시로 저장한다.

⚠️ cats 분리 (screen_all.py §핵심): 신규 int8/float32 피처는 범주형이 아니다.
    cats = CAT_COLS + ["platoon", "count_state"] (cats = CAT_COLS + F3_EXTRA 금지).

Adopted 캐시 (feats = base + platoon + count_state + asof_n_bucket + score_diff_binary):
  cache/fe_adopted_primary_lgb_f3.npy  (≤2023→2024 전체)
  cache/fe_adopted_r2022_lgb_f3.npy    (≤2021→2022 R-only)
  cache/fe_adopted_r2023_lgb_f3.npy
  cache/fe_adopted_r2024_lgb_f3.npy
Baseline 캐시 (feats = base + platoon + count_state, 채택/신규 피처 없음):
  cache/fe_base_primary_lgb_f3.npy
  cache/fe_base_r2022_lgb_f3.npy
  cache/fe_base_r2023_lgb_f3.npy
  cache/fe_base_r2024_lgb_f3.npy
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common

SEEDS = list(range(42, 52))  # 10시드
CATS = common.CAT_COLS + ["platoon", "count_state"]  # ⚠️ 신규 int8/float32는 numeric
ADOPTED_EXTRA = ["platoon", "count_state", "asof_n_bucket", "score_diff_binary"]
BASE_EXTRA = ["platoon", "count_state"]

OUT_ADOPTED = {
    "primary": "cache/fe_adopted_primary_lgb_f3.npy",
    "r2022": "cache/fe_adopted_r2022_lgb_f3.npy",
    "r2023": "cache/fe_adopted_r2023_lgb_f3.npy",
    "r2024": "cache/fe_adopted_r2024_lgb_f3.npy",
}
OUT_BASE = {
    "primary": "cache/fe_base_primary_lgb_f3.npy",
    "r2022": "cache/fe_base_r2022_lgb_f3.npy",
    "r2023": "cache/fe_base_r2023_lgb_f3.npy",
    "r2024": "cache/fe_base_r2024_lgb_f3.npy",
}


def run_fold(train, feats, cats, tr_m, va_m):
    """한 폴드 10시드 로짓 평균 반환."""
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
    return np.mean(zs, axis=0)


def main():
    t0 = time.time()
    print("[gen_adopted] LGB F3 4폴드 10시드 로짓 예측 생성 "
          "(채택: asof_n_bucket + score_diff_binary)", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns
    base_feats = [c for c in test_cols if c != common.ID]
    feats_adopted = base_feats + ADOPTED_EXTRA
    feats_base = base_feats + BASE_EXTRA
    print(f"train: {train.shape} | adopted_feats: {len(feats_adopted)} "
          f"(base {len(base_feats)} + {len(ADOPTED_EXTRA)}) | cats: {CATS}",
          flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    scores = {}
    # ── Adopted 캐시 (채택 2피처 포함) ──
    for fn, (tr_m, va_m) in folds.items():
        t1 = time.time()
        yv = train.loc[va_m, common.TARGET].values
        z = run_fold(train, feats_adopted, CATS, tr_m, va_m)
        np.save(OUT_ADOPTED[fn], z)
        p = common.sigmoid(z)
        bss = common.score(p, yv)
        scores.setdefault("adopted", {})[fn] = dict(bss=float(bss), pred_mean=float(p.mean()))
        print(f"  [adopted/{fn}] BSS={bss:.1f} pred_mean={p.mean():.4f} "
              f"저장 {OUT_ADOPTED[fn]} ({time.time()-t1:.0f}s)", flush=True)

    # ── Baseline 캐시 (채택 피처 없는 F3, deterministic=True) ──
    for fn, (tr_m, va_m) in folds.items():
        t1 = time.time()
        yv = train.loc[va_m, common.TARGET].values
        z = run_fold(train, feats_base, CATS, tr_m, va_m)
        np.save(OUT_BASE[fn], z)
        p = common.sigmoid(z)
        bss = common.score(p, yv)
        scores.setdefault("base", {})[fn] = dict(bss=float(bss), pred_mean=float(p.mean()))
        print(f"  [base/{fn}] BSS={bss:.1f} pred_mean={p.mean():.4f} "
              f"저장 {OUT_BASE[fn]} ({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 88, flush=True)
    print("채택 2피처 10시드 gain (adopted − base, deterministic=True):", flush=True)
    for fn in folds:
        g = scores["adopted"][fn]["bss"] - scores["base"][fn]["bss"]
        print(f"  {fn:<8s} adopted={scores['adopted'][fn]['bss']:.1f} "
              f"base={scores['base'][fn]['bss']:.1f} gain={g:+.1f}", flush=True)
    print(f"\n완료 (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
