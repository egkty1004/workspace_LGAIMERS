#!/usr/bin/env python3
"""combo_check.py — COMBO(F1+F2) 5시드 × 4폴드 스크리닝 (경계 판정용, 2026-08-09).

F_post2023 + three_balls 동시 적용 → 계획서 경계 로직("통합 시 이득 확인") 검증.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEEDS = list(range(42, 47))
t0 = time.time()
print("[combo] F1+F2 통합 5시드 × 4폴드", flush=True)
train, _ = common.load_train()
common.preprocess_for_submission(train)
feats = common.get_feature_cols(
    pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
cats = common.CAT_COLS + common.F3_EXTRA
train["F_post2023"] = ((train["game_type"] == "F") & (train["season"] >= 2023)).astype("int8")
train["three_balls"] = (train["balls_before"] == 3).astype("int8")
feats = feats + ["F_post2023", "three_balls"]

isR = train["game_type"] == "R"
folds = {
    "primary": (train["season"] <= 2023, train["season"] == 2024),
    "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
    "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
    "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
}
out = {}
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
    z = np.mean(zs, axis=0)
    p = common.sigmoid(z)
    out[fn] = float(common.score(p, yv))
    print(f"  [{fn}] BSS={out[fn]:.1f} ({time.time()-t0:.0f}s)", flush=True)

base = {"primary": 726.3, "r2022": 568.9, "r2023": 542.8, "r2024": 710.3}
print("\nCOMBO vs baseline:")
for fn in ["primary", "r2022", "r2023", "r2024"]:
    print(f"  {fn}: {out[fn]:.1f} (Δ{out[fn]-base[fn]:+.1f})", flush=True)
r_imp = sum(1 for fn in ["r2022", "r2023", "r2024"] if out[fn] > base[fn])
print(f"R-only 개선: {r_imp}/3", flush=True)
json.dump({"combo": out, "baseline": base, "r_only_improved": r_imp,
           "total_time": time.time()-t0},
          open("experiments/combo_check.json", "w"), indent=2)
print("저장: experiments/combo_check.json", flush=True)
