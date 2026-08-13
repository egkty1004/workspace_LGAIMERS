#!/usr/bin/env python3
"""Post-training cross-check: run deployed LGB+MLP models on 2024 rows (primary fold proxy)
to measure the new pipeline's actual natural mean; compare vs champion 0.4871."""
import os
import sys
import json
import importlib

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = os.path.abspath(os.path.expanduser(os.environ.get(
    "LGAIMERS_REPRO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)))
PROJECT_ROOT = os.path.dirname(REPO)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(REPO)
from repro_979 import common  # noqa: E402

MODEL_DIR = os.path.join(REPO, "submit_plato3b2", "model")
FEATURE = "count_platoon_3b2_same"
SEEDS = list(range(42, 52))
W_LGB = 0.51
CHAMPION_MEAN = 0.4871


def logit(p):
    return common.logit(p)


def sigmoid(z):
    return common.sigmoid(z)


def main():
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv(os.path.join(REPO, "open", "data", "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    feats = common.get_feature_cols(test_cols) + [FEATURE]
    X = train.loc[train["season"] == 2024, feats]
    print(f"2024 rows: {len(X)} | feats: {len(feats)}", flush=True)

    z_lgb = np.zeros(len(X), dtype=np.float64)
    for seed in SEEDS:
        bst = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"f3_s{seed}.txt"))
        z_lgb += logit(bst.predict(X))
    z_lgb /= len(SEEDS)
    p_lgb = sigmoid(z_lgb)
    print(f"LGB 10-seed 2024 mean: {p_lgb.mean():.6f}", flush=True)

    sys.path.insert(0, os.path.join(REPO, "submit_plato3b2"))
    mlp_model = importlib.import_module("mlp_model")
    prep, mlps = mlp_model.load(MODEL_DIR, SEEDS)
    test_full = train.loc[train["season"] == 2024].copy()
    z_mlp = mlp_model.predict_z(test_full, prep, mlps)
    p_mlp = sigmoid(z_mlp)
    print(f"MLP 10-seed 2024 mean: {p_mlp.mean():.6f}", flush=True)

    p_new = W_LGB * p_lgb.mean() + (1 - W_LGB) * p_mlp.mean()
    print(f"\nnew pipeline natural mean (2024, deployed models): {p_new:.6f}")
    print(f"champion 0.4871 | diff: {p_new - CHAMPION_MEAN:+.6f}")
    c = logit(0.477) - logit(p_new)
    print(f"C_LOGIT re-derived (measured): {c:.6f}")
    print(f"|diff|>0.001 -> re-align: {abs(p_new - CHAMPION_MEAN) > 0.001}")

    # cache-based (task-prescribed) comparison
    zc_lgb = np.load(os.path.join(REPO, "cache", "plato3b2_cand_primary_lgb.npy"))
    zc_mlp = np.load(os.path.join(REPO, "cache", "plato3b2_cand_primary_mlp.npy"))
    p_c = W_LGB * sigmoid(zc_lgb).mean() + (1 - W_LGB) * sigmoid(zc_mlp).mean()
    print(f"\ncache-based 2024-proxy mean: {p_c:.6f} | C_LOGIT(cache): "
          f"{logit(0.477) - logit(p_c):.6f}")

    out = dict(
        measured_2024_mean_lgb=float(p_lgb.mean()),
        measured_2024_mean_mlp=float(p_mlp.mean()),
        measured_new_pipeline_mean=float(p_new),
        champion_mean=CHAMPION_MEAN,
        diff=float(p_new - CHAMPION_MEAN),
        c_logit_measured=float(c),
        cache_proxy_mean=float(p_c),
        c_logit_cache=float(logit(0.477) - logit(p_c)),
        target_2025=0.477,
    )
    with open(os.path.join(REPO, "experiments", "plato3b2_mean_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n-> experiments/plato3b2_mean_check.json")


if __name__ == "__main__":
    main()
