# -*- coding: utf-8 -*-
"""Step9: 실질 달성 상한 (학습셋 셀평균을 2024에 적용) vs oracle 상한 + 구간 관계"""
import pandas as pd
import numpy as np
import pickle

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "pitcher_id", "control_success"])
tr = tr.rename(columns={"pitcher_id": "pid"})

prof = pd.read_pickle(OUT + "pitcher_season_profile.pkl")
prof["prior_season"] = prof.season + 1
prof = prof[prof.prior_season.isin([2019, 2020, 2021, 2022, 2023, 2024])].drop(columns=["season"])
prof = prof.rename(columns={"prior_season": "season"})
feats = ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
         "mix_fast", "mix_break", "mix_off"]
tr = tr.merge(prof[feats + ["pid", "season"]], on=["pid", "season"], how="left")
has = tr[feats].notna().all(axis=1)
tr = tr[has].copy()

Xtr = tr[tr.season < 2024]
Xva = tr[tr.season == 2024]
print("train", len(Xtr), "val", len(Xva))

def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))

print(f"\n{'feature':<12} {'oracle_upper':>12} {'train->val(실질)':>14} {'train_bin_r':>10}")
res = {}
for f in feats:
    # oracle: 검증셋 자체 셀평균
    dv = Xva[[f, "control_success"]].dropna().copy()
    try:
        dv["b"] = pd.qcut(dv[f], 10, labels=False, duplicates="drop")
    except Exception:
        continue
    r = dv.control_success.mean()
    grp = dv.groupby("b")["control_success"]
    w = grp.size() / len(dv); o_k = grp.mean()
    dRes = (w * (o_k - r) ** 2).sum()
    oracle = dRes / (r * (1 - r)) * 1e5

    # train bin 경계로 val bin 매핑 (qcut은 train 기준)
    dt = Xtr[[f, "control_success"]].dropna().copy()
    try:
        bins = pd.qcut(dt[f], 10, labels=False, duplicates="drop")
    except Exception:
        continue
    dt["b"] = bins
    bin_mean = dt.groupby("b")["control_success"].mean()
    # val에 동일 qcut 경계 적용
    edges = pd.qcut(dt[f], 10, retbins=True, duplicates="drop")[1]
    edges[0] = -np.inf; edges[-1] = np.inf
    vb = pd.cut(dv[f], edges, labels=False)
    pred = vb.map(bin_mean).astype(float)
    av = dv[~pred.isna()]
    real = bss(av.control_success, pred[av.index])
    res[f] = (round(oracle, 1), round(real, 1))
    print(f"{f:<12} {oracle:>12.1f} {real:>14.1f}")

# 상위 후보들의 2024 구간별 성공률 (단조성 확인)
for f in ["spin_rate", "rel_side", "mix_break", "rel_height", "rel_speed"]:
    d = Xva[[f, "control_success"]].dropna()
    try:
        d["b"] = pd.qcut(d[f], 5, labels=False, duplicates="drop")
    except Exception:
        continue
    g = d.groupby("b").control_success.mean()
    print(f"\n{f} 2024 quintile success rates: {[round(x, 4) for x in g.values]}")

pickle.dump(res, open(OUT + "honest_ub.pkl", "wb"))
