# -*- coding: utf-8 -*-
"""Step10: per-pitch TM 피처의 train->val 전이 BSS (매칭 행만)"""
import pandas as pd
import numpy as np
import pickle

from portability_paths import TMP_DATA_DIR

OUT = TMP_DATA_DIR
mdf = pd.read_pickle(OUT + "matched_features.pkl")

Xtr = mdf[mdf.season < 2024]
Xva = mdf[mdf.season == 2024]
print("matched train rows:", len(Xtr), "matched val rows:", len(Xva))

def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))

print(f"\n{'feature':<16} {'val_r':>6} {'train->val':>10} {'oracle':>8}")
for f in ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side", "pitch_type_group"]:
    dt = Xtr[[f, "control_success"]].dropna()
    dv = Xva[[f, "control_success"]].dropna()
    if len(dt) == 0 or len(dv) == 0:
        continue
    if dt[f].dtype == object or dv[f].dtype == object:
        continue
    try:
        edges = pd.qcut(dt[f], 10, retbins=True, duplicates="drop")[1]
        edges[0] = -np.inf; edges[-1] = np.inf
    except Exception:
        continue
    dt["b"] = pd.cut(dt[f], edges, labels=False)
    bin_mean = dt.groupby("b").control_success.mean()
    dv["b"] = pd.cut(dv[f], edges, labels=False)
    pred = dv["b"].map(bin_mean).astype(float)
    av = dv[~pred.isna()]
    real = bss(av.control_success, pred[av.index])

    # oracle (2024 셀평균)
    r = dv.control_success.mean()
    grp = dv.groupby("b")["control_success"]
    w = grp.size() / len(dv); o_k = grp.mean()
    dRes = (w * (o_k - r) ** 2).sum()
    oracle = dRes / (r * (1 - r)) * 1e5
    print(f"{f:<16} {r:>6.3f} {real:>10.1f} {oracle:>8.1f}")

# 구종(train->val)
dt = Xtr[["pitch_type_group", "control_success"]]
dv = Xva[["pitch_type_group", "control_success"]]
bm = dt.groupby("pitch_type_group").control_success.mean()
pred = dv.pitch_type_group.map(bm)
r = dv.control_success.mean()
print(f"\npitch_type_group train->val BSS = {bss(dv.control_success, pred):.1f}")
