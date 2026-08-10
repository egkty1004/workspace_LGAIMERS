# -*- coding: utf-8 -*-
"""Step3: 투수 시즌별 TM 프로필(asof) 구축 + 2024 검증셋 BSS 이론상한"""
import pandas as pd
import numpy as np
import time, pickle

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"
OUT = "/home/gpu_01/workspace_LGAIMERS/experiments/tmpdata/"
t0 = time.time()

with open(OUT + "match_pairs.pkl", "rb") as f:
    MP = pickle.load(f)
p_mapping = MP["p_mapping"]
rev_p = {v: k for k, v in p_mapping.items()}

tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "pitcher_trackman_id",
              "pitch_type_group", "rel_speed", "spin_rate", "induced_vert_break",
              "horz_break", "extension", "rel_height", "rel_side"])
tm = tm[tm.pitcher_trackman_id.isin(rev_p)].copy()
tm["pid"] = tm.pitcher_trackman_id.map(rev_p)
print("TM rows for mapped pitchers:", len(tm), time.time() - t0, flush=True)

feats = {
    "rel_speed": ["mean"],
    "spin_rate": ["mean"],
    "induced_vert_break": ["mean"],
    "horz_break": ["mean"],
    "extension": ["mean"],
    "rel_height": ["mean"],
    "rel_side": ["mean"],
}
prof = tm.groupby(["pid", "season"]).agg(feats)
prof.columns = ["_".join(c) for c in prof.columns]
prof = prof.reset_index()
prof.columns = ["pid", "season", "rel_speed", "spin_rate", "ivb", "hb", "extension",
                "rel_height", "rel_side"]

pitch_mix = tm.groupby(["pid", "season", "pitch_type_group"]).size().unstack(fill_value=0)
pitch_mix = pitch_mix.div(pitch_mix.sum(axis=1), axis=0).reset_index()
pitch_mix.columns = ["pid", "season", "mix_fast", "mix_break", "mix_off", "mix_other"]

prof = prof.merge(pitch_mix, on=["pid", "season"], how="left")
prof.to_pickle(OUT + "pitcher_season_profile.pkl")
print("pitcher-season profiles:", len(prof), time.time() - t0, flush=True)
print(prof.head())

# ---- 2024 검증셋 ----
tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "pitcher_id", "control_success"])
tr = tr.rename(columns={"pitcher_id": "pid"})
v24 = tr[tr.season == 2024].copy()
print("2024 rows:", len(v24))

def upper_bound(df, feat, r=None, nbins=20):
    d = df[[feat]].join(df["control_success"])
    d = d.dropna()
    if len(d) == 0:
        return np.nan
    if r is None:
        r = d.control_success.mean()
    x = d[feat]
    if x.dtype == object:
        cats = x
    else:
        try:
            cats = pd.qcut(x, nbins, labels=False, duplicates="drop")
        except Exception:
            cats = pd.cut(x, nbins, labels=False)
    d = d.copy(); d["c"] = cats
    grp = d.groupby("c")["control_success"]
    w = grp.size() / len(d)
    o_k = grp.mean(); obar = d.control_success.mean()
    dRes = (w * (o_k - obar) ** 2).sum()
    return dRes / (r * (1 - r)) * 1e5

# (a) asof: 이전 시즌 프로필 (2019-2023 평균), 현재 시즌 프로필 없이
prior = prof[prof.season < 2024].groupby("pid").mean(numeric_only=True).reset_index()
prior["season"] = 2024
v_a = v24.merge(prior, on=["pid", "season"], how="left")
print("\n=== asof(이전시즌 2019-2023 평균) 투수 프로필 2024 이론상한 ===")
print("2024 rows with prior profile:", v_a[["rel_speed", "spin_rate", "ivb"]].notna().any(axis=1).sum(), "/", len(v_a))
for f in ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
          "mix_fast", "mix_break", "mix_off"]:
    ub = upper_bound(v_a, f, r=None)
    print(f"  {f}: upper={ub:.1f}")

# (b) leaky: 같은 시즌(2024) 프로필 — 피처 자체의 투수레벨 이론상한 참고용
cur = prof[prof.season == 2024].drop(columns=["season"])
v_b = v24.merge(cur, on="pid", how="left")
print("\n=== 같은시즌(leaky 참고) 투수 프로필 2024 이론상한 ===")
print("2024 rows with same-season profile:", v_b[["rel_speed", "spin_rate", "ivb"]].notna().any(axis=1).sum(), "/", len(v_b))
for f in ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
          "mix_fast", "mix_break", "mix_off"]:
    ub = upper_bound(v_b, f, r=None)
    print(f"  {f}: upper={ub:.1f}")

print("\ntime", time.time() - t0)
