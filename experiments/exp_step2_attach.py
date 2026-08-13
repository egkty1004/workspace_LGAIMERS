# -*- coding: utf-8 -*-
"""Step2: 매칭 게임 피치-레벨 TM 피처 부착 + 2024 커버리지 + 피치 레벨 이론상한"""
import pandas as pd
import numpy as np
import time, collections, pickle

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR
t0 = time.time()

with open(OUT + "match_pairs.pkl", "rb") as f:
    MP = pickle.load(f)
pairs = MP["pairs"]
p_mapping = MP["p_mapping"]
b_mapping = MP["b_mapping"]
print("pairs:", len(pairs))

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "inning", "top_bottom",
              "balls_before", "strikes_before", "outs_before",
              "pitcher_id", "batter_id", "control_success",
              "pitcher_team_id", "batter_team_id"])
home2 = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away2 = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home2 * 1000 + away2
gid = np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1
tr["gid"] = gid
tr = tr.drop(columns=["pitcher_team_id", "batter_team_id"])
print("train loaded", len(tr), time.time() - t0, flush=True)

# TM: 매칭 게임만의 피처
tm_feat_cols = ["trackman_game_id", "pitch_no", "rel_speed", "spin_rate", "induced_vert_break",
                "horz_break", "extension", "rel_height", "rel_side", "pitch_type_group"]
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=tm_feat_cols)
tm = tm.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)
tm_g = tm.groupby("trackman_game_id", sort=False)
tm_by_g = {g: sub for g, sub in tm_g}
print("TM loaded", time.time() - t0, flush=True)

# 매칭 게임에 피치 레벨 부착
matched_gids = set(g for g, tg, n in pairs)
tm_map_gid = {g: tg for g, tg, n in pairs}
tr["tm_game"] = tr.gid.map(tm_map_gid)
tr_match = tr[tr.gid.isin(matched_gids)].copy()
print("matched rows:", len(tr_match), "of", len(tr), f"({len(tr_match)/len(tr):.1%})", flush=True)

# 게임별 위치 align
feats = []
for g, tg, n in pairs:
    sub = tr_match[tr_match.gid == g]
    tsub = tm_by_g[tg]
    if len(sub) != len(tsub):
        continue
    sub = sub.reset_index(drop=True)
    tsub = tsub.reset_index(drop=True)
    tmp = pd.DataFrame({
        "row_id": sub.row_id.values,
        "season": sub.season.values,
        "pitcher_id": sub.pitcher_id.values,
        "batter_id": sub.batter_id.values,
        "control_success": sub.control_success.values,
        "rel_speed": tsub.rel_speed.values,
        "spin_rate": tsub.spin_rate.values,
        "ivb": tsub.induced_vert_break.values,
        "hb": tsub.horz_break.values,
        "extension": tsub.extension.values,
        "rel_height": tsub.rel_height.values,
        "rel_side": tsub.rel_side.values,
        "pitch_type_group": tsub.pitch_type_group.values,
    })
    feats.append(tmp)
match_df = pd.concat(feats, ignore_index=True)
print("aligned matched rows:", len(match_df), time.time() - t0, flush=True)
match_df.to_parquet if False else None
match_df.to_pickle(OUT + "matched_features.pkl")
print("saved matched_features.pkl")

# ---- 2024 커버리지 ----
tr["tm_matched"] = tr.gid.isin(matched_gids)
cov24 = tr[tr.season == 2024]
print("\n2024 rows:", len(cov24), "matched:", cov24.tm_matched.sum(),
      f"({cov24.tm_matched.mean():.1%})")
for s in [2019, 2020, 2021, 2022, 2023, 2024]:
    sub = tr[tr.season == s]
    print(f"  {s}: matched {sub.tm_matched.mean():.1%} ({sub.tm_matched.sum()}/{len(sub)})")

# 투수 매핑 커버리지
tr["p_mapped"] = tr.pitcher_id.map(p_mapping).notna()
cov_p = tr.groupby("season").p_mapped.mean()
print("\npitcher-mapped coverage by season:")
print(cov_p.round(3))
print("2024 rows with pitcher mapped:", tr[(tr.season == 2024)].p_mapped.sum(), "/", len(tr[tr.season == 2024]))

# ---- 2024 피치 레벨 이론상한 ----
def upper_bound(df, ycol, feat, r=None, nbins=20):
    d = df[[ycol, feat]].dropna()
    if len(d) == 0:
        return np.nan
    if r is None:
        r = d[ycol].mean()
    x = d[feat]
    if x.dtype == object:
        cats = x
    else:
        try:
            cats = pd.qcut(x, nbins, labels=False, duplicates="drop")
        except Exception:
            cats = pd.cut(x, nbins, labels=False)
    d = d.copy()
    d["c"] = cats
    grp = d.groupby("c")[ycol]
    w = grp.size() / len(d)
    o_k = grp.mean()
    obar = d[ycol].mean()
    dRes = (w * (o_k - obar) ** 2).sum()
    return dRes / (r * (1 - r)) * 1e5

m24 = match_df[match_df.season == 2024]
r24 = tr[(tr.season == 2024)].control_success.mean()
print("\n2024 r =", r24)
print("\n=== 2024 피치-레벨 TM 피처 이론상한 (매칭 2024 행) ===")
print("matched 2024 rows:", len(m24))
for f in ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side", "pitch_type_group"]:
    ub = upper_bound(m24, "control_success", f, r=r24)
    print(f"  {f}: upper={ub:.1f}  (n={m24[f].notna().sum()})")

print("time", time.time() - t0)
