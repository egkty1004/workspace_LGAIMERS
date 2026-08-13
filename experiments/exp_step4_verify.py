# -*- coding: utf-8 -*-
"""검증: 1) test 투수 매핑 커버리지 2) 매핑 시즌 안정성/신뢰도 3) per-pitch 상한 신뢰성(상관)"""
import pandas as pd
import numpy as np
import pickle

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR

with open(OUT + "match_pairs.pkl", "rb") as f:
    MP = pickle.load(f)
p_mapping = MP["p_mapping"]
p_conf = MP["p_conf"]

te = pd.read_csv(DATA + "test.csv", usecols=["pitcher_id", "batter_id"])
tp = set(te.pitcher_id.unique()); tb = set(te.batter_id.unique())
print("test pitchers:", tp, "-> mapped:", [x for x in tp if x in p_mapping])
print("test batters:", tb, "-> mapped:", [x for x in tb if x in p_mapping])

tr = pd.read_csv(DATA + "train.csv", usecols=["season", "pitcher_id", "control_success"])
tr = tr.rename(columns={"pitcher_id": "pid"})
mapped = tr.pid.map(p_mapping).notna()
print("\ntrain rows with mapped pitcher:", mapped.mean().round(4))
print("2024 rows with mapped:", tr[mapped & (tr.season == 2024)].shape[0], "/", tr[tr.season == 2024].shape[0])

conf = pd.Series([v[0] for v in p_conf.values()])
print("\nmapping confidence (majority share): min", conf.min(), "median", conf.median())
low = {k: v for k, v in p_conf.items() if v[0] < 0.95}
print("pitchers with conf<0.95:", len(low))

# 매핑 시즌 안정성: 동일 train pitcher가 서로 다른 시즌에 다른 TM pitcher로 매핑되었는지
# step1의 p_mapping은 모든 시즌 통합. 시즌별 매핑 재계산은 비용이 크므로,
# 시즌별로 일관된지 확인: 매핑을 매칭게임별로 뽑아 시즌별 일치도 검증
tr2 = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "pitcher_id", "batter_id",
              "control_success", "inning", "top_bottom", "balls_before", "strikes_before",
              "outs_before", "pitcher_team_id", "batter_team_id"])
home2 = np.where(tr2.top_bottom.values == "T", tr2.pitcher_team_id.values, tr2.batter_team_id.values)
away2 = np.where(tr2.top_bottom.values == "T", tr2.batter_team_id.values, tr2.pitcher_team_id.values)
pair = home2 * 1000 + away2
gid = np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1
tr2["gid"] = gid

# 매칭 게임에서 피치별 pitcher 매핑 쌍의 시즌별 일관성
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["trackman_game_id", "pitch_no",
              "pitcher_trackman_id", "batter_trackman_id"])
tm = tm.sort_values(["trackman_game_id", "pitch_no"])
tm_g = tm.groupby("trackman_game_id", sort=False)
tm_by_g = {g: sub for g, sub in tm_g}

pairs = MP["pairs"]
tm_map = {g: tg for g, tg, n in pairs}
tr2["tm_game"] = tr2.gid.map(tm_map)
mrows = tr2[tr2.tm_game.notna()].copy()
print("\nmatching rows for consistency check:", len(mrows))

import collections
inconsist = 0
for g, sub in mrows.groupby("tm_game", sort=False):
    tg = g
    tsub = tm_by_g[tg]
    if len(sub) != len(tsub):
        continue
    ttp = tsub.pitcher_trackman_id.values
    for i, (tp_id, exp_tm) in enumerate(zip(sub.pitcher_id.values, ttp)):
        if tp_id in p_mapping and p_mapping[tp_id] != exp_tm:
            inconsist += 1
print("pitch-level mapping inconsistencies:", inconsist)

# per-pitch 상한 신뢰성: 2024 매칭 행에서 ivb/rel_speed/pitch_type와 success 상관
mdf = pd.read_pickle(OUT + "matched_features.pkl")
m24 = mdf[mdf.season == 2024]
print("\n2024 matched rows:", len(m24), "success mean:", m24.control_success.mean().round(4),
      "vs all 2024:", tr[tr.season == 2024].control_success.mean().round(4))
for f in ["rel_speed", "ivb", "hb", "spin_rate", "extension", "rel_side", "rel_height"]:
    d = m24[[f, "control_success"]].dropna()
    print(f"  {f}: corr with success = {d[f].corr(d.control_success):+.4f}")

# 구간별 평균 (ivb 상한 신뢰)
d = m24[["ivb", "control_success"]].dropna()
d["bin"] = pd.qcut(d.ivb, 10, labels=False, duplicates="drop")
print("\nivb decile -> success rate:")
print(d.groupby("bin").agg(n=("control_success", "size"), rate=("control_success", "mean")).round(4).to_string())

d = m24[["rel_speed", "control_success"]].dropna()
d["bin"] = pd.qcut(d.rel_speed, 10, labels=False, duplicates="drop")
print("\nrel_speed decile -> success rate:")
print(d.groupby("bin").agg(n=("control_success", "size"), rate=("control_success", "mean")).round(4).to_string())

d = m24[["pitch_type_group", "control_success"]]
print("\npitch_type_group -> success rate:")
print(d.groupby("pitch_type_group").agg(n=("control_success", "size"), rate=("control_success", "mean")).round(4).to_string())
