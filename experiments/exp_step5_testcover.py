# -*- coding: utf-8 -*-
"""1) test 투수가 train에 존재하는지 (매핑 확장 가능성) 2) 팀/핸드/시즌 확률적 매칭 ambiguity"""
import pandas as pd
import numpy as np
import collections, pickle

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR

with open(OUT + "match_pairs.pkl", "rb") as f:
    MP = pickle.load(f)
p_mapping = MP["p_mapping"]

test_pitchers = [21813, 24198, 24745, 24713]

tr = pd.read_csv(DATA + "train.csv", usecols=["season", "pitcher_id", "batter_id",
              "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
              "inning", "top_bottom", "balls_before", "strikes_before", "outs_before"])
tr = tr.rename(columns={"pitcher_id": "pid", "batter_id": "bid"})
home2 = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away2 = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home2 * 1000 + away2
gid = np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1
tr["gid"] = gid

print("=== test pitcher 존재/매핑 현황 ===")
for tp in test_pitchers:
    sub = tr[tr.pid == tp]
    seasons = sorted(sub.season.unique()) if len(sub) else []
    mapped = tp in p_mapping
    tm_id = p_mapping.get(tp)
    n_games = sub.gid.nunique() if len(sub) else 0
    print(f"  pitcher {tp}: in train={len(sub)>0} seasons={seasons} games={n_games} mapped={mapped} -> tm_id={tm_id}")

# test pitcher가 투구한 게임이 매칭되었는지
tm_map = {g: tg for g, tg, n in MP["pairs"]}
for tp in test_pitchers:
    sub = tr[tr.pid == tp]
    if len(sub) == 0: continue
    gs = set(sub.gid.unique())
    matched_g = [g for g in gs if g in tm_map]
    print(f"  pitcher {tp}: matched games {len(matched_g)}/{len(gs)}")

print("\n=== 확률적 매칭 (팀+핸드+시즌) ambiguity ===")
# train 투수별 (season, team, hand) 카운트
tr["ph"] = tr.pitcher_hand.astype(int)
prof = tr.groupby(["season", "pitcher_team_id", "ph"]).pid.nunique().rename("n_train_pitchers")
print(prof.describe())
# 팀+핸드+시즌당 평균 투수 수
print("\nper (season,team,hand) pitcher count: mean=%.2f median=%.0f" % (prof.mean(), prof.median()))
print("share of cells with 1 pitcher: %.1f%%" % ((prof == 1).mean() * 100))
print("share of cells with 2 pitchers: %.1f%%" % ((prof == 2).mean() * 100))

# TM 쪽도 같은 분포
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "pitcher_trackman_id",
              "pitcher_hand", "pitcher_team", "game_date"])
tmap_team = {12:"DOO_BEA",13:"LG_TWI",14:"KIW_HER",15:"LOT_GIA",16:"KIA_TIG",17:"HAN_EAG",
             18:"SAM_LIO",19:"NC_DIN",20:"KT_WIZ",21:"SSG_LAN",22:"KBO_POL",23:"KBO_ARM"}
tm = tm[tm.pitcher_team.isin(tmap_team.values())].copy()
tm["ph"] = tm.pitcher_hand.map({"Left":1,"Right":2,"L":1,"R":2})
tm["tid"] = tm.pitcher_team.map({v:k for k,v in tmap_team.items()})
tprof = tm.groupby(["season", "tid", "ph"]).pitcher_trackman_id.nunique().rename("n_tm_pitchers")
print("\nTM per (season,team,hand) pitcher count: mean=%.2f median=%.0f" % (tprof.mean(), tprof.median()))
print("TM share of cells with 1 pitcher: %.1f%%" % ((tprof == 1).mean() * 100))

# train에서 미매핑된 139명 중 몇 명이 (team,hand,season) 셀에서 유일한가
tr["mapped"] = tr.pid.map(p_mapping).notna()
unmapped = tr[~tr.mapped]
cells = unmapped.groupby(["season", "pitcher_team_id", "ph"]).pid.nunique()
print("\nunmapped train rows pitcher-cells: mean=%.2f" % cells.mean())
# 미매핑 투수 중 다른 미매핑과 동일 셀 공유 비율
dup = (cells > 1).mean()
print("cells with >1 unmapped pitcher: %.1f%%" % (dup * 100))

# test 투수들의 (season=2024, team, hand)
te = pd.read_csv(DATA + "test.csv", usecols=["pitcher_id", "pitcher_hand", "pitcher_team_id"])
print("\ntest pitcher (2025, team, hand):")
for _, r in te.drop_duplicates("pitcher_id").iterrows():
    print(f"  {r.pitcher_id}: team {r.pitcher_team_id} hand {r.pitcher_hand}")
