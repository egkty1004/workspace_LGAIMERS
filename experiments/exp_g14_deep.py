# -*- coding: utf-8 -*-
"""gid=14 (DOO_BEA vs KIW_HER, 2019-03-26, Jamsil) train/TM 시퀀스 직접 비교"""
import pandas as pd
import numpy as np

from portability_paths import DATA_DIR

DATA = DATA_DIR

# train gid=14
tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "game_type", "inning", "top_bottom",
                                              "balls_before", "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id"])
home = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home * 1000 + away
gid = (np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1)
tr["gid"] = gid

g14 = tr[tr.gid == 14].reset_index(drop=True)
print("train gid=14 rows:", len(g14))
print(g14.iloc[180:200][["row_id", "inning", "top_bottom", "balls_before", "strikes_before",
                         "outs_before", "pitcher_hand", "batter_hand", "pitcher_id", "batter_id"]].to_string())
print("\n... and around 186 ...")

# TM game
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_date", "game_month",
                  "game_dayofweek", "trackman_game_id", "pitch_no", "inning", "top_bottom",
                  "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team", "pitcher_trackman_id", "batter_trackman_id",
                  "pitch_type_group"])
tm = tm.sort_values(["trackman_game_id", "pitch_no"])
g = tm[tm.trackman_game_id == "20190326-Jamsil-1"].reset_index(drop=True)
print("\nTM game rows:", len(g))
print(g.iloc[180:200][["pitch_no", "inning", "top_bottom", "balls_before", "strikes_before",
                       "outs_before", "pitcher_hand", "batter_hand", "pitcher_trackman_id",
                       "batter_trackman_id", "pitch_type_group"]].to_string())

# 186번째 지점 전후 상태 비교
print("\n-- train state[185:190] vs TM state[185:190] --")
enc_t = (g14.inning.astype(int) * 10000 + (g14.top_bottom == "T").astype(int) * 1000 +
         g14.balls_before * 100 + g14.strikes_before * 10 + g14.outs_before)
enc_m = (g.inning.astype(int) * 10000 + (g.top_bottom == "Top").astype(int) * 1000 +
         g.balls_before * 100 + g.strikes_before * 10 + g.outs_before)
print("train:", enc_t.iloc[184:190].tolist())
print("TM   :", enc_m.iloc[184:190].tolist())

# 두 경기의 상태 멀티셋 비교 (시퀀스 무시)
from collections import Counter
ct = Counter(enc_t); cm = Counter(enc_m)
common = sum((ct & cm).values())
print("\nstate-set overlap:", common, "/", len(g14), " train-only:", len(ct - cm), " TM-only:", len(cm - ct))

# pitcher/batter 페어(팀+핸드)로 게임 내 투구 수 비교: train과 TM의 pitcher_hand/batter_hand 시퀀스
ph_tr = (g14.pitcher_hand * 3 + g14.batter_hand).tolist()
ph_tm = ((g.pitcher_hand.map({"Left":1,"Right":2}) * 3 + g.batter_hand.map({"Left":1,"Right":2}))).tolist()
print("\nhand-seq equal:", ph_tr == ph_tm)
# 어디까지 일치하는지
d = [i for i,(a,b) in enumerate(zip(ph_tr, ph_tm)) if a != b]
print("first hand mismatch at:", d[0] if d else None, "num mismatches:", len(d))
