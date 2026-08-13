# -*- coding: utf-8 -*-
"""핑거프린트 매칭: train 게임 카운트워크를 TM 게임과 비교."""
import pandas as pd
import numpy as np
import time, collections

from portability_paths import DATA_DIR

DATA = DATA_DIR
t0 = time.time()

# ---------- train ----------
tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "game_type", "inning", "top_bottom",
                                              "balls_before", "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id"])
home = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home * 1000 + away
gid = (np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1)
tr["gid"] = gid

# TM top_bottom 매핑: Top->0, Bottom->1
tb_map = {"T": 0, "B": 1, "Top": 0, "Bottom": 1}
tr["tb"] = tr.top_bottom.map(tb_map)
tr["state"] = (tr.inning * 8 + tr.tb * 4 + tr.balls_before * 1 + 0)  # placeholder
# 실제 시퀀스 인코딩: (inning, tb, balls, strikes, outs) -> int
enc = tr.inning.astype(np.int16) * 10000 + tr.tb.astype(np.int16) * 1000 + \
      tr.balls_before.astype(np.int16) * 100 + tr.strikes_before.astype(np.int16) * 10 + \
      tr.outs_before.astype(np.int16)
# + pitcher_hand/batter_hand 포함 버전
enc_hand = enc * 10 + tr.pitcher_hand.astype(np.int16) * 3 + tr.batter_hand.astype(np.int16)

tr["enc"] = enc
tr["enc_hand"] = enc_hand

def game_walk(df_enc):
    """게임별 시퀀스를 bytes로 (빠른 비교). df_enc는 정렬된 인코딩 값."""
    return df_enc.values.astype(np.int32).tobytes()

tr_walks = {}
for g, sub in tr.groupby("gid"):
    tr_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                    int(home[sub.index[0]]), int(away[sub.index[0]]),
                    game_walk(sub["enc"]), game_walk(sub["enc_hand"]), sub.game_type.iloc[0], len(sub))
print("train walks built", time.time() - t0, "s; games:", len(tr_walks))

# ---------- TM ----------
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_date", "game_month",
                  "game_dayofweek", "trackman_game_id", "pitch_no", "inning", "top_bottom",
                  "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team"])
tm["tb"] = tm.top_bottom.map(tb_map)
tm_hand_map = {"Left": 1, "Right": 2, "L": 1, "R": 2}
tm["ph"] = tm.pitcher_hand.map(tm_hand_map)
tm["bh"] = tm.batter_hand.map(tm_hand_map)
tm_enc = tm.inning.astype(np.int16) * 10000 + tm.tb.astype(np.int16) * 1000 + \
         tm.balls_before.astype(np.int16) * 100 + tm.strikes_before.astype(np.int16) * 10 + \
         tm.outs_before.astype(np.int16)
tm_enc_hand = tm_enc * 10 + tm.ph.astype(np.int16) * 3 + tm.bh.astype(np.int16)
tm["enc"] = tm_enc
tm["enc_hand"] = tm_enc_hand

# TM 홈/원정 (top_bottom 기준)
tm_home = np.where(tm.tb.values == 0, tm.pitcher_team.values, tm.batter_team.values)
tm_away = np.where(tm.tb.values == 0, tm.batter_team.values, tm.pitcher_team.values)

# TM 게임별 walk. trackman_game_id로 정렬 보장: pitch_no 순 정렬 필요
tm = tm.sort_values(["trackman_game_id", "pitch_no"])
tm_walks = {}
for g, sub in tm.groupby("trackman_game_id"):
    tm_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                   tm_home[sub.index[0]], tm_away[sub.index[0]],
                   game_walk(sub["enc"]), game_walk(sub["enc_hand"]), len(sub))
print("TM walks built", time.time() - t0, "s; games:", len(tm_walks))

# TM 인덱스: (season, month, dayofweek, walk) -> list of game_ids
tm_by_key = collections.defaultdict(list)
for g, (s, mo, dw, h, a, w, wh, n) in tm_walks.items():
    tm_by_key[(s, mo, dw, w)].append(g)

# train 게임 매칭
matched = 0
matched_exact_len = 0
collisions = 0
unmatched = []
for g, (s, mo, dw, h, a, w, wh, gt, n) in tr_walks.items():
    cands = tm_by_key.get((s, mo, dw, w), [])
    if len(cands) == 0:
        unmatched.append((g, gt, n))
    elif len(cands) == 1:
        matched += 1
        tg = cands[0]
        if tm_walks[tg][7] == n:
            matched_exact_len += 1
    else:
        collisions += 1

print("\n==== RESULT ====")
print("train games:", len(tr_walks))
print("matched (walk == TM game walk):", matched)
print("  of which exact row count match:", matched_exact_len)
print("collisions (multiple TM games same walk):", collisions)
print("unmatched:", len(unmatched))

gt_counter = collections.Counter()
for g, gt, n in unmatched:
    gt_counter[gt] += 1
print("unmatched by game_type:", dict(gt_counter))

# 매칭된 게임의 게임타입 분포
print("\nsample matched:", list(tr_walks.keys())[:3], "->", [tm_by_key.get((tr_walks[g][0], tr_walks[g][1], tr_walks[g][2], tr_walks[g][5]), []) for g in list(tr_walks.keys())[:3]])
print("time", time.time() - t0)
