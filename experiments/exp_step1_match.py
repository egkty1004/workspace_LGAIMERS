# -*- coding: utf-8 -*-
"""Step1: train/TM 게임 완전매칭 + pitcher/batter ID 매핑 학습"""
import pandas as pd
import numpy as np
import time, collections, pickle

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR
t0 = time.time()

def load_train():
    tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                  "game_type", "inning", "top_bottom", "balls_before", "strikes_before",
                  "outs_before", "pitcher_hand", "batter_hand", "pitcher_team_id",
                  "batter_team_id", "pitcher_id", "batter_id"])
    home = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
    away = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
    pair = home * 1000 + away
    gid = np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1
    tr["gid"] = gid
    tb = (tr.top_bottom.values == "T").astype(np.int16)
    enc = tr.inning.values.astype(np.int16) * 10000 + tb * 1000 + \
          tr.balls_before.values.astype(np.int16) * 100 + tr.strikes_before.values.astype(np.int16) * 10 + \
          tr.outs_before.values.astype(np.int16)
    enc_hand = enc * 10 + tr.pitcher_hand.values.astype(np.int16) * 3 + tr.batter_hand.values.astype(np.int16)
    tr["enc"] = enc; tr["enc_hand"] = enc_hand
    return tr

def load_tm():
    tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_month", "game_dayofweek",
                  "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
                  "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team", "pitcher_trackman_id", "batter_trackman_id",
                  "pitch_type_group", "rel_speed", "spin_rate", "induced_vert_break",
                  "horz_break", "extension", "rel_height", "rel_side"])
    tb = (tm.top_bottom.values == "Top").astype(np.int16)
    hand = {"Left": 1, "Right": 2, "L": 1, "R": 2}
    ph = tm.pitcher_hand.map(hand).values.astype(np.int16)
    bh = tm.batter_hand.map(hand).values.astype(np.int16)
    enc = tm.inning.values.astype(np.int16) * 10000 + tb * 1000 + \
          tm.balls_before.values.astype(np.int16) * 100 + tm.strikes_before.values.astype(np.int16) * 10 + \
          tm.outs_before.values.astype(np.int16)
    enc_hand = enc * 10 + ph * 3 + bh
    tm["enc"] = enc; tm["enc_hand"] = enc_hand
    tm["ph"] = ph; tm["bh"] = bh
    return tm.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)

tr = load_train()
tm = load_tm()
print("loaded", time.time() - t0, flush=True)

def walk_bytes(enc_arr):
    return enc_arr.astype(np.int32).tobytes()

# TM 게임 인덱스
tm_by_key = collections.defaultdict(list)
tm_meta = {}
for g, sub in tm.groupby("trackman_game_id", sort=False):
    tm_meta[g] = (sub.season.iloc[0], int(sub.game_month.iloc[0]), int(sub.game_dayofweek.iloc[0]),
                  walk_bytes(sub.enc.values), walk_bytes(sub.enc_hand.values), len(sub),
                  sub[["pitcher_trackman_id", "batter_trackman_id"]].values)
    tm_by_key[(tm_meta[g][0], tm_meta[g][1], tm_meta[g][2], tm_meta[g][3])].append(g)
print("TM games indexed", time.time() - t0, flush=True)

# train 게임 매칭
match_pairs = []   # (gid, tm_game, n)
unmatched_R = []
for g, sub in tr.groupby("gid", sort=False):
    s = int(sub.season.iloc[0]); mo = int(sub.game_month.iloc[0]); dw = int(sub.game_dayofweek.iloc[0])
    w = walk_bytes(sub.enc.values)
    wh = walk_bytes(sub.enc_hand.values)
    cands = tm_by_key.get((s, mo, dw, w), [])
    if len(cands) == 1:
        tg = cands[0]
        if tm_meta[tg][4] == wh and tm_meta[tg][5] == len(sub):
            match_pairs.append((g, tg, len(sub)))
    elif len(cands) == 0:
        if sub.game_type.iloc[0] == "R":
            unmatched_R.append(g)

print("matched games:", len(match_pairs), "unmatched R:", len(unmatched_R), flush=True)
print("match time", time.time() - t0, flush=True)

# --- pitcher/batter 매핑 수집 (피치 위치별 align) ---
pid_map = collections.defaultdict(collections.Counter)
bid_map = collections.defaultdict(collections.Counter)
for g, tg, n in match_pairs:
    sub = tr[tr.gid == g]
    tmv = tm_meta[tg][6]
    if len(sub) != len(tmv):
        continue
    tp_ids = sub.pitcher_id.values
    bt_ids = sub.batter_id.values
    ttp_ids = tmv[:, 0]
    tbt_ids = tmv[:, 1]
    for i in range(len(tp_ids)):
        pid_map[tp_ids[i]][ttp_ids[i]] += 1
        bid_map[bt_ids[i]][tbt_ids[i]] += 1
print("mapping collected", time.time() - t0, flush=True)

# 1:1 매핑 해석
p_mapping = {}
for k, v in pid_map.items():
    top, cnt = v.most_common(1)[0]
    total = sum(v.values())
    p_mapping[k] = (top, cnt / total, total)
b_mapping = {}
for k, v in bid_map.items():
    top, cnt = v.most_common(1)[0]
    total = sum(v.values())
    b_mapping[k] = (top, cnt / total, total)

uniq_p = set(x[0] for x in p_mapping.values())
uniq_b = set(x[0] for x in b_mapping.values())
print("pitcher mapped:", len(p_mapping), "unique tm pitchers:", len(uniq_p))
print("batter mapped:", len(b_mapping), "unique tm batters:", len(uniq_b))
# 1:1 여부
rev_p = collections.defaultdict(list)
for k, (v, conf, tot) in p_mapping.items():
    rev_p[v].append(k)
multi_p = {v: ks for v, ks in rev_p.items() if len(ks) > 1}
print("tm pitcher with >1 train pitcher:", len(multi_p))
rev_b = collections.defaultdict(list)
for k, (v, conf, tot) in b_mapping.items():
    rev_b[v].append(k)
multi_b = {v: ks for v, ks in rev_b.items() if len(ks) > 1}
print("tm batter with >1 train batter:", len(multi_b))

with open(OUT + "match_pairs.pkl", "wb") as f:
    pickle.dump({"pairs": match_pairs, "p_mapping": {k: v[0] for k, v in p_mapping.items()},
                 "b_mapping": {k: v[0] for k, v in b_mapping.items()},
                 "p_conf": {k: (v[1], v[2]) for k, v in p_mapping.items()},
                 "b_conf": {k: (v[1], v[2]) for k, v in b_mapping.items()}}, f)
print("saved. time", time.time() - t0)
