# -*- coding: utf-8 -*-
"""1) 매칭 게임으로 팀 매핑 학습 2) 미매칭 R 게임 진단 (왜 매칭 안 되나)"""
import pandas as pd
import numpy as np
import time, collections

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"
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
tb_map = {"T": 0, "B": 1, "Top": 0, "Bottom": 1}
tr["tb"] = tr.top_bottom.map(tb_map)
enc = tr.inning.astype(np.int16) * 10000 + tr.tb.astype(np.int16) * 1000 + \
      tr.balls_before.astype(np.int16) * 100 + tr.strikes_before.astype(np.int16) * 10 + \
      tr.outs_before.astype(np.int16)
enc_hand = enc * 10 + tr.pitcher_hand.astype(np.int16) * 3 + tr.batter_hand.astype(np.int16)
tr["enc"] = enc; tr["enc_hand"] = enc_hand

def walk(df, col):
    return df[col].values.astype(np.int32).tobytes()

tr_walks = {}
for g, sub in tr.groupby("gid"):
    tr_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                   int(home[sub.index[0]]), int(away[sub.index[0]]),
                   walk(sub, "enc"), walk(sub, "enc_hand"), sub.game_type.iloc[0], len(sub))

# ---------- TM ----------
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_date", "game_month",
                  "game_dayofweek", "trackman_game_id", "pitch_no", "inning", "top_bottom",
                  "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team"])
tm["tb"] = tm.top_bottom.map(tb_map)
hand_map = {"Left": 1, "Right": 2, "L": 1, "R": 2}
tm["ph"] = tm.pitcher_hand.map(hand_map); tm["bh"] = tm.batter_hand.map(hand_map)
tm_enc = tm.inning.astype(np.int16) * 10000 + tm.tb.astype(np.int16) * 1000 + \
         tm.balls_before.astype(np.int16) * 100 + tm.strikes_before.astype(np.int16) * 10 + \
         tm.outs_before.astype(np.int16)
tm_enc_hand = tm_enc * 10 + tm.ph.astype(np.int16) * 3 + tm.bh.astype(np.int16)
tm["enc"] = tm_enc; tm["enc_hand"] = tm_enc_hand
tm_home = np.where(tm.tb.values == 0, tm.pitcher_team.values, tm.batter_team.values)
tm_away = np.where(tm.tb.values == 0, tm.batter_team.values, tm.pitcher_team.values)
tm = tm.sort_values(["trackman_game_id", "pitch_no"])

tm_walks = {}
for g, sub in tm.groupby("trackman_game_id"):
    tm_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                   tm_home[sub.index[0]], tm_away[sub.index[0]],
                   walk(sub, "enc"), walk(sub, "enc_hand"), len(sub))

tm_by_key = collections.defaultdict(list)
for g, (s, mo, dw, h, a, w, wh, n) in tm_walks.items():
    tm_by_key[(s, mo, dw, w)].append(g)

matched_games = {}
unmatched_games = {}
for g, (s, mo, dw, h, a, w, wh, gt, n) in tr_walks.items():
    cands = tm_by_key.get((s, mo, dw, w), [])
    if len(cands) == 1:
        matched_games[g] = cands[0]
    else:
        unmatched_games[g] = (s, mo, dw, h, a, w, wh, gt, n)

print("matched:", len(matched_games), "unmatched:", len(unmatched_games))

# ---- 팀 매핑 학습 ----
mapping = collections.defaultdict(collections.Counter)
for g, tg in matched_games.items():
    s, mo, dw, h, a, w, wh, gt, n = tr_walks[g]
    ts, tmo, tdw, th, ta, tw, twh, tn = tm_walks[tg]
    mapping[h][th] += 1
    mapping[a][ta] += 1
print("\n-- team id -> team code mapping (from matched games) --")
for tid in sorted(mapping):
    print(f"  {tid}: {mapping[tid].most_common()}")

# ---- 미매칭 진단: 같은 날짜/시즌의 TM 게임과 길이 비교 ----
print("\n-- unmatched R game diagnosis --")
day_match_len = []
for g, (s, mo, dw, h, a, w, wh, gt, n) in unmatched_games.items():
    if gt != "R":
        continue
    # 같은 (s,mo,dw) TM 게임들
    same_day = [tg for tg in tm_walks if tm_walks[tg][0] == s and tm_walks[tg][1] == mo and tm_walks[tg][2] == dw]
    if not same_day:
        day_match_len.append((n, None))
        continue
    # 최대 공통 prefix 길이 계산 (첫 TM 후보 5개만)
    best = 0; best_tg = None
    for tg in same_day[:5]:
        tw = tm_walks[tg][5]
        # prefix 비교
        l = min(len(w)//4, len(tw)//4)
        pref = 0
        wa = np.frombuffer(w, dtype=np.int32)[:l]
        tb = np.frombuffer(tw, dtype=np.int32)[:l]
        eq = (wa == tb)
        if eq.all():
            pref = l
        else:
            pref = int(np.argmax(~eq)) if (~eq).any() else l
        if pref > best:
            best = pref; best_tg = tg
    day_match_len.append((n, best))

c = collections.Counter()
for n, best in day_match_len:
    if best is None: c["no TM game same day"] += 1
    elif best == n: c["len match, walk diff"] += 1
    elif best == 0: c["walk diff from start"] += 1
    elif best >= n * 0.9: c["prefix >=90%"] += 1
    elif best >= n * 0.5: c["prefix 50-90%"] += 1
    else: c["prefix <50%"] += 1
print("unmatched R diagnosis:", dict(c))

# 미매칭 R 게임의 팀이 매핑에 존재하는가
unmatched_teams = collections.Counter()
for g, (s, mo, dw, h, a, w, wh, gt, n) in unmatched_games.items():
    if gt == "R":
        unmatched_teams[h] += 1
        unmatched_teams[a] += 1
print("unmatched R game team ids freq:", dict(sorted(unmatched_teams.items())))

print("time", time.time() - t0)
