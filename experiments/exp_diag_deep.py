# -*- coding: utf-8 -*-
"""미매칭 R 게임 정밀 진단: 병합(두 경기 합침) vs 투구집합 차이 vs TM에 없음"""
import pandas as pd
import numpy as np
import time, collections

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"
t0 = time.time()

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
tb_map = {"T": 0, "B": 1}
tr["tb"] = tr.top_bottom.map(tb_map)
enc = tr.inning.astype(np.int16) * 10000 + tr.tb.astype(np.int16) * 1000 + \
      tr.balls_before.astype(np.int16) * 100 + tr.strikes_before.astype(np.int16) * 10 + \
      tr.outs_before.astype(np.int16)
tr["enc"] = enc

def walk(df, col="enc"):
    return df[col].values.astype(np.int32).tobytes()

tr_walks = {}
for g, sub in tr.groupby("gid"):
    tr_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                   int(home[sub.index[0]]), int(away[sub.index[0]]),
                   walk(sub), sub.game_type.iloc[0], len(sub))

tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_date", "game_month",
                  "game_dayofweek", "trackman_game_id", "pitch_no", "inning", "top_bottom",
                  "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team"])
tm["tb"] = tm.top_bottom.map({"Top": 0, "Bottom": 1})
tm_enc = tm.inning.astype(np.int16) * 10000 + tm.tb.astype(np.int16) * 1000 + \
         tm.balls_before.astype(np.int16) * 100 + tm.strikes_before.astype(np.int16) * 10 + \
         tm.outs_before.astype(np.int16)
tm["enc"] = tm_enc
tm_home = np.where(tm.tb.values == 0, tm.pitcher_team.values, tm.batter_team.values)
tm_away = np.where(tm.tb.values == 0, tm.batter_team.values, tm.pitcher_team.values)
tm = tm.sort_values(["trackman_game_id", "pitch_no"])

tm_walks = {}
for g, sub in tm.groupby("trackman_game_id"):
    tm_walks[g] = (sub.season.iloc[0], sub.game_month.iloc[0], sub.game_dayofweek.iloc[0],
                   tm_home[sub.index[0]], tm_away[sub.index[0]],
                   walk(sub), len(sub))

# 같은 날 TM 게임들을 (season, month, dayofweek)으로 인덱싱
tm_day = collections.defaultdict(list)
for g, w in tm_walks.items():
    tm_day[(w[0], w[1], w[2])].append(g)

def lcp(a, b):
    """bytes a,b의 최대 공통 prefix 길이 (int 단위)"""
    la, lb = len(a) // 4, len(b) // 4
    l = min(la, lb)
    if l == 0: return 0
    A = np.frombuffer(a, dtype=np.int32)[:l]
    B = np.frombuffer(b, dtype=np.int32)[:l]
    neq = np.nonzero(A != B)[0]
    return int(neq[0]) if len(neq) else l

# unmatched R 게임 중 일부 샘플의 best TM 매칭 분석
unmatched_R = [(g, w) for g, w in tr_walks.items() if w[6] == "R" and
               (w[0], w[1], w[2], w[5]) not in {(tw[0], tw[1], tw[2], tw[5]) for tw in tm_walks.values()}]
print("unmatched R count:", len(unmatched_R))

# 통계: 같은 날 TM 게임들과 최대 lcp
best_lcp = []
no_tm_same_day = 0
for g, (s, mo, dw, h, a, w, gt, n) in unmatched_R:
    cands = tm_day.get((s, mo, dw), [])
    if not cands:
        no_tm_same_day += 1
        best_lcp.append((n, 0))
        continue
    bl = 0
    for tg in cands:
        tw = tm_walks[tg][5]
        l = lcp(w, tw)
        if l > bl: bl = l
    best_lcp.append((n, bl))

print("unmatched R with NO tm game same day:", no_tm_same_day)
b = np.array(best_lcp)
frac = b[:, 1] / np.maximum(b[:, 0], 1)
print("best-lcp fraction stats: median %.2f, mean %.2f, >0.95: %d, 0.9-0.95: %d, 0.5-0.9: %d, <0.5: %d" %
      (np.median(frac), frac.mean(), (frac > 0.95).sum(), ((frac > 0.9) & (frac <= 0.95)).sum(),
       ((frac > 0.5) & (frac <= 0.9)).sum(), (frac <= 0.5).sum()))

# 샘플: lcp가 0.5 부근이면 병합 의심. 한 개 게임 깊이 보기
print("\n-- deep dive: an unmatched R game with frac~0.5 --")
target = None
for idx, (g, (s, mo, dw, h, a, w, gt, n)) in enumerate(unmatched_R):
    if 0.45 < frac[idx] < 0.6:
        target = g; break
if target is not None:
    s, mo, dw, h, a, w, gt, n = tr_walks[target]
    print("train game gid", target, "season", s, "month", mo, "day", dw, "home", h, "away", a, "n", n)
    cands = tm_day[(s, mo, dw)]
    print("same-day TM games:", len(cands))
    # 이 게임이 두 TM 게임의 연결인지 확인: 앞부분이 TM game1과 일치하는지
    arr = np.frombuffer(w, dtype=np.int32)
    for tg in cands:
        tw = tm_walks[tg][5]
        arr2 = np.frombuffer(tw, dtype=np.int32)
        l = lcp(w, tw)
        print(f"  TM {tg}: n={len(arr2)} lcp={l} ({l/len(arr):.2f}) teams {tm_walks[tg][3]}-{tm_walks[tg][4]}")
    # 두 번째 절반이 같은 날 다른 TM 게임과 일치하는지
    second = arr[len(arr)//2:].tobytes()
    for tg in cands:
        tw = tm_walks[tg][5]
        l2 = lcp(second, tw)
        if l2 > 100:
            print(f"  second-half matches TM {tg} with lcp {l2}")
print("time", time.time() - t0)
