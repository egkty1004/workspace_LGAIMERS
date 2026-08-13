# -*- coding: utf-8 -*-
"""보충: 부분(prefix)매칭 커버리지 + 최근시즌(2023) asof 상한 + 결과 종합 저장"""
import pandas as pd
import numpy as np
import collections, json, time

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR
t0 = time.time()

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
              "game_type", "inning",
              "top_bottom", "balls_before", "strikes_before", "outs_before",
              "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"])
home2 = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away2 = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home2 * 1000 + away2
gid = np.concatenate([[True], pair[1:] != pair[:-1]]).cumsum() - 1
tr["gid"] = gid
tb = (tr.top_bottom.values == "T").astype(np.int16)
enc = tr.inning.values.astype(np.int16) * 10000 + tb * 1000 + \
      tr.balls_before.values.astype(np.int16) * 100 + tr.strikes_before.values.astype(np.int16) * 10 + \
      tr.outs_before.values.astype(np.int16)
enc_hand = enc * 10 + tr.pitcher_hand.values.astype(np.int16) * 3 + tr.batter_hand.values.astype(np.int16)
tr["enc_hand"] = enc_hand

tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_month", "game_dayofweek",
              "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
              "strikes_before", "outs_before", "pitcher_hand", "batter_hand"])
tbt = (tm.top_bottom.values == "Top").astype(np.int16)
hand = {"Left": 1, "Right": 2}
tm_enc = tm.inning.values.astype(np.int16) * 10000 + tbt * 1000 + \
         tm.balls_before.values.astype(np.int16) * 100 + tm.strikes_before.values.astype(np.int16) * 10 + \
         tm.outs_before.values.astype(np.int16)
tm_enc_hand = tm_enc * 10 + tm.pitcher_hand.map(hand).values.astype(np.int16) * 3 + tm.batter_hand.map(hand).values.astype(np.int16)
tm["enc_hand"] = tm_enc_hand
tm = tm.sort_values(["trackman_game_id", "pitch_no"])
tm_by_key = collections.defaultdict(list)
tm_by_day = collections.defaultdict(list)
for g, sub in tm.groupby("trackman_game_id", sort=False):
    w = sub.enc_hand.values.astype(np.int32).tobytes()
    tm_by_key[(int(sub.season.iloc[0]), int(sub.game_month.iloc[0]), int(sub.game_dayofweek.iloc[0]), w)].append(g)
    tm_by_day[(int(sub.season.iloc[0]), int(sub.game_month.iloc[0]), int(sub.game_dayofweek.iloc[0]))].append(
        (g, sub.enc_hand.values.astype(np.int32)))

def lcp(a, b):
    l = min(len(a), len(b))
    if l == 0: return 0
    neq = np.nonzero(a[:l] != b[:l])[0]
    return int(neq[0]) if len(neq) else l

rows_covered = {0.5: 0, 0.75: 0, 0.9: 0, 1.0: 0}
games_by_frac = collections.Counter()
for g, sub in tr.groupby("gid", sort=False):
    s = int(sub.season.iloc[0]); mo = int(sub.game_month.iloc[0]); dw = int(sub.game_dayofweek.iloc[0])
    w = sub.enc_hand.values.astype(np.int32)
    n = len(w)
    cands = tm_by_key.get((s, mo, dw, w.tobytes()), [])
    best = n if len(cands) == 1 else 0
    if best == 0:
        for tg, tw in tm_by_day.get((s, mo, dw), []):
            l = lcp(w, tw)
            if l > best: best = l
    frac = best / n
    games_by_frac[frac] += 1
    for thr in [0.5, 0.75, 0.9, 1.0]:
        if frac >= thr:
            rows_covered[thr] += n
print("row coverage by prefix threshold:", {k: f"{v/len(tr):.1%}" for k, v in rows_covered.items()})

# 2024 기준
cov24 = {}
for thr in [0.5, 0.75, 0.9, 1.0]:
    tot = 0
    for g, sub in tr.groupby("gid", sort=False):
        if int(sub.season.iloc[0]) != 2024: continue
        s = int(sub.season.iloc[0]); mo = int(sub.game_month.iloc[0]); dw = int(sub.game_dayofweek.iloc[0])
        w = sub.enc_hand.values.astype(np.int32); n = len(w)
        cands = tm_by_key.get((s, mo, dw, w.tobytes()), [])
        best = n if len(cands) == 1 else 0
        if best == 0:
            for tg, tw in tm_by_day.get((s, mo, dw), []):
                l = lcp(w, tw)
                if l > best: best = l
        if best / n >= thr:
            tot += n
    cov24[thr] = tot
print("2024 coverage:", {k: f"{v/253507:.1%}" for k, v in cov24.items()})

# 최근시즌(직전 1년) asof 상한
prof = pd.read_pickle(OUT + "pitcher_season_profile.pkl")
trv = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "pitcher_id", "control_success"])
trv = trv.rename(columns={"pitcher_id": "pid"})
v24 = trv[trv.season == 2024].copy()

def upper_bound(df, feat, nbins=20):
    d = df[[feat]].join(df["control_success"]).dropna()
    if len(d) == 0: return np.nan
    r = d.control_success.mean()
    x = d[feat]
    try:
        cats = pd.qcut(x, nbins, labels=False, duplicates="drop")
    except Exception:
        cats = pd.cut(x, nbins, labels=False)
    d = d.copy(); d["c"] = cats
    grp = d.groupby("c")["control_success"]
    w = grp.size() / len(d); o_k = grp.mean(); obar = d.control_success.mean()
    dRes = (w * (o_k - obar) ** 2).sum()
    return dRes / (r * (1 - r)) * 1e5

recent = prof[prof.season == 2023].drop(columns=["season"])
v_r = v24.merge(recent, on="pid", how="left")
print("\n=== 직전시즌(2023) 프로필 asof 2024 이론상한 ===")
res = {}
for f in ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
          "mix_fast", "mix_break", "mix_off"]:
    ub = upper_bound(v_r, f)
    res[f] = round(ub, 1)
    print(f"  {f}: upper={ub:.1f}")

summary = {
    "match_games": 2418, "total_games": 4605,
    "row_coverage": {str(k): rows_covered[k] / len(tr) for k in rows_covered},
    "cov24": {str(k): cov24[k] / 253507 for k in cov24},
    "pitcher_mapped": 653, "train_pitchers": 792,
    "pitcher_mapping_conf_min": 0.976,
    "asof_prior_season_ub": res,
}
with open(OUT + "summary_partial.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved summary_partial.json", time.time() - t0)
