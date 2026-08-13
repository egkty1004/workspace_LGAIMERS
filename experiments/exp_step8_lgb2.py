# -*- coding: utf-8 -*-
"""Step8: 정제 실험 — 1) 프로필 존재 행만 비교 2) NaN 보간 방식 3) 5시드 평균"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time, pickle, json

from portability_paths import DATA_DIR, TMP_DATA_DIR

DATA = DATA_DIR
OUT = TMP_DATA_DIR
t0 = time.time()

base_cols = ["game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
             "balls_before", "strikes_before", "outs_before", "run_top_before",
             "run_bot_before", "run_total_before", "score_diff_home",
             "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
             "num_runners_on", "base_state", "home_win_expectancy", "away_win_expectancy",
             "li", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
             "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
             "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
             "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
             "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
             "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
             "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
             "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
tm_feats = ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
            "mix_fast", "mix_break", "mix_off"]

tr = pd.read_csv(DATA + "train.csv", usecols=base_cols + ["row_id", "season", "pitcher_id", "control_success"])
tr = tr.rename(columns={"pitcher_id": "pid"})
for c in ["top_bottom", "game_type", "base_state"]:
    tr[c] = tr[c].astype("category")

prof = pd.read_pickle(OUT + "pitcher_season_profile.pkl")
prof["prior_season"] = prof.season + 1
prof = prof[prof.prior_season.isin([2019, 2020, 2021, 2022, 2023, 2024])].drop(columns=["season"])
prof = prof.rename(columns={"prior_season": "season"})
tr = tr.merge(prof[tm_feats + ["pid", "season"]], on=["pid", "season"], how="left")
tr["season_cat"] = tr.season.astype("category")
print("coverage:", tr[tm_feats[0]].notna().mean().round(3), time.time() - t0, flush=True)

# 프로필 존재 행만 (모든 TM 피처 non-null)
has = tr[tm_feats].notna().all(axis=1)
print("rows with full prior profile:", has.mean().round(3))
X = tr[has].copy()

# 상위 피처 일부만 사용 + 나머지 보간
feats_sel = ["rel_speed", "spin_rate", "ivb", "hb", "extension", "rel_height", "rel_side",
             "mix_fast", "mix_break", "mix_off"]
Xva_full = X[X.season == 2024]
Xtr = X[X.season < 2024]
print("train rows:", len(Xtr), "val rows:", len(Xva_full), flush=True)

def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))

params = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
    "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "n_jobs": 8,
}

def run(feats, tag, seeds=(42,)):
    preds = []
    for sd in seeds:
        p = dict(params); p["seed"] = sd
        dt = lgb.Dataset(Xtr[feats], label=Xtr.control_success.values)
        dv = lgb.Dataset(Xva_full[feats], label=Xva_full.control_success.values, reference=dt)
        m = lgb.train(p, dt, num_boost_round=800, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        preds.append(m.predict(Xva_full[feats], num_iteration=m.best_iteration))
    avg = np.mean(preds, axis=0)
    s = bss(Xva_full.control_success.values, avg)
    print(f"[{tag}] 2024 BSS = {s:.2f}", flush=True)
    return s

base_feats = base_cols + ["season_cat"]
s_base = run(base_feats, "baseline(no TM, 프로필존재 행)", seeds=(1, 2, 3, 4, 5))
s_tm = run(base_feats + feats_sel, "+TM asof profile", seeds=(1, 2, 3, 4, 5))
print("delta:", round(s_tm - s_base, 2))

# 피처 중요도 (TM 피처들의 순위)
cols = base_feats + feats_sel
dt = lgb.Dataset(Xtr[cols], label=Xtr.control_success.values)
m = lgb.train(dict(params, seed=42), dt, num_boost_round=400)
imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
print("\nTM 피처 중요도 순위:")
for f in tm_feats:
    print(f"  {f}: rank {list(imp.index).index(f)} (gain {imp[f]:.0f})")

with open(OUT + "lgb_result2.json", "w") as f:
    json.dump({"baseline": round(s_base, 2), "with_tm": round(s_tm, 2),
               "delta": round(s_tm - s_base, 2)}, f, indent=2)
print("\nDONE", time.time() - t0)
