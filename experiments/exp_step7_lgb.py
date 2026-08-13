# -*- coding: utf-8 -*-
"""Step7: LightGBM 2024 검증 실험 — baseline vs +Trackman asof 투수 프로필 피처
학습: 2019-2023, 검증: 2024. 시드 1개(비교 목적)."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time, pickle

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
print("train loaded", time.time() - t0, flush=True)

# asof 투수 프로필: 시즌 S 행에는 S-1 시즌 프로필 사용
prof = pd.read_pickle(OUT + "pitcher_season_profile.pkl")
prof["prior_season"] = prof.season + 1
prof = prof[prof.prior_season.isin([2019, 2020, 2021, 2022, 2023, 2024])].drop(columns=["season"])
prof = prof.rename(columns={"prior_season": "season"})
tr = tr.merge(prof[tm_feats + ["pid", "season"]], on=["pid", "season"], how="left")
print("TM features merged, coverage:", tr[tm_feats[0]].notna().mean().round(3), time.time() - t0, flush=True)

# season 범주화 (추세 반영)
tr["season_cat"] = tr.season.astype("category")

Xtr = tr[tr.season < 2024]
Xva = tr[tr.season == 2024]
print("train rows:", len(Xtr), "val rows:", len(Xva), flush=True)

def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))

params = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
    "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "n_jobs": 8,
}

def run(feats, tag):
    cols = [c for c in feats if c in Xtr.columns]
    dt = lgb.Dataset(Xtr[cols], label=Xtr.control_success.values)
    dv = lgb.Dataset(Xva[cols], label=Xva.control_success.values, reference=dt)
    m = lgb.train(params, dt, num_boost_round=600, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(Xva[cols], num_iteration=m.best_iteration)
    s = bss(Xva.control_success.values, p)
    print(f"[{tag}] 2024 BSS = {s:.2f}  (best_iter={m.best_iteration})", flush=True)
    return s

base_feats = base_cols + ["season_cat"]
s_base = run(base_feats, "baseline(no TM)")

full_feats = base_feats + tm_feats
s_tm = run(full_feats, "+TM asof profile")

# TM 피처만으로 이득 확인 (baseline에 하나씩 추가)
single_impr = {}
for f in tm_feats:
    cols = base_feats + [f]
    dt = lgb.Dataset(Xtr[cols], label=Xtr.control_success.values)
    dv = lgb.Dataset(Xva[cols], label=Xva.control_success.values, reference=dt)
    m = lgb.train(params, dt, num_boost_round=600, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(Xva[cols], num_iteration=m.best_iteration)
    s = bss(Xva.control_success.values, p)
    single_impr[f] = round(s - s_base, 2)
print("\nper-feature incremental BSS (vs baseline):")
for k, v in sorted(single_impr.items(), key=lambda x: -x[1]):
    print(f"  +{k}: {v:+.2f}")

with open(OUT + "lgb_result.json", "w") as f:
    import json
    json.dump({"baseline": round(s_base, 2), "with_tm": round(s_tm, 2),
               "delta": round(s_tm - s_base, 2), "per_feature": single_impr}, f, indent=2)
print("\nDONE", time.time() - t0)
