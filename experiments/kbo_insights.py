# -*- coding: utf-8 -*-
"""kbo_insights.py — KBO 도메인 가설 실측 검증 (control_success).

모든 가설을 실데이터(train.csv)로 성공률 + 표본수로 검증.
전체(2019-24)와 최근(2023-24)로 나눠 드리프트 고려.
결과: stdout 표 + CSV 저장 (experiments/kbo_insights_out/).
"""
import os
import numpy as np
import pandas as pd

OUT = "/home/gpu_01/workspace_LGAIMERS/experiments/kbo_insights_out"
os.makedirs(OUT, exist_ok=True)

COLS = ["season", "inning", "top_bottom", "game_type", "balls_before", "strikes_before",
        "outs_before", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "base_state", "num_runners_on", "li", "pitcher_hand", "batter_hand", "asof_pitcher_n",
        "asof_batter_success_rate", "asof_pitcher_success_rate", "control_success"]

df = pd.read_csv("/home/gpu_01/workspace_LGAIMERS/데이터/open/data/train.csv", usecols=COLS)
df = df.dropna(subset=["control_success"])
print(f"loaded {len(df):,} rows")

# ---------- derived features ----------
df["recent"] = np.where(df["season"] >= 2023, "2023-24", "2019-22")
df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)

# count advantage: pitcher-ahead (waste counts) vs batter-ahead (must-throw-strike)
def count_role(b, s):
    d = b - s
    if d <= -1:
        return "pitcher_ahead(0-1,0-2,1-2,2-2)"   # striker count
    if d >= 2:
        return "batter_ahead(3-0,3-1,2-0)"
    if d == 1:
        return "batter_slight(1-0,2-1,3-2)"
    return "even(0-0,1-1)"
df["count_role"] = df.apply(lambda r: count_role(r["balls_before"], r["strikes_before"]), axis=1)

# platoon: same hand vs opposite (pitcher_hand/batter_hand: 1=?, 2=?; use equality)
df["platoon"] = np.where(df["pitcher_hand"] == df["batter_hand"], "same_hand", "opp_hand")
# left/right detail
hand_map = {1: "L", 2: "R"}
df["platoon_detail"] = (df["pitcher_hand"].map(hand_map) + "-" + df["batter_hand"].map(hand_map))

# runner pressure
df["risp"] = (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
df["runner_pressure"] = np.select(
    [df["base_state"] == "123", df["risp"], df["num_runners_on"] > 0],
    ["bases_loaded", "risp", "runner_no_risp"], default="none")

# inning phase
df["inning_phase"] = pd.cut(df["inning"], bins=[0, 3, 6, 9, 99],
                            labels=["1-3", "4-6", "7-9", "10+"], right=True)
# fatigue buckets by cumulative pitches (asof_pitcher_n = pitches so far in game)
df["fatigue"] = pd.cut(df["asof_pitcher_n"], bins=[-1, 25, 50, 75, 100, 10000],
                       labels=["0-25", "26-50", "51-75", "76-100", "100+"])

# LI buckets (standard leverage index buckets)
df["li_bucket"] = pd.cut(df["li"], bins=[-0.01, 0.5, 1.0, 2.0, 100],
                         labels=["<0.5", "0.5-1.0", "1.0-2.0", ">2.0"])
# score diff buckets
ad = df["score_diff_pitcher_team"].abs()
df["score_bucket"] = np.select(
    [ad <= 1, ad <= 3, ad <= 5],
    ["close(<=1)", "mod(2-3)", "gapped(4-5)"], default="blowout(>=6)")
df["lead_dir"] = np.select(
    [df["score_diff_pitcher_team"] > 0, df["score_diff_pitcher_team"] < 0],
    ["leading", "trailing"], default="tied")

# batter quality quintiles (asof_batter_success_rate)
df["batter_q"] = pd.qcut(df["asof_batter_success_rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
# pitcher skill quintiles (asof_pitcher_success_rate)
df["pitcher_q"] = pd.qcut(df["asof_pitcher_success_rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

# top_bottom (T=away attack / B=home attack)
df["side_label"] = np.where(df["top_bottom"] == "T", "away_attack(T)", "home_attack(B)")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def tab(grp, key, cols=None):
    g = grp.groupby(key)["control_success"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "succ", "count": "n"})
    rec = grp[grp["recent"] == "2023-24"].groupby(key)["control_success"].agg(["mean", "count"])
    rec = rec.rename(columns={"mean": "succ_rec", "count": "n_rec"})
    out = g.join(rec, how="left")
    if cols is not None:
        out = out.reindex(cols)
    return out


def show(grp, key, name, cols=None, sort=False):
    t = tab(grp, key, cols)
    if sort:
        t = t.sort_values("succ")
    print("\n" + "=" * 100)
    print(f"[{name}]")
    print(t.to_string())
    t.to_csv(f"{OUT}/{name}.csv")
    return t


# ============ 0) baseline ============
print("\n### BASELINE")
print("overall:", df["control_success"].mean(), "n=", len(df))
print(df.groupby("season")["control_success"].agg(["mean", "count"]).to_string())

# ============ 1) count hypotheses ============
show(df, "count_state", "H_count_state", cols=[f"{b}-{s}" for b in range(4) for s in range(3)])
show(df, "count_role", "H_count_role")
show(df, "count_state", "H_first_vs_full", cols=["0-0", "3-2"])
# count role by recent
show(df, "count_role", "H_count_role_recent", cols=["pitcher_ahead(0-1,0-2,1-2,2-2)", "even(0-0,1-1)",
                                       "batter_slight(1-0,2-1,3-2)", "batter_ahead(3-0,3-1,2-0)"])

# ============ 2) runner hypotheses ============
show(df, "runner_pressure", "H_runner_pressure")
show(df, "base_state", "H_base_state", cols=["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"])
show(df, "risp", "H_risp_binary")

# ============ 3) inning & fatigue ============
show(df, "inning_phase", "H_inning_phase", cols=["1-3", "4-6", "7-9", "10+"])
show(df, "inning", "H_inning_raw")
show(df, "fatigue", "H_fatigue", cols=["0-25", "26-50", "51-75", "76-100", "100+"])
# fatigue × recent
show(df[df["recent"] == "2023-24"], "fatigue", "H_fatigue_recent", cols=["0-25", "26-50", "51-75", "76-100", "100+"])

# ============ 4) platoon ============
show(df, "platoon", "H_platoon")
show(df, "platoon_detail", "H_platoon_detail")
show(df[df["recent"] == "2023-24"], "platoon_detail", "H_platoon_recent")

# ============ 5) LI ============
show(df, "li_bucket", "H_li_bucket", cols=["<0.5", "0.5-1.0", "1.0-2.0", ">2.0"])
show(df[df["recent"] == "2023-24"], "li_bucket", "H_li_bucket_recent", cols=["<0.5", "0.5-1.0", "1.0-2.0", ">2.0"])

# ============ 6) score diff ============
show(df, "score_bucket", "H_score_bucket")
show(df, "lead_dir", "H_lead_dir")

# ============ 7) batter quality ============
show(df, "batter_q", "H_batter_q", cols=["Q1", "Q2", "Q3", "Q4", "Q5"])
show(df, "pitcher_q", "H_pitcher_q", cols=["Q1", "Q2", "Q3", "Q4", "Q5"])

# ============ 8) game type ============
gt = df.groupby(["game_type", "season"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n### H_game_type_x_season")
print(gt.to_string())
gt.to_csv(f"{OUT}/H_game_type_x_season.csv")

# ============ 9) outs / top_bottom ============
show(df, "outs_before", "H_outs", cols=[0, 1, 2])
show(df, "side_label", "H_top_bottom")

print("\nDONE")
