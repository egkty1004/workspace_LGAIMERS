# -*- coding: utf-8 -*-
"""kbo_cross.py — 2-way 교차 신호 + ABS 2024 분석.

최소 표본 3,000 규칙 적용. 전체 + 최근(2023-24) split.
"""
import os
import numpy as np
import pandas as pd

OUT = "/home/gpu_01/workspace_LGAIMERS/experiments/kbo_insights_out"
os.makedirs(OUT, exist_ok=True)

COLS = ["season", "inning", "game_type", "balls_before", "strikes_before", "outs_before",
        "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "base_state", "num_runners_on", "li", "pitcher_hand", "batter_hand", "control_success"]

df = pd.read_csv("/home/gpu_01/workspace_LGAIMERS/데이터/open/data/train.csv", usecols=COLS)
df = df.dropna(subset=["control_success"])
df["recent"] = np.where(df["season"] >= 2023, "2023-24", "2019-22")
df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
df["platoon"] = np.where(df["pitcher_hand"] == df["batter_hand"], "same_hand", "opp_hand")
df["risp"] = (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
df["li_bucket"] = pd.cut(df["li"], bins=[-0.01, 0.5, 1.0, 2.0, 100],
                         labels=["<0.5", "0.5-1.0", "1.0-2.0", ">2.0"])
ad = df["score_diff_pitcher_team"].abs()
df["score_bucket"] = np.select([ad <= 1, ad <= 3, ad <= 5],
                               ["close(<=1)", "mod(2-3)", "gapped(4-5)"], default="blowout(>=6)")
df["inning_phase"] = pd.cut(df["inning"], bins=[0, 3, 6, 9, 99],
                            labels=["1-3", "4-6", "7-9", "10+"], right=True)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
MINN = 3000


def cross(k1, k2, name, recent_only=False, sub=None):
    d = df[df["recent"] == "2023-24"] if recent_only else df
    if sub is not None:
        d = d.query(sub)
    g = d.groupby([k1, k2])["control_success"].agg(["mean", "count"]).rename(
        columns={"mean": "succ", "count": "n"})
    g = g[g["n"] >= MINN]
    g = g.sort_values("succ")
    print("\n" + "=" * 100)
    print(f"[{name}] recent_only={recent_only} sub={sub}  (min n={MINN})")
    print(g.to_string())
    g.to_csv(f"{OUT}/{name}.csv")
    return g


cross("count_state", "risp", "X_count_x_risp")
cross("count_state", "runner_on_3b", "X_count_x_r3")
cross("count_state", "platoon", "X_count_x_platoon")
cross("count_state", "li_bucket", "X_count_x_li")
cross("count_state", "outs_before", "X_count_x_outs")
cross("count_state", "game_type", "X_count_x_gtype")
cross("count_state", "score_bucket", "X_count_x_score")
cross("platoon", "inning_phase", "X_platoon_x_inning")
cross("risp", "li_bucket", "X_risp_x_li")
cross("inning_phase", "li_bucket", "X_inning_x_li")
cross("inning_phase", "game_type", "X_inning_x_gtype")
cross("outs_before", "risp", "X_outs_x_risp")
cross("score_bucket", "game_type", "X_score_x_gtype")

# 최근년만 — 같은 교차 중 의미 있는 것 재검
cross("count_state", "risp", "X_count_x_risp_recent", recent_only=True)
cross("count_state", "platoon", "X_count_x_platoon_recent", recent_only=True)
cross("count_state", "li_bucket", "X_count_x_li_recent", recent_only=True)
cross("count_state", "score_bucket", "X_count_x_score_recent", recent_only=True)

# ============ ABS 2024: R 경기 season x count ============
rd = df[df["game_type"] == "R"]
g = rd.groupby(["season", "count_state"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n" + "=" * 100)
print("[ABS_R_season_x_count]")
print(g.to_string())
g.to_csv(f"{OUT}/ABS_R_season_x_count.csv")

# 2023 vs 2024 차이를 카운트별로
piv = g["succ"].unstack(level=0)
delta = piv[2024] - piv[2023]
print("\n### R games 2024-2023 delta per count")
print(delta.sort_values().to_string())
pd.DataFrame({"delta_2024_minus_2023": delta}).to_csv(f"{OUT}/ABS_R_count_delta.csv")

# R season x inning_phase (ABS 효과가 후반부에?)
g2 = rd.groupby(["season", "inning_phase"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n" + "=" * 100)
print("[ABS_R_season_x_inning]")
print(g2.to_string())
g2.to_csv(f"{OUT}/ABS_R_season_x_inning.csv")

# R season x platoon
g3 = rd.groupby(["season", "platoon"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n" + "=" * 100)
print("[ABS_R_season_x_platoon]")
print(g3.to_string())
g3.to_csv(f"{OUT}/ABS_R_season_x_platoon.csv")

# R season x base_state 간단
g4 = rd.groupby(["season", "base_state"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n" + "=" * 100)
print("[ABS_R_season_x_base]")
print(g4.to_string())
g4.to_csv(f"{OUT}/ABS_R_season_x_base.csv")

print("\nDONE")
