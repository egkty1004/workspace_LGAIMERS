# -*- coding: utf-8 -*-
"""kbo_supp.py — 보충 검증.

1) 최근년(2023-24) 내에서 re-quintile한 batter/pitcher 성공률 → 시대 혼동 검증
2) 2024년만의 F vs R count 구조 (레짐 수렴 여부)
3) 3-way: count x platoon x risp (3-2 same_hand 중심)
4) F 게임 2023-24 카운트 분포
"""
import numpy as np
import pandas as pd

from portability_paths import DATA_DIR, KBO_INSIGHTS_DIR

OUT = KBO_INSIGHTS_DIR

COLS = ["season", "game_type", "balls_before", "strikes_before",
        "runner_on_2b", "runner_on_3b", "pitcher_hand", "batter_hand",
        "asof_batter_success_rate", "asof_pitcher_success_rate", "control_success"]
df = pd.read_csv(DATA_DIR + "train.csv", usecols=COLS)
df = df.dropna(subset=["control_success"])
df["recent"] = np.where(df["season"] >= 2023, "2023-24", "2019-22")
df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
df["platoon"] = np.where(df["pitcher_hand"] == df["batter_hand"], "same_hand", "opp_hand")
df["risp"] = (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# 1) within-recent re-quintile
rec = df[df["recent"] == "2023-24"].copy()
rec["bq_re"] = pd.qcut(rec["asof_batter_success_rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
rec["pq_re"] = pd.qcut(rec["asof_pitcher_success_rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
for name, key in [("S_batter_q_within_recent", "bq_re"), ("S_pitcher_q_within_recent", "pq_re")]:
    g = rec.groupby(key)["control_success"].agg(["mean", "count"]).rename(columns={"mean": "succ", "count": "n"})
    print(f"\n[{name}]")
    print(g.to_string())
    g.to_csv(f"{OUT}/{name}.csv")

# 2) 2024년만 F vs R count
d24 = df[df["season"] == 2024]
g = d24.groupby(["game_type", "count_state"])["control_success"].agg(["mean", "count"]).rename(
    columns={"mean": "succ", "count": "n"})
print("\n[S_2024_F_vs_R_count]")
print(g.to_string())
g.to_csv(f"{OUT}/S_2024_F_vs_R_count.csv")

# 3) 3-way count x platoon x risp (3-2, 3-1, 0-1, 0-0 focus)
for c in ["3-2", "3-1", "0-1", "0-0", "2-2"]:
    sub = df[df["count_state"] == c]
    g = sub.groupby(["platoon", "risp"])["control_success"].agg(["mean", "count"]).rename(
        columns={"mean": "succ", "count": "n"})
    g = g[g["n"] >= 3000]
    print(f"\n[S_3way_{c}]")
    print(g.to_string())
    g.to_csv(f"{OUT}/S_3way_{c}.csv")
    # recent only
    g = sub[sub["recent"] == "2023-24"].groupby(["platoon", "risp"])["control_success"].agg(
        ["mean", "count"]).rename(columns={"mean": "succ", "count": "n"})
    g = g[g["n"] >= 3000]
    print(f"[S_3way_{c}_recent]")
    print(g.to_string())

# 4) F 2023-24 count 구조 vs R 2023-24
for gt in ["F", "R"]:
    sub = df[(df["game_type"] == gt) & (df["recent"] == "2023-24")]
    g = sub.groupby("count_state")["control_success"].agg(["mean", "count"]).rename(
        columns={"mean": "succ", "count": "n"})
    print(f"\n[S_{gt}_2023_24_count]")
    print(g.to_string())
    g.to_csv(f"{OUT}/S_{gt}_2023_24_count.csv")

# F 게임 연도별 n (2023 이후 퓨처스 규모)
print("\n[S_F_season_n]")
print(df[df["game_type"] == "F"].groupby("season").size().to_string())

print("\nDONE")
