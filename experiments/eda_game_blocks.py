# -*- coding: utf-8 -*-
"""train 게임 경계 복원 시도: 연속 run 탐지. 경기(블록) 단위 구조 파악."""
import pandas as pd
import numpy as np

from portability_paths import DATA_DIR

DATA = DATA_DIR

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "inning", "top_bottom", "balls_before",
                                              "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id",
                                              "game_type", "control_success"])

# 게임 블록 = (season, month, dayofweek, game_type, home/away 무관 팀 쌍) 연속 run
# 팀 쌍을 정렬된 튜플로 (홈/어웨이 무관)
teams = list(zip(tr.pitcher_team_id, tr.batter_team_id))
home = tr.top_bottom == "T"  # top이면 pitcher가 home? 아님. top_bottom과 팀 관계 확인 필요.
# 임시: 무정렬 쌍
pair = pd.Series([tuple(sorted((a, b))) for a, b in teams], dtype="object")
key = pd.concat([tr.season.astype(str), tr.game_month.astype(str),
                 tr.game_dayofweek.astype(str), tr.game_type.astype(str),
                 pair.astype(str)], axis=1).agg("|".join, axis=1)

# 연속 run id
grp = (key != key.shift()).cumsum()
tr["block"] = grp
print("total blocks (naive):", grp.max())

block_stats = tr.groupby("block").agg(
    n=("row_id", "size"),
    n_inning=("inning", "nunique"),
    season=("season", "first"),
    pair=("pitcher_team_id", lambda s: tuple(sorted(set(s)))),
    has_T=("top_bottom", lambda s: (s == "T").any()),
    has_B=("top_bottom", lambda s: (s == "B").any()),
).reset_index()

print("\n--- block size distribution ---")
print(block_stats.n.describe())
print("\nblocks by size bucket:")
b = pd.cut(block_stats.n, [0, 50, 100, 200, 300, 400, 500, 1000, 10000])
print(block_stats.groupby(b, observed=True).size())

# 비정상 블록 (한쪽 top만, 이닝 1개만 등)
print("\nblocks with only T or only B:", ((block_stats.has_T) != (block_stats.has_B)).sum())
print("blocks with n_inning==1:", (block_stats.n_inning == 1).sum())

# 상위 몇 블록 샘플 확인
print("\n--- sample of first blocks ---")
print(tr[tr.block <= 3][["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
                          "balls_before", "strikes_before", "outs_before", "pitcher_hand",
                          "batter_hand", "pitcher_team_id", "batter_team_id"]].to_string())
