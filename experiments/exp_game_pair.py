# -*- coding: utf-8 -*-
"""게임 경계 = (홈팀, 원정팀) 순서쌍 변화로 복원."""
import pandas as pd
import numpy as np

from portability_paths import DATA_DIR

DATA = DATA_DIR

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "game_type", "inning", "top_bottom",
                                              "balls_before", "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id"])
print("loaded", len(tr))

# 홈팀: top_bottom=='T' 이면 투수가 홈팀. home = pitcher team in T, away = batter team in T
home = np.where(tr.top_bottom.values == "T", tr.pitcher_team_id.values, tr.batter_team_id.values)
away = np.where(tr.top_bottom.values == "T", tr.batter_team_id.values, tr.pitcher_team_id.values)
pair = home * 1000 + away  # ordered
pair_changed = np.concatenate([[True], pair[1:] != pair[:-1]])

gid = pair_changed.cumsum() - 1
tr["gid"] = gid
print("games by pair-change:", gid.max() + 1)

gs = tr.groupby("gid").size()
print("size desc:\n", gs.describe())
print("\nsizes by bucket:")
print(pd.cut(gs, [0, 50, 100, 200, 300, 400, 500], include_lowest=True).value_counts())

# 이제 게임별: 매 게임 시작이 (1,T,0-0-0)인지 (게임 시작 검증)
st = tr.groupby("gid").agg(
    first_inning=("inning", "first"),
    first_tb=("top_bottom", "first"),
    n=("row_id", "size"),
    season=("season", "first"),
)
print("\nfirst pitch state of games (should be 1/T):")
print(st.groupby(["first_inning", "first_tb"]).size())

print("\ngames per season:", st.groupby("season").size().to_dict())
print("games by game_type:")
gt = tr.groupby("gid").game_type.first()
print(gt.value_counts())

# 게임당 행수 통계 (정상 게임만, size>50)
norm = gs[gs > 50]
print("\nnormal games (size>50):", len(norm), "median", norm.median(), "mean", round(norm.mean(),1))

# 다음 스텝용 저장
tr[["row_id", "gid", "season", "game_month", "game_dayofweek", "game_type", "inning",
    "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id"]].to_parquet("/tmp/opencode/tr_game.parquet")
print("saved tr_game.parquet")
