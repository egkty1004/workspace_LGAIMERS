# -*- coding: utf-8 -*-
"""train/test 프로파일: 팀 ID 분포, 행수, 시즌."""
import pandas as pd
import numpy as np

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "pitcher_id", "batter_id",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id",
                                              "inning", "top_bottom", "balls_before",
                                              "strikes_before", "outs_before",
                                              "game_type", "control_success"])
te = pd.read_csv(DATA + "test.csv")
print("train rows:", len(tr), "test rows:", len(te))
print("train seasons:", sorted(tr.season.unique()))
print("test seasons:", sorted(te.season.unique()))
print("train pitcher_id range:", tr.pitcher_id.min(), tr.pitcher_id.max(), "nunique:", tr.pitcher_id.nunique())
print("train batter_id range:", tr.batter_id.min(), tr.batter_id.max(), "nunique:", tr.batter_id.nunique())
print("test pitcher_id range:", te.pitcher_id.min(), te.pitcher_id.max(), "nunique:", te.pitcher_id.nunique())
print("test batter_id range:", te.batter_id.min(), te.batter_id.max(), "nunique:", te.batter_id.nunique())

print("\n--- train pitcher_team_id ---")
print(tr.pitcher_team_id.value_counts().sort_index())
print("\n--- train batter_team_id ---")
print(tr.batter_team_id.value_counts().sort_index())

print("\n--- test pitcher_team_id ---")
print(te.pitcher_team_id.value_counts().sort_index())

print("\n--- train game_type ---")
print(tr.game_type.value_counts())

print("\n--- control_success mean by season ---")
print(tr.groupby("season").control_success.mean())

print("\n--- pitcher_id intersection with trackman range? (tm ids ~50008-71775155) ---")
print("train pitcher_id min/max:", tr.pitcher_id.min(), tr.pitcher_id.max())
print("test pitcher_id min/max:", te.pitcher_id.min(), te.pitcher_id.max())
