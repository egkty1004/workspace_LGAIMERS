# -*- coding: utf-8 -*-
"""Trackman_history 기본 프로파일: season 분포, ID 체계, 팀 매핑, 구종."""
import pandas as pd
import numpy as np

from portability_paths import DATA_DIR

DATA = DATA_DIR

tm_cols = ["trackman_id", "season", "game_date", "game_month", "game_dayofweek",
           "trackman_game_id", "pitch_no", "inning", "top_bottom",
           "balls_before", "strikes_before", "outs_before", "pitch_of_pa",
           "pitcher_trackman_id", "batter_trackman_id",
           "pitcher_hand", "batter_hand", "pitcher_team", "batter_team",
           "tagged_pitch_type", "auto_pitch_type", "pitch_type_group",
           "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]

tm = pd.read_csv(DATA + "trackman_history.csv", usecols=tm_cols)
print("TM rows:", len(tm), "cols:", tm.shape[1])
print("TM seasons:", sorted(tm.season.unique()))
print("\n--- rows per season ---")
print(tm.groupby("season").size())
print("\n--- game_date range per season ---")
g = tm.groupby("season")["game_date"].agg(["min", "max"])
print(g)

print("\n--- NA counts (key cols) ---")
for c in ["pitcher_trackman_id", "batter_trackman_id", "pitcher_team", "batter_team",
          "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "pitch_type_group", "trackman_game_id"]:
    print(f"  {c}: {tm[c].isna().sum()}")

print("\n--- pitch_type_group ---")
print(tm.pitch_type_group.value_counts(dropna=False).head(10))

print("\n--- tagged vs auto agreement ---")
print((tm.tagged_pitch_type == tm.auto_pitch_type).mean())

print("\n--- unique ids ---")
print("pitcher_trackman_id:", tm.pitcher_trackman_id.nunique(), "range", tm.pitcher_trackman_id.min(), tm.pitcher_trackman_id.max())
print("batter_trackman_id:", tm.batter_trackman_id.nunique(), "range", tm.batter_trackman_id.min(), tm.batter_trackman_id.max())
print("trackman_game_id:", tm.trackman_game_id.nunique())
print("pitcher_team:", tm.pitcher_team.nunique())
print("batter_team:", tm.batter_team.nunique())

print("\n--- teams sample ---")
print(sorted(tm.pitcher_team.dropna().unique()))
