# -*- coding: utf-8 -*-
"""게임 시작 마커 오탐 디버그: 마커 인접성, 크기 분포, 컬럼 이상값 확인."""
import pandas as pd
import numpy as np

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"

tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "game_type", "inning", "top_bottom",
                                              "balls_before", "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id",
                                              "pitcher_id", "batter_id"])

print("unique top_bottom:", tr.top_bottom.unique())
print("unique game_type:", tr.game_type.unique())
print("inning range:", tr.inning.min(), tr.inning.max())
print("balls range:", tr.balls_before.min(), tr.balls_before.max())

is_start = (tr.inning == 1) & (tr.top_bottom == "T") & (tr.balls_before == 0) & \
           (tr.strikes_before == 0) & (tr.outs_before == 0)
start_idx = np.where(is_start.values)[0]
print("num markers:", len(start_idx))

# 인접 마커 거리
d = np.diff(start_idx)
print("min gap between markers:", d.min(), "max:", d.max())
print("gaps histogram (bins of 50):")
import collections
c = collections.Counter((d//50)*50)
for k in sorted(c): print(f"  gap {k}-{k+49}: {c[k]}")

# 게임 크기 분포
gid = is_start.cumsum() - 1
gs = pd.Series(gid).value_counts().sort_index()
print("\nsizes by bucket:")
print(pd.cut(gs, [0, 10, 50, 100, 200, 300, 400, 500], include_lowest=True).value_counts())

# 마커가 연속 2회 이상 (gap==1)?
print("\nconsecutive markers (gap==1):", (d == 1).sum())

# 몇 번째 마커에서 작은 게임이 시작되는지
small = gs[gs < 50].index
print("num small games (<50 rows):", len(small))
if len(small):
    ex = small[0]
    idx = np.where(gid == ex)[0]
    print("example small game rows:")
    print(tr.iloc[idx][["row_id", "season", "inning", "top_bottom", "balls_before",
                        "strikes_before", "outs_before", "pitcher_team_id", "batter_team_id"]].to_string())

# row_id 파싱해서 순서 확인
rn = tr.row_id.str.split("_").str[1].astype(int)
print("\nrow_id monotonic:", rn.is_monotonic_increasing)
