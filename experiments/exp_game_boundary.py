# -*- coding: utf-8 -*-
"""1) train 게임경계 정밀 복원 2) TM 게임 서명(카운트워크) 구축 3) 핑거프린트 매칭 실험"""
import pandas as pd
import numpy as np
import time

DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/"

t0 = time.time()
# ---------- train ----------
tr = pd.read_csv(DATA + "train.csv", usecols=["row_id", "season", "game_month", "game_dayofweek",
                                              "game_type", "inning", "top_bottom",
                                              "balls_before", "strikes_before", "outs_before",
                                              "pitcher_hand", "batter_hand",
                                              "pitcher_team_id", "batter_team_id"])
print("train loaded", time.time() - t0, "s", flush=True)

# 게임 시작 = (inning==1, top_bottom=='T', 0-0-0)
is_start = (tr.inning == 1) & (tr.top_bottom == "T") & (tr.balls_before == 0) & \
           (tr.strikes_before == 0) & (tr.outs_before == 0)
print("game-start markers:", is_start.sum())
print("expected 2019-24 KBO games: ~720*6 = 4320 (regular R)")

# 게임 인덱스
game_id = is_start.cumsum() - 1
tr["gid"] = game_id
n_train_games = tr.gid.nunique()
print("train games:", n_train_games)
print("games by game_type:")
print(tr.groupby("gid").game_type.first().value_counts())

# 게임별 행수
gs = tr.groupby("gid").size()
print("game size desc:\n", gs.describe())

# 같은 gid 내 시작마커가 1개만인지 (게임이 제대로 분리되었는지)
cnt_start = tr.groupby("gid").apply(lambda d: int((d.inning==1)&(d.top_bottom=="T")&(d.balls_before==0)&(d.strikes_before==0)&(d.outs_before==0)), include_groups=False)
print("games with start markers ==1:", (cnt_start == 1).sum(), "!=1:", (cnt_start != 1).sum())

# top_bottom으로 홈/원정 팀 확정: top이면 pitcher_team이 홈?
# 게임 내 pitcher_team_id 변화 횟수 (교체 있음)
pt_change = tr.groupby("gid")["pitcher_team_id"].nunique()
print("games where pitcher_team changes:", (pt_change > 1).sum())

# ---------- TM ----------
tm = pd.read_csv(DATA + "trackman_history.csv", usecols=["season", "game_date", "game_month",
                  "game_dayofweek", "trackman_game_id", "pitch_no", "inning", "top_bottom",
                  "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
                  "pitcher_team", "batter_team", "pitch_type_group"])
print("TM loaded", time.time() - t0, "s", flush=True)

# TM 게임별 행수
tms = tm.groupby("trackman_game_id").size()
print("TM games:", tm.trackman_game_id.nunique())
print("TM game size desc:\n", tms.describe())

# TM top_bottom mapping
print("\nTM top_bottom values:", tm.top_bottom.unique())
print("TM season-game count by year:")
print(tm.groupby("season").trackman_game_id.nunique())

# TM 게임당 홈/원정 팀 고정 여부 (pitcher_team 변화)
tmpc = tm.groupby("trackman_game_id")["pitcher_team"].nunique()
print("TM games where pitcher_team changes(>1):", (tmpc > 1).sum(), "/", len(tmpc))

# 저장 (다음 스텝에 재사용)
tr[["row_id", "gid", "season", "game_month", "game_dayofweek", "game_type", "inning",
    "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id"]].to_parquet("/tmp/opencode/tr_game.parquet")
tm.to_parquet("/tmp/opencode/tm_game.parquet")
print("saved", time.time() - t0, "s")
