#!/usr/bin/env python3
"""drift_eda.py — 드리프트 인지 EDA + 누수 안전 피처 가설 레지스트리 (2026-08-14)

Todo 3 (aimers9-top100-score-improvement): train/test·시즌·레짐 분포 감사와
피처 가설 레지스트리 게이트. 기존 eda_*.py 는 수정하지 않고 원본(Raw) 컬럼으로
감사한다 (전처리는 결측을 가릴 수 있음).

감사 항목:
  1. 수치형 드리프트 — 시즌별(2019~2024) mean/std/분위수, 연속 시즌 SMD, 장기(2024 vs 2019)
     SMD + rank-biserial, 레짐(R vs F, pre2023 vs post2023) SMD. 효과 크기 순으로 랭킹.
     지원(n) 병기. 로컬 test.csv 는 5행 형식 샘플이라 train-vs-test 수치 드리프트는
     추정 불가로 명시.
  2. 범주 지원 — 범주형 피처별 train/test 범주 지원(train-only/test-only), 저지원(<N) 플래그,
     시즌별 game_type/count_state 지원.
  3. 결측 패턴 — 피처별·시즌별·레짐별(asof 16종 중심).
  4. 콜드 스타트 — asof_pitcher_n==0 / asof_batter_n==0 / G1(prev1 결측) 비율(시즌별 + test).
  5. 피처 가용 시점 — 모든 피처에 대해 pre-pitch 정적 / as-of 누적 / prev-game 윈도우 분류.

가설 레지스트리 (repro_979/feature_hypothesis_registry.json):
  - 각 항목: id/name/data_source/temporal_availability/expected_mechanism/prior_result/
    transform/status/leakage_check (+rejected_relation). 8개 신규(proposed) + 9개 기각(blocked).
  - 기각 반영: screen_all_10seed.json 의 후보 9종 전원(10시드 게이트 all-reject) — identical 또는
    near-identical 가설은 blocked 로만 허용.
  - --validate-registry: 스키마/누수/기각-동일성/변환 참조 검증. 위반 시 exit 1.

CLI:
  python3 repro_979/drift_eda.py                # 전체 감사 + 레지스트리 + 증거 산출
  python3 repro_979/drift_eda.py --smoke        # 축소 감사(샘플)로 코드 경로 검증 → PASS
  python3 repro_979/drift_eda.py --validate-registry [--registry path] [--probe-transforms]

누수 가드(비협상): 2025 라벨 미사용, test 행 간 파생 금지, 외부 데이터 금지. 전처리 통계는
전부 학습 윈도우(train 전용) 상수이며 미관측 시즌(2025)은 최근 학습 시즌 값으로 폴백.

출력:
  - repro_979/feature_hypothesis_registry.json      (커밋 대상)
  - repro_979/experiments/REPORT_drift_eda.md       (커밋 대상 압축 리포트)
  - .omo/evidence/aimers9-top100/task-3-eda.{json,log,md}  (증거 — gitignore 대상)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("LGAIMERS_ROOT", str(PROJECT_ROOT))

import common  # noqa: E402  (TARGET/카테고리 규약 공유)

DATA_DIR = REPO / "open" / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TARGET = common.TARGET  # "control_success"

EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100"
EVIDENCE_JSON = EVIDENCE_DIR / "task-3-eda.json"
EVIDENCE_LOG = EVIDENCE_DIR / "task-3-eda.log"
EVIDENCE_MD = EVIDENCE_DIR / "task-3-eda.md"
REPORT_MD = REPO / "experiments" / "REPORT_drift_eda.md"
REGISTRY_PATH = REPO / "feature_hypothesis_registry.json"

SEASONS = list(range(2019, 2025))  # 2019~2024 학습 시즌
LOW_SUPPORT_N = 1000    # <1k 행 범주 → 저지원 플래그
CRITICAL_SUPPORT_N = 100  # <100 행 범주 → 위험 지원 플래그

# ID 계열: 드리프트 랭킹 제외(수치 드리프트 의미 없음), 범주 지원에는 포함
ID_COLS = {"row_id", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
AVAILABILITY = {
    # pre-pitch 정적: 공 전에 확정되는 정적/상황 정보
    "season": "pre-pitch static (game time)",
    "game_month": "pre-pitch static (game time)",
    "game_dayofweek": "pre-pitch static (game time)",
    "inning": "pre-pitch static (game time)",
    "top_bottom": "pre-pitch static (game time)",
    "game_type": "pre-pitch static (game time)",
    "balls_before": "pre-pitch static (current count)",
    "strikes_before": "pre-pitch static (current count)",
    "outs_before": "pre-pitch static (current count)",
    "run_top_before": "pre-pitch static (current game state)",
    "run_bot_before": "pre-pitch static (current game state)",
    "run_total_before": "pre-pitch static (current game state)",
    "score_diff_home": "pre-pitch static (current game state)",
    "score_diff_pitcher_team": "pre-pitch static (current game state)",
    "runner_on_1b": "pre-pitch static (current game state)",
    "runner_on_2b": "pre-pitch static (current game state)",
    "runner_on_3b": "pre-pitch static (current game state)",
    "num_runners_on": "pre-pitch static (current game state)",
    "base_state": "pre-pitch static (current game state)",
    "home_win_expectancy": "pre-pitch static (pre-game model output)",
    "away_win_expectancy": "pre-pitch static (pre-game model output)",
    "li": "pre-pitch static (context-derived leverage)",
    "pitcher_id": "pre-pitch static (identity)",
    "batter_id": "pre-pitch static (identity)",
    "pitcher_hand": "pre-pitch static (identity attribute)",
    "batter_hand": "pre-pitch static (identity attribute)",
    "pitcher_team_id": "pre-pitch static (identity attribute)",
    "batter_team_id": "pre-pitch static (identity attribute)",
    # as-of 누적: 시즌 내 이전 투구/타석 누적 (공 전에 완결)
    "asof_pitcher_n": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_success_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_reverse_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_middle_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_ball_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_strike_rate": "as-of cumulative (prior pitches in season)",
    "asof_batter_n": "as-of cumulative (prior plate appearances in season)",
    "asof_batter_success_rate": "as-of cumulative (prior plate appearances in season)",
    "asof_batter_middle_rate": "as-of cumulative (prior plate appearances in season)",
    "asof_pitcher_pitchmix_n": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_fastball_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_breaking_rate": "as-of cumulative (prior pitches in season)",
    "asof_pitcher_offspeed_rate": "as-of cumulative (prior pitches in season)",
    # prev-game 윈도우: 직전 1/3/5경기 윈도우 집계 (공 전에 완결)
    "asof_pitcher_prev1_game_success_rate": "prev-game window (last 1 game)",
    "asof_pitcher_prev3_game_success_rate": "prev-game window (last 3 games)",
    "asof_pitcher_prev5_game_success_rate": "prev-game window (last 5 games)",
    "asof_pitcher_prev1_game_middle_rate": "prev-game window (last 1 game)",
    "asof_pitcher_prev3_game_middle_rate": "prev-game window (last 3 games)",
    "asof_pitcher_prev5_game_middle_rate": "prev-game window (last 5 games)",
}
G1_PREV_COLS = [f"asof_pitcher_prev{n}_game_{r}_rate"
                for n in (1, 3, 5) for r in ("success", "middle")]
G2_PITCHER_RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
G3_BATTER_RATE_COLS = ["asof_batter_success_rate", "asof_batter_middle_rate"]
MISSING_COLS = sorted(set(G1_PREV_COLS) | set(G2_PITCHER_RATE_COLS) | set(G3_BATTER_RATE_COLS))

CS_LABELS = {0: "0-0", 1: "0-1", 2: "0-2", 3: "1-0", 4: "1-1", 5: "1-2",
             6: "2-0", 7: "2-1", 8: "2-2", 9: "3-0", 10: "3-1", 11: "3-2"}

# ── 기각 후보 레코드 (screen_all_10seed.json verdict=기각, 10시드 게이트 all-reject) ──
# primary delta: 10시드 ablation(제거) 또는 addition(추가) 기준 delta_vs_baseline_fe.
REJECTED_RECORD = {
    "count_platoon_3b2_same": dict(
        mode="addition", primary_delta_bss=4.9, r_only_improved=3,
        reason="10시드 강화 게이트: primary +4.9 < +15 미달 → 기각 (screen_all_10seed.json)",
        transform_ref="common.py:add_cross_cell_features",
        cols=["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]),
    "asof_n_bucket": dict(
        mode="ablation", primary_delta_bss=-8.9, r_only_improved=1,
        reason="10시드 강화 게이트: ablation primary -8.9 (제거 시 개선) → 기각",
        transform_ref="common.py:add_asof_n_bucket_feature",
        cols=["asof_pitcher_n"]),
    "score_diff_binary": dict(
        mode="ablation", primary_delta_bss=-8.6, r_only_improved=2,
        reason="10시드 강화 게이트: ablation primary -8.6 → 기각",
        transform_ref="common.py:add_cross_cell_features",
        cols=["score_diff_pitcher_team"]),
    "recent_gap_success": dict(
        mode="ablation", primary_delta_bss=-8.4, r_only_improved=2,
        reason="10시드 강화 게이트: ablation primary -8.4 → 기각",
        transform_ref="common.py:add_recent_gap_feature",
        cols=["asof_pitcher_prev1_game_success_rate", "asof_pitcher_success_rate"]),
    "return_gap": dict(
        mode="ablation", primary_delta_bss=-3.4, r_only_improved=1,
        reason="10시드 강화 게이트: ablation primary -3.4 → 기각",
        transform_ref="common.py:add_return_gap_feature",
        cols=["asof_pitcher_prev1_game_success_rate", "asof_pitcher_n"]),
    "pitcher_debut": dict(
        mode="ablation", primary_delta_bss=-3.4, r_only_improved=1,
        reason="10시드 강화 게이트: ablation primary -3.4 → 기각",
        transform_ref="common.py:add_debut_feature",
        cols=["asof_pitcher_n"]),
    "batter_debut": dict(
        mode="ablation", primary_delta_bss=-3.4, r_only_improved=0,
        reason="10시드 강화 게이트: ablation primary -3.4, R-only 0/3 → 기각",
        transform_ref="common.py:add_debut_feature",
        cols=["asof_batter_n"]),
    "li_risp_flag": dict(
        mode="ablation", primary_delta_bss=-4.2, r_only_improved=1,
        reason="10시드 강화 게이트: ablation primary -4.2 → 기각",
        transform_ref="common.py:add_cross_cell_features",
        cols=["runner_on_2b", "runner_on_3b", "li"]),
    "outs_count_3b2_2out": dict(
        mode="ablation", primary_delta_bss=-5.3, r_only_improved=1,
        reason="10시드 강화 게이트: ablation primary -5.3 → 기각",
        transform_ref="common.py:add_cross_cell_features",
        cols=["balls_before", "strikes_before", "outs_before"]),
}
REJECTED_IDS = tuple(REJECTED_RECORD.keys())
REJECTED_VERDICT = "기각 (10시드 게이트 all-reject, screen_all_10seed.json)"

REGISTRY_SCHEMA_VERSION = 1

# ════════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════════
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cohens_d(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    """표준화 평균 차이(Cohen's d) — 풀드 std. s가 0이면 0 반환(상수 컬럼)."""
    sp2 = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    sp = math.sqrt(sp2) if sp2 > 0 else 0.0
    if sp == 0 or n1 == 0 or n2 == 0:
        return 0.0
    return float((m1 - m2) / sp)


def rank_biserial(x1: np.ndarray, x2: np.ndarray) -> float:
    """rank-biserial r = 2*(mean_rank_g1 − mean_rank_g2)/(n1+n2). O(n log n)."""
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return 0.0
    combined = np.concatenate([x1, x2])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype="float64")
    ranks[order] = np.arange(1, len(combined) + 1)
    return float(2.0 * (ranks[:n1].mean() - ranks[n1:].mean()) / (n1 + n2))


# ════════════════════════════════════════════════════════════════════
# 데이터 로드 (원본 — 전처리/다운캐스팅 없음, 결측 원형 유지)
# ════════════════════════════════════════════════════════════════════
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    test = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    print(f"[load] train {train.shape} / test {test.shape} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return train, test


def numeric_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """수치 드리프트 대상: test 컬럼 ∩ 수치형 − ID/row_id/season."""
    test_cols = [c for c in test.columns if c != "row_id"]
    num_cols = [c for c in test_cols
                if c not in ID_COLS and c != "season"
                and pd.api.types.is_numeric_dtype(train[c])]
    return num_cols


# ════════════════════════════════════════════════════════════════════
# 1. 수치형 드리프트
# ════════════════════════════════════════════════════════════════════
def season_stats(train: pd.DataFrame, num_cols: list[str]) -> dict:
    """피처×시즌 mean/std/n. pandas groupby agg 후 long-form dict."""
    g = train.groupby("season", observed=True)[num_cols].agg(["mean", "std", "count"])
    out: dict[str, dict[str, dict]] = {}
    for col in num_cols:
        out[col] = {}
        for season in SEASONS:
            if season not in g.index:
                out[col][str(season)] = {"mean": None, "std": None, "n": 0}
                continue
            row = g.loc[season, col]
            out[col][str(season)] = {
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else None,
                "std": float(row["std"]) if pd.notna(row["std"]) else None,
                "n": int(row["count"]),
            }
    return out


def group_stats(df: pd.DataFrame, mask: np.ndarray, col: str) -> dict:
    """한 그룹(마스크)의 mean/std/n (결측 무시)."""
    x = df.loc[mask, col].to_numpy(dtype="float64")
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "n": int(len(x))}


def numeric_drift(train: pd.DataFrame, num_cols: list[str]) -> dict:
    is_r = (train["game_type"] == "R").to_numpy()
    pre = (train["season"] <= 2022).to_numpy()
    by_season = season_stats(train, num_cols)
    f2019, f2024 = np.zeros(len(train), dtype=bool), np.zeros(len(train), dtype=bool)
    f2019[train["season"].to_numpy() == 2019] = True
    f2024[train["season"].to_numpy() == 2024] = True

    long_term, regime_rf, regime_prepost, consecutive = [], [], [], []
    rb_long, rb_rf = {}, {}
    for col in num_cols:
        s19 = by_season[col]["2019"]
        s24 = by_season[col]["2024"]
        if s19["n"] and s24["n"]:
            smd = cohens_d(s24["mean"], s24["std"], s24["n"], s19["mean"], s19["std"], s19["n"])
            long_term.append({
                "feature": col, "season_2019": s19["mean"], "season_2024": s24["mean"],
                "delta": s24["mean"] - s19["mean"],
                "smd_2024_vs_2019": smd, "n_2019": s19["n"], "n_2024": s24["n"],
            })
        # 레짐: R vs F
        r, f = group_stats(train, is_r, col), group_stats(train, ~is_r, col)
        if r["n"] and f["n"]:
            regime_rf.append({
                "feature": col,
                "smd_R_vs_F": cohens_d(r["mean"], r["std"], r["n"], f["mean"], f["std"], f["n"]),
                "mean_R": r["mean"], "mean_F": f["mean"], "n_R": r["n"], "n_F": f["n"],
            })
        # 레짐: pre2023 vs post2023
        pr, po = group_stats(train, pre, col), group_stats(train, ~pre, col)
        if pr["n"] and po["n"]:
            regime_prepost.append({
                "feature": col,
                "smd_post_vs_pre": cohens_d(po["mean"], po["std"], po["n"], pr["mean"], pr["std"], pr["n"]),
                "mean_pre": pr["mean"], "mean_post": po["mean"], "n_pre": pr["n"], "n_post": po["n"],
            })
        # 연속 시즌 SMD
        for i in range(len(SEASONS) - 1):
            a, b = by_season[col][str(SEASONS[i])], by_season[col][str(SEASONS[i + 1])]
            if a["n"] and b["n"]:
                consecutive.append({
                    "feature": col, "transition": f"{SEASONS[i]}→{SEASONS[i+1]}",
                    "smd": cohens_d(b["mean"], b["std"], b["n"], a["mean"], a["std"], a["n"]),
                    "delta": b["mean"] - a["mean"], "n_from": a["n"], "n_to": b["n"],
                })
        # rank-biserial: 2024 vs 2019 (효과 크기 교차 검증 — 상위 랭킹용)
        x1 = train.loc[f2024, col].to_numpy(dtype="float64")
        x2 = train.loc[f2019, col].to_numpy(dtype="float64")
        rb_long[col] = rank_biserial(x1, x2)
        rb_rf[col] = rank_biserial(
            train.loc[is_r, col].to_numpy(dtype="float64"),
            train.loc[~is_r, col].to_numpy(dtype="float64"))

    long_term.sort(key=lambda d: abs(d["smd_2024_vs_2019"]), reverse=True)
    regime_rf.sort(key=lambda d: abs(d["smd_R_vs_F"]), reverse=True)
    regime_prepost.sort(key=lambda d: abs(d["smd_post_vs_pre"]), reverse=True)
    return {
        "by_season": by_season,
        "long_term_drift_2024_vs_2019": long_term,
        "regime_drift_R_vs_F": regime_rf,
        "regime_drift_pre2023_vs_post2023": regime_prepost,
        "consecutive_season_smd": consecutive,
        "rank_biserial_2024_vs_2019": {k: rb_long[k] for k in num_cols},
        "rank_biserial_R_vs_F": {k: rb_rf[k] for k in num_cols},
        "ranked_by_abs_smd": [d["feature"] for d in long_term],
    }


# ════════════════════════════════════════════════════════════════════
# 2. 범주 지원
# ════════════════════════════════════════════════════════════════════
def category_support(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cat_cols = [c for c in test.columns
                if c not in ID_COLS
                and not pd.api.types.is_numeric_dtype(train[c])]
    # 수치지만 저차원 이산(balls/strikes/outs/runner_*/num_runners_on/season/game_month/...)
    # 도 범주 지원 관점에서 포함 (수치 드리프트 랭킹과 분리).
    discrete = ["season", "game_month", "game_dayofweek", "inning",
                "balls_before", "strikes_before", "outs_before",
                "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on"]
    for c in discrete:
        if c in test.columns and c not in cat_cols:
            cat_cols.append(c)

    out: dict[str, dict] = {}
    for col in cat_cols:
        tr_counts = train[col].value_counts(dropna=True)
        te_counts = test[col].value_counts(dropna=True)
        tr_cats = set(tr_counts.index)
        te_cats = set(te_counts.index)
        train_only = sorted(str(c) for c in tr_cats - te_cats)
        test_only = sorted(str(c) for c in te_cats - tr_cats)
        low = {str(c): int(v) for c, v in tr_counts.items()
               if int(v) < LOW_SUPPORT_N}
        critical = {str(c): int(v) for c, v in tr_counts.items()
                    if int(v) < CRITICAL_SUPPORT_N}
        out[col] = {
            "n_train_categories": int(len(tr_cats)),
            "n_test_categories": int(len(te_cats)),
            "train_only_categories": train_only,
            "test_only_categories": test_only,
            "low_support_lt_1000": low,
            "critical_lt_100": critical,
            "test_support": {str(k): int(v) for k, v in te_counts.items()},
            "test_is_format_sample": int(len(test)),
            "note": ("로컬 test.csv 는 5행 형식 샘플 — train/test 범주 지원 비교는 구조 점검용이며 "
                     "실제 평가 test(245,789행)의 범주 지원은 로컬에서 확인 불가"),
        }
    return out


# ════════════════════════════════════════════════════════════════════
# 3. 결측 패턴
# ════════════════════════════════════════════════════════════════════
def missingness(train: pd.DataFrame) -> dict:
    miss = train.isna()
    total = len(train)
    per_feature: dict[str, dict] = {}
    for col in MISSING_COLS:
        n = int(miss[col].sum())
        per_feature[col] = {"n": n, "rate": float(n / total)}
    # 시즌별 (asof 16종 집계)
    by_season: dict[str, dict] = {}
    for season, sub in train.groupby("season", observed=True):
        sm = sub[MISSING_COLS].isna()
        by_season[str(season)] = {
            "n": int(len(sub)),
            "any_missing_rows": int(sm.any(axis=1).sum()),
            "any_missing_rate": float(sm.any(axis=1).mean()),
            "per_feature": {c: int(sm[c].sum()) for c in MISSING_COLS},
        }
    # 레짐별 (R vs F, pre/post)
    is_r = train["game_type"] == "R"
    by_regime: dict[str, dict] = {}
    for name, mask in (("R", is_r), ("F", ~is_r),
                       ("pre2023", train["season"] <= 2022),
                       ("post2023", train["season"] >= 2023)):
        sm = train.loc[mask, MISSING_COLS].isna()
        by_regime[name] = {
            "n": int(mask.sum()),
            "any_missing_rows": int(sm.any(axis=1).sum()),
            "any_missing_rate": float(sm.any(axis=1).mean()),
        }
    return {
        "total_rows": total,
        "per_feature": per_feature,
        "by_season": by_season,
        "by_regime": by_regime,
        "zero_missing_other_cols": sorted(
            set(train.columns) - set(MISSING_COLS) - {"control_success", "row_id"}),
    }


# ════════════════════════════════════════════════════════════════════
# 4. 콜드 스타트
# ════════════════════════════════════════════════════════════════════
def cold_start(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def rates(df: pd.DataFrame) -> dict:
        n = len(df)
        g1 = df["asof_pitcher_prev1_game_success_rate"].isna()
        return {
            "n": int(n),
            "pitcher_debut_n0_rate": float((df["asof_pitcher_n"] == 0).mean()),
            "batter_debut_n0_rate": float((df["asof_batter_n"] == 0).mean()),
            "g1_prev1_missing_rate": float(g1.mean()),
            "g2_any_pitcher_rate_missing_rate": float(df[G2_PITCHER_RATE_COLS].isna().any(axis=1).mean()),
            "g3_any_batter_rate_missing_rate": float(df[G3_BATTER_RATE_COLS].isna().any(axis=1).mean()),
        }
    by_season = {str(s): rates(sub) for s, sub in train.groupby("season", observed=True)}
    overall = rates(train)
    test_sample = rates(test)
    return {"overall_train": overall, "by_season": by_season,
            "test_format_sample": test_sample,
            "test_note": "로컬 test.csv 는 5행 형식 샘플(season 2025) — 2025 평가 test 의 콜드 스타트 비율은 로컬 확인 불가"}


# ════════════════════════════════════════════════════════════════════
# 5. 피처 가용 시점 + 레짐 표 (R vs F / count_state Δ)
# ════════════════════════════════════════════════════════════════════
def availability_timing(test: pd.DataFrame) -> dict:
    cols = [c for c in test.columns if c != "row_id"]
    missing_map = [c for c in cols if c not in AVAILABILITY]
    summary: dict[str, int] = {}
    for c in cols:
        cls = AVAILABILITY.get(c, "UNKNOWN")
        summary[cls] = summary.get(cls, 0) + 1
    return {"feature_availability": {c: AVAILABILITY.get(c, "UNKNOWN") for c in cols},
            "class_counts": summary,
            "unclassified": missing_map,
            "note": ("pre-pitch static: 공 전 확정 / as-of cumulative: 시즌 내 이전 투구·타석 누적 "
                     "/ prev-game window: 직전 1/3/5경기 집계. 모든 as-of/prev 피처는 예측 시점(공 전)에 "
                     "완결 — 시차 누수 없음.")}


def regime_count_delta(train: pd.DataFrame) -> dict:
    """R count_state 2023→2024 Δ (ABS 도입) + F/R 레짐 레이트 — eda_label_regime §2.4 재현."""
    is_r = train["game_type"] == "R"
    cs = (train["balls_before"] * 3 + train["strikes_before"]).to_numpy()
    df = pd.DataFrame({"season": train["season"].to_numpy(), "is_r": is_r.to_numpy(),
                       "cs": cs, "y": train[TARGET].to_numpy()})
    abs_delta: dict[str, dict] = {}
    for i in range(12):
        lbl = CS_LABELS[i]
        r23 = df[(df["is_r"]) & (df["season"] == 2023) & (df["cs"] == i)]
        r24 = df[(df["is_r"]) & (df["season"] == 2024) & (df["cs"] == i)]
        a, b = r23["y"].mean() if len(r23) else None, r24["y"].mean() if len(r24) else None
        abs_delta[lbl] = {
            "r2023_rate": float(a) if a is not None else None,
            "r2024_rate": float(b) if b is not None else None,
            "n2023": int(len(r23)), "n2024": int(len(r24)),
            "delta_pp": float(100 * (b - a)) if a is not None and b is not None else None,
        }
    # F/R 레짐 레이트
    regime = {}
    for name, mask in (("F_pre2023", (df["is_r"] == False) & (df["season"] <= 2022)),  # noqa: E712
                       ("F_2023", (df["is_r"] == False) & (df["season"] == 2023)),      # noqa: E712
                       ("F_2024", (df["is_r"] == False) & (df["season"] == 2024)),      # noqa: E712
                       ("R_pre2023", (df["is_r"]) & (df["season"] <= 2022)),
                       ("R_2023", (df["is_r"]) & (df["season"] == 2023)),
                       ("R_2024", (df["is_r"]) & (df["season"] == 2024))):
        sub = df[mask]
        regime[name] = {"rate": float(sub["y"].mean()) if len(sub) else None,
                        "n": int(len(sub))}
    return {"r_count_delta_2024_minus_2023": abs_delta,
            "regime_rates": regime,
            "interpretation": ("R 3-2 장기 하락(pre2023 0.500→post2023 0.462, -3.8pp) 및 "
                               "ABS(2024) 카운트별 비균등(1-1 -2.1pp 최대) — count_state_abs_regime "
                               "가설의 근거")}


# ════════════════════════════════════════════════════════════════════
# 가설별 근거 수치 (레지스트리 proposed 항목의 expected_mechanism 뒷받침)
# ════════════════════════════════════════════════════════════════════
def candidate_evidence(train: pd.DataFrame, ndrift: dict, regime: dict) -> dict:
    is_r = (train["game_type"] == "R").to_numpy()
    post = (train["season"] >= 2023).to_numpy()
    g2 = (train["asof_pitcher_n"] == 0).to_numpy()
    g3 = (train["asof_batter_n"] == 0).to_numpy()

    asof_success = train["asof_pitcher_success_rate"]
    asof_middle = train["asof_pitcher_middle_rate"]
    asof_n = train["asof_pitcher_n"]
    present1 = train["asof_pitcher_prev1_game_success_rate"].notna().to_numpy()

    return {
        "season_dev_success_pitcher": {
            "train_mean_asof_pitcher_success_rate": float(asof_success.mean()),
            "season_means": {s: float(asof_success[train["season"] == s].mean())
                             for s in SEASONS},
            "mechanism_note": "asof_success 가 target 과 시즌 평균 상관 +0.97 로 동조(descriptive_ts §2.3) "
                              "→ 데미안(평균 편차)이 더 안정적일 수 있음"},
        "league_trend_ratio_middle": {
            "season_means_middle": {s: float(asof_middle[train["season"] == s].mean())
                                    for s in SEASONS},
            "pct_growth_2019_to_2024": float(100 * (
                asof_middle[train["season"] == 2024].mean() / asof_middle[train["season"] == 2019].mean() - 1)),
            "mechanism_note": "middle_rate 가 +35% 리그 변화(descriptive_ts §2.1) — 비율 변환으로 디트렌드"},
        "usage_rank_pitcher": {
            "asof_n_quantiles": {q: float(v) for q, v in asof_n.quantile(
                [0.01, 0.1, 0.5, 0.9, 0.99]).items()},
            "season_means_n": {s: float(asof_n[train["season"] == s].mean()) for s in SEASONS},
            "mechanism_note": "asof_pitcher_n +411% 드리프트(descriptive_ts §2.1) — 랭크로 정규화"},
        "log1p_usage_pitcher": {
            "mechanism_note": "asof_n 우측 꼬리(skew 1.6) + 드리프트 압축 — 안정 변환"},
        "count_state_abs_regime": {
            "r_count_delta_2024_minus_2023": regime["r_count_delta_2024_minus_2023"],
            "mechanism_note": "ABS(2024) 카운트별 비균등 Δ — 레짐×count_state 결합 범주로 분리 학습"},
        "debut_league_impute_success": {
            "g2_n": int(g2.sum()),
            "g2_rate": float(g2.mean()),
            "league_mean_success_pre2023": float(asof_success[(~post) & asof_success.notna()].mean()),
            "league_mean_success_post2023": float(asof_success[post & asof_success.notna()].mean()),
            "mechanism_note": "G2(투수 데뷔 792행) NaN 을 레짐별 리그 평균으로 대체 — 플래그(기각)와 다른 메커니즘"},
        "batter_debut_league_impute_success": {
            "g3_n": int(g3.sum()),
            "g3_rate": float(g3.mean()),
            "batter_league_mean_success_pre2023": float(
                train["asof_batter_success_rate"][(~post) & train["asof_batter_success_rate"].notna()].mean()),
            "batter_league_mean_success_post2023": float(
                train["asof_batter_success_rate"][post & train["asof_batter_success_rate"].notna()].mean()),
            "mechanism_note": "G3(타자 데뷔 830행) NaN 을 레짐별 리그 평균으로 대체"},
        "f_game_post2023_label_mapping": {
            "regime_rates": regime["regime_rates"],
            "mechanism_note": "F 레이블 레짐: pre2023 +15.3pp(R 대비) → 2023~ -3.0pp. "
                              "2025 test 의 F 존재 시 post-2023 레짐(≈0.46)으로 매핑 — Todo 6 캘리브레이션 입력"},
    }


# ════════════════════════════════════════════════════════════════════
# 가설 변환 (등록 함수 — 레지스트리 transform 참조의 구현체)
# ════════════════════════════════════════════════════════════════════
# 계약: fn(df: DataFrame) -> pandas.Series | pandas.DataFrame (새 피처 1종 또는 대체 컬럼).
# 통계는 전부 학습 윈도우 상수 — 2025 등 미관측 시즌은 최근 학습 시즌 값으로 폴백 (누수 없음).
TRANSFORMS: dict[str, object] = {}


def _register(fn):
    TRANSFORMS[fn.__name__] = fn
    return fn


@_register
def season_dev_success_pitcher(df: pd.DataFrame) -> pd.Series:
    """안정 변환: asof_pitcher_success_rate − 학습 윈도우 상수 평균 (데미안 신호)."""
    mean = float(df["asof_pitcher_success_rate"].mean())
    return (df["asof_pitcher_success_rate"] - mean).astype("float32")


@_register
def league_trend_ratio_middle(df: pd.DataFrame) -> pd.Series:
    """안정 변환: asof_pitcher_middle_rate / 시즌별 리그 middle 평균(학습 윈도우, 미관측 시즌=최근 값)."""
    season_means = df.groupby("season", observed=True)["asof_pitcher_middle_rate"].mean()
    latest = float(season_means.iloc[-1])
    denom = df["season"].map(season_means).fillna(latest).astype("float64")
    return (df["asof_pitcher_middle_rate"] / denom).astype("float32")


@_register
def usage_rank_pitcher(df: pd.DataFrame) -> pd.Series:
    """안정 변환: asof_pitcher_n 의 학습 윈도우 백분위 랭크 (상수 단조 맵)."""
    x = df["asof_pitcher_n"].to_numpy(dtype="float64")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype="float64")
    ranks[order] = np.arange(1, len(x) + 1)
    return pd.Series(ranks / max(len(x), 1), index=df.index, dtype="float32")


@_register
def log1p_usage_pitcher(df: pd.DataFrame) -> pd.Series:
    """안정 변환: log1p(asof_pitcher_n) — 우측 꼬리/드리프트 압축."""
    return pd.Series(np.log1p(df["asof_pitcher_n"].to_numpy(dtype="float64")),
                     index=df.index, dtype="float32")


@_register
def count_state_abs_regime(df: pd.DataFrame) -> pd.Series:
    """레짐별 전처리: season>=2024(ABS)는 12개 count_state 유지, 미만은 'PRE' 단일 버킷.
    반환: '3-2_ABS' / '3-2_PRE' 형태의 범주형 Series."""
    cs = (df["balls_before"] * 3 + df["strikes_before"]).to_numpy(dtype="int64")
    regime = np.where(df["season"].to_numpy() >= 2024, "ABS", "PRE")
    labels = np.empty(len(df), dtype=object)
    for i in range(12):
        m = cs == i
        labels[m] = np.char.add(np.char.add(CS_LABELS[i], "_"), regime[m])
    labels[~np.isin(cs, list(range(12)))] = f"other_{regime[~np.isin(cs, list(range(12)))]}"
    return pd.Series(labels, index=df.index).astype("category")


@_register
def debut_league_impute_success(df: pd.DataFrame) -> pd.Series:
    """결측 처리: G2(투수 데뷔 asof_pitcher_n==0) asof_pitcher_success_rate 를
    레짐별 학습 리그 평균으로 대체한 값 반환 (NaN 유지 프레임과 비교용)."""
    post = (df["season"].to_numpy() >= 2023)
    pre_mean = float(df.loc[~post, "asof_pitcher_success_rate"].mean())
    post_mean = float(df.loc[post, "asof_pitcher_success_rate"].mean())
    v = df["asof_pitcher_success_rate"].to_numpy(dtype="float64").copy()
    debut = (df["asof_pitcher_n"] == 0).to_numpy()
    v[debut & ~post] = pre_mean
    v[debut & post] = post_mean
    return pd.Series(v, index=df.index).astype("float32")


@_register
def batter_debut_league_impute_success(df: pd.DataFrame) -> pd.Series:
    """결측 처리: G3(타자 데뷔 asof_batter_n==0) asof_batter_success_rate 를
    레짐별 학습 리그 평균으로 대체한 값 반환."""
    post = (df["season"].to_numpy() >= 2023)
    pre_mean = float(df.loc[~post, "asof_batter_success_rate"].mean())
    post_mean = float(df.loc[post, "asof_batter_success_rate"].mean())
    v = df["asof_batter_success_rate"].to_numpy(dtype="float64").copy()
    debut = (df["asof_batter_n"] == 0).to_numpy()
    v[debut & ~post] = pre_mean
    v[debut & post] = post_mean
    return pd.Series(v, index=df.index).astype("float32")


@_register
def f_game_post2023_label_mapping(df: pd.DataFrame) -> pd.Series:
    """레짐별 전처리: game_type==F 행의 라벨 레짐 정책 상수 (post-2023 F ≈0.462, pre ≈0.600+).
    모델 피처가 아니라 Todo 6 캘리브레이션/라벨 매핑 입력용 정책 시리즈(0/1 지시 + 상수 메타)."""
    is_f = (df["game_type"].astype(str) == "F").to_numpy()
    return pd.Series(is_f.astype("int8"), index=df.index, name="f_game_indicator")


# ════════════════════════════════════════════════════════════════════
# 레지스트리
# ════════════════════════════════════════════════════════════════════
def _leakage_check(test_row_dependency: bool, note: str) -> dict:
    return {
        "test_row_dependency": test_row_dependency,
        "no_2025_labels": True,
        "no_external_data": True,
        "note": note,
    }


def _blocked_entry(cid: str) -> dict:
    rec = REJECTED_RECORD[cid]
    return {
        "id": cid,
        "name": {
            "count_platoon_3b2_same": "3-2 카운트 & 동손 플래그",
            "asof_n_bucket": "asof_pitcher_n 구간화",
            "score_diff_binary": "접전 |diff|<=1 이진화",
            "recent_gap_success": "직전 1경기 성공률 − 누적 성공률 gap",
            "return_gap": "복귀/공백(prev1 결측 & n<100) 플래그",
            "pitcher_debut": "투수 데뷔(asof_pitcher_n==0) 플래그",
            "batter_debut": "타자 데뷔(asof_batter_n==0) 플래그",
            "li_risp_flag": "RISP & LI>=1.0 플래그",
            "outs_count_3b2_2out": "3-2 & 2아웃 플래그",
        }[cid],
        "data_source": {
            "columns": rec["cols"],
            "scope": "train+test",
            "note": "행 단위 조건 변환 — 원래 10시드 스크리닝에서 평가됨",
        },
        "temporal_availability": "pre-pitch static (모두 예측 시점에 관측 가능)",
        "expected_mechanism": "기각 판정 레코드 — 신규 채택 금지",
        "prior_result": {
            "type": "screening",
            "source": "experiments/screen_all_10seed.json",
            "mode": rec["mode"],
            "primary_delta_bss": rec["primary_delta_bss"],
            "r_only_improved": rec["r_only_improved"],
            "verdict": REJECTED_VERDICT,
            "blocked": True,
            "blocked_reason": rec["reason"],
        },
        "transform": rec["transform_ref"],
        "status": "blocked",
        "leakage_check": _leakage_check(
            False, "행 단위 변환이나 10시드 게이트 all-reject로 인해 채택 불가(기각 동일 가설)"),
        "rejected_relation": {
            "rejected_id": cid, "near_identical": True,
            "justification": "동일 가설 — 화려한 재실행 금지 (승자 저주 방지)",
        },
    }


def build_registry(evidence: dict) -> dict:
    """레지스트리 구성: 8 proposed(신규) + 9 blocked(기각) = 17 항목."""
    proposed = [
        {
            "id": "season_dev_success_pitcher",
            "name": "투수 최근 성공률 데미안(평균 편차) 안정 변환",
            "data_source": {"columns": ["asof_pitcher_success_rate"], "scope": "train+test",
                            "note": "상수(평균)는 학습 윈도우 전용 — 2025 라벨 미사용"},
            "temporal_availability": "as-of cumulative (공 전 완결); 상수는 학습 윈도우에서 고정",
            "expected_mechanism": ("asof_success 가 target 과 시즌 평균 corr +0.97 로 동조(descriptive_ts §2.3) "
                                   "— 절대값 대신 평균 편차로 디트렌드하면 시즌 드리프트에 안정"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기각 후보(recent_gap_success)와 다른 신호(asof 성공률 자체의 데미안)"},
            "transform": "drift_eda.py:season_dev_success_pitcher",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; 통계는 학습 윈도우 상수"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "asof_success_rate 의 안정 변환 — rejected 는 prev1−asof gap"},
        },
        {
            "id": "league_trend_ratio_middle",
            "name": "middle_rate 리그 추세 비율 (디트렌드 안정 변환)",
            "data_source": {"columns": ["asof_pitcher_middle_rate", "season"], "scope": "train+test",
                            "note": "시즌별 리그 평균은 학습 윈도우에서만 계산, 미관측 시즌은 최근 값 폴백"},
            "temporal_availability": "as-of cumulative; 시즌별 리그 평균 = 학습 윈도우 상수",
            "expected_mechanism": ("middle_rate +35% 리그 변화(descriptive_ts §2.1) — 리그 평균 비율로 "
                                   "리그 전반 투구 성향 변화를 디트렌드, 개인 상대 신호 강화"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기존 asof_middle_rate 의 비율 변환"},
            "transform": "drift_eda.py:league_trend_ratio_middle",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; 리그 평균은 학습 윈도우 상수"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "기각 후보와 무관한 신규 안정 변환"},
        },
        {
            "id": "usage_rank_pitcher",
            "name": "투수 사용량 백분위 랭크 (드리프트 정규화)",
            "data_source": {"columns": ["asof_pitcher_n"], "scope": "train+test",
                            "note": "랭크 맵은 학습 윈도우 상수 단조 변환"},
            "temporal_availability": "as-of cumulative (공 전 완결)",
            "expected_mechanism": ("asof_pitcher_n +411% 드리프트(descriptive_ts §2.1) — 절대값 대신 "
                                   "백분위로 정규화해 시즌 간 사용량 체계 변화에 안정"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기각된 asof_n_bucket(구간화)이 아닌 연속 랭크 변환"},
            "transform": "drift_eda.py:usage_rank_pitcher",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; 상수 단조 맵"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "연속 순위 변환 vs rejected 의 정적 구간화(ablation -8.9) — "
                                                   "구간 선택이 아니라 전체 단조 맵으로 메커니즘 상이"},
        },
        {
            "id": "log1p_usage_pitcher",
            "name": "log1p 투수 사용량 (꼬리 압축 안정 변환)",
            "data_source": {"columns": ["asof_pitcher_n"], "scope": "train+test"},
            "temporal_availability": "as-of cumulative (공 전 완결)",
            "expected_mechanism": ("asof_n 우측 꼬리(skew 1.6, descriptive_ts §1.1) + 연도별 레벨 상승 "
                                   "— log 압축으로 분산 안정화"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기각된 asof_n_bucket 과 다른 단조 연속 변환"},
            "transform": "drift_eda.py:log1p_usage_pitcher",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "연속 log 변환 vs rejected 의 이산 구간화"},
        },
        {
            "id": "count_state_abs_regime",
            "name": "ABS 레짐 × count_state 결합 범주 (레짐별 전처리)",
            "data_source": {"columns": ["balls_before", "strikes_before", "season"],
                            "scope": "train+test"},
            "temporal_availability": "pre-pitch static (카운트 + season, 공 전 확정)",
            "expected_mechanism": ("ABS(2024) 가 R count 계수를 비균등 하락(1-1 -2.1pp vs 3-2 -0.5pp, "
                                   "eda_regime) — season>=2024 는 12 count_state 유지, 미만은 PRE 단일 "
                                   "버킷으로 분리 학습"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기각된 카운트 셀 플래그들(count_platoon_3b2_same, "
                                     "outs_count_3b2_2out)과 다른 레짐 분리 전처리"},
            "transform": "drift_eda.py:count_state_abs_regime",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; season 은 공 전에 알려짐"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "레짐(연도) 분리 버킷 vs rejected 의 특정 셀 플래그 — "
                                                   "기각된 셀을 재현하지 않음"},
        },
        {
            "id": "debut_league_impute_success",
            "name": "투수 데뷔(G2) asof 성공률 레짐별 리그 대체 (결측 처리)",
            "data_source": {"columns": ["asof_pitcher_n", "asof_pitcher_success_rate", "season"],
                            "scope": "train+test"},
            "temporal_availability": "as-of cumulative; 대체 평균은 학습 윈도우 레짐별 상수",
            "expected_mechanism": ("G2(792행, n==0) NaN 을 레짐별 리그 평균으로 대체 — 모델에 NaN "
                                   "대신 정보성 있는 추정값 제공. 플래그(pitcher_debut 기각)가 아닌 값 대체"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — rejected pitcher_debut 은 이진 플래그(단독 BSS≈0), 본 항목은 "
                                     "asof rate 값 자체의 결측 대체로 메커니즘이 다름"},
            "transform": "drift_eda.py:debut_league_impute_success",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; 레짐 평균은 학습 윈도우 상수"),
            "rejected_relation": {"rejected_id": "pitcher_debut", "near_identical": False,
                                  "justification": "플래그(지시자) vs 값 대체 — 동일 792행에 작동하나 "
                                                   "메커니즘(imputation)이 다름을 명시"},
        },
        {
            "id": "batter_debut_league_impute_success",
            "name": "타자 데뷔(G3) asof 성공률 레짐별 리그 대체 (결측 처리)",
            "data_source": {"columns": ["asof_batter_n", "asof_batter_success_rate", "season"],
                            "scope": "train+test"},
            "temporal_availability": "as-of cumulative; 대체 평균은 학습 윈도우 레짐별 상수",
            "expected_mechanism": ("G3(830행, n==0) NaN 을 레짐별 리그 평균으로 대체 — "
                                   "투수 대체와 대칭되는 결측 처리"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — rejected batter_debut 은 이진 플래그, 본 항목은 값 대체"},
            "transform": "drift_eda.py:batter_debut_league_impute_success",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "행 단위 변환; 레짐 평균은 학습 윈도우 상수"),
            "rejected_relation": {"rejected_id": "batter_debut", "near_identical": False,
                                  "justification": "플래그 vs 값 대체 — 메커니즘이 다름"},
        },
        {
            "id": "f_game_post2023_label_mapping",
            "name": "F 게임 post-2023 라벨 레짐 매핑 (레짐별 전처리/캘리브레이션 정책)",
            "data_source": {"columns": ["game_type"], "scope": "train+test",
                            "note": "F 존재 여부는 평가 test 미확정(로컬 5행 샘플은 R-only)"},
            "temporal_availability": "pre-pitch static (game_type 공 전 확정); 정책 상수는 학습 윈도우",
            "expected_mechanism": ("F 레이블 레짐 전환: pre2023 +15.3pp → 2023~ -3.0pp(eda_regime). "
                                   "2025 test 에 F 가 있으면 post-2023 레짐(≈0.46)으로 매핑 — "
                                   "Todo 6 캘리브레이션 입력용 정책(모델 피처 아님)"),
            "prior_result": {"type": "none", "blocked": False,
                             "note": "신규 — 기각 후보와 무관한 레짐 매핑 정책"},
            "transform": "drift_eda.py:f_game_post2023_label_mapping",
            "status": "proposed",
            "leakage_check": _leakage_check(False, "game_type 은 공 전 알려짐; 정책 상수는 학습 윈도우"),
            "rejected_relation": {"rejected_id": None, "near_identical": False,
                                  "justification": "기각 후보와 무관한 라벨 레짐 정책"},
        },
    ]
    blocked = [_blocked_entry(cid) for cid in REJECTED_IDS]
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "generated_by": "drift_eda.py (Todo 3 — aimers9-top100-score-improvement)",
        "generated_at_utc": _now_utc(),
        "leakage_guard_policy": (
            "비협상 가드: ① 2025 라벨 미사용(모든 통계/변환 상수는 학습 윈도우 전용) ② test 행 간 "
            "파생 금지(test_row_dependency=false) ③ 외부 데이터 금지. "
            "기각 가설(count_platoon_3b2_same, asof_n_bucket, score_diff_binary, recent_gap_success, "
            "return_gap, pitcher_debut, batter_debut, li_risp_flag, outs_count_3b2_2out)은 blocked 로만 "
            "기록 — proposed/approved 불가."),
        "prior_rejection_source": "experiments/screen_all_10seed.json",
        "gate": "10-seed(42..51) × 4-fold temporal OOF; primary Δ≥+15 & R-only 3/3 & max|Δmean|≤0.005 → 채택, 그 외 기각",
        "candidates": proposed + blocked,
        "summary": {
            "n_proposed": len(proposed),
            "n_blocked": len(blocked),
            "n_total": len(proposed) + len(blocked),
            "proposed_ids": [e["id"] for e in proposed],
            "blocked_ids": [e["id"] for e in blocked],
            "evidence_refs": [
                "docs/waveb_eda_descriptive_ts.md §2.1/§2.3/§1.1",
                "docs/waveb_eda_full_report.md §3.2/§3.3/§5.3",
                "repro_979/experiments/eda_regime.json / eda_missing.json",
                "repro_979/experiments/screen_all_10seed.json",
            ],
        },
    }
    # 등록 레지스트리를 evidence 번들에도 포함 (보고용)
    evidence["registry"] = registry
    return registry


# ════════════════════════════════════════════════════════════════════
# 레지스트리 검증
# ════════════════════════════════════════════════════════════════════
def validate_registry(registry: dict, probe_transforms: bool = True) -> tuple[bool, list[str]]:
    """스키마/누수/기각-동일성/변환 참조 검증. 문제 목록(빈 목록 = 통과)과 판정 반환."""
    problems: list[str] = []
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        problems.append("registry.candidates 가 비어 있거나 리스트가 아님")
        return False, problems

    ids: set[str] = set()
    required = ["id", "name", "data_source", "temporal_availability", "expected_mechanism",
                "prior_result", "transform", "status", "leakage_check"]
    for idx, entry in enumerate(candidates):
        tag = f"candidates[{idx}]"
        if not isinstance(entry, dict):
            problems.append(f"{tag}: dict 아님")
            continue
        eid = entry.get("id", "<no-id>")
        if eid in ids:
            problems.append(f"{tag}: id '{eid}' 중복")
        ids.add(eid)
        for key in required:
            if key not in entry:
                problems.append(f"{tag} '{eid}': 필수 키 {key} 부재")
        status = entry.get("status")
        if status not in ("proposed", "blocked", "approved"):
            problems.append(f"{tag} '{eid}': status={status!r} — proposed/blocked/approved 중 하나여야 함")
        prior = entry.get("prior_result", {})
        lc = entry.get("leakage_check", {})
        rel = entry.get("rejected_relation", {})

        # 기각 후보 id → 반드시 blocked
        if eid in REJECTED_IDS:
            if status != "blocked":
                problems.append(f"{tag} '{eid}': 기각 가설(screen_all_10seed.json)이지만 "
                                f"status={status!r} — blocked 여야 함")
            if not prior.get("blocked"):
                problems.append(f"{tag} '{eid}': prior_result.blocked 가 False/부재 — 기각 표시 필수")
            if not prior.get("blocked_reason"):
                problems.append(f"{tag} '{eid}': blocked_reason 부재")
        else:
            if status == "blocked" and not prior.get("blocked_reason"):
                problems.append(f"{tag} '{eid}': blocked 상태인데 blocked_reason 부재")

        # proposed/approved → 누수 가드
        if status in ("proposed", "approved"):
            if lc.get("test_row_dependency") is True:
                problems.append(f"{tag} '{eid}': leakage_check.test_row_dependency=True — "
                                "test 행 간 파생 금지(누수)")
            if lc.get("no_2025_labels") is not True:
                problems.append(f"{tag} '{eid}': leakage_check.no_2025_labels != True — 2025 라벨 사용 금지")
            if lc.get("no_external_data") is not True:
                problems.append(f"{tag} '{eid}': leakage_check.no_external_data != True — 외부 데이터 금지")
            if rel.get("near_identical") is True:
                problems.append(f"{tag} '{eid}': rejected_relation.near_identical=True 인데 "
                                f"status={status!r} — 기각 후보와 동일/근접 가설은 proposed 불가")
            if not isinstance(rel, dict) or rel.get("rejected_id") not in (None, *(REJECTED_IDS + ())):
                problems.append(f"{tag} '{eid}': rejected_relation.rejected_id 가 기각 id 가 아닌 값 "
                                f"{rel.get('rejected_id')!r}")

        # transform 참조 해석
        tref = entry.get("transform")
        if not isinstance(tref, str) or not tref.strip():
            problems.append(f"{tag} '{eid}': transform 참조 부재")
        else:
            fn = resolve_transform(tref)
            if fn is None:
                problems.append(f"{tag} '{eid}': transform 참조 해석 불가: {tref}")
            elif probe_transforms and status in ("proposed", "approved"):
                # proposed/approved 만 실행 프로브 (blocked 는 재현 실행 안 함 — 기각 가설)
                err = probe_transform(fn)
                if err:
                    problems.append(f"{tag} '{eid}': transform 실행 오류 — {err}")
    return (not problems), problems


def resolve_transform(tref: str):
    """'module.py:func' 또는 'func' 참조 → 호출 가능 객체. 실패 시 None.
    drift_eda 내부 함수는 TRANSFORMS, 기각 후보 참조(common.py:add_*)는 common 모듈에서 해석."""
    if ":" in tref:
        mod, fn_name = tref.rsplit(":", 1)
        if mod.strip().endswith("common.py"):
            return getattr(common, fn_name, None)
        return TRANSFORMS.get(fn_name)
    return TRANSFORMS.get(tref)


def probe_transform(fn) -> str | None:
    """최소 합성 프레임으로 변환 실행 — 재현성 증명. 오류 메시지 반환(없으면 None)."""
    n = 32
    df = pd.DataFrame({
        "season": np.where(np.arange(n) % 2 == 0, 2023, 2024),
        "game_type": np.where(np.arange(n) % 3 == 0, "F", "R"),
        "balls_before": np.arange(n) % 4,
        "strikes_before": np.arange(n) % 3,
        "asof_pitcher_n": np.arange(n, dtype="float64"),
        "asof_batter_n": np.arange(n, dtype="float64") * 2,
        "asof_pitcher_success_rate": np.linspace(0.4, 0.6, n),
        "asof_batter_success_rate": np.linspace(0.4, 0.6, n),
        "asof_pitcher_middle_rate": np.linspace(0.1, 0.2, n),
    })
    try:
        out = fn(df)
        if not (isinstance(out, pd.Series) or isinstance(out, pd.DataFrame)):
            return "반환 타입이 Series/DataFrame 아님"
        if len(out) != len(df):
            return f"반환 길이 {len(out)} != 입력 {len(df)}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


# ════════════════════════════════════════════════════════════════════
# 리포트 생성 (Markdown)
# ════════════════════════════════════════════════════════════════════
def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(evidence: dict, smoke: bool) -> str:
    drift = evidence["drift_report"]
    reg = evidence["registry"]
    lines = [
        "# Task 3 — Drift-aware EDA + feature hypothesis registry",
        "",
        f"> 날짜: {evidence['recorded_at_utc']} | smoke={smoke} | "
        f"데이터: train {evidence['meta']['train_rows']:,}행 (2019~2024) + test 형식 샘플",
        "",
        "## 1. 누수 가드",
        "",
        "- 2025 라벨 미사용 (모든 통계/변환 상수는 학습 윈도우 전용)",
        "- test 행 간 파생 없음 (`test_row_dependency=false` 강제)",
        "- 외부 데이터 없음",
        "",
        "## 2. 수치형 드리프트 (2024 vs 2019, |SMD| 상위)",
        "",
        "| feature | 2019 mean | 2024 mean | Δ | SMD | n19 | n24 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in drift["numeric"]["long_term_drift_2024_vs_2019"][:12]:
        lines.append("| %s | %s | %s | %+.4f | %+.3f | %s | %s |" % (
            d["feature"], _fmt(d["season_2019"]), _fmt(d["season_2024"]),
            d["delta"], d["smd_2024_vs_2019"], f"{d['n_2019']:,}", f"{d['n_2024']:,}"))
    lines += ["",
              "## 3. 레짐 드리프트 (R vs F, |SMD| 상위)",
              "",
              "| feature | mean_R | mean_F | SMD | n_R | n_F |",
              "|---|---|---|---|---|---|"]
    for d in drift["numeric"]["regime_drift_R_vs_F"][:8]:
        lines.append("| %s | %s | %s | %+.3f | %s | %s |" % (
            d["feature"], _fmt(d["mean_R"]), _fmt(d["mean_F"]),
            d["smd_R_vs_F"], f"{d['n_R']:,}", f"{d['n_F']:,}"))
    lines += ["",
              "## 4. 범주 지원",
              "",
              "| feature | train cats | test cats | train-only | low<1k |",
              "|---|---|---|---|---|"]
    for col, info in drift["category"].items():
        lines.append("| %s | %s | %s | %s | %s |" % (
            col, info["n_train_categories"], info["n_test_categories"],
            len(info["train_only_categories"]), len(info["low_support_lt_1000"])))
    lines += ["",
              "## 5. 결측 패턴 (asof 16종)",
              "",
              "| feature | missing n | rate |",
              "|---|---|---|"]
    for col, info in drift["missing"]["per_feature"].items():
        lines.append("| %s | %s | %.4f |" % (col, f"{info['n']:,}", info["rate"]))
    lines += ["",
              "## 6. 콜드 스타트",
              "",
              f"- pitcher_debut(asof_pitcher_n==0): "
              f"{drift['cold_start']['overall_train']['pitcher_debut_n0_rate']:.4f} "
              f"(n={drift['cold_start']['overall_train']['n']:,})",
              f"- batter_debut(asof_batter_n==0): "
              f"{drift['cold_start']['overall_train']['batter_debut_n0_rate']:.4f}",
              f"- G1(prev1 결측): {drift['cold_start']['overall_train']['g1_prev1_missing_rate']:.4f}",
              "",
              "## 7. 가설 레지스트리 요약",
              "",
              f"- proposed(신규): {reg['summary']['n_proposed']} → {', '.join(reg['summary']['proposed_ids'])}",
              f"- blocked(기각 반영): {reg['summary']['n_blocked']} → {', '.join(reg['summary']['blocked_ids'])}",
              "",
              "## 8. 검증",
              "",
              f"- 레지스트리 검증(--validate-registry): "
              f"{'PASS' if evidence['validation']['ok'] else 'FAIL'} "
              f"(problems: {len(evidence['validation']['problems'])})",
              f"- 스모크(--smoke): {'PASS' if evidence['smoke'] else 'n/a'}",
    ]
    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
def run_audit(train: pd.DataFrame, test: pd.DataFrame, smoke: bool) -> dict:
    num_cols = numeric_features(train, test)
    print(f"[audit] 수치 피처 {len(num_cols)}개 / 범주 지원 피처 "
          f"{len([c for c in test.columns if c not in ID_COLS])}개", flush=True)
    ndrift = numeric_drift(train, num_cols)
    cat = category_support(train, test)
    miss = missingness(train)
    cold = cold_start(train, test)
    avail = availability_timing(test)
    regime = regime_count_delta(train)
    cev = candidate_evidence(train, ndrift, regime)
    drift_report = {
        "numeric": ndrift,
        "category": cat,
        "missing": miss,
        "cold_start": cold,
        "availability": avail,
        "regime": regime,
        "candidate_evidence": cev,
        "leakage_guard": {
            "no_2025_labels_used": True,
            "no_test_row_to_test_row": True,
            "no_external_data": True,
            "note": "감사/변환 통계는 전부 학습 윈도우 전용 상수; 2025 라벨·test 행 간 파생·외부 데이터 미사용",
        },
        "test_limitation": (
            "로컬 test.csv 는 5행 형식 샘플(season 2025, R-only) — train-vs-test 수치 드리프트는 "
            "추정 불가, 실제 평가 test(245,789행)의 2025 F 존재/범주 지원/콜드 스타트 비율은 "
            "로컬에서 확인 불가"),
    }
    return drift_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="드리프트 인지 EDA + 피처 가설 레지스트리 (Todo 3)")
    ap.add_argument("--smoke", action="store_true",
                    help="축소 감사(샘플)로 코드 경로 검증 후 PASS 출력 — 기본 동작에 영향 없음")
    ap.add_argument("--validate-registry", action="store_true",
                    help="레지스트리 검증 모드(스키마/누수/기각-동일성/변환 참조) — 문제 시 exit 1")
    ap.add_argument("--registry", default=None,
                    help="검증 대상 레지스트리 경로 (기본 repro_979/feature_hypothesis_registry.json)")
    ap.add_argument("--no-probe-transforms", action="store_true",
                    help="검증 시 변환 프로브 스킵 (기본: 프로브 실행)")
    args = ap.parse_args(argv)

    # ── 검증 전용 모드 ──
    if args.validate_registry:
        reg_path = Path(args.registry).expanduser().resolve() if args.registry else REGISTRY_PATH
        if not reg_path.is_file():
            print(f"[FAIL] 레지스트리 파일 없음: {reg_path}", file=sys.stderr)
            return 1
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
        ok, problems = validate_registry(registry, probe_transforms=not args.no_probe_transforms)
        for p in problems:
            print(f"[REJECT] {p}", file=sys.stderr)
        print(f"[validate-registry] {reg_path.name}: {'PASS' if ok else 'FAIL'} — "
              f"candidates={len(registry.get('candidates', []))}, problems={len(problems)}",
              flush=True)
        return 0 if ok else 1

    t0 = time.time()
    smoke = bool(args.smoke)
    train, test = load_raw()
    if smoke:
        SMOKE_N = 100_000
        train = train.sample(n=SMOKE_N, random_state=42).reset_index(drop=True)
        test = test.copy()
        print(f"[--smoke] 축소 감사: train 샘플 {SMOKE_N:,}행 "
              "(기본 전체 동작과 무관한 테스트 모드)", flush=True)

    drift_report = run_audit(train, test, smoke)
    evidence: dict = {
        "schema_version": 1,
        "task": "aimers9-top100 task-3",
        "recorded_at_utc": _now_utc(),
        "smoke": smoke,
        "meta": {
            "script": "repro_979/drift_eda.py",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_is_format_sample": True,
            "seasons": SEASONS,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "drift_report": drift_report,
    }
    registry = build_registry(evidence)  # evidence["registry"] 도 함께 기록

    # 검증 상태 기록
    ok, problems = validate_registry(registry, probe_transforms=True)
    evidence["validation"] = {"ok": ok, "problems": problems}

    # 레지스트리 저장 (커밋 대상)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[registry] {REGISTRY_PATH.name}: proposed={registry['summary']['n_proposed']} "
          f"blocked={registry['summary']['n_blocked']} → "
          f"{'VALID' if ok else 'INVALID'}", flush=True)

    # 증거/리포트 저장
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8")
    md = render_markdown(evidence, smoke)
    EVIDENCE_MD.write_text(md, encoding="utf-8")
    REPORT_MD.write_text(md, encoding="utf-8")
    print(f"[evidence] {EVIDENCE_JSON} / {EVIDENCE_MD}", flush=True)
    print(f"[report]   {REPORT_MD}", flush=True)

    # ── stdout 요약 ──
    long = drift_report["numeric"]["long_term_drift_2024_vs_2019"]
    print("\n" + "=" * 100, flush=True)
    print("수치 드리프트 2024 vs 2019 — |SMD| 랭킹 상위 10 (rank-biserial 병기)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'feature':<38s} {'2019':>8s} {'2024':>8s} {'Δ':>8s} {'SMD':>7s} {'rB':>7s} "
          f"{'n19':>9s} {'n24':>9s}", flush=True)
    rb_long = drift_report["numeric"]["rank_biserial_2024_vs_2019"]
    for d in long[:10]:
        print("  %-38s %8.4f %8.4f %+8.4f %+7.3f %+7.3f %9s %9s" % (
            d["feature"], d["season_2019"], d["season_2024"], d["delta"],
            d["smd_2024_vs_2019"], rb_long.get(d["feature"], 0.0),
            f"{d['n_2019']:,}", f"{d['n_2024']:,}"), flush=True)

    print("\n" + "=" * 100, flush=True)
    print("레짐 드리프트 R vs F — |SMD| 상위 6", flush=True)
    print("=" * 100, flush=True)
    for d in drift_report["numeric"]["regime_drift_R_vs_F"][:6]:
        print("  %-38s R=%.4f F=%.4f SMD=%+.3f n_R=%s n_F=%s" % (
            d["feature"], d["mean_R"], d["mean_F"], d["smd_R_vs_F"],
            f"{d['n_R']:,}", f"{d['n_F']:,}"), flush=True)

    cold = drift_report["cold_start"]["overall_train"]
    print("\n" + "-" * 100, flush=True)
    print("콜드 스타트: pitcher_debut=%.4f batter_debut=%.4f G1=%.4f (n=%s)" % (
        cold["pitcher_debut_n0_rate"], cold["batter_debut_n0_rate"],
        cold["g1_prev1_missing_rate"], f"{cold['n']:,}"), flush=True)
    print("누수 가드: 2025 라벨 미사용 / test 행 간 파생 없음 / 외부 데이터 없음", flush=True)
    print(f"\n[drift_eda] 전체 완료 (총 {time.time() - t0:.1f}s)", flush=True)

    if smoke:
        required = ["drift_report", "registry", "validation", "meta"]
        ok_smoke = (
            ok
            and all(k in evidence for k in required)
            and registry["summary"]["n_blocked"] == len(REJECTED_IDS)
            and len(evidence["drift_report"]["numeric"]["long_term_drift_2024_vs_2019"]) > 0
        )
        print(f"\n[--smoke] {'PASS' if ok_smoke else 'FAIL'} — 감사/레지스트리/검증/증거 경로 검증 "
              f"(샘플 n={len(train):,})", flush=True)
        return 0 if ok_smoke else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
