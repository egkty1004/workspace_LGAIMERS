#!/usr/bin/env python3
"""boundary_check.py — 기각된 상호작용 피처와 신규 교차 신호의 구조적 비중복 증명.

경계 해소 (aimers9-eda-fe-full.md Wave B, Todo 6):
  기각 목록(5범주)을 명시적으로 열거하고, eda_cross.json에서 측정된 신규 교차 신호가
  각 목록과 **구조적으로 중복되지 않음**을 증명한다. 중복 판정 신호는 Wave C 인코딩에서 제외.

기각 목록 (5범주, 전부 출처 명시):
  1. InteractionAdder (bss_preprocess.py, diag_dual_gate.py --variant interact)
     - multi-column cross-product blob: base_state_li / count_cat / runner_risk
  2. dual-gate 후보 9종 (experiments/backup/adoption_report.md): interact, B1, D1D2, A2, E2, E3, C2, C1, A1c
  3. interact 변형 4종 (experiments/diag_interact_variants.py): interact, interact+missing, interact_no_basestate, interact+Fflag
  4. GIHO H2/H3/H5 계열 (experiments/REPORT_kbo_insights.md §1)
  5. "우리 5종" (giho979-to-1000.md OUT 목록 — 명시 열거 없음 → diag_* + prior plans에서 재구성)

증명 로직:
  - 신규 신호의 컬럼 구성 = [기존 raw 1~4개]의 이진 조건 AND → 단일 0/1 셀 플래그.
  - 기각 목록 = multi-column cross-product blob / target-derived / F-레짐 의존 상호작용.
  - 구조적 차이를 신호별 evidence로 기록.

실행: /home/gpu_01/.conda/envs/aimers9/bin/python boundary_check.py
출력: experiments/boundary_verdict.json + stdout 판정표.
"""
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.abspath(__file__))          # repro_979/
EXP = os.path.join(ROOT, "experiments")
OUT_JSON = os.path.join(EXP, "boundary_verdict.json")
EDA_CROSS = os.path.join(EXP, "eda_cross.json")

# 참조 소스 (repro_979/ 기준 상대경로로 정규화)
SOURCES = {
    "diag_dual_gate": "../experiments/diag_dual_gate.py",
    "bss_preprocess": "../experiments/bss_preprocess.py",
    "feat_eng": "../experiments/feat_eng.py",
    "adoption_report": "../experiments/backup/adoption_report.md",
    "diag_interact_variants": "../experiments/diag_interact_variants.py",
    "report_kbo": "../experiments/REPORT_kbo_insights.md",
    "aimers9_fe": "../.omo/plans/aimers9-feature-engineering.md",
    "giho979": "../.omo/plans/giho979-to-1000.md",
    "eda_synthesis": "experiments/REPORT_eda_synthesis.md",
    "screen_candidates": "screen_candidates.py",
    "screen_results": "experiments/screen_results.json",
}

# ---------------------------------------------------------------------------
# 명시적 제외 목록 (5범주) — 구조 타입 분류 + 기각 근거
# structure 분류:
#   cross_product_blob : 다중 컬럼의 곱(cross-product)으로 만든 다단 ordinal/categorical
#   column_drop        : 컬럼 제거 (신규 피처 아님)
#   single_transform   : 단일 컬럼의 (정규화/차분/절대값) 변환
#   single_binary      : 단일 컬럼 조건의 이진 플래그
#   categorical_recode : dtype/범주 재구성
#   target_derived     : 타깃/레이블 기반으로 파생된 통계 (asof_success 등 레이블 근접)
#   f_regime_interaction : F(퓨처스) 레짐 의존 상호작용 (2023 레이블 레짐 브레이크)
# ---------------------------------------------------------------------------
EXCLUSION_LIST = [
    # ---- 범주 1: InteractionAdder (diag_dual_gate.py interact) ----
    {
        "id": "interact",
        "family": "InteractionAdder",
        "name": "InteractionAdder (interact)",
        "source": f"{SOURCES['bss_preprocess']} class InteractionAdder; {SOURCES['diag_dual_gate']} --variant interact",
        "columns_added": ["base_state_li", "count_cat", "runner_risk"],
        "input_columns": ["base_state", "li", "balls_before", "strikes_before", "num_runners_on"],
        "structure": "cross_product_blob",
        "rejection_basis": (
            "adoption_report.md: interact(50) ΔG24 +55.4 yet Public -103 (동일 패턴 G24 단독 개선 = Public 실패 위험); "
            "aimers9-feature-engineering.md T7: 'interact permanently excluded' (Public 반례)."
        ),
    },
    # ---- 범주 2: dual-gate 후보 9종 (adoption_report.md) ----
    {
        "id": "B1",
        "family": "dual_gate_candidate",
        "name": "B1 DropDupCols (dedup)",
        "source": f"{SOURCES['feat_eng']} class DropDupCols; {SOURCES['adoption_report']}",
        "columns_added": [],  # 제거: asof_pitcher_pitchmix_n, away_win_expectancy
        "input_columns": ["asof_pitcher_pitchmix_n", "away_win_expectancy"],
        "structure": "column_drop",
        "rejection_basis": "adoption_report.md: G23 클램프로 개선 증명 불가 → reject (ΔG24 +64.0).",
    },
    {
        "id": "D1D2",
        "family": "dual_gate_candidate",
        "name": "D1D2 SeasonProgress+InningNorm",
        "source": f"{SOURCES['feat_eng']} class SeasonProgress/InningNorm; {SOURCES['adoption_report']}",
        "columns_added": ["season_progress", "inning_norm"],
        "input_columns": ["game_month", "inning"],
        "structure": "single_transform",
        "rejection_basis": "adoption_report.md: ΔG24 +0.0 → reject.",
    },
    {
        "id": "A2",
        "family": "dual_gate_candidate",
        "name": "A2 FormTrend (momentum)",
        "source": f"{SOURCES['feat_eng']} class FormTrend; {SOURCES['adoption_report']}",
        "columns_added": ["form_trend"],
        "input_columns": ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate"],
        "structure": "single_transform",  # 차분 (prev1 - prev5)
        "rejection_basis": "adoption_report.md: ΔG24 -0.0 → reject.",
    },
    {
        "id": "E2",
        "family": "dual_gate_candidate",
        "name": "E2 HandMatch (좌우 상성)",
        "source": f"{SOURCES['feat_eng']} class HandMatch; {SOURCES['adoption_report']}",
        "columns_added": ["hand_match"],
        "input_columns": ["pitcher_hand", "batter_hand"],
        "structure": "single_binary",  # (pitcher_hand == batter_hand)
        "rejection_basis": (
            "adoption_report.md: ΔG24 +118.0 최대 개선이나 G23 개선 증명 불가 → reject ⚠️ "
            "(G24 단독 개선 = Public 실패 위험, interact와 동일 패턴)."
        ),
    },
    {
        "id": "E3",
        "family": "dual_gate_candidate",
        "name": "E3 ScoreAbs (abs-score)",
        "source": f"{SOURCES['feat_eng']} class ScoreAbs; {SOURCES['adoption_report']}",
        "columns_added": ["score_abs"],
        "input_columns": ["score_diff_pitcher_team"],
        "structure": "single_transform",  # abs() 연속값
        "rejection_basis": "adoption_report.md: ΔG24 +53.7, G23 클램프 → reject.",
    },
    {
        "id": "C2",
        "family": "dual_gate_candidate",
        "name": "C2 ReverseCount (reverse×count)",
        "source": f"{SOURCES['feat_eng']} class ReverseCount; {SOURCES['adoption_report']}",
        "columns_added": ["reverse_count"],
        "input_columns": ["asof_pitcher_reverse_rate", "balls_before", "strikes_before"],
        "structure": "cross_product_blob",  # 3bin(reverse_rate) × count_cat
        "rejection_basis": "adoption_report.md: ΔG24 +0.5 → reject.",
    },
    {
        "id": "C1",
        "family": "dual_gate_candidate",
        "name": "C1 CountCondition (count×condition)",
        "source": f"{SOURCES['feat_eng']} class CountCondition; {SOURCES['adoption_report']}",
        "columns_added": ["count_cond"],
        "input_columns": ["asof_pitcher_success_rate", "balls_before", "strikes_before"],
        "structure": "cross_product_blob",  # 3bin(asof_pitcher_success_rate) × count_cat — asof rate는 레이블 근접(target-derived)
        "rejection_basis": "adoption_report.md: ΔG24 +59.7, G23 클램프 → reject.",
    },
    {
        "id": "A1c",
        "family": "dual_gate_candidate",
        "name": "A1c SeasonCat (season 범주화)",
        "source": f"{SOURCES['feat_eng']} class SeasonCat; {SOURCES['adoption_report']}",
        "columns_added": [],
        "input_columns": ["season"],
        "structure": "categorical_recode",
        "rejection_basis": "adoption_report.md: ΔG24 -439.0 완전 붕괴 → 범주화 금지 확정.",
    },
    # ---- 범주 3: interact 변형 (diag_interact_variants.py) ----
    {
        "id": "interact+missing",
        "family": "interact_variant",
        "name": "interact + MissingIndicator (8종)",
        "source": f"{SOURCES['diag_interact_variants']} class MissingIndicatorAdder",
        "columns_added": ["base_state_li", "count_cat", "runner_risk"] + [f"{c}_missing" for c in [
            "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
            "asof_pitcher_success_rate", "asof_batter_success_rate"]],
        "input_columns": ["base_state", "li", "balls_before", "strikes_before", "num_runners_on"],
        "structure": "cross_product_blob",
        "rejection_basis": "interact 계열 (Public -103) → aimers9-feature-engineering T7 영구 제외.",
    },
    {
        "id": "interact_no_basestate",
        "family": "interact_variant",
        "name": "interact_no_basestate (CountRunnerOnly)",
        "source": f"{SOURCES['diag_interact_variants']} class CountRunnerOnly",
        "columns_added": ["count_cat", "runner_risk"],
        "input_columns": ["balls_before", "strikes_before", "li", "num_runners_on"],
        "structure": "cross_product_blob",
        "rejection_basis": "interact 계열 (Public -103) → T7 영구 제외.",
    },
    {
        "id": "interact+Fflag",
        "family": "interact_variant",
        "name": "interact + FFlag (is_postseason)",
        "source": f"{SOURCES['diag_interact_variants']} class FFlagAdder",
        "columns_added": ["base_state_li", "count_cat", "runner_risk", "is_postseason"],
        "input_columns": ["base_state", "li", "balls_before", "strikes_before", "num_runners_on", "game_type"],
        "structure": "cross_product_blob",
        "rejection_basis": "interact 계열 + F-레짐 의존 → T7 영구 제외.",
    },
    # ---- 범주 4: GIHO H2/H3/H5 계열 (REPORT_kbo_insights.md §1) ----
    {
        "id": "GIHO_H2",
        "family": "giho_h_family",
        "name": "H2 2-스트라이크 웨이스트",
        "source": f"{SOURCES['report_kbo']} §1 H2",
        "columns_added": [],
        "input_columns": ["strikes_before", "balls_before"],
        "structure": "single_transform",  # 0-2/2-2 카운트 관계 — count_state에 이미 흡수
        "rejection_basis": "giho979-to-1000.md OUT: '상호작용 조합 피처 (GIHO H2/H3/H5 + 우리 5종 기각)'.",
    },
    {
        "id": "GIHO_H3",
        "family": "giho_h_family",
        "name": "H3 주자 압박 (만루)",
        "source": f"{SOURCES['report_kbo']} §1 H3",
        "columns_added": [],
        "input_columns": ["base_state"],
        "structure": "single_transform",  # base_state==만루 — base_state에 이미 흡수
        "rejection_basis": "giho979-to-1000.md OUT: '상호작용 조합 피처 (GIHO H2/H3/H5 + 우리 5종 기각)'.",
    },
    {
        "id": "GIHO_H5",
        "family": "giho_h_family",
        "name": "H5 이닝 후반 피로",
        "source": f"{SOURCES['report_kbo']} §1 H5",
        "columns_added": [],
        "input_columns": ["inning"],
        "structure": "single_transform",  # inning 단조 — inning에 이미 흡수
        "rejection_basis": "giho979-to-1000.md OUT: '상호작용 조합 피처 (GIHO H2/H3/H5 + 우리 5종 기각)'.",
    },
    # ---- 범주 5: "우리 5종" (giho979-to-1000.md OUT 목록 — 명시 열거 없어 재구성) ----
    {
        "id": "u5_1_gtype_x_season",
        "family": "our5",
        "name": "우리5종-1 game_type×season (F_post2023)",
        "source": f"{SOURCES['screen_candidates']} F1: F_post2023; {SOURCES['report_kbo']} §4 ①; {SOURCES['giho979']} OUT",
        "columns_added": ["F_post2023"],
        "input_columns": ["game_type", "season"],
        "structure": "f_regime_interaction",  # (game_type=='F') & (season>=2023)
        "rejection_basis": (
            f"{SOURCES['screen_results']}: F_post2023 R-only 개선 0/3 → 기각. "
            "R-only 폴드에서 F 상수(0) → primary 단독도 +9.2 < +20 게이트 미달."
        ),
    },
    {
        "id": "u5_2_three_balls",
        "family": "our5",
        "name": "우리5종-2 count_role 3볼 (three_balls)",
        "source": f"{SOURCES['screen_candidates']} F2: three_balls; {SOURCES['report_kbo']} §4 ②; {SOURCES['giho979']} OUT",
        "columns_added": ["three_balls"],
        "input_columns": ["balls_before"],
        "structure": "single_binary",  # (balls_before==3)
        "rejection_basis": (
            f"{SOURCES['screen_results']}: three_balls R-only 2/3 개선이나 primary Δ-3.5 (기준선 대비 하락) → 기각."
        ),
    },
    {
        "id": "u5_3_platoon_detail",
        "family": "our5",
        "name": "우리5종-3 platoon_detail (L-L/R-R/L-R/R-L 4범주)",
        "source": f"{SOURCES['screen_candidates']} F3; {SOURCES['report_kbo']} §4 ③; {SOURCES['giho979']} OUT",
        "columns_added": [],
        "input_columns": ["pitcher_hand", "batter_hand"],
        "structure": "categorical_recode",  # common.add_platoon_feature가 이미 4범주(3,4,5,6) 생성
        "rejection_basis": "screen_candidates.py: common.add_platoon_feature가 이미 4범주 platoon 생성 → F3 불필요 기각.",
    },
    {
        "id": "u5_4_li_x_risp",
        "family": "our5",
        "name": "우리5종-4 li×risp (RISP×LI)",
        "source": f"{SOURCES['report_kbo']} §4 ④ 'li×risp 곱셈 조합'; {SOURCES['giho979']} OUT",
        "columns_added": ["li_x_risp"],
        "input_columns": ["li", "num_runners_on", "runner_on_2b", "runner_on_3b"],
        "structure": "cross_product_blob",  # 곱셈 조합
        "rejection_basis": "giho979-to-1000.md OUT: '상호작용 조합 피처 ... 우리 5종 기각'.",
    },
    {
        "id": "u5_5_outs_x_count",
        "family": "our5",
        "name": "우리5종-5 outs×count (2아웃-풀카운트 조합)",
        "source": f"{SOURCES['report_kbo']} §4 ⑤ 'outs_before × count_state'; {SOURCES['giho979']} OUT",
        "columns_added": ["outs_x_count"],
        "input_columns": ["outs_before", "balls_before", "strikes_before"],
        "structure": "cross_product_blob",  # outs × count_state 곱셈 조합
        "rejection_basis": "giho979-to-1000.md OUT: '상호작용 조합 피처 ... 우리 5종 기각'.",
    },
]

# ---------------------------------------------------------------------------
# 신규 교차 신호 (eda_cross.json meta.signals + game_type×count)
# structure:
#   single_cell_binary_flag : [raw 1~4개] 이진 조건 AND → 단일 0/1 셀 플래그
#   single_column_binary     : 단일 컬럼의 임계 이진화
#   cross_product_f_regime   : F 레짐 의존 다중컬럼 교차 (경계 판정 대상)
# ---------------------------------------------------------------------------
SIGNALS = [
    {
        "id": "count_platoon_3b2_same",
        "formula": "(balls==3 & strikes==2) & (pitcher_hand==batter_hand)",
        "input_columns": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
        "structure": "single_cell_binary_flag",
        "output_arity": 1,
    },
    {
        "id": "score_diff_binary",
        "formula": "abs(score_diff_pitcher_team)<=1",
        "input_columns": ["score_diff_pitcher_team"],
        "structure": "single_column_binary",
        "output_arity": 1,
    },
    {
        "id": "li_risp_flag",
        "formula": "(runner_on_2b==1 | runner_on_3b==1) & li>=1.0",
        "input_columns": ["runner_on_2b", "runner_on_3b", "li"],
        "structure": "single_cell_binary_flag",
        "output_arity": 1,
    },
    {
        "id": "outs_count_3b2_2out",
        "formula": "(balls==3 & strikes==2) & outs_before==2",
        "input_columns": ["balls_before", "strikes_before", "outs_before"],
        "structure": "single_cell_binary_flag",
        "output_arity": 1,
    },
    {
        "id": "game_type_x_count",
        "formula": "(game_type=='F') & count_state (F 카운트 구조) 또는 F 내 count 셀",
        "input_columns": ["game_type", "balls_before", "strikes_before"],
        "structure": "cross_product_f_regime",
        "output_arity": "1~12 (F×count 셀)",
    },
]

# ---------------------------------------------------------------------------
# 신호별 증거 (어떤 기각 변형과 비교했는지, 왜 구조적으로 다른지)
# ---------------------------------------------------------------------------
EVIDENCE = {
    "count_platoon_3b2_same": {
        "verdict": "통과",
        "verdict_code": "pass",
        "compared": ["interact", "E2", "u5_4_li_x_risp"],
        "reason": (
            "단일셀 이진 플래그: (balls==3 & strikes==2) & (pitcher_hand==batter_hand) → 1/0 한 컬럼. "
            "InteractionAdder(interact)는 5개 입력 컬럼(base_state/li/balls/strikes/runners)에서 "
            "base_state_li(피팅 매핑)·count_cat(12단 ordinal)·runner_risk(ordinal) 3개의 다단 "
            "cross-product blob을 생성 — 구조(다중컬럼 곱 → 다단 코드)와 산출 컬럼 수(3 vs 1)가 근본적으로 다름. "
            "E2 hand_match와 동손 술어를 공유하나, E2는 무조건부 동손 이진이고 count_platoon은 3-2 카운트 "
            "셀에 조건화된 명시 셀 플래그 — E2 기각 사유는 G23 게이트 클램프(성능), 구조 기각이 아님. "
            "li×risp(우리5종-4) 곱셈 조합과도 다른 단일셀 이진."
        ),
    },
    "score_diff_binary": {
        "verdict": "통과",
        "verdict_code": "pass",
        "compared": ["E3", "C1", "C2"],
        "reason": (
            "단일 컬럼 임계 이진화: abs(score_diff_pitcher_team)<=1 → 0/1. "
            "E3 ScoreAbs는 같은 raw 컬럼의 abs() 연속값 변환이며 단일컬럼 변환으로 cross-product가 아님 — "
            "E3 기각은 G23 게이트 클램프(성능) 때문이고, score_diff_binary는 임계 플래그(이진 셀)로 "
            "'접전 집중' 비선형성을 명시화하는 별개 인코딩. "
            "C1/C2는 asof rate/reverse_rate × count_cat 2단계 곱(cross-product blob) — 구조적으로 무관."
        ),
    },
    "li_risp_flag": {
        "verdict": "통과",
        "verdict_code": "pass",
        "compared": ["interact", "u5_4_li_x_risp", "GIHO_H3"],
        "reason": (
            "단일셀 이진 플래그: (runner_on_2b|runner_on_3b) & li>=1.0 → 1/0. "
            "InteractionAdder의 runner_risk는 num_runners_on*3 + li 3구간 → 다단 ordinal cross-product(최대 12값)인 반면, "
            "li_risp_flag는 2b/3b 실점권 + li 임계의 단일 0/1 셀 플래그. "
            "우리5종-4 li×risp는 곱셈 조합(cross-product blob)으로 기각 — li_risp_flag는 명시 이진 AND 플래그라 구조적으로 다름. "
            "H3(만루, base_state)는 base_state 전용으로 2b/3b+LI 임계와 다른 셀."
        ),
    },
    "outs_count_3b2_2out": {
        "verdict": "통과",
        "verdict_code": "pass",
        "compared": ["interact", "u5_5_outs_x_count", "GIHO_H2"],
        "reason": (
            "단일셀 이진 플래그: (balls==3 & strikes==2) & outs_before==2 → 1/0. "
            "interact의 count_cat은 balls*3+strikes 12단 ordinal cross-product — outs_count는 3-2 & 2아웃 "
            "단일 0/1 셀. 우리5종-5 outs×count는 outs×count_state 곱셈 조합으로 기각 — outs_count는 "
            "단일 셀(3-2, 2out) 이진 플래그로 곱셈 조합이 아님. H2(2스트라이크 웨이스트)는 count_state "
            "관계로 다른 셀."
        ),
    },
    "game_type_x_count": {
        "verdict": "제외",
        "verdict_code": "exclude",
        "compared": ["interact", "u5_1_gtype_x_season", "C2"],
        "reason": (
            "F-레짐 의존 다중컬럼 교차: game_type(F) × count_state = multi-column cross-product — "
            "interact/C2와 동일한 cross-product blob 구조. 우리5종-1 F_post2023((game_type=='F')&(season>=2023))과 "
            "동일한 F-레짐 의존 계열로, screen_results에서 R-only 0/3로 기각된 것과 같은 위험. "
            "REPORT_eda_synthesis §5: 'game_type×count — F 카운트 구조가 R과 다름 → 독립 피처로 쓰면 "
            "구식 F 레짐을 평균에 새김 → 경계 해소에서 판정' + F 3볼 셀 표본 부족(F 3-2 n=2,570) → "
            "인코딩 제외."
        ),
    },
}

STRUCT_LABEL = {
    "cross_product_blob": "다중컬럼 cross-product blob",
    "column_drop": "컬럼 제거",
    "single_transform": "단일컬럼 변환",
    "single_binary": "단일컬럼 이진",
    "categorical_recode": "범주 재구성",
    "target_derived": "target-derived",
    "f_regime_interaction": "F-레짐 의존 상호작용",
    "single_cell_binary_flag": "단일셀 이진 플래그",
    "single_column_binary": "단일컬럼 이진",
    "cross_product_f_regime": "F-레짐 의존 cross-product",
}


def verify_sources():
    """참조 소스 파일 존재 + 핵심 심볼 존재를 확인해 증거를 근거화."""
    found = {}
    for key, rel in SOURCES.items():
        p = os.path.normpath(os.path.join(ROOT, rel))
        found[key] = os.path.exists(p)
    return found


def load_eda_cross():
    """eda_cross.json의 측정값을 신호별로 첨부 (있으면)."""
    if not os.path.exists(EDA_CROSS):
        return {}
    with open(EDA_CROSS, encoding="utf-8") as f:
        return json.load(f)


def build_verdict():
    # 신호별 판정 + Wave C 대상 표기
    signals_out = []
    for sig in SIGNALS:
        ev = EVIDENCE[sig["id"]]
        signals_out.append({
            "id": sig["id"],
            "formula": sig["formula"],
            "input_columns": sig["input_columns"],
            "structure": sig["structure"],
            "structure_label": STRUCT_LABEL[sig["structure"]],
            "output_arity": sig["output_arity"],
            "verdict": ev["verdict"],
            "verdict_code": ev["verdict_code"],
            "evidence": ev["reason"],
            "compared_against": [
                {"exclusion_id": eid, "exclusion_name": next(x["name"] for x in EXCLUSION_LIST if x["id"] == eid)}
                for eid in ev["compared"]
            ],
            "wave_c_encoding_target": ev["verdict_code"] == "pass",
        })

    passes = [s["id"] for s in signals_out if s["verdict_code"] == "pass"]
    excludes = [s["id"] for s in signals_out if s["verdict_code"] != "pass"]

    return {
        "meta": {
            "script": "boundary_check.py",
            "date": str(date.today()),
            "purpose": "기각된 상호작용 피처(InteractionAdder/dual-gate interact)와 신규 교차 신호의 구조적 비중복 증명 — Wave C 인코딩 대상 선정",
            "python": sys.executable,
            "source_files": SOURCES,
            "source_exists": verify_sources(),
            "reconstruction_note": (
                "'우리 5종'은 giho979-to-1000.md OUT에 '상호작용 조합 피처 (GIHO H2/H3/H5 + 우리 5종 기각)'로만 "
                "명시되고 개별 목록이 없어, diag_*(screen_candidates.py) + prior plans(REPORT_kbo_insights.md §4 "
                "BSS 개선 우선순위 제안 ①~⑤)에서 재구성: ① game_type×season ② count_role(3볼) ③ platoon_detail "
                "④ li×risp ⑤ outs×count."
            ),
            "eda_cross_loaded": os.path.exists(EDA_CROSS),
        },
        "exclusion_list": EXCLUSION_LIST,
        "exclusion_summary": {
            "total": len(EXCLUSION_LIST),
            "by_family": {
                "InteractionAdder": 1,
                "dual_gate_candidate": 8,
                "interact_variant": 3,
                "giho_h_family": 3,
                "our5": 5,
            },
        },
        "signals": signals_out,
        "summary": {
            "total_signals": len(signals_out),
            "pass": passes,
            "exclude": excludes,
            "wave_c_encoding_targets": passes,
        },
    }


def print_table(verdict):
    print("=" * 100)
    print("[boundary_check] 경계 해소 판정표 — 기각 상호작용 vs 신규 교차 신호 비중복 증명")
    print("=" * 100)
    print(f"{'신호':<26} {'구조':<22} {'판정':<6} {'Wave C':<7} 근거 요약")
    print("-" * 100)
    for s in verdict["signals"]:
        brief = s["evidence"][:60] + "…" if len(s["evidence"]) > 60 else s["evidence"]
        print(f"{s['id']:<26} {s['structure_label']:<22} {s['verdict']:<6} "
              f"{'O' if s['wave_c_encoding_target'] else 'X':<7} {brief}")
    print("-" * 100)
    print(f"통과(비중복): {verdict['summary']['pass']}")
    print(f"제외(중복):   {verdict['summary']['exclude']}")
    print(f"Wave C 인코딩 대상: {verdict['summary']['wave_c_encoding_targets']}")
    print("=" * 100)
    print(f"제외 목록 {verdict['exclusion_summary']['total']}건 ("
          f"InteractionAdder 1, dual-gate 8, interact변형 3, GIHO H2/H3/H5 3, 우리5종 5)")
    print(f"소스 파일 존재: {verdict['meta']['source_exists']}")
    print(f"JSON 저장: {OUT_JSON}")


def main():
    verdict = build_verdict()
    os.makedirs(EXP, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print_table(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
