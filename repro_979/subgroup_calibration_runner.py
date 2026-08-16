#!/usr/bin/env python3
"""subgroup_calibration_runner.py — Todo 8: bounded train-only subgroup-intercept calibration.

이 라운드(aimers9-score-improvement-next-round)의 Wave 3 태스크: Task 3(Primary CatBoost
boundary gate) 또는 Task 7(promotion) 을 통과한 **단일 base 후보** 위에, 오직 primary
causal OOF 만 사용해 **유계(bounded) subgroup-intercept 로짓 캘리브레이션**을 평가한다.

동결 스펙 (plan lines 125-131 — THE FROZEN CONTRACT):
  - 정확히 세 구성: `game_type`, `count_state`, 가산(additive) `game_type+count_state`.
    그룹 키 = raw frozen-MLP categorical 문자열; 미관측(seen) 키는 delta 0 으로 매핑.
  - 각 term: mean BCE + lambda*sum(beta_g^2) 를 가중 primary-population
    `sum(n_g*beta_g)=0` (weighted zero-sum) 제약 하에 최소화 — 결정적 LBFGS, tol 1e-8.
    lambda grid {0.0001, 0.001, 0.01, 0.1}, k grid {5000, 20000, 50000}.
  - 고정 연산 순서: (1) 제약 zero-sum raw beta → (2) n_g/(n_g+k) 로 수축 →
    (3) term 별 [-0.05, 0.05] 클립 → (4) 가산 term 합산 → (5) total delta
    [-0.05, 0.05] 클립 → `p = sigmoid(z_base + C_LOGIT + delta)` (C_LOGIT=-0.0404 동결,
    slope 은 1 로 동결 — 비단위 slope 피팅 금지).
  - 인과 패널: all-row 2021-2024 one-year-ahead OOF. 2021 = cold-start delta 0.
    outer year Y 의 구성은 패널 연도 `<Y` 에서만 피팅. game-type `F` 는 2023+ OOF 가
    패널에 존재하기 전에는 피팅하지 않는다(레짐 플립 0.673→0.470/0.474; ideation line 215
    과 정합 — F 계수에는 pre-2023 행을 절대 사용하지 않는다). 2025 배포 리핏은
    through-2024 OOF 사용.
  - 동점 시 lexicographic 최저 (lambda, k, configuration) 선택 (BSS 동일할 때).
  - 채택 게이트 (본 라운드 = primary fold 만): primary ΔBSS > 1 AND 정의된 paired
    bootstrap LB5 > 0 AND primary mean shift <= 0.005.
  - R 폴드는 절대 로드하지 않는다 (Task 10 전 embargo; 구조 가드 + --fixture
    attempted-r-fold-read → exit 2). C_LOGIT 보존.

SKIPPED_NO_BASE 분기 (이 태스크가 실제로 실행하는 경로):
  - Task 3 증거 verdict == PRIMARY_PASS 또는 Task 7 증거 verdict == PRIMARY_PROMOTED 인
    base 가 하나도 없으면, 캘리브레이션 라벨을 **하나도 읽지 않고**
    (label_sources=[], labels_read=false) SKIPPED_NO_BASE 를 출력하고 exit 0.
  - 현재 실적: Task 3 = PRIMARY_REJECT, Task 7 = SKIPPED → 이 분기가 실행된다.

모드/퇴장 코드:
  - `--primary-causal` (기본): PASS | REJECT | SKIPPED_NO_BASE → exit 0.
  - `--fixture <이름>` (실패 주입, 전부 exit 2):
      same-year-residual      — 스코어 연도의 잔차로 피팅 시도 차단 (누수 가드)
      nonunit-slope           — slope != 1 피팅 시도 차단 (slope 동결 가드)
      cap-violation           — term/total 캡 위반 delta 주입 차단 (캡 가드)
      future-year-coefficient — outer year Y 의 계수를 >=Y 연도 행으로 피팅 시도 차단
      attempted-r-fold-read   — r2022 라벨 읽기 시도 차단 (R-fold embargo)
  - exit 1 = 치명적 입력 오류.

증거: .omo/evidence/aimers9-next-round/task-8-calibration.{json,log,md}
      (+ 사전 등록 config: task-8-calibration-config.json)
      (+ fixture 로그: task-8-calibration-fixture-<name>.log)

구현 원칙: SKIPPED_NO_BASE 는 데이터/라벨 로드 없이 종료한다 (task-7 SKIPPED 컨벤션).
전체 캘리브레이션 경로(패널/피팅/선택/게이트)는 미래 base 를 위한 **결정 완전**
구현으로 포함되며, 순수 함수는 단위 테스트(합성 데이터)로 검증된다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너와 동일

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import repro_979.common as common  # noqa: E402
# Task 2 동결 챔피언 컨트롤 재사용 (읽기 전용 — 모듈 로드 시 자기 무결성 게이트 포함).
from repro_979.qualification_runner import (  # noqa: E402
    FOLDS, R_FOLDS, C_LOGIT, CLIP_LO, CLIP_HI, _sha256,
)
from repro_979.next_round_policy import (  # noqa: E402
    validate_policy, _canonical_sha256, load_json,
)

# numpy 배열 타입 별칭 — basedpyright 제네릭 요구 충족 (catboost_boundary_selector 컨벤션)
F64 = np.dtype(np.float64)
FloatArray = np.ndarray[Any, F64]
I64 = np.dtype(np.int64)
IntArray = np.ndarray[Any, I64]
JsonDict = dict[str, Any]

# ════════════════════════════════════════════════════════════════════
# 동결 제어 상수 (계획 Task 8 — 수정 금지)
# ════════════════════════════════════════════════════════════════════
SCHEMA_VERSION = 1
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
POLICY_PATH = REPO / "next_round_policy.json"

# base 후보 자격 증거 (이 태스크의 분기 입력)
TASK3_EVIDENCE = EVIDENCE_DIR / "task-3-cat-boundary-gate.json"
TASK7_EVIDENCE = EVIDENCE_DIR / "task-7-promotion.json"

# 캘리브레이션 그리드 (사전 선언 — 라벨로 튜닝 금지)
LAMBDA_GRID = (0.0001, 0.001, 0.01, 0.1)
K_GRID = (5000, 20000, 50000)
# 구성 순서 = lexicographic tie-break 순서 (plan listing + ideation hierarchy)
CONFIGURATIONS = ("game_type", "count_state", "game_type+count_state")

# 유계 (cap)
TERM_CAP = 0.05      # term 별 클립 [-0.05, 0.05]
TOTAL_CAP = 0.05     # total delta 클립 [-0.05, 0.05]

# 결정적 LBFGS
LBFGS_TOL = 1e-8
LBFGS_MAX_ITER = 500
LBFGS_MEM = 10
SLOPE_FROZEN = 1.0   # slope 은 1 로 동결 (비단위 slope 피팅 금지)

# game-type F 레짐 플립 가드: F 계수는 season >= F_MIN_SEASON OOF 행에서만 피팅.
# (plan: "do not fit game-type F until pre-2023 OOF exists" + ideation line 215 —
#  pre-2023 rate 0.673 → 2023/2024 rate 0.470/0.474. F 는 2023+ OOF 가 패널에 존재하기
#  전에는 피팅하지 않고, 피팅할 때도 pre-2023 행을 절대 사용하지 않는다.)
F_MIN_SEASON = 2023

# 인과 패널 (all-row 2021-2024 one-year-ahead OOF)
PANEL_YEARS = (2021, 2022, 2023, 2024)   # 2021 = cold-start delta 0 (패널 연도 <2021 없음)
SCORED_YEAR = 2024                       # 본 라운드 게이팅 = primary fold 만 (2024 all rows)
DEPLOY_REFIT_YEAR = 2025                 # 배포 리핏: through-2024 OOF 에서 선택 구성 재피팅

# 채택 게이트 (primary fold)
GATE_DELTA_BSS_MIN = 1.0      # primary ΔBSS > 1 (STRICT)
GATE_MEAN_SHIFT_MAX = 0.005   # primary mean shift <= 0.005
GATE_LB5_MIN = 0.0            # paired bootstrap LB5 > 0 (STRICT)

# paired game-cluster bootstrap (계획 검증 전략: 10,000 paired game-cluster resamples,
# seed 20260816). game_id 컬럼이 없으므로 game-cluster 키는 (season, game_month,
# game_dayofweek, pitcher_team_id, batter_team_id) 로 결정적으로 정의한다 (하루에 같은
# 팀 페어의 경기는 1경기).
BOOT_SEED = 20260816
BOOT_N_RESAMPLES = 10000
CLUSTER_KEYS = ("season", "game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id")

# 그룹 키 컬럼 (raw frozen-MLP categorical 문자열)
GROUP_COLUMNS = {"game_type": "game_type", "count_state": "count_state"}

FIXTURES = ("same-year-residual", "nonunit-slope", "cap-violation",
            "future-year-coefficient", "attempted-r-fold-read")


class PolicyViolation(RuntimeError):
    """동결 스펙/정책 위반 — exit 2."""


class LeakageError(RuntimeError):
    """R-only 폴드 라벨/로짓 읽기 또는 스코어 폴드 피팅 누수 — exit 2."""


# ── 작은 헬퍼 ────────────────────────────────────────────────────────
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    import subprocess  # noqa: PLC0415
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}


def _file_sha256(path: Path) -> str:
    return _sha256(path)


# ════════════════════════════════════════════════════════════════════
# 구조 가드 (동결 스펙 — 실패 주입 fixture 가 직접 호출)
# ════════════════════════════════════════════════════════════════════
def _assert_unit_slope(slope: float) -> None:
    """slope 는 1 로 동결 — 비단위 slope 피팅은 PolicyViolation (exit 2)."""
    if slope != SLOPE_FROZEN:
        raise PolicyViolation(
            f"[POLICY] 비단위 slope 피팅 시도: a={slope} != {SLOPE_FROZEN} — "
            f"slope 은 1 로 동결 (subgroup-intercept 캘리브레이션은 오직 가산 delta 만 허용, exit 2)")


def _assert_cap_respected(values: list[float], cap: float = TERM_CAP) -> None:
    """term/total delta 가 캡 내에 있는지 사후 조건 검증 — 위반 시 PolicyViolation (exit 2)."""
    bad = [v for v in values if abs(float(v)) > cap + 1e-12]
    if bad:
        raise PolicyViolation(
            f"[POLICY] 캡 위반 delta: {bad} — |delta| <= {cap} 필수 "
            f"(term/total 클립이 스킵되면 이 가드가 중단, exit 2)")


def _assert_no_future_fit(fit_years: list[int], outer_year: int) -> None:
    """outer year Y 의 계수는 패널 연도 < Y 에서만 피팅 — >= Y 연도 행 사용은 PolicyViolation."""
    future = [y for y in fit_years if y >= outer_year]
    if future:
        raise PolicyViolation(
            f"[POLICY] future-year 계수 피팅 시도: outer_year={outer_year} 피팅 연도에 "
            f"{future} 포함 — outer year Y 의 구성은 패널 연도 < Y 에서만 피팅 (exit 2)")


def _assert_no_same_year_residual(fit_row_ids: IntArray, scored_row_ids: IntArray) -> None:
    """피팅 셋에 스코어 폴드 행이 하나라도 있으면 누수 — LeakageError (exit 2)."""
    if len(fit_row_ids) == 0 or len(scored_row_ids) == 0:
        return
    overlap = np.intersect1d(np.sort(fit_row_ids), np.sort(scored_row_ids))
    if len(overlap) > 0:
        raise LeakageError(
            f"[LEAKAGE] same-year residual 피팅 시도: 피팅 셋에 스코어 폴드 행 "
            f"{len(overlap)} 개 포함 — 스코어 연도의 잔차로 피팅 금지 (exit 2)")


def _read_label_fold(labels: dict[str, FloatArray], fold: str) -> FloatArray:
    """라벨 폴드 읽기 가드 — R-only 폴드는 Task 10 전 절대 읽지 않는다."""
    if fold in R_FOLDS:
        raise LeakageError(
            f"[LEAKAGE] R-only 폴드 라벨 {fold!r} 읽기 차단 — Task 10 전 R-only 폴드는 "
            f"reject-only (exit 2)")
    if fold not in labels:
        raise KeyError(f"라벨 폴드 {fold!r} 부재 — labels = {sorted(labels)}")
    return labels[fold]


def _assert_primary_only_labels(labels: dict[str, FloatArray]) -> None:
    """선택/게이팅 경로가 primary 라벨만 보유했는지 구조 검증."""
    if set(labels) != {"primary"}:
        raise LeakageError(
            f"[LEAKAGE] 게이팅 경로에 primary 외 라벨 존재: {sorted(labels)} — "
            f"primary 라벨만 허용 (exit 2)")


# ════════════════════════════════════════════════════════════════════
# 결정적 LBFGS (순수 numpy — 의존성 없음, RNG 없음)
# ════════════════════════════════════════════════════════════════════
def _line_search(obj_grad: Callable[[FloatArray], tuple[float, FloatArray]],
                 q: FloatArray, f: float, g: FloatArray, d: FloatArray,
                 ) -> tuple[FloatArray, float, FloatArray] | None:
    """Armijo backtracking — 만족 step 을 못 찾으면 None (현재 q 유지, zero-sum 불변)."""
    step = 1.0
    slope0 = float(np.dot(g, d))
    for _ in range(60):
        cand = q + step * d
        f_cand, g_cand = obj_grad(cand)
        if f_cand <= f + 1e-4 * step * slope0:
            return cand, f_cand, g_cand
        step *= 0.5
    return None


def _lbfgs(obj_grad: Callable[[FloatArray], tuple[float, FloatArray]], q0: FloatArray,
           tol: float = LBFGS_TOL, max_iter: int = LBFGS_MAX_ITER, mem: int = LBFGS_MEM):
    """결정적 LBFGS (two-loop recursion) + Armijo backtracking.

    obj_grad(q) -> (f, grad). 반환: (q, f, n_iter). 수렴 = ||grad||_inf <= tol."""
    q = np.asarray(q0, dtype=np.float64).copy()
    if q.size == 0:
        return q, float(obj_grad(q)[0]), 0
    f, g = obj_grad(q)
    n_iter = 0
    if float(np.max(np.abs(g))) <= tol:
        return q, float(f), n_iter
    S: list[FloatArray] = []
    Y: list[FloatArray] = []
    while n_iter < max_iter:
        # ── two-loop recursion (결정적) ──
        if not S:
            d = -g.copy()
        else:
            m = len(S)
            rho = np.empty(m)
            for i in range(m):
                denom = float(np.dot(Y[i], S[i]))
                rho[i] = 1.0 / (denom if denom > 1e-30 else 1e-30)
            alphas = np.empty(m)
            r = g.copy()
            for i in range(m - 1, -1, -1):
                alphas[i] = rho[i] * float(np.dot(S[i], r))
                r = r - alphas[i] * Y[i]
            for i in range(m):
                beta_i = rho[i] * float(np.dot(Y[i], r))
                r = r + S[i] * (alphas[i] - beta_i)
            d = -r
        # ── Armijo backtracking line search ──
        ls = _line_search(obj_grad, q, f, g, d)
        if ls is None:  # line search 실패 — 현재 q 반환 (zero-sum 은 항상 유지)
            break
        q_new, f_new, g_new = ls
        s_ = q_new - q
        y_ = g_new - g
        if float(np.dot(y_, s_)) > 1e-12:
            S.append(s_)
            Y.append(y_)
            if len(S) > mem:
                S.pop(0)
                Y.pop(0)
        q, f, g = q_new, f_new, g_new
        n_iter += 1
        if float(np.max(np.abs(g))) <= tol:
            break
    return q, float(f), n_iter


def _null_space_basis(n: FloatArray) -> FloatArray:
    """n^T 의 영공간 정규직교 기저 (결정적, SVD 없음). n: (G,) 양수 카운트.
    반환 (G, G-1) for G>=2, (G, 0) for G<=1. 프로젝션 M = I - nn^T 의 QR (Householder)."""
    G = int(len(n))
    if G <= 1:
        return np.zeros((G, 0), dtype=np.float64)
    nn = n / float(np.linalg.norm(n))
    M = np.eye(G) - np.outer(nn, nn)
    Q, _ = np.linalg.qr(M[:, :G - 1])
    keep = [j for j in range(Q.shape[1]) if np.linalg.norm(Q[:, j]) > 1e-8]
    return Q[:, keep]


def _obj_grad_q(q: FloatArray, z: FloatArray, y: FloatArray,
                group_ids: IntArray, n_g: FloatArray, lam: float,
                N: FloatArray) -> tuple[float, FloatArray]:
    """q-공간 목적/그래디언트: beta = N@q (항상 sum(n_g*beta)=0 을 만족).
    목적 = mean BCE(z_i + beta_{g_i}) + lam * sum(beta_g^2)."""
    beta = N @ q
    p = np.clip(common.sigmoid(z + beta[group_ids]), 1e-12, 1.0 - 1e-12)
    obj = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    obj += lam * float(np.dot(beta, beta))
    res = p - y
    grad_beta = np.zeros(len(n_g))
    np.add.at(grad_beta, group_ids, res)
    grad_beta /= max(len(z), 1)
    grad_beta += 2.0 * lam * beta
    return obj, N.T @ grad_beta


def _fit_zero_sum_lbfgs(z: FloatArray, y: FloatArray, group_ids: IntArray,
                        n_g: FloatArray, lam: float) -> tuple[FloatArray, JsonDict]:
    """제약 zero-sum raw beta 피팅: min mean_BCE + lam*sum(beta^2) s.t. sum(n_g*beta_g)=0.
    결정적 LBFGS (tol 1e-8) — 순수 numpy. 반환 (beta_raw, diag)."""
    G = int(len(n_g))
    if G <= 1 or len(z) == 0:
        return np.zeros(G), {"n_free_dims": 0, "note": "constraint forces beta=0 (G<=1)"}
    N = _null_space_basis(n_g)
    if N.shape[1] == 0:
        return np.zeros(G), {"n_free_dims": 0, "note": "empty null space"}
    q0 = np.zeros(N.shape[1], dtype=np.float64)
    q, f, n_iter = _lbfgs(
        lambda qq: _obj_grad_q(qq, z, y, group_ids, n_g, lam, N), q0)
    beta = N @ q
    _, grad_q = _obj_grad_q(q, z, y, group_ids, n_g, lam, N)
    diag = {
        "n_free_dims": int(N.shape[1]),
        "objective": f,
        "grad_inf_norm": float(np.max(np.abs(grad_q))),
        "lbfgs_iter": n_iter,
        "lbfgs_tol": LBFGS_TOL,
        "zero_sum_residual": float(np.dot(n_g, beta)),
    }
    return beta, diag


# ════════════════════════════════════════════════════════════════════
# term 피팅 — 고정 연산 순서 (동결 스펙)
#   1) 제약 zero-sum raw beta  →  2) n_g/(n_g+k) 수축  →  3) term 별 클립
# ════════════════════════════════════════════════════════════════════
def fit_term(z_fit: FloatArray, y_fit: FloatArray, keys_fit: FloatArray,
             lam: float, k: int) -> tuple[dict[str, float], JsonDict]:
    """하나의 term (game_type 또는 count_state) 그룹 인터셉트 피팅.

    keys_fit: raw frozen-MLP categorical 문자열 배열. 반환 (delta_map, diag).
    delta_map: {key: 최종 beta} — 미관측 키는 apply 단계에서 0."""
    z = np.asarray(z_fit, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.float64)
    keys = np.asarray(keys_fit)
    if len(z) == 0:
        return {}, {"n_groups": 0, "note": "fit rows 없음 — delta 0"}
    uniq = sorted(np.unique(keys).tolist())  # 결정적 순서
    G = len(uniq)
    order = {key: i for i, key in enumerate(uniq)}
    group_ids = np.asarray([order[key] for key in keys], dtype=np.int64)
    n_g = np.asarray([int((group_ids == i).sum()) for i in range(G)], dtype=np.float64)

    beta_raw, lbfgs_diag = _fit_zero_sum_lbfgs(z, y, group_ids, n_g, lam)
    beta_shrunk = beta_raw * n_g / (n_g + float(k))          # 순서 2: 수축
    beta_final = np.clip(beta_shrunk, -TERM_CAP, TERM_CAP)   # 순서 3: term 클립
    _assert_cap_respected(beta_final.tolist(), cap=TERM_CAP)  # 사후 조건 가드

    delta_map = {uniq[i]: float(beta_final[i]) for i in range(G)}
    diag = {
        "n_groups": G,
        "groups": uniq,
        "n_g": [int(v) for v in n_g.tolist()],
        "n_rows": int(len(z)),
        "lambda": float(lam),
        "k": int(k),
        "beta_raw": [float(v) for v in beta_raw.tolist()],
        "beta_shrunk": [float(v) for v in beta_shrunk.tolist()],
        "beta_final": [float(v) for v in beta_final.tolist()],
        "order_of_operations": [
            "constrained zero-sum raw beta (LBFGS tol 1e-8)",
            "shrink beta_g *= n_g/(n_g+k)",
            "clip each term to [-0.05, 0.05]",
        ],
        **lbfgs_diag,
    }
    return delta_map, diag


# ════════════════════════════════════════════════════════════════════
# 구성 피팅 — 인과 패널 + game-type F 가드
# ════════════════════════════════════════════════════════════════════
def fit_configuration_for_year(panel: dict[str, FloatArray], outer_year: int,
                               configuration: str, lam: float, k: int,
                               ) -> tuple[list[dict[str, float]], list[JsonDict]]:
    """outer year Y 의 구성 피팅 — 패널 연도 < Y 행만 사용 (미래/동년 피팅 가드 포함).

    panel: {"year": int[], "z": float[], "y": float[], "game_type": str[], "count_state": str[]}.
    반환: (delta_terms, diags) — 가산 구성은 term 2개, 단일 구성은 term 1개.
    패널 연도 < Y 가 없으면 cold-start → delta 0."""
    years = np.asarray(panel["year"], dtype=np.int64)
    fit_mask = years < outer_year
    # 구조 가드: 실제 피팅 셋의 연도가 반드시 < outer_year (마스크 회귀 시 차단)
    _assert_no_future_fit(sorted({int(v) for v in years[fit_mask]}), outer_year)
    if not bool(fit_mask.any()):
        return [], [{"note": f"outer_year={outer_year} cold-start: 패널 연도 <Y 없음 → delta 0"}]

    z_f = np.asarray(panel["z"], dtype=np.float64)[fit_mask]
    y_f = np.asarray(panel["y"], dtype=np.float64)[fit_mask]
    gt_f = np.asarray(panel["game_type"]).astype(str)[fit_mask]
    cs_f = np.asarray(panel["count_state"]).astype(str)[fit_mask]
    yr_f = years[fit_mask]

    # game-type F 가드: F 행은 season >= F_MIN_SEASON 만 피팅 모집단에 포함
    # (F 계수에는 pre-2023 행을 절대 사용하지 않음 — 레짐 플립 0.673→0.470/0.474).
    keep_gt = (gt_f != "F") | (yr_f >= F_MIN_SEASON)

    if configuration == "game_type":
        delta_map, diag = fit_term(z_f[keep_gt], y_f[keep_gt], gt_f[keep_gt], lam, k)
        diag["f_guard"] = ("F rows with season >= %d only; pre-2023 F rows excluded "
                           "(regime-flip guard)" % F_MIN_SEASON)
        return [delta_map], [diag]
    if configuration == "count_state":
        delta_map, diag = fit_term(z_f, y_f, cs_f, lam, k)
        return [delta_map], [diag]
    if configuration == "game_type+count_state":
        m1, d1 = fit_term(z_f[keep_gt], y_f[keep_gt], gt_f[keep_gt], lam, k)
        m2, d2 = fit_term(z_f, y_f, cs_f, lam, k)
        return [m1, m2], [d1, d2]
    raise ValueError(f"미등록 구성: {configuration!r} — {CONFIGURATIONS}")


def apply_delta(z_base: FloatArray, delta_terms: list[dict[str, float]],
                key_arrays: list[FloatArray]) -> FloatArray:
    """행별 delta = Σ_term term.get(key, 0.0) (미관측 키 → 0), total 클립 [-TOTAL_CAP, TOTAL_CAP].
    순서 4(합산) + 순서 5(total 클립) — p = sigmoid(z_base + C_LOGIT + delta) 는 스코어에서."""
    delta = np.zeros(len(z_base), dtype=np.float64)
    for term, keys in zip(delta_terms, key_arrays):
        keys = np.asarray(keys).astype(str)
        for key, beta in term.items():
            mask = keys == key
            delta[mask] += beta
    delta = np.clip(delta, -TOTAL_CAP, TOTAL_CAP)  # 순서 5: total 클립
    # 사후 조건 가드: 클립을 제거하는 회귀가 생기면 이 가드가 중단 (exit 2)
    _assert_cap_respected([float(np.max(np.abs(delta)))], cap=TOTAL_CAP)
    return delta


# ════════════════════════════════════════════════════════════════════
# 스코어링 (배포 산식: clip + sigmoid(z_base + C_LOGIT + delta)) + 부트스트랩
# ════════════════════════════════════════════════════════════════════
def _score_p(p: FloatArray, y: FloatArray) -> float:
    return float(common.score(p, y))


def score_base(z_base: FloatArray, y: FloatArray) -> tuple[float, float]:
    """미캘리브레이션 base: p = clip(sigmoid(z_base + C_LOGIT), 0.30, 0.70). (bss, p_mean)"""
    p = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    return _score_p(p, y), float(p.mean())


def score_calibrated(z_base: FloatArray, delta: FloatArray, y: FloatArray) -> tuple[float, float]:
    """캘리브레이션 적용: p = clip(sigmoid(z_base + C_LOGIT + delta), 0.30, 0.70). (bss, p_mean)"""
    p = np.clip(common.sigmoid(z_base + C_LOGIT + delta), CLIP_LO, CLIP_HI)
    return _score_p(p, y), float(p.mean())


def paired_bootstrap_lb5(z_base: FloatArray, delta: FloatArray, y: FloatArray,
                         cluster_ids: FloatArray,
                         n_resamples: int = BOOT_N_RESAMPLES,
                         seed: int = BOOT_SEED) -> float:
    """정의된 paired bootstrap LB5: 10,000 paired game-cluster resamples 의 ΔBSS 5th percentile.

    paired = 같은 클러스터 재표본으로 base/cal 둘 다 스코어. game-cluster = CLUSTER_KEYS
    조합 (game_id 컬럼 부재 — 계획 검증 전략 정의를 결정적으로 구현)."""
    if len(z_base) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    r = float(y.mean())
    var = r * (1.0 - r)
    if var <= 0.0:
        return float("nan")
    p_base = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    p_cal = np.clip(common.sigmoid(z_base + C_LOGIT + delta), CLIP_LO, CLIP_HI)
    clusters, inv = np.unique(cluster_ids, return_inverse=True)
    cnts = np.bincount(inv, minlength=int(len(clusters)))
    w = cnts / float(cnts.sum())
    mse_base = np.bincount(inv, weights=(p_base - y) ** 2, minlength=int(len(clusters))) / cnts
    mse_cal = np.bincount(inv, weights=(p_cal - y) ** 2, minlength=int(len(clusters))) / cnts
    counts = rng.multinomial(len(z_base), w, size=n_resamples).astype(np.float64)
    mse_base_r = (counts @ mse_base) / float(len(z_base))
    mse_cal_r = (counts @ mse_cal) / float(len(z_base))
    d_bss = 100000.0 * (mse_base_r - mse_cal_r) / var
    return float(np.percentile(d_bss, 5))


# ════════════════════════════════════════════════════════════════════
# 선택 (max primary BSS, 동점 → lexicographic 최저 (lambda, k, configuration)) + 게이트
# ════════════════════════════════════════════════════════════════════
def _lexico_key(rec: JsonDict) -> tuple[int, int, int]:
    return (int(rec["li"]), int(rec["ki"]), int(rec["ci"]))


def _better(a: JsonDict, b: JsonDict) -> bool:
    """a 가 b 보다 우선인가: BSS 최대, 동점 시 lexicographic 최저 (li, ki, ci)."""
    if a["bss"] != b["bss"]:
        return a["bss"] > b["bss"]
    return _lexico_key(a) < _lexico_key(b)


def _select_best(records: list[JsonDict]) -> JsonDict:
    assert records, "선택할 구성 레코드 없음"
    best = records[0]
    for rec in records[1:]:
        if _better(rec, best):
            best = rec
    return best


def evaluate_calibration_grid(panel: dict[str, FloatArray],
                              scored_z: FloatArray, scored_y: FloatArray,
                              scored_keys: dict[str, FloatArray],
                              ) -> tuple[JsonDict, list[JsonDict]]:
    """(lambda, k, configuration) 그리드 전부를 스코어 연도(primary)에 대해 평가.
    반환 (best, records). best = max primary BSS, 동점 시 lexicographic 최저."""
    records: list[JsonDict] = []
    for li, lam in enumerate(LAMBDA_GRID):
        for ki, k in enumerate(K_GRID):
            for ci, configuration in enumerate(CONFIGURATIONS):
                delta_terms, diags = fit_configuration_for_year(
                    panel, SCORED_YEAR, configuration, lam, k)
                key_arrays = _key_arrays_for(scored_keys, configuration)
                delta = apply_delta(scored_z, delta_terms, key_arrays)
                bss, p_mean = score_calibrated(scored_z, delta, scored_y)
                records.append({
                    "li": li, "ki": ki, "ci": ci,
                    "lambda": float(lam), "k": int(k),
                    "configuration": configuration,
                    "bss": bss, "p_mean": p_mean,
                    "max_abs_delta": float(np.max(np.abs(delta))),
                    "delta_terms": delta_terms, "fit_diags": diags,
                })
    return _select_best(records), records


def _key_arrays_for(scored_keys: dict[str, FloatArray], configuration: str) -> list[FloatArray]:
    if configuration == "game_type":
        return [scored_keys["game_type"]]
    if configuration == "count_state":
        return [scored_keys["count_state"]]
    if configuration == "game_type+count_state":
        return [scored_keys["game_type"], scored_keys["count_state"]]
    raise ValueError(f"미등록 구성: {configuration!r}")


def gate_verdict(best: JsonDict, base_bss: float, base_p_mean: float,
                 lb5: float) -> tuple[str, JsonDict]:
    """채택 게이트 (본 라운드 = primary fold): ΔBSS > 1 AND LB5 > 0 AND mean shift <= 0.005."""
    delta_bss = best["bss"] - base_bss
    mean_shift = abs(best["p_mean"] - base_p_mean)
    gate = {
        "delta_bss": delta_bss,
        "delta_bss_ok": bool(delta_bss > GATE_DELTA_BSS_MIN),
        "bootstrap_lb5": lb5,
        "bootstrap_lb5_ok": bool(lb5 > GATE_LB5_MIN),
        "mean_shift": mean_shift,
        "mean_shift_ok": bool(mean_shift <= GATE_MEAN_SHIFT_MAX),
        "thresholds": {
            "delta_bss_min": GATE_DELTA_BSS_MIN,
            "bootstrap_lb5_min": GATE_LB5_MIN,
            "mean_shift_max": GATE_MEAN_SHIFT_MAX,
        },
        "base_bss": base_bss,
        "base_p_mean": base_p_mean,
    }
    verdict = "PASS" if (gate["delta_bss_ok"] and gate["bootstrap_lb5_ok"]
                         and gate["mean_shift_ok"]) else "REJECT"
    return verdict, gate


# ════════════════════════════════════════════════════════════════════
# 인과 패널 (all-row 2021-2024 one-year-ahead OOF) — 계약 검증
# ════════════════════════════════════════════════════════════════════
def build_causal_panel(per_year_oof: dict[int, JsonDict]) -> dict[str, FloatArray]:
    """연도별 one-year-ahead OOF → 패널 집합 (계약 검증 포함).

    per_year_oof[year] = {"row_ids": int[], "z_base": float[], "y": float[],
                          "game_type": str[], "count_state": str[],
                          "cluster_id": str[]} — year in PANEL_YEARS.
    one-year-ahead 계약: season=t 행의 z_base 는 <=t-1 학습 모델의 예측 (호출자가 보장;
    여기서는 연도 범위/2025 부재/행 분리/유한 로짓을 검증). 2021 은 cold-start —
    패널 연도 <2021 이 없으므로 피팅에 영원히 사용되지 않는다."""
    if not per_year_oof:
        raise PolicyViolation("[POLICY] 빈 인과 패널 — 피팅/게이팅 불가 (exit 2)")
    panel = {"year": [], "row_ids": [], "z": [], "y": [],
             "game_type": [], "count_state": [], "cluster_id": []}
    seen_rows: set[int] = set()
    for year in sorted(per_year_oof):
        rec = per_year_oof[year]
        if year not in PANEL_YEARS:
            raise PolicyViolation(
                f"[POLICY] 패널 연도 {year} ∉ {PANEL_YEARS} — all-row 2021-2024 one-year-ahead "
                f"OOF 계약 위반 (exit 2)")
        n = len(rec["row_ids"])
        if n == 0 or len(rec["z_base"]) != n or len(rec["y"]) != n:
            raise PolicyViolation(f"[POLICY] 연도 {year} 패널 행 길이 불일치 (exit 2)")
        z = np.asarray(rec["z_base"], dtype=np.float64)
        if not np.isfinite(z).all():
            raise PolicyViolation(f"[POLICY] 연도 {year} z_base 에 비유한 값 (exit 2)")
        row_ids = np.asarray(rec["row_ids"], dtype=np.int64)
        dup = seen_rows & set(row_ids.tolist())
        if dup:
            raise PolicyViolation(f"[POLICY] 패널 행 ID 중복: {sorted(dup)[:5]} (exit 2)")
        seen_rows.update(row_ids.tolist())
        panel["year"].extend([int(year)] * n)
        panel["row_ids"].extend(row_ids.tolist())
        panel["z"].extend(z.tolist())
        panel["y"].extend(np.asarray(rec["y"], dtype=np.float64).tolist())
        panel["game_type"].extend(np.asarray(rec["game_type"]).astype(str).tolist())
        panel["count_state"].extend(np.asarray(rec["count_state"]).astype(str).tolist())
        panel["cluster_id"].extend(np.asarray(rec["cluster_id"]).astype(str).tolist())
    out = {k: np.asarray(v) for k, v in panel.items()}
    if 2021 in per_year_oof:
        out["_note_2021_cold_start"] = np.asarray(
            ["2021 = cold-start delta 0 (패널 연도 <2021 없음)"] * len(out["year"]))
    return out


# ════════════════════════════════════════════════════════════════════
# base 후보 자격 판정 (Task 3 / Task 7)
# ════════════════════════════════════════════════════════════════════
def _evidence_verdict(path: Path) -> tuple[str, JsonDict]:
    """증거 JSON 로드 → (verdict, record). 없거나 파싱 불가면 ("MISSING", {})."""
    rec = load_json(path)
    if rec is None:
        return "MISSING", {}
    return str(rec.get("verdict", "MISSING")), rec


def base_eligibility() -> tuple[bool, JsonDict, str]:
    """Task 3 (PRIMARY_PASS) 또는 Task 7 (PRIMARY_PROMOTED) base 여부 판정.
    반환: (eligible, base_info|{}, reason). 라벨은 읽지 않는다 — 결정 증거 JSON 만 읽음."""
    t3_verdict, t3_rec = _evidence_verdict(TASK3_EVIDENCE)
    t7_verdict, t7_rec = _evidence_verdict(TASK7_EVIDENCE)

    if t3_verdict == "PRIMARY_PASS":
        frozen = t3_rec.get("gate", {}).get("frozen", {})
        base = {
            "source": "task-3",
            "candidate_id": frozen.get("candidate_id"),
            "weights_by_member": frozen.get("weights_by_member"),
            "primary_bss": frozen.get("primary_bss"),
        }
        return True, base, f"task-3 verdict={t3_verdict} → base {base.get('candidate_id')}"
    if t7_verdict == "PRIMARY_PROMOTED":
        base = {
            "source": "task-7",
            "candidate_id": t7_rec.get("promoted_candidate_id") or t7_rec.get("candidate_id"),
            "evidence": str(TASK7_EVIDENCE),
        }
        return True, base, f"task-7 verdict={t7_verdict} → base {base.get('candidate_id')}"
    return False, {}, (
        f"task-3 verdict={t3_verdict} (필요 PRIMARY_PASS) | task-7 verdict={t7_verdict} "
        f"(필요 PRIMARY_PROMOTED) → base 후보 없음 → SKIPPED_NO_BASE")


# ════════════════════════════════════════════════════════════════════
# 전체 경로 (미래 base 용 — 이 태스크에서는 실행되지 않음; 순수 함수는 단위 테스트 검증)
# ════════════════════════════════════════════════════════════════════
def run_full_calibration(train: pd.DataFrame, base: JsonDict,
                         panel_dir: Path) -> JsonDict:
    """base 가 존재할 때의 전체 인과 캘리브레이션 (primary fold 게이팅).

    panel_dir: base 후보의 one-year-ahead OOF 패널 아티팩트
      (<panel_dir>/year_<t>.npy + meta.json) — Task 9/10 replay 메커니즘이 생산하는
      문서화된 계약. all-row 2021-2024, R-fold 아티팩트 절대 미사용.
    이 태스크(SKIPPED_NO_BASE)에서는 절대 호출되지 않는다."""
    from repro_979.qualification_runner import build_folds, _check_leakage  # noqa: PLC0415

    meta_path = panel_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"[FAIL] base 캘리브레이션 패널 메타 없음: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("base_candidate_id") != base.get("candidate_id"):
        raise PolicyViolation(
            f"[POLICY] 패널 base {meta.get('base_candidate_id')} != 후보 {base.get('candidate_id')} "
            f"(exit 2)")
    per_year: dict[int, JsonDict] = {}
    for year in PANEL_YEARS:
        path = panel_dir / f"year_{year}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"[FAIL] 패널 연도 {year} 아티팩트 없음: {path}")
        arr = np.load(path, allow_pickle=True).item()
        per_year[year] = arr
    panel = build_causal_panel(per_year)

    # 스코어 연도 행 = primary fold val (2024 all rows) — 마스크만 사용 (라벨 가드 경유)
    folds = build_folds(train)
    leak_problems = _check_leakage(folds, train)
    if leak_problems:
        raise PolicyViolation("[POLICY] 폴드 누수 가드 실패 (exit 2): " + "; ".join(leak_problems))
    scored_mask = folds["primary"][1]
    scored_row_ids = np.asarray(train.loc[scored_mask, "row_id"].values, dtype=np.int64)
    _assert_no_same_year_residual(
        np.asarray(panel["row_ids"][panel["year"] < SCORED_YEAR], dtype=np.int64),
        scored_row_ids)

    scored_idx = np.where(panel["year"] == SCORED_YEAR)[0]
    z_scored = panel["z"][scored_idx]
    y_scored = panel["y"][scored_idx]
    # primary fold 라벨만 (구조 가드) — R-fold 라벨 절대 미로드
    labels = {"primary": y_scored}
    _assert_primary_only_labels(labels)
    _read_label_fold(labels, "primary")

    scored_keys = {
        "game_type": panel["game_type"][scored_idx],
        "count_state": panel["count_state"][scored_idx],
    }
    cluster_ids = panel["cluster_id"][scored_idx]
    base_bss, base_p_mean = score_base(z_scored, y_scored)

    best, records = evaluate_calibration_grid(panel, z_scored, y_scored, scored_keys)
    key_arrays = _key_arrays_for(scored_keys, best["configuration"])
    delta = apply_delta(z_scored, best["delta_terms"], key_arrays)
    lb5 = paired_bootstrap_lb5(z_scored, delta, y_scored, cluster_ids)
    verdict, gate = gate_verdict(best, base_bss, base_p_mean, lb5)

    # 배포(2025) 리핏 스냅샷 — through-2024 패널에서 선택 구성 재피팅 (동결 산식용)
    deploy_terms, deploy_diags = fit_configuration_for_year(
        panel, DEPLOY_REFIT_YEAR, best["configuration"],
        best["lambda"], best["k"])

    return {
        "base": base,
        "panel": {
            "years": sorted({int(v) for v in panel["year"]}),
            "n_rows_per_year": {int(y): int((panel["year"] == y).sum()) for y in PANEL_YEARS},
            "n_scored_rows": int(len(scored_idx)),
        },
        "best": {k: best[k] for k in ("lambda", "k", "configuration", "bss", "p_mean",
                                      "max_abs_delta")},
        "grid_records": [
            {k: r[k] for k in ("li", "ki", "ci", "lambda", "k", "configuration", "bss")}
            for r in records
        ],
        "selection_rule": ("max primary BSS; ties → lowest (lambda, k, configuration) "
                           "lexicographically"),
        "gate": gate,
        "verdict": verdict,
        "deployment_refit": {
            "outer_year": DEPLOY_REFIT_YEAR,
            "fit_years": "through-2024 panel (all years < 2025)",
            "delta_terms": deploy_terms,
            "fit_diags": deploy_diags,
        },
        "r_fold_embargo": {
            "loaded_folds": ["primary"],
            "note": "R-fold 아티팩트/라벨 절대 미사용 — 패널은 one-year-ahead 재실행 산출물",
        },
    }


# ════════════════════════════════════════════════════════════════════
# 정책 체크 + 증거 작성
# ════════════════════════════════════════════════════════════════════
def _check_policy() -> tuple[JsonDict, list[JsonDict], str]:
    policy = load_json(POLICY_PATH)
    if policy is None:
        raise PolicyViolation(f"[POLICY] 정책 JSON 읽기 불가: {POLICY_PATH}")
    violations = validate_policy(policy)
    return policy, violations, _canonical_sha256(policy)


def _frozen_spec() -> JsonDict:
    """동결 캘리브레이션 스펙 서술 (미래 base 용 결정 완전성 — SKIPPED_NO_BASE 증거에도 기록)."""
    return {
        "configurations": list(CONFIGURATIONS),
        "group_keys": ("raw frozen-MLP categorical strings (game_type: 'R'/'F'; "
                       "count_state: str(balls_before*3 + strikes_before)); unseen keys → delta 0"),
        "objective": ("mean BCE + lambda*sum(beta_g^2), subject to weighted primary-population "
                      "sum(n_g*beta_g)=0, deterministic LBFGS tolerance 1e-8"),
        "lambda_grid": list(LAMBDA_GRID),
        "k_grid": list(K_GRID),
        "order_of_operations": [
            "constrained zero-sum raw beta",
            "shrink beta_g *= n_g/(n_g+k)",
            "clip each term to [-0.05, 0.05]",
            "sum additive terms",
            "clip total delta to [-0.05, 0.05]",
            "p = sigmoid(z_base + C_LOGIT + delta)",
        ],
        "term_cap": TERM_CAP,
        "total_cap": TOTAL_CAP,
        "slope_frozen": SLOPE_FROZEN,
        "f_guard": ("game-type F coefficient fit ONLY from season >= 2023 OOF rows (rate flip "
                    "0.673 pre-2023 -> 0.470/0.474 in 2023/2024; plan 'do not fit game-type F "
                    "until pre-2023 OOF exists' read together with ideation line 215); F delta 0 "
                    "for outer years whose fitting panel has no 2023+ OOF"),
        "causal_panel": ("all-row 2021-2024 one-year-ahead OOF; outer year Y fits only from panel "
                         "years < Y; 2021 cold-start delta 0; 2025 deployment refit uses "
                         "through-2024 OOF"),
        "scored_year": SCORED_YEAR,
        "scored_fold": "primary (PRIMARY fold only for this round's gating)",
        "selection_rule": ("max primary BSS; ties → lowest (lambda, k, configuration) "
                           "lexicographically"),
        "advance_gate": {
            "primary_delta_bss_min": GATE_DELTA_BSS_MIN,
            "paired_bootstrap_lb5_min": GATE_LB5_MIN,
            "primary_mean_shift_max": GATE_MEAN_SHIFT_MAX,
            "definition": ("advance only if primary ΔBSS > 1 AND defined paired bootstrap LB5 > 0 "
                           "AND primary mean shift <= 0.005"),
        },
        "bootstrap": {
            "n_resamples": BOOT_N_RESAMPLES,
            "seed": BOOT_SEED,
            "cluster_definition": ("game-cluster = (season, game_month, game_dayofweek, "
                                   "pitcher_team_id, batter_team_id) — no game_id column"),
            "definition": "empirical 5th percentile of ΔBSS from 10,000 paired game-cluster resamples",
        },
        "c_logit_frozen": C_LOGIT,
        "scoring_formula": "common.score(clip(sigmoid(z_base + C_LOGIT + delta), 0.30, 0.70), y)",
        "r_fold_embargo": ("r2022/r2023/r2024 never loaded before Task 10 — structural guards "
                           "(_read_label_fold/_assert_primary_only_labels) + --fixture "
                           "attempted-r-fold-read exits 2"),
    }


class _Tee:
    def __init__(self, path: Path):
        self.fh = path.open("w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s: str):
        self.out.write(s)
        self.fh.write(s)

    def flush(self):
        self.out.flush()
        self.fh.flush()


def _write_md(record: JsonDict, path: Path) -> Path:
    checks = record.get("checks", [])
    lines = [
        f"# {record['title']} — {record['verdict']} (exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
        f"- **config_hash**: `{record['config_hash']}`",
        f"- **policy_config_hash**: `{record['policy_config_hash']}`",
        f"- **label_sources**: {record['label_sources']} (labels_read={record['labels_read']})",
        "",
        "## Base eligibility",
        "",
        f"- **task-3** (`{Path(str(record.get('base_eligibility', {}).get('task3_file', ''))).name}`): "
        f"verdict `{record.get('base_eligibility', {}).get('task3_verdict')}` "
        f"(required `PRIMARY_PASS`)",
        f"- **task-7** (`{Path(str(record.get('base_eligibility', {}).get('task7_file', ''))).name}`): "
        f"verdict `{record.get('base_eligibility', {}).get('task7_verdict')}` "
        f"(required `PRIMARY_PROMOTED`)",
        f"- **eligible**: {record.get('base_eligibility', {}).get('eligible')}",
        f"- **reason**: {record.get('base_eligibility', {}).get('reason')}",
        "",
        "## Frozen calibration spec (decision-complete for a future base)",
        "",
    ]
    spec = record.get("frozen_spec", {})
    for k, v in spec.items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Checks",
        "",
    ]
    for c in checks:
        mark = "PASS" if c.get("ok") else ("FAIL" if c.get("ok") is False else "INFO")
        lines.append(f"- **[{mark}]** {c.get('rule')}: {c.get('reason', '')}")
    lines += ["", f"## Verdict: **{record['verdict']}**", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════
# 실패 주입 fixture — 전부 exit 2 (구조 가드 증명, 데이터/라벨 로드 없음)
# ════════════════════════════════════════════════════════════════════
def run_fixture(fixture: str) -> int:
    if fixture == "same-year-residual":
        print("\n[fixture same-year-residual] 스코어 연도 잔차 피팅 주입 — 누수 가드 검증…",
              flush=True)
        try:
            _assert_no_same_year_residual(np.array([1, 2, 3]), np.array([3, 4]))
        except LeakageError as exc:
            print(f"[fixture same-year-residual] PASS — 누수 가드가 동년 잔차 피팅을 "
                  f"차단했습니다: {exc}", flush=True)
            return 2
        print("[fixture same-year-residual] FAIL — LeakageError 가 발생하지 않았습니다",
              file=sys.stderr)
        return 1
    if fixture == "nonunit-slope":
        print("\n[fixture nonunit-slope] 비단위 slope 피팅 주입 — slope 동결 가드 검증…",
              flush=True)
        try:
            _assert_unit_slope(0.98)
        except PolicyViolation as exc:
            print(f"[fixture nonunit-slope] PASS — slope 동결 가드가 a=0.98 을 "
                  f"차단했습니다: {exc}", flush=True)
            return 2
        print("[fixture nonunit-slope] FAIL — PolicyViolation 이 발생하지 않았습니다",
              file=sys.stderr)
        return 1
    if fixture == "cap-violation":
        print("\n[fixture cap-violation] 캡 위반 delta 주입 — 캡 가드 검증…", flush=True)
        try:
            _assert_cap_respected([0.06], cap=TERM_CAP)
        except PolicyViolation as exc:
            print(f"[fixture cap-violation] PASS — 캡 가드가 |delta|=0.06 > 0.05 를 "
                  f"차단했습니다: {exc}", flush=True)
            return 2
        print("[fixture cap-violation] FAIL — PolicyViolation 이 발생하지 않았습니다",
              file=sys.stderr)
        return 1
    if fixture == "future-year-coefficient":
        print("\n[fixture future-year-coefficient] future-year 계수 피팅 주입 — 연도 경계 가드 "
              "검증…", flush=True)
        try:
            _assert_no_future_fit([2021, 2022, 2024], outer_year=2024)
        except PolicyViolation as exc:
            print(f"[fixture future-year-coefficient] PASS — 연도 경계 가드가 피팅 연도 2024 "
                  f"(>= outer_year 2024) 를 차단했습니다: {exc}", flush=True)
            return 2
        print("[fixture future-year-coefficient] FAIL — PolicyViolation 이 발생하지 않았습니다",
              file=sys.stderr)
        return 1
    if fixture == "attempted-r-fold-read":
        print("\n[fixture attempted-r-fold-read] R-only 폴드 라벨 읽기 주입 (r2022)…", flush=True)
        try:
            _read_label_fold({"primary": np.zeros(1, dtype=np.float64)}, "r2022")
        except LeakageError as exc:
            print(f"[fixture attempted-r-fold-read] PASS — 누수 가드가 r2022 읽기를 "
                  f"차단했습니다: {exc}", flush=True)
            return 2
        print("[fixture attempted-r-fold-read] FAIL — LeakageError 가 발생하지 않았습니다",
              file=sys.stderr)
        return 1
    print(f"[FAIL] 미등록 fixture: {fixture!r} — {FIXTURES}", file=sys.stderr)
    return 1


# ════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 8: bounded train-only subgroup-intercept calibration "
                    "(--primary-causal → PASS | REJECT | SKIPPED_NO_BASE)")
    parser.add_argument("--primary-causal", action="store_true",
                        help="primary causal OOF 캘리브레이션 게이팅 (기본 모드)")
    parser.add_argument("--fixture", choices=list(FIXTURES), default=None,
                        help="실패 QA (전부 exit 2): same-year-residual / nonunit-slope / "
                             "cap-violation / future-year-coefficient / attempted-r-fold-read")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 task-8-calibration.json)")
    parser.add_argument("--log", default=None, help="증거 로그 경로 (기본 task-8-calibration.log)")
    parser.add_argument("--md", default=None, help="증거 MD 경로 (기본 task-8-calibration.md)")
    args = parser.parse_args(argv)

    if args.fixture:
        base_name = f"task-8-calibration-fixture-{args.fixture}"
    else:
        base_name = "task-8-calibration"
    evidence_path = (Path(args.evidence).expanduser().resolve() if args.evidence
                     else EVIDENCE_DIR / f"{base_name}.json")
    log_path = (Path(args.log).expanduser().resolve() if args.log
                else EVIDENCE_DIR / f"{base_name}.log")
    md_path = (Path(args.md).expanduser().resolve() if args.md
               else EVIDENCE_DIR / f"{base_name}.md")
    config_path = EVIDENCE_DIR / ("task-8-calibration-config.json" if not args.fixture
                                  else f"task-8-calibration-config-{args.fixture}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    t0 = time.time()
    print(f"[subgroup_calibration_runner] Todo 8 — bounded subgroup-intercept calibration "
          f"(mode=primary-causal, fixture={args.fixture})", flush=True)

    # ── 실패 주입: 데이터/라벨 로드 전 차단 (exit 2) ──
    if args.fixture:
        return run_fixture(args.fixture)

    # ── 1) 정책 검증 (읽기 전용) ──
    try:
        policy, policy_violations, policy_hash = _check_policy()
    except PolicyViolation as exc:
        print(f"[FAIL] 정책 가드: {exc}", file=sys.stderr)
        return 2
    if policy_violations:
        print("[FAIL] 정책 위반:\n  " + "\n  ".join(v["reason"] for v in policy_violations),
              file=sys.stderr)
        return 2
    print(f"[policy] PASS — label_source={policy['selection']['label_source']} "
          f"sort_keys={policy['selection']['sort_keys']} policy_hash={policy_hash[:16]}…",
          flush=True)

    # ── 2) 설정 사전 등록 — base 자격/라벨 결과를 읽기 **이전**에 기록 ──
    config = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-8-calibration-config",
        "title": "Todo 8 — pre-registered bounded subgroup-intercept calibration config",
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "selection_fold": "primary",
        "label_sources": [],
        "labels_read": False,
        "labels_read_note": ("Config is written BEFORE any base-eligibility/label read "
                             "(pre-registration contract). SKIPPED_NO_BASE branch reads no "
                             "calibration labels at all."),
        "mode": "primary-causal",
        "frozen_controls": {
            "c_logit": C_LOGIT, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
            "slope_frozen": SLOPE_FROZEN,
            "rollback_baseline_candidate_id": policy.get("rollback_baseline_candidate_id"),
        },
        "calibration_spec": _frozen_spec(),
        "base_eligibility": {
            "rule": ("task-3 evidence verdict PRIMARY_PASS OR task-7 evidence verdict "
                     "PRIMARY_PROMOTED required; otherwise emit SKIPPED_NO_BASE without reading "
                     "calibration labels (label_sources=[], labels_read=false), exit 0"),
            "task3_evidence": str(TASK3_EVIDENCE),
            "task7_evidence": str(TASK7_EVIDENCE),
        },
        "fixtures": list(FIXTURES),
    }
    config_hash = _canonical_sha256(config)
    config["config_hash"] = config_hash
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[pre-register] config hash={config_hash[:16]}… → {config_path} "
          f"(라벨 읽기 이전 기록)", flush=True)

    # ── 3) base 후보 자격 판정 (결정 증거 JSON 만 — 라벨 미로드) ──
    eligible, base_info, reason = base_eligibility()
    t3_verdict, _ = _evidence_verdict(TASK3_EVIDENCE)
    t7_verdict, _ = _evidence_verdict(TASK7_EVIDENCE)
    print(f"[base-eligibility] task-3={t3_verdict} | task-7={t7_verdict} | eligible={eligible}",
          flush=True)
    print(f"[base-eligibility] reason: {reason}", flush=True)

    # ── 4) SKIPPED_NO_BASE 분기 (현재 실행 경로) ──
    if not eligible:
        checks = [
            {"rule": "policy_check", "ok": not policy_violations,
             "reason": f"{len(policy_violations)} violations — label_source=primary (동결 정책)"},
            {"rule": "pre_registration", "ok": True,
             "reason": f"config hash={config_hash[:16]}… written BEFORE base-eligibility/label "
                       f"read ({config_path.name})"},
            {"rule": "base_eligibility_task3", "ok": t3_verdict == "PRIMARY_PASS",
             "reason": f"task-3 verdict={t3_verdict} (PRIMARY_PASS 필요) — base 자격 없음"},
            {"rule": "base_eligibility_task7", "ok": t7_verdict == "PRIMARY_PROMOTED",
             "reason": f"task-7 verdict={t7_verdict} (PRIMARY_PROMOTED 필요) — base 자격 없음"},
            {"rule": "no_base_detected", "ok": not eligible,
             "reason": reason},
            {"rule": "label_free_skip", "ok": True,
             "reason": "SKIPPED_NO_BASE: 캘리브레이션 라벨 미로드 (label_sources=[], "
                       "labels_read=false) — train/라벨 로드 없이 종료"},
            {"rule": "r_fold_embargo", "ok": True,
             "reason": "R-only 폴드(r2022/r2023/r2024) 라벨/아티팩트 미로드 — 구조 가드 "
                       "(_read_label_fold/_assert_primary_only_labels) + --fixture "
                       "attempted-r-fold-read exit 2"},
            {"rule": "frozen_spec_complete", "ok": True,
             "reason": "동결 캘리브레이션 스펙(구성 3종/그리드/연산 순서/F 가드/게이트) 전체 "
                       "구현 — 미래 base 용 결정 완전 (frozen_spec 참조)"},
        ]
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 8 — bounded train-only subgroup-intercept calibration",
            "task": "aimers9-next-round/task-8-calibration",
            "mode": "primary-causal",
            "verdict": "SKIPPED_NO_BASE",
            "exit_code": 0,
            "recorded_at_utc": _now_utc(),
            "git_head": _git_commit(),
            "config_hash": config_hash,
            "config_pre_registered": {
                "file": str(config_path), "sha256": _file_sha256(config_path),
                "written_before_labels_read": True,
                "labels_read_at_registration": False,
            },
            "policy_path": str(POLICY_PATH),
            "policy_config_hash": policy_hash,
            "label_sources": [],
            "labels_read": False,
            "labels_read_note": ("Neither Task 3 (PRIMARY_PASS) nor Task 7 (PRIMARY_PROMOTED) "
                                 "provided a passing base candidate → SKIPPED_NO_BASE without "
                                 "reading any calibration labels."),
            "reason": ("No Task-3 (PRIMARY_PASS) or Task-7 (PRIMARY_PROMOTED) base candidate "
                       "exists — " + reason),
            "base_eligibility": {
                "task3_file": str(TASK3_EVIDENCE), "task3_verdict": t3_verdict,
                "task7_file": str(TASK7_EVIDENCE), "task7_verdict": t7_verdict,
                "eligible": False,
                "reason": reason,
            },
            "frozen_spec": _frozen_spec(),
            "r_fold_embargo": {
                "loaded_folds": [],
                "guards": ["_read_label_fold", "_assert_primary_only_labels",
                           "fixture attempted-r-fold-read → exit 2"],
                "note": "Task 10 전 R-only 폴드는 reject-only — SKIPPED_NO_BASE 경로는 아무 "
                        "폴드도 로드하지 않음",
            },
            "checks": checks,
            "violations": [],
            "environment": _environment(),
            "total_time_s": time.time() - t0,
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        _write_md(record, md_path)
        print(f"\n[subgroup_calibration_runner] 증거 JSON → {evidence_path}", flush=True)
        print(f"[subgroup_calibration_runner] 증거 MD   → {md_path}", flush=True)
        print(f"[subgroup_calibration_runner] 총 {time.time() - t0:.0f}s", flush=True)
        print(f"\n[--primary-causal] SKIPPED_NO_BASE — Task 3/7 통과 base 없음 → "
              f"캘리브레이션 라벨 미로드, exit 0", flush=True)
        return 0

    # ── 5) 전체 경로 (미래 base 용 — 지금은 미도달) ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    panel_dir = REPO / "cache" / "calibration" / "next-round" / str(base_info["candidate_id"])
    print(f"[full-path] base {base_info.get('candidate_id')} — 캘리브레이션 실행 시작 "
          f"(panel_dir={panel_dir})", flush=True)
    result = run_full_calibration(train, base_info, panel_dir)
    verdict = result["verdict"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 8 — bounded train-only subgroup-intercept calibration",
        "task": "aimers9-next-round/task-8-calibration",
        "mode": "primary-causal",
        "verdict": verdict,
        "exit_code": 0,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": config_hash,
        "config_pre_registered": {
            "file": str(config_path), "sha256": _file_sha256(config_path),
            "written_before_labels_read": True,
            "labels_read_at_registration": False,
        },
        "policy_path": str(POLICY_PATH),
        "policy_config_hash": policy_hash,
        "label_sources": ["primary"],
        "labels_read": True,
        "base_eligibility": {
            "task3_file": str(TASK3_EVIDENCE), "task3_verdict": t3_verdict,
            "task7_file": str(TASK7_EVIDENCE), "task7_verdict": t7_verdict,
            "eligible": True, "base": base_info, "reason": reason,
        },
        "frozen_spec": _frozen_spec(),
        "calibration": result,
        "checks": [
            {"rule": "policy_check", "ok": not policy_violations, "reason": "0 violations"},
            {"rule": "primary_only_labels", "ok": True,
             "reason": "labels=[primary] — R-only 라벨 미로드 (구조 가드)"},
            {"rule": "gate_verdict_rule", "ok": verdict in ("PASS", "REJECT"),
             "reason": json.dumps(result["gate"], ensure_ascii=False)},
        ],
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    _write_md(record, md_path)
    print(f"\n[subgroup_calibration_runner] 증거 JSON → {evidence_path}", flush=True)
    print(f"[subgroup_calibration_runner] 총 {time.time() - t0:.0f}s", flush=True)
    print(f"\n[--primary-causal] {verdict} — ΔBSS={result['gate']['delta_bss']:+.4f} | "
          f"LB5={result['gate']['bootstrap_lb5']:.4f} | mean_shift="
          f"{result['gate']['mean_shift']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
