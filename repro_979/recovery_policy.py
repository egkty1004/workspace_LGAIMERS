#!/usr/bin/env python3
"""recovery_policy.py — Todo 2 (aimers9-top100-score-recovery): immutable rolling-origin
recovery policy + terminal-label firewall.

This module is the single source of truth for the pre-registered, immutable policy
that the recovery evaluator (recovery_evaluator.py) enforces. It defines:

  - Chronological selection origins (R-only) r2022 / r2023, the historically reused
    terminal/stress origin `primary` (all 2024), and the diagnostic origin `r2024`
    (overlaps primary; branch-inert until Task 8).
  - The immutable deployed-formula scoring
        common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y)
    with the reconciled v93 6-leg baseline as the comparison baseline
    (C_LOGIT = -0.0461645795229729, clip [.30, .70]).
  - The paired row-bootstrap definition and the candidate screen gate.
  - The terminal-label firewall: primary labels are structurally unreadable until
    the firewall state is FROZEN (set only by --freeze in recovery_evaluator.py).

Structural firewall contract (core Task 2 requirement):
  - Candidate-selection code (the --screen path) reads ONLY r2022/r2023 labels via
    read_selection_labels(). It never imports or calls read_primary_labels().
  - read_primary_labels() raises TerminalFirewallError unless the module-level
    firewall state is FROZEN. The --screen path therefore cannot read primary
    labels even if it tried — the guard is structural, not just a convention.
  - r2024 may be reported only after Task 8 (diagnostic), never as a sort key and
    never to decide a branch.

Shared utilities reused by the evaluator and its audits (canonical sha256,
evidence-name regex, forbidden-path-token + upload-marker scans) live here so the
evaluator and the F1/F2/F4 audits agree on one definition.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.common / repro_979.* 패키지 import 용

JSON = dict[str, Any]

# ── 상수 ──────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1

# ── 동결 배포 산식 (불변) ─────────────────────────────────────────────
# 계획 검증 전략: common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y).
# Task 1 이 조정한 reconciled v93 6-leg baseline 이 비교 기준선이다
# (C_LOGIT=-0.0461645795229729, clip [.30,.70]) — rollback .30/.35/.35 계약이 아니다.
C_LOGIT = -0.0461645795229729
CLIP_LO, CLIP_HI = 0.30, 0.70

# ── 기원 (chronological rolling origins) ──────────────────────────────
# r2022=(train<=2021,R)->(2022,R); r2023=(train<=2022,R)->(2023,R);
# primary=(train<=2023,all)->(2024,all); r2024=(train<=2023,R)->(2024,R) [진단 전용].
SELECTION_ORIGINS = ("r2022", "r2023")     # 후보 선택이 읽을 수 있는 유일한 라벨 기원
TERMINAL_ORIGIN = "primary"                # Task 8 단일 스트레스 체크 (동결 후에만)
DIAGNOSTIC_ORIGIN = "r2024"                # Task 8 이후 진단 전용, branch-inert
ALL_ORIGINS = ("r2022", "r2023", "primary", "r2024")
R_ORIGINS = ("r2022", "r2023", "r2024")

# ── 선택 키 (정렬/선택에 사용 가능한 유일한 키) ────────────────────────
# 선택은 mean(r2022,r2023) ΔBSS 단일 키로만. r2024/Public 값은 정렬 금지.
SELECTION_KEYS = ("mean_selection_delta_bss",)
FORBIDDEN_SORT_KEYS = ("r2024", "public", "leaderboard", "boot_lb5")

# ── 후보 스크린 게이트 (계획 검증 전략) ────────────────────────────────
GATE_DELTA_BSS_MIN = 1.0        # 각 선택 기원에서 ΔBSS > 1.0
GATE_BRIER_REDUCE = True        # 각 선택 기원에서 Brier 감소
GATE_BOOTSTRAP_LB5_MIN = 0.0    # 각 기원 paired row-bootstrap LB5(ΔBSS) > 0
GATE_MEAN_SHIFT_MAX = 0.005     # max|Δmean| <= 0.005 (deployed clipped 확률)
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 20260817  # rng = default_rng(BOOTSTRAP_SEED_BASE + outer_year)

# ── 터미널 스트레스 게이트 (Task 8 단일 2024 체크) ─────────────────────
# 계획 검증 전략: Task 8 은 primary ΔBSS > 3.0, primary paired row-bootstrap
# LB5 > 0, mean shift <= .005 일 때만 STATISTICAL_PASS; 그 외 NO_PROMOTION.
TERMINAL_DELTA_BSS_MIN = 3.0        # primary ΔBSS > 3.0
TERMINAL_BOOTSTRAP_LB5_MIN = 0.0    # primary paired row-bootstrap LB5(ΔBSS) > 0
TERMINAL_MEAN_SHIFT_MAX = 0.005     # max|Δmean| <= 0.005 (deployed clipped 확률)
TERMINAL_OUTER_YEAR = 2024          # primary = (train<=2023, all) -> (2024, all)

# ── 터미널 라벨 파이어월 상태 ─────────────────────────────────────────
# UNFROZEN: primary 라벨 읽기 차단 (구조적). FROZEN: --freeze 가 후보를 동결한 후에만.
FIREWALL_UNFROZEN = "UNFROZEN"
FIREWALL_FROZEN = "FROZEN"
_firewall_state = FIREWALL_UNFROZEN

# ── 증거/범위 상수 ────────────────────────────────────────────────────
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100-recovery"
EVIDENCE_NAME_RE = re.compile(r"^(?:task-\d+-[a-z0-9_-]+|f[1-4]-[a-z0-9_-]+)\.json$")
FORBIDDEN_PATH_TOKENS = ("데이터/", "/model/", "/cache/", ".zip", "submit_sim", "submit_sim_")
UPLOAD_KEY_NORMS = {"upload", "uploaded", "daconupload", "daconsubmission",
                    "submitted", "uploadattempt", "uploadmarker"}

# Task 2-10 증거 아티팩트 패턴 (baseline-block 감사에서 부재해야 함).
TASKS_2_10_RE = re.compile(r"^task-(?:[2-9]|10)-[a-z0-9_-]+\.(?:json|md)$")

# 코드 해시 대상 — 평가기/정책/공용 스코어링 모듈 (라벨 무관, 불변성 증명용).
CODE_HASH_FILES = ("recovery_policy.py", "recovery_evaluator.py", "common.py")


class TerminalFirewallError(RuntimeError):
    """동결 전에 primary 라벨을 읽으려 하면 발생 (구조적 파이어월, exit 2)."""


class PolicyViolation(RuntimeError):
    """정책 위반 (r2024/Public 정렬 키 등) — exit 2."""


class ManifestError(RuntimeError):
    """매니페스트 해시/구조 불일치 — exit 2."""


# ── 유틸 ──────────────────────────────────────────────────────────────
def _norm_key(k: str) -> str:
    return re.sub(r"[^0-9a-z]", "", k.lower())


def _canonical_sha256(payload: JSON) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> JSON | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iter_strs(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_iter_strs(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_strs(v))
    return out


def _iter_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_dicts(v)


def scan_scope(record: JSON, path: Path) -> list[str]:
    """증거 기록 범위 스캔: 금지 경로 토큰 + 업로드 마커 (F4 scope audit 재사용)."""
    problems: list[str] = []
    for s in _iter_strs(record):
        for tok in FORBIDDEN_PATH_TOKENS:
            if tok in s:
                problems.append(f"forbidden path token {tok!r} in {path.name}")
                break
    for key in _iter_strs(record):
        if _norm_key(key) in UPLOAD_KEY_NORMS:
            problems.append(f"upload marker key {key!r} in {path.name}")
            break
    return problems


# ── 기원 마스크 ───────────────────────────────────────────────────────
def build_origin_masks(train) -> dict[str, tuple[Any, Any]]:
    """시계열 기원 마스크 — qualification_runner.build_folds 와 동일한 정의.

    반환: origin -> (train_mask, val_mask). 마스크만 계산 — 라벨은 읽지 않는다.
    """
    is_r = train["game_type"] == "R"
    return {
        "r2022": ((train["season"] <= 2021) & is_r, (train["season"] == 2022) & is_r),
        "r2023": ((train["season"] <= 2022) & is_r, (train["season"] == 2023) & is_r),
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2024": ((train["season"] <= 2023) & is_r, (train["season"] == 2024) & is_r),
    }


def check_leakage(masks: dict[str, tuple[Any, Any]], train) -> list[str]:
    """누수 가드: 검증 마스크에 2025 시즌 행이 절대 없어야 한다."""
    problems: list[str] = []
    for fn, (_tr, va) in masks.items():
        if int(va.sum()) == 0:
            problems.append(f"{fn}: 검증 마스크가 비어 있음")
        if (train.loc[va, "season"] == 2025).any():
            problems.append(f"{fn}: 검증 마스크에 2025 시즌 포함 — 누수!")
    return problems


def assert_row_disjointness(masks: dict[str, tuple[Any, Any]], train) -> dict[str, JSON]:
    """row-ID 겹침 단언 기록.

    r2022/r2023 은 primary 와 row-ID 완전 분리 (하드 assert).
    r2024 는 계획상 overlapping 진단 폴드 — primary(2024 전체)에 포함되는 2024-R 행으로
    설계됨 (branch-inert). 겹침은 문서화한다.
    """
    va_primary = masks["primary"][1]
    overlap_note: dict[str, JSON] = {}
    for rfold in R_ORIGINS:
        overlap = int((va_primary & masks[rfold][1]).sum())
        if rfold == "r2024":
            overlap_note[rfold] = {
                "overlap_rows_with_primary": overlap, "expected": True,
                "reason": "r2024 val = (season==2024) & R subset of primary val "
                          "(overlapping diagnostic origin, branch-inert)"}
        elif overlap != 0:
            raise PolicyViolation(
                f"[POLICY] row-ID disjointness 위반: primary ∩ {rfold} ≠ ∅ "
                f"({overlap} rows)")
        else:
            overlap_note[rfold] = {"overlap_rows_with_primary": 0, "expected": True,
                                   "reason": "row-ID disjoint from primary"}
    return overlap_note


# ── 라벨 읽기 (구조적 파이어월) ────────────────────────────────────────
def read_selection_labels(train, masks: dict[str, tuple[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
    """선택 기원(r2022/r2023) 라벨만 읽는다 — --screen 경로가 사용하는 유일한 라벨 접근.

    primary/r2024 라벨은 절대 여기서 읽지 않는다 (구조적 파이어월).
    """
    labels: dict[str, np.ndarray[Any, Any]] = {}
    for origin in SELECTION_ORIGINS:
        va = masks[origin][1]
        labels[origin] = train.loc[va, "control_success"].values.astype(np.float64)
    return labels


def read_primary_labels(train, masks: dict[str, tuple[Any, Any]]) -> np.ndarray[Any, Any]:
    """primary 라벨 읽기 — 동결(FROZEN) 전에는 구조적으로 불가능.

    --screen 경로는 이 함수를 호출하지 않는다. --terminal-check 는 --freeze 가
    파이어월을 FROZEN 으로 바꾼 뒤에만 호출할 수 있다.
    """
    if _firewall_state != FIREWALL_FROZEN:
        raise TerminalFirewallError(
            f"[FIREWALL] primary 라벨은 동결 전에 읽을 수 없습니다 "
            f"(firewall_state={_firewall_state!r}, 필요 {FIREWALL_FROZEN!r}) — exit 2")
    va = masks[TERMINAL_ORIGIN][1]
    return train.loc[va, "control_success"].values.astype(np.float64)


def set_firewall(state: str) -> None:
    """파이어월 상태 설정 (--freeze 가 FROZEN 으로, 테스트가 리셋)."""
    global _firewall_state
    if state not in (FIREWALL_UNFROZEN, FIREWALL_FROZEN):
        raise ValueError(f"알 수 없는 파이어월 상태: {state!r}")
    _firewall_state = state


def firewall_state() -> str:
    return _firewall_state


# ── 배포 산식 스코어링 ────────────────────────────────────────────────
def deployed_score(z: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """불변 배포 산식: common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y)."""
    import repro_979.common as common  # noqa: PLC0415  (동일 스코어링)
    p = np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)
    return float(common.score(p, y))


def deployed_probs(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """배포 클리핑 확률 (부트스트랩/mean-shift 진단용)."""
    import repro_979.common as common  # noqa: PLC0415
    return np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)


def raw_brier(z: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """raw-logit Brier 진단 (선택 산식 아님 — 선택은 deployed_score 만)."""
    import repro_979.common as common  # noqa: PLC0415
    return float(np.mean((common.sigmoid(z) - y) ** 2))


# ── paired row-bootstrap (계획 검증 전략) ─────────────────────────────
def paired_row_bootstrap(cand_p: np.ndarray[Any, Any], base_p: np.ndarray[Any, Any],
                         y: np.ndarray[Any, Any],
                         outer_year: int, n_resamples: int = BOOTSTRAP_N_RESAMPLES,
                         seed_base: int = BOOTSTRAP_SEED_BASE) -> float:
    """각 기원의 paired row-bootstrap LB5(ΔBSS).

    정확히 n 개 행 인덱스를 복원 추출하고, baseline/candidate 의 deployed clipped
    확률에 동일 인덱스를 재사용하며, 단일 클래스 리샘플은 버리고 다시 뽑는다.
    common.score(candidate*) - common.score(baseline*) 의 경험적 5번째 백분위를 반환.
    """
    import repro_979.common as common  # noqa: PLC0415
    rng = np.random.default_rng(seed_base + outer_year)
    n = int(len(y))
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        while len(np.unique(yb)) < 2:  # 단일 클래스 리샘플 → 버리고 재추출
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
        deltas[i] = common.score(cand_p[idx], yb) - common.score(base_p[idx], yb)
    deltas.sort()
    return float(np.percentile(deltas, 5))


# ── 후보 스크린 게이트 ────────────────────────────────────────────────
def screen_gate(cand: dict[str, np.ndarray[Any, Any]], base: dict[str, np.ndarray[Any, Any]],
                y: dict[str, np.ndarray[Any, Any]]) -> JSON:
    """후보 스크린 게이트 — 각 선택 기원에서 계획 검증 전략의 모든 조건을 검사.

    cand/base: origin -> deployed clipped 확률 (배포 산식 적용 후).
    y: origin -> 라벨.
    반환: {origins: {...}, passed, verdict, violations}.
    """
    import repro_979.common as common  # noqa: PLC0415
    results: dict[str, JSON] = {}
    violations: list[str] = []
    for origin in SELECTION_ORIGINS:
        cp = np.asarray(cand[origin], dtype=np.float64)
        bp = np.asarray(base[origin], dtype=np.float64)
        yo = np.asarray(y[origin], dtype=np.float64)
        outer_year = int(origin.replace("r", ""))
        delta_bss = float(common.score(cp, yo) - common.score(bp, yo))
        brier_cand = float(np.mean((cp - yo) ** 2))
        brier_base = float(np.mean((bp - yo) ** 2))
        lb5 = paired_row_bootstrap(cp, bp, yo, outer_year)
        mean_shift = float(np.max(np.abs(cp.mean() - bp.mean())))
        finite = bool(np.all(np.isfinite(cp)))
        checks = {
            "delta_bss_gt_1": bool(delta_bss > GATE_DELTA_BSS_MIN),
            "brier_reduced": bool(brier_cand < brier_base),
            "bootstrap_lb5_gt_0": bool(lb5 > GATE_BOOTSTRAP_LB5_MIN),
            "mean_shift_le_005": bool(mean_shift <= GATE_MEAN_SHIFT_MAX),
            "finite": bool(finite),
        }
        if not all(checks.values()):
            violations.append(
                f"{origin}: ΔBSS={delta_bss:+.4f}(>{GATE_DELTA_BSS_MIN}?{checks['delta_bss_gt_1']}) "
                f"Brier {brier_cand:.6f}<{brier_base:.6f}?{checks['brier_reduced']} "
                f"LB5={lb5:+.4f}(>0?{checks['bootstrap_lb5_gt_0']}) "
                f"mean_shift={mean_shift:.6f}(<={GATE_MEAN_SHIFT_MAX}?{checks['mean_shift_le_005']}) "
                f"finite={finite}")
        results[origin] = {
            "delta_bss": delta_bss, "brier_candidate": brier_cand,
            "brier_baseline": brier_base, "bootstrap_lb5": lb5,
            "mean_shift": mean_shift, "finite": finite, "checks": checks,
        }
    passed = not violations
    return {
        "origins": results,
        "passed": passed,
        "verdict": "PASS" if passed else "REJECT",
        "violations": violations,
    }


# ── 터미널 스트레스 게이트 (Task 8) ───────────────────────────────────
def terminal_gate(cand_p: np.ndarray[Any, Any], base_p: np.ndarray[Any, Any],
                  y_primary: np.ndarray[Any, Any]) -> JSON:
    """2024 터미널 스트레스 게이트 — Task 8 단일 체크.

    cand_p/base_p: primary deployed clipped 확률 (배포 산식 적용 후).
    y_primary: primary 라벨 (파이어월 FROZEN 후에만 읽음).
    반환: {delta_bss, brier_candidate, brier_baseline, bootstrap_lb5, mean_shift,
           finite, passed, verdict, violations}.
    """
    import repro_979.common as common  # noqa: PLC0415
    cp = np.asarray(cand_p, dtype=np.float64)
    bp = np.asarray(base_p, dtype=np.float64)
    yo = np.asarray(y_primary, dtype=np.float64)
    delta_bss = float(common.score(cp, yo) - common.score(bp, yo))
    brier_cand = float(np.mean((cp - yo) ** 2))
    brier_base = float(np.mean((bp - yo) ** 2))
    lb5 = paired_row_bootstrap(cp, bp, yo, TERMINAL_OUTER_YEAR)
    mean_shift = float(np.max(np.abs(cp.mean() - bp.mean())))
    finite = bool(np.all(np.isfinite(cp)) and np.all(np.isfinite(bp)))
    checks = {
        "delta_bss_gt_3": bool(delta_bss > TERMINAL_DELTA_BSS_MIN),
        "bootstrap_lb5_gt_0": bool(lb5 > TERMINAL_BOOTSTRAP_LB5_MIN),
        "mean_shift_le_005": bool(mean_shift <= TERMINAL_MEAN_SHIFT_MAX),
        "finite": bool(finite),
    }
    violations: list[str] = []
    if not all(checks.values()):
        violations.append(
            f"primary: ΔBSS={delta_bss:+.4f}(>{TERMINAL_DELTA_BSS_MIN}?"
            f"{checks['delta_bss_gt_3']}) "
            f"LB5={lb5:+.4f}(>0?{checks['bootstrap_lb5_gt_0']}) "
            f"mean_shift={mean_shift:.6f}(<={TERMINAL_MEAN_SHIFT_MAX}?"
            f"{checks['mean_shift_le_005']}) finite={finite}")
    passed = not violations
    return {
        "delta_bss": delta_bss,
        "brier_candidate": brier_cand,
        "brier_baseline": brier_base,
        "bootstrap_lb5": lb5,
        "mean_shift": mean_shift,
        "finite": finite,
        "checks": checks,
        "passed": passed,
        "verdict": "PASS" if passed else "REJECT",
        "violations": violations,
    }


# ── 매니페스트 (불변) ─────────────────────────────────────────────────
def code_hashes() -> dict[str, str]:
    """코드 해시 — 평가기/정책/공용 스코어링 모듈 (라벨 무관)."""
    out: dict[str, str] = {}
    for name in CODE_HASH_FILES:
        path = REPO / name
        out[name] = _sha256_file(path) if path.is_file() else "missing"
    return out


def data_hash() -> str:
    """공식 학습 데이터 해시 (train.csv). 없으면 'unavailable'."""
    path = REPO / "open" / "data" / "train.csv"
    return _sha256_file(path) if path.is_file() else "unavailable"


def policy_config() -> JSON:
    """정책 구성의 정규화 표현 — config_hash 의 원천 (라벨 무관)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "c_logit": C_LOGIT,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "selection_origins": list(SELECTION_ORIGINS),
        "terminal_origin": TERMINAL_ORIGIN,
        "diagnostic_origin": DIAGNOSTIC_ORIGIN,
        "all_origins": list(ALL_ORIGINS),
        "selection_keys": list(SELECTION_KEYS),
        "forbidden_sort_keys": list(FORBIDDEN_SORT_KEYS),
        "gate": {
            "delta_bss_min": GATE_DELTA_BSS_MIN,
            "brier_reduce": GATE_BRIER_REDUCE,
            "bootstrap_lb5_min": GATE_BOOTSTRAP_LB5_MIN,
            "mean_shift_max": GATE_MEAN_SHIFT_MAX,
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
        },
        "firewall_unfrozen": FIREWALL_UNFROZEN,
        "firewall_frozen": FIREWALL_FROZEN,
    }


def policy_config_hash() -> str:
    return _canonical_sha256(policy_config())


def build_manifest(candidate_id: str, *, seeds: list[int] | None = None,
                   formula_note: str | None = None) -> JSON:
    """불변 매니페스트 구성 — 라벨 로드 이전에 해시/기원/산식/부트스트랩/파이어월 기록."""
    seeds = list(seeds) if seeds is not None else []
    manifest: JSON = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "code_hashes": code_hashes(),
        "config_hash": policy_config_hash(),
        "data_hash": data_hash(),
        "origins": {
            "selection": list(SELECTION_ORIGINS),
            "terminal": TERMINAL_ORIGIN,
            "diagnostic": DIAGNOSTIC_ORIGIN,
            "all": list(ALL_ORIGINS),
        },
        "seeds": seeds,
        "formula": {
            "scoring": "common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y)",
            "c_logit": C_LOGIT,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "baseline": "reconciled v93 6-leg contract (C_LOGIT=-0.0461645795229729)",
            "note": formula_note or "deployed-formula scoring is immutable; raw-logit "
                                     "Brier is diagnostic only, never a selection key",
        },
        "selection_keys": list(SELECTION_KEYS),
        "forbidden_sort_keys": list(FORBIDDEN_SORT_KEYS),
        "bootstrap": {
            "definition": ("each origin's paired row-bootstrap LB5 for delta-BSS > 0; "
                           "exactly n row indices sampled with replacement; same indices "
                           "reused for baseline/candidate deployed clipped probabilities; "
                           "single-class resample dropped and redrawn; "
                           "common.score(candidate*)-common.score(baseline*); "
                           "empirical fifth percentile reported"),
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed_base": BOOTSTRAP_SEED_BASE,
            "rng": "np.random.default_rng(20260817 + outer_year)",
        },
        "terminal_firewall": {
            "state": firewall_state(),
            "rule": ("primary labels structurally unreadable until FROZEN; "
                     "--screen reads only r2022/r2023; r2024 diagnostic-only after Task 8"),
        },
    }
    manifest["manifest_hash"] = _canonical_sha256(manifest)
    return manifest


def validate_manifest(manifest: JSON) -> list[str]:
    """매니페스트 구조/해시 검증 → violation 목록 (비면 = 유효)."""
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest not a dict"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version = {manifest.get('schema_version')!r} (필요 {SCHEMA_VERSION})")
    if not manifest.get("candidate_id"):
        problems.append("candidate_id 누락")
    if not isinstance(manifest.get("code_hashes"), dict):
        problems.append("code_hashes 누락")
    if not manifest.get("config_hash"):
        problems.append("config_hash 누락")
    if not manifest.get("data_hash"):
        problems.append("data_hash 누락")
    origins = manifest.get("origins") or {}
    if list(origins.get("selection") or []) != list(SELECTION_ORIGINS):
        problems.append(f"origins.selection = {origins.get('selection')!r} "
                        f"(필요 {list(SELECTION_ORIGINS)})")
    if origins.get("terminal") != TERMINAL_ORIGIN:
        problems.append(f"origins.terminal = {origins.get('terminal')!r} (필요 {TERMINAL_ORIGIN})")
    if list(manifest.get("selection_keys") or []) != list(SELECTION_KEYS):
        problems.append(f"selection_keys = {manifest.get('selection_keys')!r} "
                        f"(필요 {list(SELECTION_KEYS)})")
    if not isinstance(manifest.get("bootstrap"), dict):
        problems.append("bootstrap 정의 누락")
    fw = manifest.get("terminal_firewall") or {}
    if fw.get("state") not in (FIREWALL_UNFROZEN, FIREWALL_FROZEN):
        problems.append(f"terminal_firewall.state = {fw.get('state')!r} (알 수 없음)")
    # 자기 무결성: manifest_hash 재계산 대조
    recorded = manifest.get("manifest_hash")
    recomputed = _canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if recorded != recomputed:
        problems.append(f"manifest_hash 불일치: recorded={str(recorded)[:16]}… "
                        f"recomputed={recomputed[:16]}… (매니페스트 변조)")
    return problems
