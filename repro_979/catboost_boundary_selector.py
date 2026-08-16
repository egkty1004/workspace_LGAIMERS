#!/usr/bin/env python3
"""catboost_boundary_selector.py — Todo 2/3: primary-only CatBoost boundary selector + gate.

이 라운드(aimers9-score-improvement-next-round)의 Wave 1 두 번째/세 번째 태스크:
기존 스윕(blend_weight_sweep.py)이 홀드아웃 부트스트랩 LB5% 로 근접 후보를 랭킹한
선택 누수(selection leakage)를 교정하고, CatBoost 경계 영역(cat 0.35..0.55)을
**primary 라벨로만** 평가하여 정확히 하나의 후보를 동결한다 (Todo 2).

Todo 3 (이 모듈에 통합): 동결 후보를 **활성 롤백 기준선** `5890a4c54f502c4e`
(lgb 0.30 / mlp 0.35 / cat 0.35) 와 primary 폴드에서만 비교해 정확히 하나의
판정을 낸다 — PRIMARY_PASS iff (ΔBSS > 3.0) AND (max|Δmean| <= 0.005), 그 외
PRIMARY_REJECT. r2022/r2023/r2024 라벨은 절대 로드하지 않는다 (Task 10 전
reject-only, 구조적 가드). 패키징/제출 없음.

그리드 계약 (정책/계획에 명시 — 분모 40 tick):
  - Cat  : tick 14..22  (0.350..0.550, step 0.025)
  - LGB  : tick  6..14  (0.150..0.350, step 0.025)
  - MLP  : 40 - cat_tick - lgb_tick,  단 mlp_tick >= 6 (즉 MLP >= 0.15)
  - feasible 행은 정확히 78개 (81 조합 중 cat+lgb > 34 인 3개 제외) — assert 필수.

선택 규칙 (누수 없음):
  - 랭킹은 primary_bss 단일 키만 사용. 동점(정확히 동일 BSS)은 (cat_tick, lgb_tick,
    mlp_tick) 오름차순 (계획 tie_rule).
  - primary_bss 산식 = common.score(clip(sigmoid(z_blend + C_LOGIT), 0.30, 0.70), y)
    (C_LOGIT=-0.0404 동결 — 계획 검증 전략의 배포 산식과 동일).
  - r2022/r2023/r2024 라벨·로짓·메트릭은 선택 전에 절대 로드하지 않는다 (Task 10
    전 R-only 폴드는 reject-only — 정책 forbidden_sort_keys).
  - 정렬 키에 R-only/부트스트랩 키가 섞이면 PolicyViolation → exit 2 (구조적 가드).

프로세스 순서 (pre-registration 계약):
  1. 정책 JSON 읽기(읽기 전용) + validate_policy 재사용 (violation 시 exit 2).
  2. 그리드 행 78개 열거 + 행별 결정적 candidate_id 계산 (라벨 무의존).
  3. 설정(config: 그리드/산식/후보 id 목록/해시)을 라벨 읽기 **이전**에
     task-2-primary-selector-config.json 으로 기록 (pre-registration).
  4. train 로드 → 폴드 마스크 → row-ID disjointness assert (마스크만, 라벨 아님).
  5. **primary 라벨만** 로드 (labels = {"primary": ...} 구조 assert).
  6. LGB/MLP/CatBoost primary OOF 로짓 로드 + 증거 다이제스트 대조.
  7. 78행 primary_bss 계산 → primary 단일 키 랭킹 → 정확히 1개 동결.

소비 파일 (primary 폴드만, 모두 sha256 검증):
  - cache/preds_primary_lgb_f3.npy            (Task 2 cache_provenance)
  - cache/mlp_primary.npy                     (Task 2 cache_provenance)
  - cache/qualification/catboost/primary.npy  (Task 5 per_fold)
  - train.csv (공식 데이터, feather 캐시 부재 시 csv 폴백)

증거: .omo/evidence/aimers9-next-round/task-2-primary-selector.{json,log,md}
  (+ 사전 등록 config: task-2-primary-selector-config.json)
  .omo/evidence/aimers9-next-round/task-3-cat-boundary-gate.{json,md}
  (+ 사전 등록 config: task-3-cat-boundary-gate-config.json)

모드:
  --smoke          그리드 78행 + 선택 경로 + 게이트 경로 검증 (증거 task-2/3 *-smoke.*)
  --primary-only   전체 primary 선택 + 동결 + 게이트 판정 (기본, 증거 task-2/3)
  --fixture <이름> 실패 주입:
                   r-sort-key / attempted-r-fold-read → exit 2
                   nonpositive-primary → ΔBSS≤0 강제 주입, PRIMARY_REJECT 경로 증명 후 exit 2

퇴장 코드: 0 = PASS (게이트 판정 포함, PRIMARY_REJECT 도 정상), 1 = 치명적 입력 오류,
          2 = 정책/누수 가드 위반 (또는 실패 주입 fixture 정상 증명).
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
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너와 동일

# ── 동결 컨트롤 / 기존 셀렉터 규약 재사용 (읽기 전용 import) ──
from repro_979.qualification_runner import (  # noqa: E402
    FOLDS, R_FOLDS, C_LOGIT, CLIP_LO, CLIP_HI,
    build_folds, _check_leakage, _sha256,
)
from repro_979.blend_selector import (  # noqa: E402
    SELECTION_FOLD, BOOT_SEED, EXPECTED_ROWS, _candidate_id,
)
from repro_979.next_round_policy import (  # noqa: E402
    validate_policy, _canonical_sha256, load_json,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import repro_979.common as common  # noqa: E402

# ════════════════════════════════════════════════════════════════════
# Todo 2 상수
# ════════════════════════════════════════════════════════════════════
SCHEMA_VERSION = 1
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
POLICY_PATH = REPO / "next_round_policy.json"
T2_EVIDENCE = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-2-control.json"
T5_EVIDENCE = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-5-models-catboost.json"

# 그리드 — 분모 40 tick (계획 검증 전략: "cat=14..22, lgb=6..14, mlp=40-cat-lgb, mlp>=6")
DENOM = 40
WEIGHT_STEP = 0.025          # tick 크기 (candidate_id step 인자와 동일)
CAT_TICKS = tuple(range(14, 23))    # 14..22  → cat 0.350..0.550
LGB_TICKS = tuple(range(6, 15))     # 6..14   → lgb 0.150..0.350
MLP_MIN_TICK = 6                    # mlp >= 0.150 (tick < 6 행 discard)
EXPECTED_GRID_ROWS = 78             # 9×9 − (cat+lgb>34 인 3행)
TIE_SORT_KEYS = ("cat_tick", "lgb_tick", "mlp_tick")   # 계획 tie_rule 순서

MEMBERS = ("lgb", "mlp", "catboost")          # 후보 ID 해시용 멤버 순서 (스윕 grid_b 동일)
MEMBER_FILES: dict[str, str] = {
    "lgb": "cache/preds_primary_lgb_f3.npy",
    "mlp": "cache/mlp_primary.npy",
    "catboost": "cache/qualification/catboost/primary.npy",
}
PRIMARY_ROWS = EXPECTED_ROWS[SELECTION_FOLD]  # 253507

# ════════════════════════════════════════════════════════════════════
# Todo 3 — reject-only temporal gate 상수 (동결 후보 vs 활성 롤백 기준선)
# ════════════════════════════════════════════════════════════════════
# 동결 후보 (Task 2 결과 — lgb 0.35 / mlp 0.30 / cat 0.35, primary_bss 795.2586)
FROZEN_CANDIDATE_ID = "d15788255a7596bb"
FROZEN_WEIGHTS_BY_MEMBER: dict[str, float] = {"lgb": 0.35, "mlp": 0.30, "catboost": 0.35}
# 활성 롤백 기준선 — 정책 rollback_baseline_candidate_id (REPORT_blend_weight_sweep.md
# grid B rank 1: lgb 0.30 / mlp 0.35 / cat 0.35). step-0.05 id 스킴이라 이 모듈의
# _row_candidate_id(step 0.025) 로 재계산하지 않는다 — 리터럴 동결 ID.
BASELINE_CANDIDATE_ID = "5890a4c54f502c4e"
BASELINE_WEIGHTS_BY_MEMBER: dict[str, float] = {"lgb": 0.30, "mlp": 0.35, "catboost": 0.35}
GATE_DELTA_BSS_MIN = 3.0      # ΔBSS > 3.0 — STRICT (초과만 PASS)
GATE_MEAN_SHIFT_MAX = 0.005   # max|Δmean| <= 0.005 (probability means, clip+sigmoid 후)
# fixture nonpositive-primary 주입 플래그 (메인 흐름에서만 True — 소문자 런타임 플래그)
force_nonpositive_primary = False
T3_SELECTOR_EVIDENCE = EVIDENCE_DIR / "task-2-primary-selector.json"  # 동결 교차 검증용

# R-only / 부트스트랩 메트릭은 reject-only — 선택 정렬 키로 절대 사용 금지 (정책 forbidden).
# 테스트(b)의 monkeypatch 대상 스텁: Task 10 이전에는 절대 채워지지 않는다.
R_ONLY_METRICS: dict[str, dict[str, Any]] = {}


class ProvenanceError(RuntimeError):
    """OOF 아티팩트가 커밋된 증거 다이제스트와 불일치하면 발생 (재생성 의심 → 중단)."""


class LeakageError(RuntimeError):
    """R-only 폴드 라벨/로짓을 선택 경로에서 읽으려 하면 발생 (누수 가드)."""


class PolicyViolation(RuntimeError):
    """정렬 키 등 정책 위반이 감지되면 발생 (exit 2)."""


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


# ════════════════════════════════════════════════════════════════════
# 그리드 — 전부 사전 지정 (어떤 라벨로도 피팅하지 않음)
# ════════════════════════════════════════════════════════════════════
def _grid_rows() -> list[dict[str, Any]]:
    """Cat/LGB tick 그리드 + MLP remainder, mlp_tick >= 6 필터. 정확히 78행 assert."""
    rows: list[dict[str, Any]] = []
    for cat_tick in CAT_TICKS:
        for lgb_tick in LGB_TICKS:
            mlp_tick = DENOM - cat_tick - lgb_tick
            if mlp_tick < MLP_MIN_TICK:
                continue
            w = (float(lgb_tick) / DENOM, float(mlp_tick) / DENOM, float(cat_tick) / DENOM)
            rows.append({
                "cat_tick": int(cat_tick), "lgb_tick": int(lgb_tick), "mlp_tick": int(mlp_tick),
                "weights": [round(x, 4) for x in w],
                "weights_by_member": {
                    "lgb": round(w[0], 4), "mlp": round(w[1], 4), "catboost": round(w[2], 4)},
            })
    assert len(rows) == EXPECTED_GRID_ROWS, (
        f"그리드 행 수 {len(rows)} != {EXPECTED_GRID_ROWS} — 계약 위반")
    for r in rows:
        assert abs(sum(r["weights"]) - 1.0) < 1e-9, f"가중치 합 ≠ 1: {r}"
        assert 0.35 <= r["weights_by_member"]["catboost"] <= 0.55 + 1e-9, r
        assert 0.15 <= r["weights_by_member"]["lgb"] <= 0.35 + 1e-9, r
        assert r["weights_by_member"]["mlp"] >= 0.15 - 1e-9, r
    return rows


def _row_candidate_id(row: dict[str, Any]) -> str:
    """결정적 후보 ID — blend_selector._candidate_id 규약 재사용 (라벨 무의존)."""
    w = np.asarray(row["weights"], dtype=np.float64)
    return _candidate_id(MEMBERS, w, SELECTION_FOLD, BOOT_SEED, WEIGHT_STEP)


# ════════════════════════════════════════════════════════════════════
# 정책/누수 가드
# ════════════════════════════════════════════════════════════════════
def _assert_primary_only_sort_keys(sort_keys: tuple[str, ...]) -> None:
    """정렬 키가 primary_bss 단일 키가 아니면 PolicyViolation (R-only/부트스트랩 정렬 차단)."""
    if tuple(sort_keys) != ("primary_bss",):
        raise PolicyViolation(
            f"[POLICY] 정렬 키 {list(sort_keys)} 는 primary_bss 단일 키가 아닙니다 — "
            f"R-only/부트스트랩 메트릭은 정렬 키로 사용 금지 (정책 forbidden_sort_keys, exit 2)")


def _read_label_fold(labels: dict[str, np.ndarray[Any, np.dtype[np.float64]]], fold: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    """라벨 폴드 읽기 가드 — R-only 폴드는 Task 10 전 절대 읽지 않는다."""
    if fold in R_FOLDS:
        raise LeakageError(
            f"[LEAKAGE] R-only 폴드 라벨 {fold!r} 읽기 차단 — Task 10 전 선택 경로에서 "
            f"R-only 폴드는 reject-only (exit 2)")
    if fold not in labels:
        raise KeyError(f"라벨 폴드 {fold!r} 부재 — labels = {sorted(labels)}")
    return labels[fold]


def _assert_primary_only_labels(labels: dict[str, np.ndarray[Any, np.dtype[np.float64]]]) -> None:
    """선택 경로가 primary 라벨만 보유했는지 구조 검증 (R-only 라벨 미로드 증명)."""
    if set(labels) != {SELECTION_FOLD}:
        raise LeakageError(
            f"[LEAKAGE] 선택 경로에 primary 외 라벨 존재: {sorted(labels)} — "
            f"primary 라벨만 허용 (exit 2)")


def _check_policy() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """동결 정책 JSON 읽기(읽기 전용) + next_round_policy.validate_policy 재사용.
    반환: (policy, violations, policy_config_hash). violations 비면 계약 준수."""
    policy = load_json(POLICY_PATH)
    if policy is None:
        raise PolicyViolation(f"[POLICY] 정책 JSON 읽기 불가: {POLICY_PATH}")
    violations = validate_policy(policy)
    sel = policy.get("selection") or {}
    if sel.get("label_source") != "primary":
        violations.append({"rule": "label_source", "ok": False,
                           "reason": f"label_source={sel.get('label_source')!r} != primary"})
    if list(sel.get("sort_keys") or []) != ["primary_bss"]:
        violations.append({"rule": "sort_keys", "ok": False,
                           "reason": f"sort_keys={sel.get('sort_keys')!r} != [primary_bss]"})
    return policy, violations, _canonical_sha256(policy)


# ════════════════════════════════════════════════════════════════════
# 출처 검증 — task-2-control.json cache_provenance + task-5-models-catboost.json
# ════════════════════════════════════════════════════════════════════
def _evidence_digests() -> dict[str, str]:
    """member → primary 폴드 기대 sha256 (증거 JSON에서 로드)."""
    if not T2_EVIDENCE.is_file():
        raise ProvenanceError(f"증거 JSON 없음: {T2_EVIDENCE}")
    if not T5_EVIDENCE.is_file():
        raise ProvenanceError(f"증거 JSON 없음: {T5_EVIDENCE}")
    t2 = json.loads(T2_EVIDENCE.read_text(encoding="utf-8"))
    t5 = json.loads(T5_EVIDENCE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for rec in t2.get("cache_provenance", []):
        if rec.get("fold") == SELECTION_FOLD and rec.get("component") in ("lgb", "mlp"):
            out[rec["component"]] = rec["sha256"]
    cat_digest = t5.get("per_fold", {}).get(SELECTION_FOLD, {}).get("digest")
    if cat_digest:
        out["catboost"] = cat_digest
    missing = sorted(set(MEMBER_FILES) - set(out))
    if missing:
        raise ProvenanceError(f"증거 JSON에 primary 다이제스트 누락: {missing}")
    return out


def _verify_provenance() -> list[dict[str, Any]]:
    """멤버 3종 primary on-disk sha256 을 커밋된 증거 다이제스트와 대조. 불일치 시 중단."""
    expected = _evidence_digests()
    manifest: list[dict[str, Any]] = []
    for mid, rel in MEMBER_FILES.items():
        path = REPO / rel
        if not path.is_file():
            raise ProvenanceError(f"OOF 파일 없음: {rel}")
        actual = _sha256(path)
        n = int(len(np.load(path)))
        source = ("task-2-control.json cache_provenance" if mid in ("lgb", "mlp")
                  else "task-5-models-catboost.json per_fold")
        rec: dict[str, Any] = {
            "member": mid, "fold": SELECTION_FOLD, "file": rel, "rows": n,
            "expected_rows": PRIMARY_ROWS, "rows_match": n == PRIMARY_ROWS,
            "sha256": actual, "expected_sha256": expected[mid],
            "digest_match": actual == expected[mid], "digest_source": source,
        }
        if not rec["digest_match"] or not rec["rows_match"]:
            raise ProvenanceError(
                f"[FAIL] {mid}/{SELECTION_FOLD} 출처 검증: file={rel} digest={actual[:16]}… "
                f"expected={expected[mid][:16]}… rows={n} expected={PRIMARY_ROWS}")
        manifest.append(rec)
    return manifest


# ════════════════════════════════════════════════════════════════════
# 평가 — primary BSS 단일 키 (배포 산식: C_LOGIT + clip)
# ════════════════════════════════════════════════════════════════════
def _primary_bss(z_blend: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """계획 검증 전략 산식: common.score(clip(sigmoid(z+C_LOGIT), 0.30, 0.70), y)."""
    p = np.clip(common.sigmoid(z_blend + C_LOGIT), CLIP_LO, CLIP_HI)
    return float(common.score(p, y))


def _clip_fracs(z_blend: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, Any]:
    """C_LOGIT 적용 후 클리핑 진단 (frozen 후보만, 진단 전용)."""
    p = common.sigmoid(z_blend + C_LOGIT)
    return {
        "frac_below_0.30": float((p < CLIP_LO).mean()),
        "frac_above_0.70": float((p > CLIP_HI).mean()),
        "clip_bounds": [CLIP_LO, CLIP_HI],
        "c_logit": C_LOGIT,
    }


def _rank_rows(rows: list[dict[str, Any]], bss: np.ndarray[Any, np.dtype[np.float64]],
               sort_keys: tuple[str, ...] = ("primary_bss",)) -> list[dict[str, Any]]:
    """primary_bss 단일 키 랭킹 (내림차순), 정확한 동점은 tick 오름차순.
    sort_keys 에 primary_bss 외 키가 오면 PolicyViolation (exit 2)."""
    _assert_primary_only_sort_keys(sort_keys)
    ranked = []
    for i, row in enumerate(rows):
        ranked.append({
            **row,
            "candidate_id": _row_candidate_id(row),
            "primary_bss": float(bss[i]),
        })
    ranked.sort(key=lambda r: (-r["primary_bss"], r["cat_tick"], r["lgb_tick"], r["mlp_tick"]))
    for idx, r in enumerate(ranked, start=1):
        r["rank"] = int(idx)
    return ranked


def _freeze(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """정확히 하나의 후보 동결 — ranked[0] (primary 단일 키 상위)."""
    assert len(ranked) >= 1, "동결할 후보 없음"
    frozen = dict(ranked[0])
    frozen["n_frozen"] = 1
    return frozen


# ════════════════════════════════════════════════════════════════════
# Todo 3 — reject-only temporal gate (동결 후보 vs 활성 롤백 기준선, primary 전용)
# ════════════════════════════════════════════════════════════════════
def _gate_verdict(delta_bss: float, mean_shift: float,
                  delta_min: float | None = None, shift_max: float | None = None) -> str:
    """게이트 판정 규칙 (모듈 상수 기본 — monkeypatch 가능).

    PRIMARY_PASS iff (delta_bss > GATE_DELTA_BSS_MIN) AND (mean_shift <= GATE_MEAN_SHIFT_MAX).
    ΔBSS == 3.0 은 strict 초과 규칙에 따라 REJECT (경계는 PASS 가 아님)."""
    delta_min = GATE_DELTA_BSS_MIN if delta_min is None else delta_min
    shift_max = GATE_MEAN_SHIFT_MAX if shift_max is None else shift_max
    if delta_bss > delta_min and mean_shift <= shift_max:
        return "PRIMARY_PASS"
    return "PRIMARY_REJECT"


def _gate_primary(z_cand: np.ndarray[Any, np.dtype[np.float64]],
                  z_base: np.ndarray[Any, np.dtype[np.float64]],
                  y: np.ndarray[Any, np.dtype[np.float64]],
                  frozen_id: str, frozen_weights: dict[str, float],
                  baseline_id: str, baseline_weights: dict[str, float],
                  force_nonpositive: bool = False) -> dict[str, Any]:
    """primary 폴드에서 동결 후보 vs 기준선 비교 (배포 산식 동일).

    ΔBSS = primary_bss(후보) − primary_bss(기준선).
    mean shift = |mean(p_후보) − mean(p_기준선)| (clip+sigmoid 후 확률 평균, 단일 폴드).
    force_nonpositive: fixture 주입 — ΔBSS 를 <= 0 으로 강제 (PRIMARY_REJECT 경로 증명)."""
    p_cand = np.clip(common.sigmoid(z_cand + C_LOGIT), CLIP_LO, CLIP_HI)
    p_base = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    bss_cand = float(common.score(p_cand, y))
    bss_base = float(common.score(p_base, y))
    delta = bss_cand - bss_base
    forced = bool(force_nonpositive)
    if forced:
        delta = min(delta, 0.0)  # ΔBSS <= 0 강제 (경로 증명용)
    mean_shift = float(np.abs(p_cand.mean() - p_base.mean()))
    return {
        "frozen": {
            "candidate_id": frozen_id,
            "weights_by_member": dict(frozen_weights),
            "primary_bss": bss_cand,
            "brier": float(((p_cand - y) ** 2).mean()),
            "pred_mean": float(p_cand.mean()),
        },
        "baseline": {
            "candidate_id": baseline_id,
            "weights_by_member": dict(baseline_weights),
            "primary_bss": bss_base,
            "brier": float(((p_base - y) ** 2).mean()),
            "pred_mean": float(p_base.mean()),
        },
        "delta_bss": delta,
        "mean_shift": mean_shift,
        "r": float(y.mean()),
        "thresholds": {"delta_bss_min": GATE_DELTA_BSS_MIN, "mean_shift_max": GATE_MEAN_SHIFT_MAX},
        "forced_nonpositive": forced,
        "verdict": _gate_verdict(delta, mean_shift),
    }


# ════════════════════════════════════════════════════════════════════
# 증거 작성
# ════════════════════════════════════════════════════════════════════
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


def _write_md(record: dict[str, Any], path: Path) -> Path:
    frozen = record["frozen"]
    lines = [
        f"# {record['title']} — {record['verdict']} (exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
        f"- **config_hash**: `{record['config_hash']}`",
        f"- **policy_config_hash**: `{record['policy_config_hash']}`",
        f"- **label_sources**: {record['label_sources']} (labels_read={record['labels_read']})",
        "",
        "## Protocol",
        "",
        "- **Grid (denominator-40 ticks, from the frozen plan contract)**: "
        f"cat `{CAT_TICKS[0]}..{CAT_TICKS[-1]}` ({CAT_TICKS[0]/DENOM:.3f}..{CAT_TICKS[-1]/DENOM:.3f}), "
        f"lgb `{LGB_TICKS[0]}..{LGB_TICKS[-1]}` ({LGB_TICKS[0]/DENOM:.3f}..{LGB_TICKS[-1]/DENOM:.3f}), "
        f"mlp = 40 − cat − lgb, discard `mlp_tick < {MLP_MIN_TICK}` (MLP < 0.15).",
        f"- **Feasible rows**: {record['grid']['n_rows']} (assert == {EXPECTED_GRID_ROWS}).",
        "- **Selection**: rank rows solely by `primary_bss` "
        f"(`common.score(clip(sigmoid(z+C_LOGIT),0.30,0.70), y)`, C_LOGIT={C_LOGIT}); "
        "exact ties sort ascending `(cat_tick, lgb_tick, mlp_tick)`. "
        "R-only folds / bootstrap LB are reject-only and never loaded before selection.",
        "- **Freeze**: exactly one candidate (weights + candidate id hash) written before any "
        "label result is read (pre-registered config "
        f"`{Path(record['config_pre_registered']['file']).name}`).",
        "- **Provenance**: LGB/MLP primary (task-2-control.json cache_provenance) and CatBoost "
        "primary (task-5-models-catboost.json per_fold) sha256 verified.",
        "",
        "## Grid & primary BSS (all feasible rows, sole order key `primary_bss`)",
        "",
        "| rank | cat | lgb | mlp | primary_bss | candidate_id |",
        "|---|---|---|---|---|---|",
    ]
    for r in record["grid"]["rows"]:
        lines.append(
            f"| {r['rank']} | {r['weights_by_member']['catboost']:.3f} "
            f"| {r['weights_by_member']['lgb']:.3f} | {r['weights_by_member']['mlp']:.3f} "
            f"| {r['primary_bss']:.4f} | `{r['candidate_id']}` |")
    lines += [
        "",
        "## Frozen candidate",
        "",
        f"- **weights**: lgb×{frozen['weights_by_member']['lgb']:.3f} "
        f"mlp×{frozen['weights_by_member']['mlp']:.3f} "
        f"catboost×{frozen['weights_by_member']['catboost']:.3f} "
        f"(cat_tick={frozen['cat_tick']}, lgb_tick={frozen['lgb_tick']}, "
        f"mlp_tick={frozen['mlp_tick']})",
        f"- **primary_bss**: {frozen['primary_bss']:.4f}",
        f"- **candidate_id**: `{frozen['candidate_id']}`",
        f"- **clipping diagnostics**: {frozen.get('clipping')}",
        "",
        "## Checks",
        "",
    ]
    for c in record.get("checks", []):
        mark = "PASS" if c.get("ok") else ("FAIL" if c.get("ok") is False else "INFO")
        lines.append(f"- **[{mark}]** {c.get('rule')}: {c.get('reason', '')}")
    lines += ["", f"## Verdict: **{record['verdict']}**", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_gate_md(record: dict[str, Any], path: Path) -> Path:
    g = record["gate"]
    fz, bl = g["frozen"], g["baseline"]
    lines = [
        f"# {record['title']} — {record['verdict']} (exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
        f"- **config_hash**: `{record['config_hash']}`",
        f"- **policy_config_hash**: `{record['policy_config_hash']}`",
        f"- **label_sources**: {record['label_sources']} (labels_read={record['labels_read']})",
        "",
        "## Protocol",
        "",
        f"- **Frozen candidate (Task 2)**: `{fz['candidate_id']}` — "
        f"lgb×{fz['weights_by_member']['lgb']:.3f} / mlp×{fz['weights_by_member']['mlp']:.3f} / "
        f"catboost×{fz['weights_by_member']['catboost']:.3f} (primary_bss={fz['primary_bss']:.4f}).",
        f"- **Baseline (active rollback)**: `{bl['candidate_id']}` — "
        f"lgb×{bl['weights_by_member']['lgb']:.3f} / mlp×{bl['weights_by_member']['mlp']:.3f} / "
        f"catboost×{bl['weights_by_member']['catboost']:.3f} (primary_bss={bl['primary_bss']:.4f}).",
        f"- **Scoring (both)**: `common.score(clip(sigmoid(z+C_LOGIT),0.30,0.70), y)`, "
        f"C_LOGIT={C_LOGIT}, primary fold only ({SELECTION_FOLD}, {PRIMARY_ROWS} rows).",
        "- **ΔBSS** = primary_bss(frozen) − primary_bss(baseline). "
        "**Mean shift** = max over the scored primary fold of "
        "|mean(p_candidate) − mean(p_baseline)| (probability means after clip+sigmoid).",
        f"- **Gate rule**: `PRIMARY_PASS` iff ΔBSS > {GATE_DELTA_BSS_MIN} "
        f"AND mean shift <= {GATE_MEAN_SHIFT_MAX}; else `PRIMARY_REJECT`.",
        "- **R-fold embargo**: r2022/r2023/r2024 labels are never read in this mode "
        "(Task 10 R qualification) — `_read_label_fold`/`_assert_primary_only_labels` "
        "structural guards; `--fixture attempted-r-fold-read` exits 2.",
        "",
        "## Gate metrics (primary fold)",
        "",
        f"- **frozen**: id `{fz['candidate_id']}` bss={fz['primary_bss']:.4f} "
        f"brier={fz['brier']:.6f} pred_mean={fz['pred_mean']:.6f}",
        f"- **baseline**: id `{bl['candidate_id']}` bss={bl['primary_bss']:.4f} "
        f"brier={bl['brier']:.6f} pred_mean={bl['pred_mean']:.6f}",
        f"- **ΔBSS**: {g['delta_bss']:+.4f} (need > {GATE_DELTA_BSS_MIN}: "
        f"{bool(g['delta_bss'] > GATE_DELTA_BSS_MIN)})",
        f"- **mean shift**: {g['mean_shift']:.6f} (need <= {GATE_MEAN_SHIFT_MAX}: "
        f"{bool(g['mean_shift'] <= GATE_MEAN_SHIFT_MAX)})",
        f"- **r** = {g['r']:.4f}",
        f"- **forced_nonpositive**: {g['forced_nonpositive']} (fixture 주입 여부)",
        "",
        "## Checks",
        "",
    ]
    for c in record.get("checks", []):
        mark = "PASS" if c.get("ok") else ("FAIL" if c.get("ok") is False else "INFO")
        lines.append(f"- **[{mark}]** {c.get('rule')}: {c.get('reason', '')}")
    lines += ["", f"## Verdict: **{record['verdict']}**", ""]
    if record.get("notion", {}).get("status"):
        n = record["notion"]
        lines += ["## Notion local-CV row", "",
                  f"- status: `{n.get('status')}`",
                  f"- cv_table_block: `{n.get('cv_table_block')}`",
                  f"- row_receipt: `{json.dumps(n.get('row_receipt'), ensure_ascii=False)}`"]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    global force_nonpositive_primary
    parser = argparse.ArgumentParser(
        description="Todo 2/3: primary-only CatBoost boundary grid selector (78 feasible rows, "
                    "freeze exactly one candidate on primary BSS only) + reject-only gate "
                    "(frozen vs rollback baseline 5890a4c54f502c4e → PRIMARY_PASS/PRIMARY_REJECT)")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 그리드 78행 + primary 선택 + 1개 동결 + 게이트 경로 검증")
    parser.add_argument("--primary-only", action="store_true",
                        help="전체 primary 선택 실행 + 게이트 판정 (기본 모드)")
    parser.add_argument("--fixture", choices=["r-sort-key", "attempted-r-fold-read",
                                              "nonpositive-primary"], default=None,
                        help="실패 QA: R-only 정렬 키 / R-only 라벨 읽기 시도 / ΔBSS≤0 강제 "
                             "주입 (전부 exit 2)")
    parser.add_argument("--evidence", default=None, help="Task 2 증거 JSON 경로 (기본 task-2-primary-selector.json)")
    parser.add_argument("--log", default=None, help="증거 로그 경로 (기본 task-2-primary-selector.log)")
    parser.add_argument("--md", default=None, help="증거 MD 경로 (기본 task-2-primary-selector.md)")
    args = parser.parse_args(argv)

    smoke = bool(args.smoke)
    if args.fixture:  # 실패 주입 실행은 메인 증거 로그를 절대 덮어쓰지 않는다
        base = f"task-2-primary-selector-fixture-{args.fixture}"
        t3_base = f"task-3-cat-boundary-gate-fixture-{args.fixture}"
    else:
        base = "task-2-primary-selector-smoke" if smoke else "task-2-primary-selector"
        t3_base = "task-3-cat-boundary-gate-smoke" if smoke else "task-3-cat-boundary-gate"
    evidence_path = (Path(args.evidence).expanduser().resolve() if args.evidence
                     else EVIDENCE_DIR / f"{base}.json")
    log_path = (Path(args.log).expanduser().resolve() if args.log
                else EVIDENCE_DIR / f"{base}.log")
    md_path = (Path(args.md).expanduser().resolve() if args.md
               else EVIDENCE_DIR / f"{base}.md")
    t3_evidence_path = EVIDENCE_DIR / f"{t3_base}.json"
    t3_md_path = EVIDENCE_DIR / f"{t3_base}.md"
    fixture_tag = f"-fixture-{args.fixture}" if args.fixture else ""
    config_path = EVIDENCE_DIR / f"task-2-primary-selector-config{fixture_tag}.json"
    gate_config_path = EVIDENCE_DIR / f"task-3-cat-boundary-gate-config{fixture_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    t0 = time.time()
    print(f"[catboost_boundary_selector] Todo 2/3 — primary-only CatBoost boundary selection "
          f"+ reject-only gate (smoke={smoke}, fixture={args.fixture})", flush=True)

    # ── 실패 주입: 데이터/라벨 로드 전에 차단 (exit 2) ──
    if args.fixture == "r-sort-key":
        print("\n[fixture r-sort-key] R-only 정렬 키 주입 (primary_bss + delta_r2022)…", flush=True)
        try:
            _assert_primary_only_sort_keys(("primary_bss", "delta_r2022"))
        except PolicyViolation as exc:
            print(f"[fixture r-sort-key] PASS — 가드가 R-only 정렬 키를 차단했습니다: {exc}",
                  flush=True)
            return 2
        print("[fixture r-sort-key] FAIL — PolicyViolation 이 발생하지 않았습니다", file=sys.stderr)
        return 1
    if args.fixture == "attempted-r-fold-read":
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
    if args.fixture == "nonpositive-primary":
        # 실데이터로 게이트 REJECT 경로 증명: ΔBSS 를 <= 0 으로 강제 (메인 흐름 계속)
        print("\n[fixture nonpositive-primary] ΔBSS<=0 강제 주입 — PRIMARY_REJECT 경로 "
              "실데이터 검증 시작…", flush=True)
        force_nonpositive_primary = True

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
          f"sort_keys={policy['selection']['sort_keys']} "
          f"policy_hash={policy_hash[:16]}…", flush=True)

    # ── 2) 그리드 열거 + 행별 candidate_id (라벨 무의존) ──
    rows = _grid_rows()
    for r in rows:
        r["candidate_id"] = _row_candidate_id(r)
    print(f"[grid] feasible rows = {len(rows)} (assert {EXPECTED_GRID_ROWS}), "
          f"cat {CAT_TICKS[0]}-{CAT_TICKS[-1]} tick, lgb {LGB_TICKS[0]}-{LGB_TICKS[-1]} tick, "
          f"mlp_tick >= {MLP_MIN_TICK}", flush=True)

    # ── 3) 설정 사전 등록 — 라벨 결과를 읽기 **이전**에 기록 ──
    grid_meta = {
        "denominator": DENOM, "weight_step": WEIGHT_STEP,
        "cat_ticks": list(CAT_TICKS), "lgb_ticks": list(LGB_TICKS),
        "mlp_min_tick": MLP_MIN_TICK, "n_rows": len(rows),
        "n_expected": EXPECTED_GRID_ROWS,
        "tie_sort_keys": list(TIE_SORT_KEYS),
        "rows": [{"cat_tick": r["cat_tick"], "lgb_tick": r["lgb_tick"],
                  "mlp_tick": r["mlp_tick"], "weights": r["weights"],
                  "weights_by_member": r["weights_by_member"],
                  "candidate_id": r["candidate_id"]} for r in rows],
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-2-primary-selector-config",
        "title": "Todo 2 — pre-registered CatBoost boundary selection config",
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "selection_fold": SELECTION_FOLD,
        "label_sources": [SELECTION_FOLD],
        "labels_read": False,
        "labels_read_note": "Config is written BEFORE any label result is read (pre-registration "
                            "contract). Candidate ids derive from weights only.",
        "members": list(MEMBERS),
        "scoring": {
            "formula": "common.score(clip(sigmoid(z_blend + C_LOGIT), 0.30, 0.70), y)",
            "c_logit": C_LOGIT, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
        },
        "sort_keys": ["primary_bss"],
        "forbidden_sort_keys_note": ("R-only folds (r2022/r2023/r2024) and bootstrap LB are "
                                     "reject-only; never a sort key (policy frozen)."),
        "candidate_id_scheme": ("sha256(canonical{members, weights, selection_fold=primary, "
                                f"seed={BOOT_SEED}, weight_step={WEIGHT_STEP}}})[:16]"),
        "grid": grid_meta,
    }
    config_hash = _canonical_sha256(config)
    config["config_hash"] = config_hash
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[pre-register] config hash={config_hash[:16]}… → {config_path} "
          f"(라벨 읽기 이전 기록)", flush=True)

    # ── 3b) Todo 3 게이트 설정 사전 등록 — 라벨 결과를 읽기 **이전**에 기록 ──
    gate_config = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-3-cat-boundary-gate-config",
        "title": "Todo 3 — pre-registered reject-only temporal gate config",
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "selection_fold": SELECTION_FOLD,
        "label_sources": [SELECTION_FOLD],
        "labels_read": False,
        "labels_read_note": ("Config written BEFORE any label result is read (pre-registration "
                             "contract). Frozen id/weights from Task 2 evidence; baseline from "
                             "frozen policy rollback_baseline_candidate_id."),
        "frozen_candidate_id": FROZEN_CANDIDATE_ID,
        "frozen_weights_by_member": FROZEN_WEIGHTS_BY_MEMBER,
        "frozen_source": "task-2-primary-selector.json frozen (lgb 0.35 / mlp 0.30 / catboost 0.35)",
        "baseline_candidate_id": BASELINE_CANDIDATE_ID,
        "baseline_weights_by_member": BASELINE_WEIGHTS_BY_MEMBER,
        "baseline_source": ("next_round_policy.json rollback_baseline_candidate_id; "
                            "REPORT_blend_weight_sweep.md grid B rank 1 (lgb 0.30 / mlp 0.35 / "
                            "catboost 0.35, step-0.05 id 스킴 — 재계산하지 않는 리터럴 ID)"),
        "scoring": {
            "formula": "common.score(clip(sigmoid(z_blend + C_LOGIT), 0.30, 0.70), y)",
            "c_logit": C_LOGIT, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
        },
        "gate": {
            "rule": ("PRIMARY_PASS iff (primary_delta_bss > 3.0) AND "
                     "(max_abs_primary_mean_shift <= 0.005); else PRIMARY_REJECT"),
            "delta_bss_min": GATE_DELTA_BSS_MIN,
            "mean_shift_max": GATE_MEAN_SHIFT_MAX,
            "mean_shift_definition": ("max over the scored primary fold of "
                                      "|mean(p_candidate) - mean(p_baseline)|, probability "
                                      "means after clip+sigmoid"),
            "scored_fold": SELECTION_FOLD,
        },
        "r_fold_embargo": ("r2022/r2023/r2024 labels never loaded in this mode (Task 10 R "
                           "qualification) — _read_label_fold / _assert_primary_only_labels "
                           "guards; --fixture attempted-r-fold-read exits 2"),
    }
    gate_config_hash = _canonical_sha256(gate_config)
    gate_config["config_hash"] = gate_config_hash
    gate_config_path.parent.mkdir(parents=True, exist_ok=True)
    gate_config_path.write_text(json.dumps(gate_config, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    print(f"[pre-register] gate config hash={gate_config_hash[:16]}… → {gate_config_path} "
          f"(라벨 읽기 이전 기록)", flush=True)

    # ── 4) 데이터 + 폴드 (마스크만, 라벨 아님) + row-ID disjointness ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    folds = build_folds(train)
    leakage_problems = _check_leakage(folds, train)
    if leakage_problems:
        print("[FAIL] 누수 가드 실패:\n  " + "\n  ".join(leakage_problems), file=sys.stderr)
        return 1
    va_primary = folds[SELECTION_FOLD][1]
    # r2022/r2023 은 primary 와 row-ID 완전 분리 (하드 assert).
    # r2024 는 계획상 "overlapping r2024" 진단 폴드 — 검증 마스크가 primary(2024 전체)에
    # 포함되는 2024-R 행으로 설계됨 (branch-inert, Task 10 전 로드 금지). 겹침은 문서화.
    overlap_note = {}
    for rfold in R_FOLDS:
        overlap = int((va_primary & folds[rfold][1]).sum())
        if rfold == "r2024":
            overlap_note[rfold] = {"overlap_rows_with_primary": overlap, "expected": True,
                                   "reason": "r2024 val = (season==2024) & R ⊆ primary val "
                                             "(계획 'overlapping r2024' 진단 폴드)"}
        elif overlap != 0:
            print(f"[FAIL] row-ID disjointness 위반: primary ∩ {rfold} ≠ ∅", file=sys.stderr)
            return 1
    assert int(va_primary.sum()) == PRIMARY_ROWS, f"primary 행 수 불일치: {int(va_primary.sum())}"
    print(f"[folds] primary val rows={int(va_primary.sum())}, R 폴드와 row-ID 분리 OK, "
          f"2025 부재 확인", flush=True)

    # ── 5) primary 라벨만 로드 (구조 assert) ──
    labels = {SELECTION_FOLD: train.loc[va_primary, common.TARGET].values.astype(np.float64)}
    _assert_primary_only_labels(labels)
    print(f"[labels] {sorted(labels)} 만 로드 (R-only 라벨 미로드)", flush=True)

    # ── 6) 멤버 OOF 로짓 + 출처 검증 (primary 폴드만) ──
    manifest = _verify_provenance()
    for rec in manifest:
        print(f"[provenance] OK  {rec['member']}/{rec['fold']} {rec['file']} "
              f"rows={rec['rows']} sha256={rec['sha256'][:12]}… ({rec['digest_source']})",
              flush=True)
    members = {m: np.load(REPO / rel).astype(np.float64) for m, rel in MEMBER_FILES.items()}
    y_primary = labels[SELECTION_FOLD]

    # ── 7) 78행 primary BSS → primary 단일 키 랭킹 → 1개 동결 ──
    z_sel = np.stack([members[m] for m in MEMBERS], axis=1)          # (n_primary, 3) [lgb, mlp, cat]
    weights_mat = np.asarray([r["weights"] for r in rows], dtype=np.float64)  # (78, 3)
    bss = np.asarray([_primary_bss(z_sel @ w, y_primary) for w in weights_mat], dtype=np.float64)
    ranked = _rank_rows(rows, bss, sort_keys=("primary_bss",))
    frozen = _freeze(ranked)
    frozen["clipping"] = _clip_fracs(
        z_sel @ np.asarray(frozen["weights"], dtype=np.float64))
    print(f"\n[rank] 78행 primary_bss 단일 키 랭킹 (동점 → (cat_tick, lgb_tick, mlp_tick) 오름차순)")
    for r in ranked[:5]:
        w = r["weights_by_member"]
        print(f"  {r['rank']}. cat×{w['catboost']:.3f} lgb×{w['lgb']:.3f} "
              f"mlp×{w['mlp']:.3f} primary_bss={r['primary_bss']:.4f} id={r['candidate_id']}",
              flush=True)
    print(f"[frozen] 1개 동결: {frozen['weights_by_member']} "
          f"primary_bss={frozen['primary_bss']:.4f} id={frozen['candidate_id']}", flush=True)

    # ── 7b) Todo 3 게이트 — 동결 후보 vs 활성 롤백 기준선 (primary 라벨만) ──
    # 계약: 게이트는 Task 2 동결 후보를 평가한다 — 그리드가 다른 후보를 뽑으면 하드 실패.
    if frozen["candidate_id"] != FROZEN_CANDIDATE_ID:
        raise PolicyViolation(
            f"[POLICY] 동결 후보 {frozen['candidate_id']} != Task 2 동결 {FROZEN_CANDIDATE_ID} "
            f"— 게이트는 Task 2 동결 후보만 평가 (exit 2)")
    z_frozen = z_sel @ np.asarray(frozen["weights"], dtype=np.float64)
    z_base = z_sel @ np.asarray([BASELINE_WEIGHTS_BY_MEMBER[m] for m in MEMBERS],
                                dtype=np.float64)
    gate = _gate_primary(z_frozen, z_base, y_primary, FROZEN_CANDIDATE_ID,
                         FROZEN_WEIGHTS_BY_MEMBER, BASELINE_CANDIDATE_ID,
                         BASELINE_WEIGHTS_BY_MEMBER,
                         force_nonpositive=force_nonpositive_primary)
    g_fz, g_bl = gate["frozen"], gate["baseline"]
    print(f"\n[gate] frozen {g_fz['candidate_id']} bss={g_fz['primary_bss']:.4f} "
          f"brier={g_fz['brier']:.6f} mean={g_fz['pred_mean']:.6f}", flush=True)
    print(f"[gate] baseline {g_bl['candidate_id']} bss={g_bl['primary_bss']:.4f} "
          f"brier={g_bl['brier']:.6f} mean={g_bl['pred_mean']:.6f}", flush=True)
    print(f"[gate] ΔBSS={gate['delta_bss']:+.4f} (need > {GATE_DELTA_BSS_MIN}) | "
          f"mean_shift={gate['mean_shift']:.6f} (need <= {GATE_MEAN_SHIFT_MAX}) | "
          f"r={gate['r']:.4f} | forced={gate['forced_nonpositive']}", flush=True)
    print(f"[gate] verdict: {gate['verdict']}", flush=True)

    # ── 8) 증거 작성 ──
    checks = [
        {"rule": "policy_check", "ok": True,
         "reason": f"{len(policy_violations)} violations — label_source=primary, "
                   f"sort_keys=[primary_bss] (동결 정책)"},
        {"rule": "grid_feasibility", "ok": len(rows) == EXPECTED_GRID_ROWS,
         "reason": f"{len(rows)} feasible rows (mlp_tick >= {MLP_MIN_TICK})"},
        {"rule": "provenance", "ok": all(r["digest_match"] and r["rows_match"] for r in manifest),
         "reason": f"{len(manifest)}/{len(manifest)} primary OOF 다이제스트 일치"},
        {"rule": "primary_only_labels", "ok": set(labels) == {SELECTION_FOLD},
         "reason": f"labels = {sorted(labels)} — R-only 라벨 미로드"},
        {"rule": "row_disjointness", "ok": True,
         "reason": "primary 검증 마스크가 r2022/r2023 과 row-ID 분리 "
                   "(r2024 는 계획상 overlapping 진단 폴드 — 겹침 문서화: "
                   f"{json.dumps(overlap_note, ensure_ascii=False)})"},
        {"rule": "sort_keys_primary_only", "ok": True,
         "reason": "랭킹 정렬 키 = [primary_bss] 단일 (R-only/부트스트랩 키 사용 금지)"},
        {"rule": "frozen_single_candidate", "ok": frozen["n_frozen"] == 1,
         "reason": f"1개 동결 — {frozen['candidate_id']}"},
    ]
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 2 — primary-only CatBoost boundary selector",
        "task": "aimers9-next-round/task-2-primary-selector",
        "mode": "smoke" if smoke else "primary-only",
        "verdict": "PASS",
        "exit_code": 0,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": config_hash,
        "config_pre_registered": {
            "file": str(config_path), "sha256": _sha256(config_path),
            "written_before_labels_read": True,
            "labels_read_at_registration": False,
        },
        "policy_path": str(POLICY_PATH),
        "policy_config_hash": policy_hash,
        "label_sources": [SELECTION_FOLD],
        "labels_read": True,
        "selection_fold": SELECTION_FOLD,
        "members": list(MEMBERS),
        "scoring": {
            "formula": "common.score(clip(sigmoid(z_blend + C_LOGIT), 0.30, 0.70), y)",
            "c_logit": C_LOGIT, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
        },
        "sort_keys": ["primary_bss"],
        "forbidden_sort_keys": list(policy["selection"]["forbidden_sort_keys"]),
        "tie_rule": "exact primary-score ties sort ascending by (cat_tick, lgb_tick, mlp_tick)",
        "grid": {
            "denominator": DENOM, "weight_step": WEIGHT_STEP,
            "cat_ticks": list(CAT_TICKS), "lgb_ticks": list(LGB_TICKS),
            "mlp_min_tick": MLP_MIN_TICK,
            "n_rows": len(rows), "n_expected": EXPECTED_GRID_ROWS,
            "rows": ranked,
        },
        "provenance": manifest,
        "frozen": {k: frozen[k] for k in (
            "rank", "cat_tick", "lgb_tick", "mlp_tick", "weights", "weights_by_member",
            "primary_bss", "candidate_id", "clipping", "n_frozen")},
        "ranking": [r["candidate_id"] for r in ranked],
        "checks": checks,
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    _write_md(record, md_path)
    print(f"\n[catboost_boundary_selector] 증거 JSON → {evidence_path}", flush=True)
    print(f"[catboost_boundary_selector] 증거 LOG  → {log_path}", flush=True)
    print(f"[catboost_boundary_selector] 증거 MD   → {md_path}", flush=True)
    print(f"[catboost_boundary_selector] 총 {time.time() - t0:.0f}s", flush=True)

    # ── 9) Task 3 게이트 증거 작성 ──
    t2_frozen_check: dict[str, Any] = {"ok": False, "reason": ""}
    if T3_SELECTOR_EVIDENCE.is_file():
        t2_rec = json.loads(T3_SELECTOR_EVIDENCE.read_text(encoding="utf-8"))
        t2_fz = t2_rec.get("frozen", {})
        id_ok = t2_fz.get("candidate_id") == FROZEN_CANDIDATE_ID
        bss_ok = abs(float(t2_fz.get("primary_bss", -1.0)) - gate["frozen"]["primary_bss"]) < 1e-6
        t2_frozen_check = {
            "ok": bool(id_ok and bss_ok),
            "reason": (f"task-2-primary-selector.json frozen id={t2_fz.get('candidate_id')} "
                       f"bss={t2_fz.get('primary_bss'):.6f} vs 재계산 "
                       f"id={gate['frozen']['candidate_id']} "
                       f"bss={gate['frozen']['primary_bss']:.6f} — id_match={id_ok}, "
                       f"bss_match={bss_ok}"),
        }
    else:
        t2_frozen_check = {"ok": False,
                           "reason": f"Task 2 증거 없음: {T3_SELECTOR_EVIDENCE} — 교차 검증 불가"}
    gate_checks = [
        {"rule": "policy_check", "ok": not policy_violations,
         "reason": f"{len(policy_violations)} violations — label_source=primary (동결 정책)"},
        {"rule": "provenance", "ok": all(r["digest_match"] and r["rows_match"] for r in manifest),
         "reason": f"{len(manifest)}/3 primary OOF 다이제스트 일치 (task-2-control / "
                   f"task-5-models-catboost)"},
        {"rule": "primary_only_labels", "ok": set(labels) == {SELECTION_FOLD},
         "reason": f"labels = {sorted(labels)} — R-only 라벨 미로드"},
        {"rule": "r_fold_embargo", "ok": set(labels) == {SELECTION_FOLD},
         "reason": ("r2022/r2023/r2024 라벨 미로드 — 구조 가드(_read_label_fold/"
                    "_assert_primary_only_labels) + --fixture attempted-r-fold-read exit 2")},
        {"rule": "frozen_matches_task2", "ok": t2_frozen_check["ok"],
         "reason": t2_frozen_check["reason"]},
        {"rule": "gate_verdict_rule",
         "ok": gate["verdict"] in ("PRIMARY_PASS", "PRIMARY_REJECT"),
         "reason": (f"ΔBSS {gate['delta_bss']:+.4f} > {GATE_DELTA_BSS_MIN}? "
                    f"{bool(gate['delta_bss'] > GATE_DELTA_BSS_MIN)} | mean_shift "
                    f"{gate['mean_shift']:.6f} <= {GATE_MEAN_SHIFT_MAX}? "
                    f"{bool(gate['mean_shift'] <= GATE_MEAN_SHIFT_MAX)} → {gate['verdict']}")},
    ]
    notion_placeholder = {
        "status": "prepared_pending_notion",
        "cv_table_block": "3b55ed6b-28d5-8135-9341-fa1109df78af",
        "note": ("Task 3 는 실라벨 스코어 CV 결과 — AGENTS.md 따라 📊 로컬 CV 테이블에 행 추가 "
                 "(Notion MCP 가용 시; 부재 시 pending_user)"),
    }
    gate_record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 3 — CatBoost boundary gate (frozen vs active rollback baseline)",
        "task": "aimers9-next-round/task-3-cat-boundary-gate",
        "mode": ("smoke" if smoke else
                 ("fixture-nonpositive-primary" if args.fixture == "nonpositive-primary"
                  else "primary-only")),
        "verdict": gate["verdict"],
        "exit_code": 2 if args.fixture == "nonpositive-primary" else 0,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": gate_config_hash,
        "config_pre_registered": {
            "file": str(gate_config_path), "sha256": _sha256(gate_config_path),
            "written_before_labels_read": True,
            "labels_read_at_registration": False,
        },
        "policy_path": str(POLICY_PATH),
        "policy_config_hash": policy_hash,
        "label_sources": [SELECTION_FOLD],
        "labels_read": True,
        "selection_fold": SELECTION_FOLD,
        "scoring": {
            "formula": "common.score(clip(sigmoid(z_blend + C_LOGIT), 0.30, 0.70), y)",
            "c_logit": C_LOGIT, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
        },
        "gate": gate,
        "r_fold_embargo": {
            "loaded_folds": sorted(labels),
            "guards": ["_read_label_fold", "_assert_primary_only_labels",
                       "fixture attempted-r-fold-read → exit 2"],
            "note": "Task 10 전 R-only 폴드는 reject-only — 이 모드에서 절대 로드 안 함",
        },
        "checks": gate_checks,
        "violations": [],
        "notion": notion_placeholder,
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    }
    t3_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    t3_evidence_path.write_text(json.dumps(gate_record, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    _write_gate_md(gate_record, t3_md_path)
    print(f"\n[catboost_boundary_selector] 게이트 증거 JSON → {t3_evidence_path}", flush=True)
    print(f"[catboost_boundary_selector] 게이트 증거 MD   → {t3_md_path}", flush=True)

    # ── 판정 ──
    if args.fixture == "nonpositive-primary":
        if gate["verdict"] == "PRIMARY_REJECT":
            print(f"[fixture nonpositive-primary] PASS — 강제 ΔBSS<=0 ({gate['delta_bss']:+.4f}) "
                  f"주입에서 PRIMARY_REJECT 경로 실데이터 증명 (exit 2)", flush=True)
            return 2
        print("[fixture nonpositive-primary] FAIL — PRIMARY_REJECT 가 아님", file=sys.stderr)
        return 1
    ok = (
        len(rows) == EXPECTED_GRID_ROWS
        and frozen["n_frozen"] == 1
        and set(labels) == {SELECTION_FOLD}
        and all(r["digest_match"] and r["rows_match"] for r in manifest)
        and not policy_violations
    )
    verdict = "PASS" if ok else "FAIL"
    print(f"\n[{'--smoke' if smoke else '--primary-only'}] {verdict} — 그리드 {len(rows)}행 "
          f"({EXPECTED_GRID_ROWS} 필요) | 동결 {frozen['n_frozen']}개 (1 필요) | "
          f"라벨 {sorted(labels)} (primary 만 필요) | 출처 {len(manifest)}/3 | "
          f"정책 위반 {len(policy_violations)}", flush=True)
    print(f"[gate] {gate['verdict']} — ΔBSS {gate['delta_bss']:+.4f} (>{GATE_DELTA_BSS_MIN}: "
          f"{bool(gate['delta_bss'] > GATE_DELTA_BSS_MIN)}) | mean_shift "
          f"{gate['mean_shift']:.6f} (<={GATE_MEAN_SHIFT_MAX}: "
          f"{bool(gate['mean_shift'] <= GATE_MEAN_SHIFT_MAX)}) | frozen "
          f"{gate['frozen']['candidate_id']} vs baseline {gate['baseline']['candidate_id']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
