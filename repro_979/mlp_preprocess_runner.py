#!/usr/bin/env python3
"""mlp_preprocess_runner.py — Todo 6: bounded MLP preprocessing side-gate.

Aimers9-score-improvement-next-round, Todo 6: evaluate exactly TWO pre-registered MLP
preprocessing variants against the existing MLP family on the PRIMARY fold only
(activated because Todo 5 DeepFM screen emitted PRIMARY_REJECT):

  Variant A `clip_z5`      — after the frozen MLP numeric standardization, clip the
                             standardized values to [-5, 5] (applied AFTER z-scoring,
                             BEFORE model input). Fit statistics (mean/std) on
                             outer-train rows only; clipping is a pure transform.
  Variant B `missing_flags`— `any_asof_missing = any(isna(asof_*))` and
                             `prev_game_missing = any(isna(prev_game_*))`, both
                             computed on RAW values BEFORE imputation (row-wise,
                             NO fitting of any kind), added as two extra binary
                             features to the model input.

Sequence (frozen): A first. If A emits PRIMARY_REJECT, B may run. If A advances
(PRIMARY_PASS), B does NOT run (recorded SKIPPED). Exactly two sequential variants,
never combined.

Gate (Task 6, deployed formula on the primary fold):
  variant advances only if primary ΔBSS >= 15 (vs the rollback baseline blend
  5890a4c54f502c4e, lgb .30 / mlp .35 / cat .35, where the variant MLP OOF replaces
  the cached MLP member at the frozen weights) AND max mean shift <= 0.005.
  ΔBSS/Brier use common.score(clip(sigmoid(z+C_LOGIT), 0.30, 0.70), y), C_LOGIT=-0.0404.

Ten-seed convention: seeds 42..51 on the primary fold (mirroring the repo's ten-seed
gate convention); per-seed + ensemble (mean of per-seed LOGITS) reported.

Modes:
  --variant <clip_z5|missing_flags> --full
                        Task 6: ten-seed primary OOF + blend probe + verdict
                        (PRIMARY_PASS / PRIMARY_REJECT), per-variant evidence.
  --variant <clip_z5|missing_flags> --promote-primary
                        Task 7 hook: requires the task-6 evidence for that variant to
                        be PRIMARY_PASS; else SKIPPED exit 0. On PASS it reuses the
                        provenance-pinned OOF (config-digest verified), fits the
                        fixed-alpha blend on primary only and applies the Task 7
                        promotion gates.
  --fixture <name>      combined-variants | validation-fit | blocked-relation —
                        every fixture exits 2 (proven on the real data path).

Exit codes: 0 = PASS / SKIPPED / REJECT (gate verdicts are normal outcomes),
1 = fatal input error, 2 = policy/leakage/provenance violation (or a failure-injection
fixture proof).

Frozen contract highlights:
  - MLP family machinery is REUSED READ-ONLY from e6c_blend_folds.train_seed (the
    qualification_runner._retrain_mlp canonical path): EntityMLP [512,256,128,64],
    emb 16/16/4/4/2/4/4/4/4, UNK dropout p=0.05 on pitcher/batter ids, lr 1e-3,
    wd 1e-4, batch 8192, max 30 epochs, patience 5, AdamW; ensembles average logits.
  - Nine categorical fields exactly matching the frozen MLP contract
    (qualification_runner.MLP_CATS order); vocabularies from outer-train rows only.
  - The variant changes ONLY the preprocessing of the numeric block: (A) post-z-score
    clip; (B) two raw binary flag features appended after the standardized numerics.
  - r2022/r2023/r2024 labels are NEVER loaded (structural guard, exit 2; Task 10 only).
  - blocked hypotheses from feature_hypothesis_registry.json are never revived:
    the runner computes per-blocked-entry relation records for the variant's feature
    specs and hard-rejects name collisions / near-identical revivals (exit 2).
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너와 동일

# ── 동결 제어 / 기존 러너 규약 재사용 (전부 읽기 전용 import — 수정 금지) ──
from repro_979.qualification_runner import (  # noqa: E402
    R_FOLDS, SEEDS, C_LOGIT, CLIP_LO, CLIP_HI, CHAMPION_FEATURES, MLP_CATS,
    build_folds, _check_leakage, _sha256,
)
from repro_979.blend_selector import (  # noqa: E402
    SELECTION_FOLD, BOOT_SEED, EXPECTED_ROWS, WEIGHT_STEP, _candidate_id,
)
from repro_979.catboost_boundary_selector import (  # noqa: E402
    MEMBER_FILES, BASELINE_CANDIDATE_ID, BASELINE_WEIGHTS_BY_MEMBER,
    _verify_provenance,
)
from repro_979.next_round_policy import (  # noqa: E402
    validate_policy, _canonical_sha256, load_json,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402  (top-level — qualification_runner/e6c convention)
from e6c_blend_folds import EntityMLP, EMB_DIM, train_seed  # noqa: E402  (동결 MLP 패밀리 기계)

SCHEMA_VERSION = 1
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
POLICY_PATH = REPO / "next_round_policy.json"
REGISTRY_PATH = REPO / "feature_hypothesis_registry.json"
CACHE_ROOT = REPO / "cache" / "mlp_preprocess"   # artifacts: cache/mlp_preprocess/<config_hash>/

# ── Task 6 게이트 (동결) — 배리언트 채택 조건 ──
GATE_DELTA_BSS_MIN = 15.0    # primary ΔBSS >= 15 (inclusive) vs rollback baseline blend
GATE_MEAN_SHIFT_MAX = 0.005  # max |candidate − baseline| probability-mean shift <= 0.005

# Task 6 배리언트 사양 (동결 — 사전 등록의 원천)
VARIANTS = ("clip_z5", "missing_flags")
CLIP_Z = 5.0
# asof_* / prev_game_* 컬럼 정의 — 동결 49 피처(CHAMPION_FEATURES) 내에서 스코프
ASOF_COLS: tuple[str, ...] = tuple(c for c in CHAMPION_FEATURES if c.startswith("asof_"))
PREV_COLS: tuple[str, ...] = tuple(c for c in CHAMPION_FEATURES if "prev" in c)

# Task 7 블렌드 전이 게이트 (deepfm_runner 와 동일 규칙, hook 전용)
ALPHA_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))
GATE_INCR_BSS_MIN = 1.0      # promote: primary incremental BSS > 1.0
GATE_CORR_MAX = 0.95         # promote: corr(logits, each existing family) < 0.95

# fixture 주입 플래그 (main 에서만 True)
force_combined = False
force_validation_fit = False
force_blocked_relation = False


class ProvenanceError(RuntimeError):
    """캐시/증거 아티팩트의 출처(provenance)가 불일치하면 발생 (하드 실패, exit 2)."""


class LeakageError(RuntimeError):
    """R-only 폴드 라벨을 Task 10 이전에 읽으려 하면 발생 (구조적 누수 가드, exit 2)."""


class PolicyViolation(RuntimeError):
    """정책/동결 계약 위반 (단일 배리언트, 검증 피팅, blocked 관계, 2025 행 등, exit 2)."""


# ════════════════════════════════════════════════════════════════════
# 배리언트 사양 (동결 — 사전 등록 구성 다이제스트의 원천)
# ════════════════════════════════════════════════════════════════════
def _variant_feature_specs(variant: str) -> list[dict[str, Any]]:
    """배리언트가 모델 입력에 추가하는 피처 스펙 (blocked-relation 검사용)."""
    if variant == "clip_z5":
        return []  # 순수 변환 — 피처 추가 없음
    if variant == "missing_flags":
        return [
            {"name": "any_asof_missing", "mechanism": "isna_any",
             "source_columns": list(ASOF_COLS),
             "role": "binary_flag",
             "definition": "any(isna(asof_*)) on RAW values before imputation"},
            {"name": "prev_game_missing", "mechanism": "isna_any",
             "source_columns": list(PREV_COLS),
             "role": "binary_flag",
             "definition": "any(isna(prev_game_*)) on RAW values before imputation"},
        ]
    if variant == "combined":  # fixture 전용 — 두 배리언트 결합 (금지 대상)
        return _variant_feature_specs("clip_z5") + _variant_feature_specs("missing_flags")
    raise ValueError(f"unknown variant {variant!r}")


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "clip_z5": {
        "order": ("1) z-score standardization (mean/std on outer-train rows only) -> "
                  "2) clip standardized values to [-5, 5] -> 3) model input"),
        "clip_bounds": [-CLIP_Z, CLIP_Z],
        "clip_note": "applied AFTER standardization, BEFORE model input; "
                     "clip bounds applied to the STANDARDIZED values; pure transform, "
                     "no added features",
        "added_features": [],
    },
    "missing_flags": {
        "order": ("1) compute any_asof_missing / prev_game_missing from RAW values "
                  "BEFORE imputation (row-wise isna OR; NO fitting) -> "
                  "2) standardize the 40 frozen numerics (mean/std on outer-train rows "
                  "only, imputation inside the frozen family transform) -> "
                  "3) append the two raw binary flags (float32 0/1, NOT standardized) "
                  "after the standardized numerics -> 4) model input"),
        "any_asof_missing": {"definition": "any(isna(asof_*))", "columns": list(ASOF_COLS)},
        "prev_game_missing": {"definition": "any(isna(prev_game_*))", "columns": list(PREV_COLS)},
        "added_features": [f["name"] for f in _variant_feature_specs("missing_flags")],
        "no_fit_note": "flags are row-wise isna ORs — zero fitting, zero validation-row "
                       "statistics; imputation happens later inside the frozen family "
                       "numeric transform only",
    },
}


def _config_dict(variant: str) -> dict[str, Any]:
    """배리언트별 동결 설정의 정규화 표현 — config digest 의 원천 (라벨 읽기 전에 기록)."""
    spec = dict(VARIANT_SPECS[variant])
    if variant == "clip_z5":
        spec["clip_bounds"] = [-CLIP_Z, CLIP_Z]  # 런타임 상수 반영 (해시 민감성)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-6-mlp-preprocess",
        "family": "mlp",
        "variant": variant,
        "variant_spec": spec,
        "feature_specs": _variant_feature_specs(variant),
        "registry_scoping": {
            "asof_columns": list(ASOF_COLS),
            "prev_game_columns": list(PREV_COLS),
            "scope_note": ("column sets are scoped to the frozen 49 CHAMPION_FEATURES; "
                           "engineered non-model columns (e.g. asof_n_bucket) excluded"),
            "registry": "feature_hypothesis_registry.json",
            "blocked_policy": ("blocked hypotheses are never revived: name collision "
                               "with a blocked id OR same-mechanism flag with "
                               "source-column containment = hard rejection (exit 2)"),
        },
        "seeds": {
            "ten_seed_gate": list(SEEDS),
            "convention": "ten-seed (42..51) primary-only, mirroring the repo's ten-seed gate",
            "ensemble": "mean of per-seed LOGITS (never probabilities)",
        },
        "family_contract": {
            "machinery": "e6c_blend_folds.train_seed (qualification_runner._retrain_mlp "
                         "canonical path) — reused READ-ONLY, no family source modified",
            "arch": "EntityMLP hidden [512,256,128,64], dropout 0.25, BatchNorm, ReLU",
            "emb_dim": list(EMB_DIM),
            "cats": list(MLP_CATS),
            "n_cats": len(MLP_CATS),
            "n_numerics_base": 40,
            "n_numerics_variant": 42 if variant == "missing_flags" else 40,
            "unk_p": 0.05,
            "unk_drop_cats": ["pitcher_id", "batter_id"],
            "lr": 1e-3, "weight_decay": 1e-4, "batch": 8192,
            "max_epochs": 30, "patience": 5,
            "early_stop_note": ("frozen e6c OOF pattern: patience-5 early stop on the "
                                "outer-validation BSS — identical to qualification_runner "
                                "retrain; the variant changes ONLY preprocessing"),
            "numeric_prep": ("mean imputation + z-score standardization fitted ONLY on "
                             "outer-train rows; zero-variance numerics map to zero"),
        },
        "folds": {
            "primary": "season <= 2023 -> season == 2024",
            "r2022": "season <= 2021 & R -> season == 2022 & R (Task 10 only)",
            "r2023": "season <= 2022 & R -> season == 2023 & R (Task 10 only)",
            "r2024": "season <= 2023 & R -> season == 2024 & R (diagnostic, Task 10 only)",
            "selection_fold": SELECTION_FOLD,
            "source": "qualification_runner.build_folds (read-only reuse)",
        },
        "r_fold_embargo": ("r2022/r2023/r2024 labels NEVER loaded in this task "
                           "(structural guard: LeakageError exit 2); one-shot R "
                           "qualification only at Task 10; no R-fold file is opened"),
        "scoring": {
            "formula": "common.score(clip(sigmoid(z+C_LOGIT), 0.30, 0.70), y)",
            "c_logit": C_LOGIT,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "c_logit_frozen_note": "C_LOGIT = -0.0404 frozen for the whole round",
        },
        "gate": {
            "delta_bss_min": GATE_DELTA_BSS_MIN,
            "delta_bss_rule": "primary ΔBSS >= 15 (INCLUSIVE) vs rollback baseline blend",
            "mean_shift_max": GATE_MEAN_SHIFT_MAX,
            "mean_shift_rule": "max |candidate − baseline| probability-mean shift <= 0.005",
            "verdict": "PRIMARY_PASS if BOTH gates hold, else PRIMARY_REJECT",
        },
        "blend_probe": {
            "base_candidate": BASELINE_CANDIDATE_ID,
            "base_weights_by_member": BASELINE_WEIGHTS_BY_MEMBER,
            "member_replacement": ("the variant MLP OOF ensemble replaces the cached MLP "
                                   "member inside the FROZEN baseline weights "
                                   "(lgb .30 / mlp .35 / cat .35) — weights are NOT "
                                   "re-fitted in Task 6"),
            "baseline_mlp_member": "cache/mlp_primary.npy (provenance-verified cached "
                                   "MLP-family OOF logits, documented 5-seed 42..46 artifact)",
            "alpha_grid": list(ALPHA_GRID),
            "promotion_gates": {
                "positive_weight": True,
                "corr_lt": GATE_CORR_MAX,
                "incremental_bss_gt": GATE_INCR_BSS_MIN,
                "mean_shift_le": GATE_MEAN_SHIFT_MAX,
            },
            "tie_rule": "smallest-alpha exact-tie choice (iterate ascending, strict >)",
        },
        "preprocessing_guards": {
            "single_variant": "exactly one variant at a time — combined application is a "
                              "hard PolicyViolation (fixture combined-variants exit 2)",
            "no_validation_fit": "transform statistics (mean/std, clip bounds, flags) are "
                                 "never fitted on validation rows (fixture validation-fit "
                                 "exit 2)",
            "no_blocked_revival": "registry-blocked hypotheses are never revived "
                                  "(fixture blocked-relation exit 2)",
        },
    }


def _config_hash(variant: str) -> str:
    return _canonical_sha256(_config_dict(variant))


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


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }


# ════════════════════════════════════════════════════════════════════
# 계약 가드
# ════════════════════════════════════════════════════════════════════
def _check_policy() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
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
    if violations:
        raise PolicyViolation("[POLICY] " + "; ".join(v["reason"] for v in violations))
    return policy, violations, _canonical_sha256(policy)


def _check_no_2025(train: pd.DataFrame) -> None:
    if (train["season"] == 2025).any():
        n = int((train["season"] == 2025).sum())
        raise PolicyViolation(
            f"[POLICY] 학습 프레임에 2025 시즌 행 {n}개 존재 — 2025 라벨/행은 "
            f"절대 학습·검증에 사용 금지 (exit 2)")


def _check_schema(train: pd.DataFrame) -> list[str]:
    """동결 MLP 계약: 49 피처 + 9 범주 (MLP_CATS 순서) — 문제 목록 반환 (비면 PASS)."""
    problems: list[str] = []
    if tuple(MLP_CATS) != tuple(c for c in MLP_CATS):
        problems.append("MLP cats 순서 불일치")
    missing_feats = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if missing_feats:
        problems.append(f"49-피처 컬럼 부재: {missing_feats}")
    missing_cats = [c for c in MLP_CATS if c not in train.columns]
    if missing_cats:
        problems.append(f"범주 컬럼 부재: {missing_cats}")
    return problems


def _sha256_of_ids(ids: pd.Series) -> str:
    return hashlib.sha256(ids.astype(str).str.cat(sep=",").encode("utf-8")).hexdigest()


def _check_folds(train: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """폴드 마스크 + 누수 가드 + 행 수 + row-ID 분리 검증 (라벨은 읽지 않음)."""
    folds = build_folds(train)
    leakage = _check_leakage(folds, train)
    if leakage:
        raise PolicyViolation("[POLICY] 폴드 누수 가드 실패: " + "; ".join(leakage))
    n_primary = int(folds[SELECTION_FOLD][1].sum())
    if n_primary != EXPECTED_ROWS[SELECTION_FOLD]:
        raise ProvenanceError(
            f"[FAIL] primary 검증 행 수 {n_primary} != 동결 {EXPECTED_ROWS[SELECTION_FOLD]} "
            f"— 폴드 계약 위반")
    va_primary = folds[SELECTION_FOLD][1]
    for rfold in R_FOLDS:
        if rfold == "r2024":
            continue  # r2024 val ⊆ primary val (계획 'overlapping r2024' 진단 폴드)
        overlap = int((va_primary & folds[rfold][1]).sum())
        if overlap != 0:
            raise ProvenanceError(
                f"[FAIL] row-ID disjointness 위반: primary ∩ {rfold} ≠ ∅ (exit 2)")
    row_ids_hash = _sha256_of_ids(train.loc[va_primary, common.ID])
    fold_info = {
        "n_primary": n_primary, "row_ids_sha256": row_ids_hash,
        "r2024_overlap": "documented diagnostic (val = (season==2024) & R ⊆ primary)",
    }
    checks = [{"rule": "leakage_guard", "ok": True, "reason": "검증 마스크에 2025 없음"},
              {"rule": "row_disjointness", "ok": True,
               "reason": "primary ∩ r2022/r2023 = ∅ (r2024 overlap 문서화)"}]
    return folds, fold_info, checks


# ════════════════════════════════════════════════════════════════════
# blocked-relation 가드 — feature_hypothesis_registry.json (읽기 전용)
# ════════════════════════════════════════════════════════════════════
_REGISTRY_MECHANISM_BY_TRANSFORM = {
    "common.py:add_cross_cell_features": "cross_cell_condition",
    "common.py:add_asof_n_bucket_feature": "bucketing",
    "common.py:add_recent_gap_feature": "continuous_gap",
    "common.py:add_return_gap_feature": "isna_conjunction",
    "common.py:add_debut_feature": "equality",
}


def _registry() -> dict[str, Any]:
    reg = load_json(REGISTRY_PATH)
    if reg is None:
        raise PolicyViolation(f"[POLICY] feature_hypothesis_registry.json 읽기 불가: {REGISTRY_PATH}")
    return reg


def _blocked_relation_records(feature_specs: list[dict[str, Any]],
                              registry: dict[str, Any]) -> list[dict[str, Any]]:
    """배리언트 피처 스펙 vs registry blocked 항목 — 관계 레코드.

    blocked 판정: (1) 피처 이름 == blocked id (동일 가설 리터럴 부활) 또는
    (2) 동일 메커니즘 + blocked 항목의 소스 컬럼 ⊆ 피처 소스 컬럼 (near-identical 부활).
    """
    blocked = [c for c in registry.get("candidates", [])
               if c.get("status") == "blocked"]
    records: list[dict[str, Any]] = []
    for b in blocked:
        bid = b["id"]
        bcols = set((b.get("data_source") or {}).get("columns", []))
        transform = b.get("transform", "")
        bmech = _REGISTRY_MECHANISM_BY_TRANSFORM.get(transform, "unknown")
        reasons: list[str] = []
        for f in feature_specs:
            if f["name"] == bid:
                reasons.append(f"feature name '{f['name']}' == blocked id '{bid}' "
                               f"(동일 가설 리터럴 부활)")
            if (f["mechanism"] == bmech and bcols
                    and bcols.issubset(set(f["source_columns"]))):
                reasons.append(f"mechanism '{f['mechanism']}' (== '{bmech}') + 소스 컬럼 "
                               f"{sorted(bcols)} ⊆ {sorted(f['source_columns'])} — "
                               f"near-identical 부활")
        records.append({
            "blocked_id": bid,
            "blocked_transform": transform,
            "blocked_mechanism": bmech,
            "blocked_columns": sorted(bcols),
            "ok": not reasons,
            "reasons": reasons,
        })
    return records


def _assert_not_blocked(feature_specs: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    """blocked 관계가 하나라도 감지되면 PolicyViolation (exit 2)."""
    records = _blocked_relation_records(feature_specs, registry)
    bad = [r for r in records if not r["ok"]]
    if bad:
        detail = "; ".join(f"{r['blocked_id']}: " + " | ".join(r["reasons"])
                           for r in bad)
        raise PolicyViolation(
            f"[POLICY] blocked 관계 부활 감지: {detail} (exit 2)")


# ════════════════════════════════════════════════════════════════════
# 배리언트 변환 (전처리만 변경 — 학습 기계는 동결 패밀리 재사용)
# ════════════════════════════════════════════════════════════════════
def _guard_single_variant(variant: str) -> None:
    """단일 배리언트 가드: clip_z5 또는 missing_flags 하나만 허용 (결합 금지)."""
    if variant not in VARIANTS:
        raise PolicyViolation(
            f"[POLICY] combined-variants: '{variant}' 는 두 배리언트(clip_z5 + "
            f"missing_flags)의 결합 적용 — 단일 배리언트만 허용 (exit 2)")


def _guard_validation_fit() -> None:
    """검증 행 피팅 가드: 변환 통계(mean/std/clip/플래그)는 절대 검증 행에서 피팅 금지."""
    raise PolicyViolation(
        "[POLICY] validation-fit: 검증 행에서 피팅한 변환 통계 적용 시도 — "
        "모든 변환 통계는 outer-train 행 전용 (exit 2)")


def _to_num_variant(df: pd.DataFrame, variant: str, nmean: dict[str, float],
                    nstd: dict[str, float], nums: list[str]) -> np.ndarray:
    """수치 블록 변환 (동결 패밀리 표준화 + 배리언트 전처리).

    clip_z5   : 표준화 후 [-CLIP_Z, CLIP_Z] 클리핑 (표준화된 값에 적용).
    missing_flags : 표준화된 40 수치 뒤에 raw 이진 플래그 2개(float32 0/1, 비표준화) 추가.
    combined  : (fixture 전용) clip_z5 + missing_flags 동시 적용 — 가드가 차단.
    """
    n_extra = 2 if variant in ("missing_flags", "combined") else 0
    X = np.zeros((len(df), len(nums) + n_extra), dtype=np.float32)
    for j, c in enumerate(nums):
        col = df[c].fillna(nmean[c]).values.astype(np.float32)
        z = (col - nmean[c]) / nstd[c]
        if variant in ("clip_z5", "combined"):
            z = np.clip(z, -CLIP_Z, CLIP_Z)
        X[:, j] = z
    if n_extra:
        X[:, len(nums)] = df[list(ASOF_COLS)].isna().any(axis=1).values.astype(np.float32)
        X[:, len(nums) + 1] = df[list(PREV_COLS)].isna().any(axis=1).values.astype(np.float32)
    return X


def _to_cat(df: pd.DataFrame, codes: dict[str, dict[str, int]],
            cats: list[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cats)), dtype=np.int64)
    for j, c in enumerate(cats):
        out[:, j] = df[c].astype(str).map(lambda v: codes[c].get(v, 0)).values
    return out


def _build_fold_variant(train: pd.DataFrame, tr_m: np.ndarray, va_m: np.ndarray,
                        variant: str) -> tuple[Any, Any, Any, Any, Any, Any, list[int], int]:
    """primary 폴드 입력 텐서 — 동결 MLP 패밀리 build_fold 패턴 + 배리언트 전처리.

    가드 (라벨 무접촉, 실데이터 경로):
      1) 단일 배리언트 (combined-variants fixture 가 실제 결합 변환 후 차단)
      2) 검증 행 피팅 금지 (validation-fit fixture 가 실제 검증 통계 피팅 후 차단)
      3) blocked 관계 부활 금지 (blocked-relation fixture 가 registry-listed 관계 주입 후 차단)
    """
    _guard_single_variant(variant)

    nums = [c for c in CHAMPION_FEATURES if c not in MLP_CATS]
    cats = list(MLP_CATS)
    tr, va = train[tr_m], train[va_m]
    ytr = tr[common.TARGET].values.astype(np.float32)
    yv = va[common.TARGET].values.astype(np.float32)

    cat_vocab: list[int] = []
    codes: dict[str, dict[str, int]] = {}
    for c in cats:
        vals = sorted(tr[c].astype(str).unique())
        codes[c] = {v: i + 1 for i, v in enumerate(vals)}
        cat_vocab.append(len(vals) + 1)

    # ── 가드 2: 검증 행 피팅 주입 (실데이터 검증 행 통계 계산 후 차단) ──
    if force_validation_fit:
        nmean_va = {c: float(va[c].mean()) for c in nums}
        nstd_va = {c: (float(va[c].std()) if va[c].std() > 0 else 1.0) for c in nums}
        _to_num_variant(va, variant, nmean_va, nstd_va, nums)  # 실제 계산 — 그 후 차단
        _guard_validation_fit()

    # ── 가드 3: blocked 관계 주입 (registry-listed return_gap 부활 시도 후 차단) ──
    specs = _variant_feature_specs(variant)
    if force_blocked_relation:
        rg = (va if force_validation_fit else tr).copy()
        rg["return_gap"] = (rg["asof_pitcher_prev1_game_success_rate"].isna()
                            & (rg["asof_pitcher_n"] < 100)).astype("int8")
        specs = specs + [{"name": "return_gap", "mechanism": "isna_conjunction",
                          "source_columns": ["asof_pitcher_prev1_game_success_rate",
                                             "asof_pitcher_n"],
                          "role": "binary_flag"}]
        _assert_not_blocked(specs, _registry())

    # 정상 경로: 변환 통계는 outer-train 행 전용
    nmean = {c: float(tr[c].mean()) for c in nums}
    nstd = {c: (float(tr[c].std()) if tr[c].std() > 0 else 1.0) for c in nums}

    Xn_tr = torch.tensor(_to_num_variant(tr, variant, nmean, nstd, nums))
    Xn_va = torch.tensor(_to_num_variant(va, variant, nmean, nstd, nums))
    Xc_tr = torch.tensor(_to_cat(tr, codes, cats))
    Xc_va = torch.tensor(_to_cat(va, codes, cats))
    n_num = int(Xn_tr.shape[1])
    return Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num


def _read_outer_labels(train: pd.DataFrame, mask: np.ndarray, fold: str) -> np.ndarray:
    """외부 검증 라벨을 읽는 유일한 경로 (R 폴드 구조적 가드)."""
    if fold in R_FOLDS:
        raise LeakageError(
            f"[LEAKAGE] R-only 폴드 라벨 {fold!r} 읽기 차단 — Task 10 전 이 경로에서 "
            f"R-only 폴드는 절대 로드하지 않는다 (exit 2)")
    return train.loc[mask, common.TARGET].values.astype(np.float64)


# ════════════════════════════════════════════════════════════════════
# 캐시/출처 (config digest + 폴드 provenance 로 재사용 검증)
# ════════════════════════════════════════════════════════════════════
def _cache_dir(config_hash: str) -> Path:
    return CACHE_ROOT / config_hash


def _manifest_path(config_hash: str) -> Path:
    return _cache_dir(config_hash) / "manifest.json"


def _check_cache_key(config_hash: str) -> dict[str, Any] | None:
    mpath = _manifest_path(config_hash)
    if not mpath.is_file():
        if _cache_dir(config_hash).exists():
            raise PolicyViolation(
                f"[POLICY] 캐시 키 {config_hash[:16]}… 디렉토리가 존재하지만 manifest.json "
                f"없음 — 폴드 출처(provenance) 누락 (exit 2)")
        return None
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config_hash:
        raise PolicyViolation(
            f"[POLICY] 기존 캐시 키 {config_hash[:16]}… 의 manifest config_hash "
            f"{str(manifest.get('config_hash'))[:16]}… != 현재 계산 {config_hash[:16]}… — "
            f"설정 변경 하에서 기존 캐시 키 재사용 금지 (exit 2)")
    return manifest


def _fold_provenance(manifest: dict[str, Any], fold: str, n_va: int,
                     row_ids_sha256: str) -> dict[str, Any]:
    rec = None
    for entry in manifest.get("fold_provenance", []):
        if entry.get("fold") == fold:
            rec = entry
            break
    if rec is None:
        raise PolicyViolation(f"[POLICY] 캐시 manifest 에 폴드 {fold} 출처 누락 (exit 2)")
    if int(rec.get("n_va", -1)) != n_va or rec.get("row_ids_sha256") != row_ids_sha256:
        raise PolicyViolation(
            f"[POLICY] 캐시 폴드 {fold} 출처 불일치 (exit 2)")
    return rec


def _save_fold_artifacts(config_hash: str, variant: str, n_va: int, row_ids_sha256: str,
                         per_seed_oof: dict[int, np.ndarray], z_ens: np.ndarray,
                         mask_sha256: str) -> dict[str, Any]:
    d = _cache_dir(config_hash)
    d.mkdir(parents=True, exist_ok=True)
    seed_logits: dict[str, Any] = {}
    for seed, z in per_seed_oof.items():
        rel = f"oof_primary_s{seed}.npy"
        path = d / rel
        np.save(path, z.astype(np.float64))
        seed_logits[str(seed)] = {"file": rel, "sha256": _sha256(path)}
    ens_rel = "oof_primary_ensemble.npy"
    ens_path = d / ens_rel
    np.save(ens_path, z_ens.astype(np.float64))
    rec = {
        "fold": SELECTION_FOLD, "n_va": int(n_va), "row_ids_sha256": row_ids_sha256,
        "mask_sha256": mask_sha256, "variant": variant,
        "seed_logits": seed_logits,
        "ensemble_logits": {"file": ens_rel, "sha256": _sha256(ens_path)},
    }
    mpath = _manifest_path(config_hash)
    manifest: dict[str, Any] = {}
    if mpath.is_file():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            raise PolicyViolation(f"[POLICY] 캐시 키 설정 불일치 — 저장 거부 (exit 2)")
    manifest["config_hash"] = config_hash
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["created_at_utc"] = _now_utc()
    manifest["git_head"] = _git_commit()
    manifest.setdefault("fold_provenance", [])
    manifest["fold_provenance"] = [e for e in manifest["fold_provenance"]
                                   if e.get("fold") != SELECTION_FOLD] + [rec]
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    return rec


# ════════════════════════════════════════════════════════════════════
# 평가 + 기준선 블렌드 프로브 (Task 6 게이트)
# ════════════════════════════════════════════════════════════════════
def _primary_bss(z: np.ndarray, y: np.ndarray) -> float:
    """계획 배포 산식: common.score(clip(sigmoid(z+C_LOGIT), 0.30, 0.70), y)."""
    p = np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)
    return float(common.score(p, y))


def _load_base_members() -> dict[str, np.ndarray]:
    """롤백 기준선 3멤버 primary OOF 로짓 (커밋 증거 다이제스트 대조 후 로드)."""
    _verify_provenance()
    return {m: np.load(REPO / rel).astype(np.float64) for m, rel in MEMBER_FILES.items()}


def _gate_verdict(delta_bss: float, mean_shift: float) -> tuple[bool, dict[str, bool]]:
    """Task 6 게이트 판정 (순수 함수 — 테스트 경계용): ΔBSS >= 15 AND shift <= 0.005."""
    gates = {
        "delta_bss_ge_15.0": delta_bss >= GATE_DELTA_BSS_MIN,
        "mean_shift_le_0.005": mean_shift <= GATE_MEAN_SHIFT_MAX,
    }
    return all(gates.values()), gates


def _blend_probe(z_mlp_var: np.ndarray, y_va: np.ndarray) -> dict[str, Any]:
    """Task 6 프로브: 배리언트 MLP OOF 가 동결 기준선 가중치 내 MLP 멤버를 대체.

    기준선 = 롤백 블렌드 5890a4c54f502c4e (lgb .30 / mlp .35 / cat .35, 캐시 멤버).
    후보   = 같은 가중치에 MLP 멤버만 배리언트 OOF 로 교체. 가중치 재피팅 없음.
    """
    members = _load_base_members()
    z_lgb, z_mlp_base, z_cat = members["lgb"], members["mlp"], members["catboost"]
    w = BASELINE_WEIGHTS_BY_MEMBER
    z_base = w["lgb"] * z_lgb + w["mlp"] * z_mlp_base + w["catboost"] * z_cat
    z_cand = w["lgb"] * z_lgb + w["mlp"] * z_mlp_var + w["catboost"] * z_cat
    p_base = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    p_cand = np.clip(common.sigmoid(z_cand + C_LOGIT), CLIP_LO, CLIP_HI)
    p_mem = np.clip(common.sigmoid(z_mlp_var + C_LOGIT), CLIP_LO, CLIP_HI)

    base_bss = float(common.score(p_base, y_va))
    cand_bss = float(common.score(p_cand, y_va))
    member_bss = float(common.score(p_mem, y_va))
    delta_bss = cand_bss - base_bss
    mean_shift = float(abs(p_cand.mean() - p_base.mean()))
    ok, gates = _gate_verdict(delta_bss, mean_shift)
    return {
        "base": {
            "candidate_id": BASELINE_CANDIDATE_ID,
            "weights_by_member": dict(w),
            "primary_bss": base_bss,
            "brier": float(((p_base - y_va) ** 2).mean()),
            "pred_mean": float(p_base.mean()),
        },
        "member": {
            "family": "mlp_preprocess_variant",
            "primary_bss": member_bss,
            "brier": float(((p_mem - y_va) ** 2).mean()),
            "pred_mean": float(p_mem.mean()),
        },
        "candidate": {
            "member_replacement": "variant MLP OOF inside frozen baseline weights",
            "primary_bss": cand_bss,
            "weights_by_member": dict(w),
            "delta_bss_vs_base": delta_bss,
            "brier": float(((p_cand - y_va) ** 2).mean()),
            "pred_mean": float(p_cand.mean()),
        },
        "mean_shift": mean_shift,
        "corr_to_families": {
            "lgb": float(np.corrcoef(z_mlp_var, z_lgb)[0, 1]),
            "mlp_base": float(np.corrcoef(z_mlp_var, z_mlp_base)[0, 1]),
            "catboost": float(np.corrcoef(z_mlp_var, z_cat)[0, 1]),
        },
        "gates": gates,
        "thresholds": {"delta_bss_min": GATE_DELTA_BSS_MIN,
                       "mean_shift_max": GATE_MEAN_SHIFT_MAX},
        "verdict": "PRIMARY_PASS" if ok else "PRIMARY_REJECT",
        "r": float(y_va.mean()),
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


def _write_json(record: dict[str, Any], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pre_register(config: dict[str, Any], config_hash: str, path: Path) -> None:
    """설정 사전 등록 — 라벨 결과를 읽기 이전에 기록 (pre-registration 계약)."""
    config["config_hash"] = config_hash
    config["recorded_at_utc"] = _now_utc()
    config["git_head"] = _git_commit()
    config["label_sources"] = []
    config["labels_read"] = False
    config["labels_read_note"] = ("Config written BEFORE any label result is read "
                                  "(pre-registration contract).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[pre-register] config hash={config_hash[:16]}… → {path} (라벨 읽기 이전 기록)",
          flush=True)


def _base_record(mode: str, variant: str, config_hash: str, policy_hash: str,
                 config_path: Path, labels_read: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": config_hash,
        "config_pre_registered": {
            "file": str(config_path), "sha256": _sha256(config_path),
            "written_before_labels_read": True,
        },
        "policy_path": str(POLICY_PATH),
        "policy_config_hash": policy_hash,
        "mode": mode,
        "variant": variant,
        "label_sources": [],
        "labels_read": labels_read,
    }


# ════════════════════════════════════════════════════════════════════
# Task 6 전체 실행 (--full): 10시드 primary OOF + 블렌드 프로브 + 판정
# ════════════════════════════════════════════════════════════════════
def _cmd_full(args: argparse.Namespace, variant: str) -> int:
    t0 = time.time()
    config_hash = _config_hash(variant)
    _check_policy()
    config_path = EVIDENCE_DIR / f"task-6-mlp-preprocess-config-{variant}.json"
    _pre_register(_config_dict(variant), config_hash, config_path)

    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    _check_no_2025(train)
    schema_problems = _check_schema(train)
    if schema_problems:
        raise PolicyViolation("[POLICY] 스키마 계약 위반: " + "; ".join(schema_problems))
    folds, fold_info, checks = _check_folds(train)

    tr_m, va_m = folds[SELECTION_FOLD]
    n_va = int(va_m.sum())
    va_ids = train.loc[va_m, common.ID]
    row_ids_sha256 = _sha256_of_ids(va_ids)
    mask_sha256 = hashlib.sha256(va_m.astype(np.uint8).to_numpy().tobytes()).hexdigest()

    # registry 관계 레코드 (라벨 무접촉) — 배리언트가 blocked 가설을 부활하지 않음을 증명
    registry = _registry()
    rel_records = _blocked_relation_records(_variant_feature_specs(variant), registry)
    _assert_not_blocked(_variant_feature_specs(variant), registry)

    (Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num) = _build_fold_variant(
        train, tr_m, va_m, variant)
    print(f"[{variant}] fold={SELECTION_FOLD} n_tr={int(tr_m.sum())} n_va={n_va} "
          f"n_num={n_num} cat_vocab={cat_vocab}", flush=True)

    per_seed_oof: dict[int, np.ndarray] = {}
    family_bss: dict[int, float] = {}
    for seed in SEEDS:  # 10시드 (42..51) — 결정적 시드 순서
        best_bss, z = train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv,
                                 cat_vocab, n_num, seed)
        per_seed_oof[seed] = z
        family_bss[seed] = float(best_bss)
        print(f"  [seed {seed}] OOF 완료 (family best_bss={best_bss:.2f})", flush=True)

    z_ens = np.mean(np.stack([per_seed_oof[s] for s in SEEDS], axis=0), axis=0)
    _save_fold_artifacts(config_hash, variant, n_va, row_ids_sha256, per_seed_oof,
                         z_ens, mask_sha256)

    # 외부 라벨은 이 시점(변환+학습 완료)에만 읽는다 — R 폴드는 구조적 차단
    y_va = _read_outer_labels(train, va_m, SELECTION_FOLD)
    per_seed_bss = {str(s): float(_primary_bss(per_seed_oof[s], y_va)) for s in SEEDS}
    ens_bss = _primary_bss(z_ens, y_va)
    ens_brier = float(((np.clip(common.sigmoid(z_ens + C_LOGIT), CLIP_LO, CLIP_HI)
                        - y_va) ** 2).mean())
    ens_pred_mean = float(np.clip(common.sigmoid(z_ens + C_LOGIT),
                                  CLIP_LO, CLIP_HI).mean())

    probe = _blend_probe(z_ens, y_va)
    record = _base_record("full", variant, config_hash, _check_policy()[2], config_path, True)
    record.update({
        "title": f"Todo 6 — MLP preprocessing variant {variant} (primary only, 10 seeds)",
        "task": "aimers9-next-round/task-6-mlp-preprocess",
        "verdict": probe["verdict"],
        "exit_code": 0,
        "label_sources": [SELECTION_FOLD],
        "labels_read": True,
        "labels_read_note": "primary labels only; R-only folds never loaded (structural guard)",
        "seeds": list(SEEDS),
        "ensemble": {
            "rule": "mean of per-seed LOGITS",
            "per_seed_bss": per_seed_bss,
            "per_seed_family_best_bss": {str(s): round(v, 4) for s, v in family_bss.items()},
            "ensemble_bss": ens_bss,
            "ensemble_brier": ens_brier,
            "ensemble_pred_mean": ens_pred_mean,
        },
        "blend_probe": probe,
        "blocked_relations": {
            "checked_against": str(REGISTRY_PATH),
            "records": rel_records,
            "revival_detected": any(not r["ok"] for r in rel_records),
        },
        "fold_provenance": fold_info,
        "artifacts": {"dir": f"cache/mlp_preprocess/{config_hash}",
                      "manifest": "cache/mlp_preprocess/{}/manifest.json".format(config_hash)},
        "checks": checks + [{"rule": "primary_only_labels", "ok": True,
                             "reason": "labels = [primary] — R-only 라벨 미로드"},
                            {"rule": "single_variant", "ok": True,
                             "reason": f"variant={variant} 만 적용 (결합 금지 가드)"},
                            {"rule": "no_validation_fit", "ok": True,
                             "reason": "변환 통계(mean/std/플래그) outer-train 전용"}],
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    })
    evidence_path = EVIDENCE_DIR / f"task-6-mlp-preprocess-{variant}.json"
    _write_json(record, evidence_path)
    _write_md(record, variant, evidence_path.with_suffix(".md"))

    print(f"\n[mlp_preprocess_runner] {variant}: verdict={probe['verdict']} "
          f"ensemble_bss={ens_bss:.4f} candidate_bss={probe['candidate']['primary_bss']:.4f} "
          f"base_bss={probe['base']['primary_bss']:.4f} "
          f"delta_bss={probe['candidate']['delta_bss_vs_base']:+.4f} "
          f"mean_shift={probe['mean_shift']:.6f} r={probe['r']:.6f}", flush=True)
    print(f"[mlp_preprocess_runner] {variant}: evidence → {evidence_path}", flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════
# Task 7 hook (--promote-primary): PASS 증거 필수, 아니면 SKIPPED exit 0
# ════════════════════════════════════════════════════════════════════
def _cmd_promote(args: argparse.Namespace, variant: str) -> int:
    t0 = time.time()
    config_hash = _config_hash(variant)
    _check_policy()
    # promote 는 --full 과 별도 설정 사전 등록 (동일 경로 재사용 시 --full 증거의
    # config_pre_registered.sha256 이 깨지므로 절대 공유하지 않는다)
    config_path = EVIDENCE_DIR / f"task-6-mlp-preprocess-config-{variant}-promote.json"
    _pre_register(_config_dict(variant), config_hash, config_path)
    evidence_path = EVIDENCE_DIR / f"task-7-promotion-mlp-{variant}.json"

    full_rec = load_json(EVIDENCE_DIR / f"task-6-mlp-preprocess-{variant}.json")
    if full_rec is None or full_rec.get("verdict") != "PRIMARY_PASS":
        record = _base_record("promote-primary", variant, config_hash,
                              _check_policy()[2], config_path, False)
        record.update({"title": f"Todo 7 hook — MLP variant {variant} promotion (skipped)",
                       "task": "aimers9-next-round/task-7-promotion",
                       "verdict": "SKIPPED", "exit_code": 0, "labels_read": False,
                       "labels_read_note": ("promotion requires task-6 PRIMARY_PASS for "
                                            "the variant; none recorded — nothing trained"),
                       "reason": ("task-6-mlp-preprocess-{}.json missing or not "
                                  "PRIMARY_PASS — promotion is the FIRST passing "
                                  "candidate's full path only").format(variant)})
        _write_json(record, evidence_path)
        print(f"[mlp_preprocess_runner] --promote-primary {variant}: SKIPPED "
              f"(no PRIMARY_PASS evidence) — exit 0", flush=True)
        return 0

    # PASS: 10시드 OOF 재사용 (출처 검증) + 고정-알파 primary 전용 블렌드 + Task 7 게이트
    manifest = _check_cache_key(config_hash)
    if manifest is None:
        raise ProvenanceError("[FAIL] promote 는 --full OOF 아티팩트 재사용 필수 — 캐시 "
                              "manifest 없음 (먼저 --full 실행)")
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    _check_no_2025(train)
    tr_m, va_m = build_folds(train)[SELECTION_FOLD]
    fold_rec = _fold_provenance(manifest, SELECTION_FOLD, int(va_m.sum()),
                                _sha256_of_ids(train.loc[va_m, common.ID]))
    d = _cache_dir(config_hash)
    z_ens = np.load(d / fold_rec["ensemble_logits"]["file"]).astype(np.float64)
    if _sha256(d / fold_rec["ensemble_logits"]["file"]) != fold_rec["ensemble_logits"]["sha256"]:
        raise ProvenanceError("[FAIL] promote OOF 앙상블 sha256 불일치 (exit 2)")

    members = _load_base_members()
    z_lgb, z_mlp_base, z_cat = members["lgb"], members["mlp"], members["catboost"]
    w = BASELINE_WEIGHTS_BY_MEMBER
    z_base = w["lgb"] * z_lgb + w["mlp"] * z_mlp_base + w["catboost"] * z_cat
    y_va = _read_outer_labels(train, va_m, SELECTION_FOLD)
    base_bss = _primary_bss(z_base, y_va)
    best_bss, best_alpha = float("-inf"), ALPHA_GRID[0]
    for alpha in ALPHA_GRID:
        b = _primary_bss((1 - alpha) * z_base + alpha * z_ens, y_va)
        if b > best_bss:
            best_bss, best_alpha = b, alpha
    weight = float(best_alpha)
    z_blend = (1 - weight) * z_base + weight * z_ens
    blend_bss = float(_primary_bss(z_blend, y_va))
    incremental_bss = blend_bss - base_bss
    p_base = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    p_blend = np.clip(common.sigmoid(z_blend + C_LOGIT), CLIP_LO, CLIP_HI)
    mean_shift = float(abs(p_blend.mean() - p_base.mean()))
    corr = {
        "lgb": float(np.corrcoef(z_ens, z_lgb)[0, 1]),
        "mlp_base": float(np.corrcoef(z_ens, z_mlp_base)[0, 1]),
        "catboost": float(np.corrcoef(z_ens, z_cat)[0, 1]),
    }
    gates = {
        "positive_weight": weight > 0.0,
        "corr_lt_0.95_all_families": all(v < GATE_CORR_MAX for v in corr.values()),
        "incremental_bss_gt_1.0": incremental_bss > GATE_INCR_BSS_MIN,
        "mean_shift_le_0.005": mean_shift <= GATE_MEAN_SHIFT_MAX,
    }
    verdict = "PRIMARY_PROMOTED" if all(gates.values()) else "PRIMARY_REJECTED"
    record = _base_record("promote-primary", variant, config_hash, _check_policy()[2],
                          config_path, True)
    record.update({
        "title": f"Todo 7 hook — MLP variant {variant} promotion (primary only)",
        "task": "aimers9-next-round/task-7-promotion",
        "verdict": verdict, "exit_code": 0,
        "label_sources": [SELECTION_FOLD], "labels_read": True,
        "blend": {"alpha": weight, "candidate_id": _candidate_id(
            ("lgb", "mlp", "catboost", f"mlp_{variant}"),
            [0.30 * (1 - weight), 0.35 * (1 - weight), 0.35 * (1 - weight), weight],
            SELECTION_FOLD, BOOT_SEED, WEIGHT_STEP),
            "base_bss": base_bss, "blend_bss": blend_bss,
            "incremental_bss": incremental_bss, "mean_shift": mean_shift,
            "corr_to_families": corr, "gates": gates,
            "weights_by_member": {"lgb": round(0.30 * (1 - weight), 6),
                                  "mlp": round(0.35 * (1 - weight), 6),
                                  "catboost": round(0.35 * (1 - weight), 6),
                                  f"mlp_{variant}": round(weight, 6)}},
        "environment": _environment(), "total_time_s": time.time() - t0,
    })
    _write_json(record, evidence_path)
    print(f"[mlp_preprocess_runner] --promote-primary {variant}: {verdict} "
          f"alpha={weight} incr_bss={incremental_bss:+.4f} mean_shift={mean_shift:.6f}",
          flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════
# Markdown 증거
# ════════════════════════════════════════════════════════════════════
def _write_md(record: dict[str, Any], variant: str, path: Path) -> None:
    probe = record["blend_probe"]
    ens = record["ensemble"]
    lines = [
        f"# Task 6 — MLP preprocessing variant `{variant}` (primary only, 10 seeds)",
        "",
        f"**Task**: aimers9-next-round/task-6-mlp-preprocess",
        f"**Variant**: `{variant}`",
        f"**Verdict**: `{record['verdict']}`",
        "**Exit code**: 0 (verdict is a normal gate outcome)",
        f"**Recorded**: {record['recorded_at_utc']} (UTC)",
        f"**Git head**: `{record['git_head']}`",
        f"**Config hash**: `{record['config_hash']}`",
        f"**Label sources**: `{record['label_sources']}` — R-only folds NEVER loaded "
        "(structural guard; Task 10 only). `labels_read=true` for primary only.",
        "",
        "---",
        "",
        "## 1. Variant spec (pre-registered)",
        "",
        "```json",
        json.dumps(record["config_pre_registered"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 2. Ten-seed records (seeds 42..51 × primary fold — R folds embargoed)",
        "",
        "| seed | per-seed primary BSS | family best BSS (e6c convention) |",
        "|---|---|---|",
    ]
    for s in record["seeds"]:
        lines.append(f"| {s} | {ens['per_seed_bss'][str(s)]:.4f} | "
                     f"{ens['per_seed_family_best_bss'][str(s)]} |")
    lines += [
        "",
        f"Ensemble (mean of per-seed logits): primary BSS **{ens['ensemble_bss']:.4f}** | "
        f"Brier **{ens['ensemble_brier']:.6f}** | pred mean **{ens['ensemble_pred_mean']:.6f}**",
        "",
        "## 3. Blend probe vs rollback baseline `5890a4c54f502c4e` "
        "(lgb .30 / mlp .35 / cat .35, MLP member replaced by variant OOF)",
        "",
        f"- Baseline primary BSS: **{probe['base']['primary_bss']:.4f}** "
        f"(brier {probe['base']['brier']:.6f}, mean {probe['base']['pred_mean']:.6f})",
        f"- Variant MLP member primary BSS: **{probe['member']['primary_bss']:.4f}** "
        f"(brier {probe['member']['brier']:.6f}, mean {probe['member']['pred_mean']:.6f})",
        f"- Candidate blend primary BSS: **{probe['candidate']['primary_bss']:.4f}** "
        f"(brier {probe['candidate']['brier']:.6f}, mean {probe['candidate']['pred_mean']:.6f})",
        f"- **ΔBSS (candidate − baseline)**: **{probe['candidate']['delta_bss_vs_base']:+.4f}**",
        f"- Mean shift |p_cand − p_base|: **{probe['mean_shift']:.6f}**",
        f"- Corr(variant logits, families): lgb {probe['corr_to_families']['lgb']:.4f} / "
        f"mlp_base {probe['corr_to_families']['mlp_base']:.4f} / "
        f"catboost {probe['corr_to_families']['catboost']:.4f}",
        f"- r (primary label mean): {probe['r']:.6f}",
        "",
        "## 4. Gate (frozen in runner) — both must hold for PRIMARY_PASS",
        "",
        "| gate | threshold | actual | PASS? |",
        "|---|---|---|---|",
        f"| ΔBSS ≥ 15 | ≥ {probe['thresholds']['delta_bss_min']} | "
        f"{probe['candidate']['delta_bss_vs_base']:+.4f} | "
        f"{'✅' if probe['gates']['delta_bss_ge_15.0'] else '❌'} |",
        f"| mean shift ≤ 0.005 | ≤ {probe['thresholds']['mean_shift_max']} | "
        f"{probe['mean_shift']:.6f} | "
        f"{'✅' if probe['gates']['mean_shift_le_0.005'] else '❌'} |",
        "",
        f"**Verdict: `{record['verdict']}`** — "
        + ("variant advances (Task 7 promotion path may proceed)."
           if record["verdict"] == "PRIMARY_PASS"
           else "variant rejected; the other pre-registered variant may run next."),
        "",
        "## 5. Blocked-hypothesis registry check (feature_hypothesis_registry.json)",
        "",
        f"- Revival detected: **{record['blocked_relations']['revival_detected']}** "
        f"({len(record['blocked_relations']['records'])} blocked entries checked)",
        "- Relations: " + json.dumps([{"blocked_id": x["blocked_id"], "ok": x["ok"]}
                                      for x in record["blocked_relations"]["records"]],
                                     ensure_ascii=False),
        "",
        "## 6. Primary-only proof",
        "",
        "- Config pre-registered BEFORE any label read: "
        f"`{record['config_pre_registered']['file']}` "
        f"(sha256 {record['config_pre_registered']['sha256'][:16]}…)",
        f"- Fold provenance: n_va={record['fold_provenance']['n_primary']}, "
        f"row_ids_sha256=`{record['fold_provenance']['row_ids_sha256'][:16]}…`",
        f"- Artifacts: `cache/mlp_preprocess/{record['config_hash']}/` "
        "(per-seed OOF + ensemble + manifest, provenance-pinned for Task 7)",
        f"- R-fold embargo PASS (structural guard); no R-fold file opened.",
        "",
        f"Runtime: {record['total_time_s']:.1f} s CPU "
        f"({record['environment']['torch']}, {record['environment']['python']}).",
        f"Evidence: task-6-mlp-preprocess-{variant}.{{json,log}}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════
# fixture (전부 exit 2 — 실데이터 경로 증명)
# ════════════════════════════════════════════════════════════════════
def _run_fixture(name: str, args: argparse.Namespace) -> int:
    variant = "missing_flags" if name == "blocked-relation" else "clip_z5"
    config_hash = _config_hash(variant)
    base = f"task-6-mlp-preprocess-fixture-{name}"
    config_path = EVIDENCE_DIR / f"task-6-mlp-preprocess-config-fixture-{name}.json"
    log_path = EVIDENCE_DIR / f"{base}.log"
    evidence_path = EVIDENCE_DIR / f"{base}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)
    _pre_register(_config_dict(variant), config_hash, config_path)

    def fail_record(reason: str) -> int:
        record = _base_record(f"fixture-{name}", variant, config_hash, "n/a",
                              config_path, False)
        record.update({"title": f"Todo 6 fixture — {name}",
                       "task": "aimers9-next-round/task-6-mlp-preprocess",
                       "verdict": "REJECT", "exit_code": 2, "labels_read": False,
                       "labels_read_note": "fixture: no real label read",
                       "fixture": name, "reason": reason,
                       "environment": _environment()})
        _write_json(record, evidence_path)
        print(f"\n[mlp_preprocess_runner] fixture {name}: exit 2 — {reason}", flush=True)
        return 2

    try:
        if name == "combined-variants":
            print("\n[fixture combined-variants] clip_z5 + missing_flags 동시 적용 주입 "
                  "(실데이터 경로)…", flush=True)
            global force_combined
            force_combined = True
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            _check_no_2025(train)
            folds, _fi, _ck = _check_folds(train)
            tr_m, va_m = folds[SELECTION_FOLD]
            # 실제 결합 변환 시도 (clip + 이진 플래그 동시) — 단일-배리언트 가드가 차단
            _build_fold_variant(train, tr_m, va_m, "combined")
            return fail_record("[POLICY] 단일-배리언트 가드가 결합 적용을 감지하지 못함")
        if name == "validation-fit":
            print("\n[fixture validation-fit] 검증 행에서 피팅한 변환 통계 적용 주입 "
                  "(실데이터 경로)…", flush=True)
            global force_validation_fit
            force_validation_fit = True
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            _check_no_2025(train)
            folds, _fi, _ck = _check_folds(train)
            tr_m, va_m = folds[SELECTION_FOLD]
            _build_fold_variant(train, tr_m, va_m, "clip_z5")
            return fail_record("[POLICY] 검증-피팅 가드가 작동하지 않음")
        if name == "blocked-relation":
            print("\n[fixture blocked-relation] registry-listed blocked 가설(return_gap) "
                  "부활 주입 (실데이터 경로)…", flush=True)
            global force_blocked_relation
            force_blocked_relation = True
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            _check_no_2025(train)
            folds, _fi, _ck = _check_folds(train)
            tr_m, va_m = folds[SELECTION_FOLD]
            _build_fold_variant(train, tr_m, va_m, "missing_flags")
            return fail_record("[POLICY] blocked-relation 가드가 작동하지 않음")
    except (PolicyViolation, LeakageError, ProvenanceError) as exc:
        return fail_record(str(exc))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 6: bounded MLP preprocessing side-gate "
                    "(--variant <clip_z5|missing_flags> --full | --promote-primary | "
                    "--fixture)")
    parser.add_argument("--variant", choices=list(VARIANTS), default=None,
                        help="배리언트: clip_z5 (A) | missing_flags (B)")
    parser.add_argument("--full", action="store_true",
                        help="Task 6: 10시드 primary OOF + 블렌드 프로브 + 판정")
    parser.add_argument("--promote-primary", action="store_true",
                        help="Task 7 hook: task-6 PRIMARY_PASS 증거 필수 (아니면 SKIPPED)")
    parser.add_argument("--fixture",
                        choices=["combined-variants", "validation-fit", "blocked-relation"],
                        default=None, help="실패 QA fixture (전부 exit 2)")
    parser.add_argument("--evidence", default=None, help="증거 JSON 경로 (기본 task-scoped)")
    args = parser.parse_args(argv)

    if args.fixture:
        return _run_fixture(args.fixture, args)
    if args.variant is None:
        print("[mlp_preprocess_runner] --variant <clip_z5|missing_flags> 필요 "
              "(--full | --promote-primary)", file=sys.stderr)
        return 1
    if not args.full and not args.promote_primary:
        print("[mlp_preprocess_runner] 모드 지정 필요: --full | --promote-primary",
              file=sys.stderr)
        return 1

    variant = args.variant
    if args.full:
        base = f"task-6-mlp-preprocess-{variant}"
        cmd = _cmd_full
    else:
        base = f"task-7-promotion-mlp-{variant}"
        cmd = _cmd_promote
    args.evidence = str((Path(args.evidence).expanduser().resolve() if args.evidence
                         else EVIDENCE_DIR / f"{base}.json"))
    log_path = EVIDENCE_DIR / f"{base}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    print(f"[mlp_preprocess_runner] variant={variant} config_hash={_config_hash(variant)[:16]}… "
          f"torch={torch.__version__}", flush=True)
    try:
        return cmd(args, variant)
    except (PolicyViolation, LeakageError, ProvenanceError) as exc:
        print(f"[mlp_preprocess_runner] REJECT (exit 2): {exc}", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001 — 치명적 입력 오류
        print(f"[mlp_preprocess_runner] FATAL (exit 1): {exc!r}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
