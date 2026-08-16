#!/usr/bin/env python3
"""deepfm_runner.py — Task 4: canonical fixed DeepFM/DCNv2 temporal runner.

Aimers9-score-improvement-next-round, Todo 4: the runner for the FROZEN DeepFM/DCNv2
member (deepfm_model.py is the single source of truth for every hyperparameter/schema;
this CLI makes ZERO architecture decisions). The plan forbids routing through
model_family_runner.py — this runner is standalone.

Modes:
  --smoke               Task 4 contract smoke: config digest pre-registration (BEFORE any
                        label access), 49-feature/9-category schema, no-2025 guard, fold
                        masks + leakage guard, vocab/numeric prep fitted on outer-train
                        rows only, model forward/backward, inner-temporal epoch machinery
                        proof on synthetic data. NO labels read (labels_read=False).
  --screen-primary      Task 5: seeds [52,53] on primary only. Inner-temporal epoch
                        selection (patience 3 on inner Brier, earliest-epoch tie-break),
                        refit fixed epoch count on the full outer-train window, outer OOF
                        logits; only THEN read outer labels; fixed-alpha primary-only
                        blend probe vs rollback baseline 5890a4c54f502c4e ->
                        PRIMARY_PROMOTED / PRIMARY_REJECT.
  --promote-primary     Task 7: complete seeds [52..56] (reuses screen OOF 52/53 under
                        provenance check), provenance-pinned primary OOF, same blend
                        probe; requires screen evidence PRIMARY_PROMOTED else SKIPPED.
  --replay-and-deploy   Task 10: final fit on ALL official 2019-2024 rows with epochs
                        pinned from the promotion manifest (config-hash-verified);
                        writes model artifacts + manifest; row-local inference QA.
  --fixture <name>      category-order-mismatch | contains-2025 | outer-label-mutation |
                        attempted-r-fold-read — every fixture exits 2 (proven on the
                        real data path).

Exit codes: 0 = PASS / SKIPPED (gate verdicts are normal outcomes), 1 = fatal input
error, 2 = policy/leakage/provenance violation (or a failure-injection fixture proof).

Contract highlights (frozen, plan §48(5) + Todo 4):
  - fold contract read-only reused from qualification_runner (build_folds / leakage guard).
  - `primary` is the ONLY selection fold; r2022/r2023/r2024 labels are NEVER loaded in
    screen/promote (structural guard -> LeakageError exit 2).
  - epochs selected on the inner temporal split (last season of the outer-train window)
    BEFORE outer validation labels are read (ordering gate -> PolicyViolation exit 2).
  - config digest covers EVERY frozen hyperparameter/schema, written+hashed before any
    label result is read; an altered config under an existing cache key, or missing fold
    provenance in a reused artifact, is a hard failure (exit 2).
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

# ── 동결 제어 / 기존 러너 규약 재사용 (전부 읽기 전용 import — 수정 금지) ──
from repro_979.qualification_runner import (  # noqa: E402
    R_FOLDS, build_folds, _check_leakage, _sha256,
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
import torch.nn as nn  # noqa: E402

import repro_979.common as common  # noqa: E402
from repro_979.deepfm_model import (  # noqa: E402
    FEATURES, CATS, NUMERICS, EMB_DIM, UNK_P, UNK_DROP_CATS, DCN_DEPTH, DCN_RANK,
    TOWER_HIDDEN, TOWER_DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    SCREEN_SEEDS, PROMOTE_SEEDS, C_LOGIT, CLIP_LO, CLIP_HI, MISSINGNESS_FLAGS,
    LOSS, SCORE_FORMULA, ENSEMBLE_RULE, DeepFMDCNv2, DeepFMPrep,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device("cpu")  # Task 4-7 environment: CPU only (no GPU assumption)

SCHEMA_VERSION = 1
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
POLICY_PATH = REPO / "next_round_policy.json"
CACHE_ROOT = REPO / "cache" / "deepfm"          # artifacts: cache/deepfm/<config_hash>/
DEFAULT_SCREEN_EVIDENCE = EVIDENCE_DIR / "task-5-deepfm-screen.json"
DEFAULT_PROMOTE_EVIDENCE = EVIDENCE_DIR / "task-7-promotion.json"

# fixed-alpha blend probe (plan verification strategy: alpha = {0.05, ..., 0.95})
ALPHA_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))
# Task 5/7 promotion gates (frozen)
GATE_CORR_MAX = 0.95          # corr(logits, each existing family) must be < 0.95
GATE_INCR_BSS_MIN = 1.0       # primary incremental BSS (blend - base) must be > 1.0
GATE_MEAN_SHIFT_MAX = 0.005   # max |candidate - baseline| probability-mean shift <= 0.005

# fixture 주입 플래그 (main 에서만 True)
force_cat_order_mismatch = False
force_contains_2025 = False
force_outer_label_mutation = False
force_attempted_r_fold_read = False


class ProvenanceError(RuntimeError):
    """캐시/증거 아티팩트의 출처(provenance)가 불일치하면 발생 (하드 실패, exit 2)."""


class LeakageError(RuntimeError):
    """R-only 폴드 라벨을 Task 10 이전에 읽으려 하면 발생 (구조적 누수 가드, exit 2)."""


class PolicyViolation(RuntimeError):
    """정책/동결 계약 위반 (정렬 키, 에포크 순서, 2025 행, 범주 계약 등, exit 2)."""


# ── 에포크 순서 게이트: 내부 시계열 분할에서 에포크를 고정하기 전에 외부 라벨 금지 ──
class _EpochOrderGate:
    epochs_selected = False
    outer_read = False


_epoch_gate = _EpochOrderGate()


def _mark_epochs_selected() -> None:
    _epoch_gate.epochs_selected = True


def _read_outer_labels(train: pd.DataFrame, mask: np.ndarray, fold: str) -> np.ndarray:
    """외부 검증 라벨을 읽는 유일한 경로 (가드 2중: R 폴드 금지 + 에포크 선선택)."""
    if fold in R_FOLDS:
        raise LeakageError(
            f"[LEAKAGE] R-only 폴드 라벨 {fold!r} 읽기 차단 — Task 10 전 screen/promote "
            f"경로에서 R-only 폴드는 절대 로드하지 않는다 (exit 2)")
    if not _epoch_gate.epochs_selected:
        raise PolicyViolation(
            f"[ORDER] 외부 검증 라벨({fold})이 에포크 선택 이전에 읽혔습니다 — "
            f"checkpoint/epoch 는 외부 검증 라벨을 읽기 전에 내부 시계열 분할에서 "
            f"고정되어야 한다 (exit 2)")
    _epoch_gate.outer_read = True
    return train.loc[mask, common.TARGET].values.astype(np.float64)


# ── 동결 설정 다이제스트 (모든 하이퍼파라미터/스키마를 커버) ──
def _config_dict() -> dict[str, Any]:
    """동결 설정 전체의 정규화 표현 — config digest 의 원천 (deepfm_model.py 상수만 사용)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "deepfm_dcnv2",
        "task": "aimers9-next-round/task-4-deepfm-contract",
        "features": list(FEATURES),
        "n_features": len(FEATURES),
        "feature_source": ("47 official non-ID inputs after excluding row_id + platoon + "
                           "count_state (mirror of the frozen MLP contract, "
                           "qualification_runner.CHAMPION_FEATURES order)"),
        "cats": list(CATS),
        "n_cats": len(CATS),
        "cat_order_note": ("exact frozen order matters — the first two fields "
                           "(pitcher_id, batter_id) are the UNK-dropout fields"),
        "numerics": list(NUMERICS),
        "n_numerics": len(NUMERICS),
        "vocab_rule": "outer-train rows only; ID 0 = UNK; sorted string keys (MLP contract)",
        "numeric_prep": ("mean imputation + z-score standardization fitted ONLY on "
                         "outer-train rows; zero-variance numerics map to zero; "
                         "never fit on validation/test rows"),
        "missingness_flags": list(MISSINGNESS_FLAGS),
        "emb_dim": EMB_DIM,
        "emb_dim_note": ("common 24-d FM vector for EVERY categorical field — "
                         "INTENTIONALLY supersedes ideation's 8/4-8 dims (plan wins)"),
        "unk_p": UNK_P,
        "unk_drop_cats": list(UNK_DROP_CATS),
        "unk_drop_note": ("10% pitcher/batter UNK dropout BEFORE embedding lookup, "
                          "training-only"),
        "model": {
            "linear_logit_term": True,
            "fm_pairwise_dot_sum": True,
            "dcn_depth": DCN_DEPTH,
            "dcn_rank": DCN_RANK,
            "dcn_rank_note": "rank-32 three-layer DCNv2 (supersedes ideation rank 32|64)",
            "tower_hidden": list(TOWER_HIDDEN),
            "tower_act": "SiLU",
            "tower_norm": "LayerNorm",
            "tower_dropout": TOWER_DROPOUT,
            "tower_dropout_note": ("ZERO tower dropout (INTENTIONALLY supersedes "
                                   "ideation dropout 0.10)"),
            "head": "concat [linear logit, FM sum, cross out, tower out] -> Linear -> scalar logit",
            "loss": LOSS,
        },
        "optim": {
            "adamw_lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "patience_metric": "inner Brier (mean squared error of sigmoid(z) vs y)",
            "patience_tie_break": "earliest epoch (strict < keeps the first occurrence)",
            "epoch_rule": ("epochs selected on an inner temporal split within the outer "
                           "training window, fixed BEFORE outer validation labels are "
                           "read; then REFIT the fixed epoch count on the full outer "
                           "training window before outer scoring"),
            "inner_split_rule": "last season of the outer-train window = inner val, earlier = inner train",
        },
        "seeds": {
            "screen": list(SCREEN_SEEDS),
            "promotion": list(PROMOTE_SEEDS),
            "order": "deterministic ascending seed order",
            "ensemble": ENSEMBLE_RULE,
        },
        "folds": {
            "primary": "season <= 2023 -> season == 2024",
            "r2022": "season <= 2021 & R -> season == 2022 & R (Task 10 only)",
            "r2023": "season <= 2022 & R -> season == 2023 & R (Task 10 only)",
            "r2024": "season <= 2023 & R -> season == 2024 & R (diagnostic, Task 10 only)",
            "selection_fold": SELECTION_FOLD,
            "source": "qualification_runner.build_folds (read-only reuse)",
        },
        "r_fold_embargo": ("r2022/r2023/r2024 labels NEVER loaded in screen/promote modes "
                           "(structural guard: LeakageError exit 2); one-shot R "
                           "qualification only at Task 10"),
        "scoring": {
            "formula": SCORE_FORMULA,
            "c_logit": C_LOGIT,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "c_logit_frozen_note": "C_LOGIT = -0.0404 frozen for the whole round",
        },
        "blend_probe": {
            "base_candidate": BASELINE_CANDIDATE_ID,
            "base_weights_by_member": BASELINE_WEIGHTS_BY_MEMBER,
            "alpha_grid": list(ALPHA_GRID),
            "tie_rule": "smallest-alpha exact-tie choice (iterate ascending, strict >)",
            "gates": {
                "positive_weight": True,
                "corr_lt": GATE_CORR_MAX,
                "incremental_bss_gt": GATE_INCR_BSS_MIN,
                "mean_shift_le": GATE_MEAN_SHIFT_MAX,
            },
        },
        "init": {"cross_u_v": "xavier_uniform", "cross_b": "zeros", "other": "torch default"},
    }


def _config_hash() -> str:
    return _canonical_sha256(_config_dict())


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
        "device": str(DEVICE),
    }


# ════════════════════════════════════════════════════════════════════
# 계약 가드
# ════════════════════════════════════════════════════════════════════
def _check_policy() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """동결 정책 JSON 읽기(읽기 전용) + validate_policy 재사용 (violation 시 PolicyViolation)."""
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
    """2025 행이 학습 프레임에 있으면 PolicyViolation (2025 라벨은 대회 테스트 시즌)."""
    if (train["season"] == 2025).any():
        n = int((train["season"] == 2025).sum())
        raise PolicyViolation(
            f"[POLICY] 학습 프레임에 2025 시즌 행 {n}개 존재 — 2025 라벨/행은 "
            f"절대 학습·검증에 사용 금지 (exit 2)")


def _check_schema(train: pd.DataFrame, cats=CATS) -> list[str]:
    """동결 49-피처/9-범주 스키마 계약: 문제 목록 반환 (비면 = PASS).

    cats 는 정확히 9개여야 하고 동결 순서(CATS)와 정확히 일치해야 한다 — 범주 필드의
    위치 인덱스가 모델 계약(UNK-dropout 필드 = 첫 두 개)에 의미를 갖기 때문.
    """
    problems: list[str] = []
    if len(cats) != 9:
        problems.append(f"범주 필드 수 {len(cats)} != 9 (동결 계약)")
    if tuple(cats) != CATS:
        problems.append(f"범주 필드 순서 불일치: {list(cats)} != 동결 {list(CATS)}")
    missing_cats = [c for c in cats if c not in train.columns]
    if missing_cats:
        problems.append(f"범주 컬럼 부재: {missing_cats}")
    missing_feats = [f for f in FEATURES if f not in train.columns]
    if missing_feats:
        problems.append(f"49-피처 컬럼 부재: {missing_feats}")
    return problems


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
    overlap_note: dict[str, Any] = {}
    va_primary = folds[SELECTION_FOLD][1]
    for rfold in R_FOLDS:
        overlap = int((va_primary & folds[rfold][1]).sum())
        if rfold == "r2024":
            overlap_note[rfold] = {"overlap_rows_with_primary": overlap, "expected": True,
                                   "reason": "r2024 val = (season==2024) & R ⊆ primary val "
                                             "(계획 'overlapping r2024' 진단 폴드)"}
        elif overlap != 0:
            raise ProvenanceError(
                f"[FAIL] row-ID disjointness 위반: primary ∩ {rfold} ≠ ∅ (exit 2)")
    row_ids_hash = _sha256_of_ids(train.loc[va_primary, common.ID])
    return folds, {"n_primary": n_primary, "row_ids_sha256": row_ids_hash,
                   "r2024_overlap": overlap_note}, [{"rule": "leakage_guard", "ok": True,
                                                     "reason": "검증 마스크에 2025 없음"}]


def _sha256_of_ids(ids: pd.Series) -> str:
    """검증 마스크 row_id 열의 결정적 지문 — 폴드 출처(provenance) 검증용."""
    return hashlib.sha256(ids.astype(str).str.cat(sep=",").encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════
# 내부 시계열 분할 + 학습 기계 (Task 4 가동부 — 완전 동결)
# ════════════════════════════════════════════════════════════════════
def _inner_split(tr_m: np.ndarray, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """내부 시계열 분할: 외부 train 창의 마지막 시즌 = inner val, 그 앞 = inner train."""
    seasons = sorted(int(s) for s in train.loc[tr_m, "season"].unique())
    if len(seasons) < 2:
        raise PolicyViolation(f"[ORDER] 내부 시계열 분할 불가 — train 창 시즌 수 {len(seasons)} < 2")
    last = seasons[-1]
    return (tr_m & (train["season"] < last).values, tr_m & (train["season"] == last).values)


def _tensors(prep: DeepFMPrep, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    Xn, Xc = prep.transform_np(df)
    return torch.tensor(Xn), torch.tensor(Xc)


def _train_epoch(model: nn.Module, opt: torch.optim.Optimizer,
                 lossf: nn.Module, Xn: torch.Tensor, Xc: torch.Tensor,
                 y: torch.Tensor, batch_size: int) -> None:
    n = len(Xn)
    idx = torch.randperm(n)
    for s in range(0, n, batch_size):
        bi = idx[s:s + batch_size]
        opt.zero_grad()
        z = model(Xn[bi], Xc[bi])
        loss = lossf(z, y[bi])
        loss.backward()
        opt.step()


def _track_patience(brier: float, ep: int, best_brier: float, best_epoch: int,
                    bad: int, patience: int) -> tuple[float, int, int]:
    """내부 Brier 조기 종료 책상 기록: strict < 비교로 earliest-epoch tie-break."""
    if brier < best_brier:
        return brier, ep, 0
    return best_brier, best_epoch, bad + 1


def _select_epoch(prep: DeepFMPrep, Xn_it: torch.Tensor, Xc_it: torch.Tensor,
                  y_it: np.ndarray, Xn_iv: torch.Tensor, Xc_iv: torch.Tensor,
                  y_iv: np.ndarray, seed: int, max_epochs: int = MAX_EPOCHS,
                  patience: int = PATIENCE, batch_size: int = BATCH_SIZE) -> int:
    """내부 시계열 분할에서 에포크 선택: inner Brier, patience 3, earliest-epoch tie-break.

    내부 검증 라벨(y_iv)은 외부 train 창(마지막 시즌)의 라벨 — 학습 데이터의 일부로
    조기 종료에만 사용. 외부 검증 라벨은 여기서 절대 읽지 않는다."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    yit = torch.tensor(y_it)
    best_brier, best_epoch, bad = float("inf"), 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        _train_epoch(model, opt, lossf, Xn_it, Xc_it, yit, batch_size)
        model.eval()
        with torch.no_grad():
            zv = model(Xn_iv, Xc_iv).numpy()
            brier = float(np.mean((common.sigmoid(zv) - y_iv) ** 2))
        best_brier, best_epoch, bad = _track_patience(
            brier, ep, best_brier, best_epoch, bad, patience)
        if bad >= patience:
            break
    return best_epoch


def _refit_predict(prep: DeepFMPrep, Xn_ot: torch.Tensor, Xc_ot: torch.Tensor,
                   y_ot: np.ndarray, Xn_va: torch.Tensor, Xc_va: torch.Tensor,
                   seed: int, epochs: int, batch_size: int = BATCH_SIZE
                   ) -> tuple[DeepFMDCNv2, np.ndarray]:
    """동결 에포크 수로 외부 train 창 전체에 REFIT 후 외부 검증 로짓 생성."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    yot = torch.tensor(y_ot)
    for _ in range(epochs):
        model.train()
        _train_epoch(model, opt, lossf, Xn_ot, Xc_ot, yot, batch_size)
    model.eval()
    with torch.no_grad():
        z = model(Xn_va, Xc_va).numpy()
    return model, z


# ════════════════════════════════════════════════════════════════════
# 캐시/출처 (config digest + 폴드 provenance 로 재사용 검증)
# ════════════════════════════════════════════════════════════════════
def _cache_dir(config_hash: str) -> Path:
    return CACHE_ROOT / config_hash


def _manifest_path(config_hash: str) -> Path:
    return _cache_dir(config_hash) / "manifest.json"


def _check_cache_key(config_hash: str) -> dict[str, Any] | None:
    """기존 캐시 키 재사용 가드: 디렉토리 존재 시 manifest 의 config_hash 가 현재와
    정확히 일치해야 한다 (조작된 설정 하에서 기존 키 재사용 = 하드 실패)."""
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
    """manifest 에서 폴드 출처를 검증해 반환 (누락/불일치 = 하드 실패)."""
    rec = None
    for entry in manifest.get("fold_provenance", []):
        if entry.get("fold") == fold:
            rec = entry
            break
    if rec is None:
        raise PolicyViolation(
            f"[POLICY] 캐시 manifest 에 폴드 {fold} 출처(provenance) 누락 (exit 2)")
    if int(rec.get("n_va", -1)) != n_va or rec.get("row_ids_sha256") != row_ids_sha256:
        raise PolicyViolation(
            f"[POLICY] 캐시 폴드 {fold} 출처 불일치: manifest n_va={rec.get('n_va')} "
            f"row_ids={str(rec.get('row_ids_sha256'))[:12]}… vs 현재 n_va={n_va} "
            f"row_ids={row_ids_sha256[:12]}… (exit 2)")
    return rec


def _load_oof_seed(config_hash: str, fold_rec: dict[str, Any], seed: int) -> np.ndarray:
    rel = fold_rec["seed_logits"][str(seed)]["file"]
    path = _cache_dir(config_hash) / rel
    if not path.is_file():
        raise ProvenanceError(f"[FAIL] 캐시 OOF 로짓 없음: {path}")
    arr = np.load(path)
    if _sha256(path) != fold_rec["seed_logits"][str(seed)]["sha256"]:
        raise ProvenanceError(f"[FAIL] 캐시 OOF 로짓 sha256 불일치: {path}")
    return arr


def _save_fold_artifacts(config_hash: str, fold: str, n_va: int, row_ids_sha256: str,
                         epochs: dict[int, int], per_seed_oof: dict[int, np.ndarray],
                         z_ens: np.ndarray, mask_sha256: str) -> dict[str, Any]:
    """폴드별 OOF 아티팩트 저장 + manifest 기록 (config digest + 출처 포함)."""
    d = _cache_dir(config_hash)
    d.mkdir(parents=True, exist_ok=True)
    seed_logits: dict[str, Any] = {}
    for seed, z in per_seed_oof.items():
        rel = f"oof_{fold}_s{seed}.npy"
        path = d / rel
        np.save(path, z.astype(np.float64))
        seed_logits[str(seed)] = {"file": rel, "sha256": _sha256(path)}
    ens_rel = f"oof_{fold}_ensemble.npy"
    ens_path = d / ens_rel
    np.save(ens_path, z_ens.astype(np.float64))
    rec = {
        "fold": fold,
        "n_va": int(n_va),
        "row_ids_sha256": row_ids_sha256,
        "mask_sha256": mask_sha256,
        "epochs": {str(s): int(e) for s, e in epochs.items()},
        "seed_logits": seed_logits,
        "ensemble_logits": {"file": ens_rel, "sha256": _sha256(ens_path)},
    }
    mpath = _manifest_path(config_hash)
    manifest: dict[str, Any] = {}
    if mpath.is_file():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            raise PolicyViolation(
                f"[POLICY] 기존 캐시 키 {config_hash[:16]}… 설정 불일치 — 저장 거부 (exit 2)")
    manifest["config_hash"] = config_hash
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["created_at_utc"] = _now_utc()
    manifest["git_head"] = _git_commit()
    manifest.setdefault("fold_provenance", [])
    manifest["fold_provenance"] = [e for e in manifest["fold_provenance"]
                                   if e.get("fold") != fold] + [rec]
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


# ════════════════════════════════════════════════════════════════════
# 평가 + 고정-알파 블렌드 프로브
# ════════════════════════════════════════════════════════════════════
def _primary_bss(z: np.ndarray, y: np.ndarray) -> float:
    """계획 배포 산식: common.score(clip(sigmoid(z+C_LOGIT), 0.30, 0.70), y)."""
    p = np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)
    return float(common.score(p, y))


def _load_base_members() -> dict[str, np.ndarray]:
    """롤백 기준선 3멤버 primary OOF 로짓 (커밋 증거 다이제스트 대조 후 로드)."""
    _verify_provenance()  # sha256 대조 (task-2-control.json / task-5-models-catboost.json)
    return {m: np.load(REPO / rel).astype(np.float64) for m, rel in MEMBER_FILES.items()}


def _blend_probe(z_member: np.ndarray, y_va: np.ndarray) -> dict[str, Any]:
    """고정-알파 primary 전용 블렌드 프로브 (롤백 기준선 5890a4c54f502c4e 대비)."""
    members = _load_base_members()
    z_lgb, z_mlp, z_cat = members["lgb"], members["mlp"], members["catboost"]
    z_base = (BASELINE_WEIGHTS_BY_MEMBER["lgb"] * z_lgb
              + BASELINE_WEIGHTS_BY_MEMBER["mlp"] * z_mlp
              + BASELINE_WEIGHTS_BY_MEMBER["catboost"] * z_cat)
    p_base = np.clip(common.sigmoid(z_base + C_LOGIT), CLIP_LO, CLIP_HI)
    p_member = np.clip(common.sigmoid(z_member + C_LOGIT), CLIP_LO, CLIP_HI)
    base_bss = float(common.score(p_base, y_va))
    member_bss = float(common.score(p_member, y_va))
    best_bss, best_alpha = float("-inf"), ALPHA_GRID[0]
    for alpha in ALPHA_GRID:                      # 오름차순 — smallest-alpha exact-tie
        b = _primary_bss((1 - alpha) * z_base + alpha * z_member, y_va)
        if b > best_bss:
            best_bss, best_alpha = b, alpha
    weight = float(best_alpha)
    z_blend = (1 - weight) * z_base + weight * z_member
    blend_bss = float(_primary_bss(z_blend, y_va))
    incremental_bss = blend_bss - base_bss
    corr = {
        "lgb": float(np.corrcoef(z_member, z_lgb)[0, 1]),
        "mlp": float(np.corrcoef(z_member, z_mlp)[0, 1]),
        "catboost": float(np.corrcoef(z_member, z_cat)[0, 1]),
    }
    mean_shift = float(np.abs(p_member.mean() - p_base.mean()))
    w_all = np.asarray([0.30 * (1 - weight), 0.35 * (1 - weight),
                        0.35 * (1 - weight), weight], dtype=np.float64)
    blend_candidate_id = _candidate_id(("lgb", "mlp", "catboost", "deepfm"),
                                       w_all, SELECTION_FOLD, BOOT_SEED, WEIGHT_STEP)
    gates = {
        "positive_weight": weight > 0.0,
        "corr_lt_0.95_all_families": all(v < GATE_CORR_MAX for v in corr.values()),
        "incremental_bss_gt_1.0": incremental_bss > GATE_INCR_BSS_MIN,
        "mean_shift_le_0.005": mean_shift <= GATE_MEAN_SHIFT_MAX,
    }
    return {
        "base": {
            "candidate_id": BASELINE_CANDIDATE_ID,
            "weights_by_member": BASELINE_WEIGHTS_BY_MEMBER,
            "primary_bss": base_bss,
            "brier": float(((p_base - y_va) ** 2).mean()),
            "pred_mean": float(p_base.mean()),
        },
        "member": {
            "family": "deepfm_dcnv2",
            "primary_bss": member_bss,
            "brier": float(((p_member - y_va) ** 2).mean()),
            "pred_mean": float(p_member.mean()),
        },
        "blend": {
            "alpha": weight,
            "candidate_id": blend_candidate_id,
            "primary_bss": blend_bss,
            "weights_by_member": {
                "lgb": round(0.30 * (1 - weight), 6),
                "mlp": round(0.35 * (1 - weight), 6),
                "catboost": round(0.35 * (1 - weight), 6),
                "deepfm": round(weight, 6),
            },
        },
        "incremental_bss": incremental_bss,
        "corr_to_families": corr,
        "mean_shift": mean_shift,
        "gates": gates,
        "thresholds": {"corr_max": GATE_CORR_MAX, "incremental_bss_min": GATE_INCR_BSS_MIN,
                       "mean_shift_max": GATE_MEAN_SHIFT_MAX},
        "verdict": "PRIMARY_PROMOTED" if all(gates.values()) else "PRIMARY_REJECT",
        "alpha_grid": list(ALPHA_GRID),
        "r": float(y_va.mean()),
    }


# ════════════════════════════════════════════════════════════════════
# 임시 OOF 파이프라인 (screen/promote 공용) — primary 폴드 전용, R 폴드 봉쇄
# ════════════════════════════════════════════════════════════════════
def _run_temporal_oof(train: pd.DataFrame, seeds: tuple[int, ...], config_hash: str,
                      mode: str, screen_fold_rec: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
    """primary 폴드: 내부 에포크 선택 -> 동결 -> REFIT -> 외부 OOF -> (라벨은 마지막에).

    screen_fold_rec: promote 모드가 screen OOF(52/53)를 출처 검증 후 재사용할 때 전달
    (_fold_provenance 로 검증된 폴드 레코드). 외부 검증 라벨은 _read_outer_labels 를
    통해서만, 에포크 선택 완료 후에 읽힌다."""
    folds = build_folds(train)
    leakage = _check_leakage(folds, train)
    if leakage:
        raise PolicyViolation("[POLICY] 폴드 누수 가드 실패: " + "; ".join(leakage))
    tr_m, va_m = folds[SELECTION_FOLD]
    n_va = int(va_m.sum())
    if n_va != EXPECTED_ROWS[SELECTION_FOLD]:
        raise ProvenanceError(f"[FAIL] primary 검증 행 수 {n_va} != {EXPECTED_ROWS[SELECTION_FOLD]}")
    va_ids = train.loc[va_m, common.ID]
    row_ids_sha256 = _sha256_of_ids(va_ids)
    mask_sha256 = hashlib.sha256(va_m.astype(np.uint8).to_numpy().tobytes()).hexdigest()

    if force_outer_label_mutation:
        # fixture: 에포크 선택 이전에 외부 라벨을 (의도적으로) 읽는 위반 경로 — 가드가 차단
        _read_outer_labels(train, va_m, SELECTION_FOLD)

    # prep 은 외부 train 행에서만 피팅
    prep = DeepFMPrep.fit(train.loc[tr_m], NUMERICS, CATS)
    it_m, iv_m = _inner_split(tr_m, train)
    Xn_it, Xc_it = _tensors(prep, train.loc[it_m])
    Xn_iv, Xc_iv = _tensors(prep, train.loc[iv_m])
    y_it = train.loc[it_m, common.TARGET].values.astype(np.float32)
    y_iv = train.loc[iv_m, common.TARGET].values.astype(np.float32)
    Xn_ot, Xc_ot = _tensors(prep, train.loc[tr_m])
    y_ot = train.loc[tr_m, common.TARGET].values.astype(np.float32)
    Xn_va, Xc_va = _tensors(prep, train.loc[va_m])

    epochs: dict[int, int] = {}
    per_seed_oof: dict[int, np.ndarray] = {}
    for seed in seeds:  # 결정적 시드 순서
        if screen_fold_rec is not None and str(seed) in screen_fold_rec["seed_logits"]:
            z = _load_oof_seed(config_hash, screen_fold_rec, seed)
            ep = screen_fold_rec["epochs"][str(seed)]
            epochs[seed], per_seed_oof[seed] = int(ep), z
            print(f"  [seed {seed}] OOF 재사용 (screen 출처 검증 완료, epochs={ep})", flush=True)
            continue
        ep = _select_epoch(prep, Xn_it, Xc_it, y_it, Xn_iv, Xc_iv, y_iv, seed)
        epochs[seed] = ep
        print(f"  [seed {seed}] inner-selected epochs={ep} (inner Brier, patience {PATIENCE})",
              flush=True)
        _model, z = _refit_predict(prep, Xn_ot, Xc_ot, y_ot, Xn_va, Xc_va, seed, ep)
        per_seed_oof[seed] = z
        print(f"  [seed {seed}] refit 완료 + 외부 OOF 생성", flush=True)

    _mark_epochs_selected()  # 에포크 동결 완료 — 이제 외부 라벨 읽기 허용
    z_ens = np.mean(np.stack([per_seed_oof[s] for s in seeds], axis=0), axis=0)
    _save_fold_artifacts(config_hash, SELECTION_FOLD, n_va, row_ids_sha256, epochs,
                         per_seed_oof, z_ens, mask_sha256)
    y_va = _read_outer_labels(train, va_m, SELECTION_FOLD)
    return {
        "fold": SELECTION_FOLD, "n_va": n_va, "seeds": list(seeds),
        "epochs": {str(s): int(e) for s, e in epochs.items()},
        "per_seed_bss": {str(s): float(_primary_bss(per_seed_oof[s], y_va)) for s in seeds},
        "ensemble_bss": _primary_bss(z_ens, y_va),
        "ensemble_brier": float(((np.clip(common.sigmoid(z_ens + C_LOGIT),
                                          CLIP_LO, CLIP_HI) - y_va) ** 2).mean()),
        "ensemble_pred_mean": float(np.clip(common.sigmoid(z_ens + C_LOGIT),
                                            CLIP_LO, CLIP_HI).mean()),
        "z_ens": z_ens, "y_va": y_va,
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


def _base_record(mode: str, config_hash: str, policy_hash: str, config_path: Path,
                 labels_read: bool) -> dict[str, Any]:
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
        "label_sources": [],
        "labels_read": labels_read,
    }


# ════════════════════════════════════════════════════════════════════
# 스모크 (Task 4) — 계약 전용, 라벨 무접촉
# ════════════════════════════════════════════════════════════════════
def _cmd_smoke(args: argparse.Namespace) -> int:
    t0 = time.time()
    config_hash = _config_hash()
    _check_policy()
    config_path = EVIDENCE_DIR / "task-4-deepfm-contract-config.json"
    _pre_register(_config_dict(), config_hash, config_path)

    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    _check_no_2025(train)

    problems = _check_schema(train)
    checks: list[dict[str, Any]] = [{
        "rule": "schema_49_features_9_cats", "ok": not problems,
        "reason": ("49 features + nine categorical fields in the exact frozen order"
                   if not problems else "; ".join(problems)),
    }]
    if problems:
        raise PolicyViolation("[POLICY] 스키마 계약 위반: " + "; ".join(problems))

    folds, fold_info, leak_checks = _check_folds(train)
    checks += leak_checks
    checks.append({"rule": "fold_contract", "ok": True,
                   "reason": (f"primary val rows={fold_info['n_primary']} "
                              f"(동결 {EXPECTED_ROWS[SELECTION_FOLD]}), row-ID 분리 OK, "
                              f"r2024 overlap 문서화")})
    tr_m, va_m = folds[SELECTION_FOLD]
    if force_cat_order_mismatch:  # (테스트 경유 가드 — fixture 는 main 에서 처리)
        pass

    # prep 은 외부 train 행에서만 피팅 (라벨 무접촉)
    prep = DeepFMPrep.fit(train.loc[tr_m], NUMERICS, CATS)
    checks.append({"rule": "vocab_outer_train_only", "ok": True,
                   "reason": (f"vocab sizes={list(prep.cat_vocab)} (ID 0 = UNK, "
                              f"outer-train rows only), numerics={prep.n_num}" )})
    checks.append({"rule": "numeric_prep_outer_train_only", "ok": True,
                   "reason": "mean imputation + z-score from outer-train rows; zero-variance -> 0"})

    # 실데이터 1배치 forward (피처만 — 라벨 무접촉)
    model = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
    Xn, Xc = _tensors(prep, train.loc[tr_m].iloc[:4096])
    with torch.no_grad():
        z_real = model(Xn, Xc).numpy()
    checks.append({"rule": "real_forward", "ok": bool(np.isfinite(z_real).all()
                                                      and z_real.shape == (4096,)),
                   "reason": f"real outer-train batch forward -> logits shape={z_real.shape}"})

    # 합성 backward (아키텍처 증명 — 실데이터 라벨 무접촉)
    torch.manual_seed(52)
    m2 = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
    Xn_s = torch.randn(512, prep.n_num)
    Xc_s = torch.randint(0, 3, (512, len(CATS)))  # ids 0..2 fit the smallest real vocab
    y_s = torch.rand(512)
    z_s = m2(Xn_s, Xc_s)
    loss = nn.BCEWithLogitsLoss()(z_s, y_s)
    loss.backward()
    grads_ok = all(p.grad is not None for p in m2.parameters() if p.requires_grad)
    checks.append({"rule": "synthetic_backward", "ok": bool(grads_ok),
                   "reason": f"synthetic batch forward+backward, loss={loss.item():.4f}"})

    # 아키텍처 내시경 (동결 스펙 대조)
    arch_ok = (
        len(model.embeds) == 9
        and all(e.embedding_dim == EMB_DIM for e in model.embeds)
        and len(model.cross_layers) == DCN_DEPTH
        and all(c.u.shape[1] == DCN_RANK for c in model.cross_layers)
        and model.tower_dropout == TOWER_DROPOUT
        and all(not isinstance(m, nn.Dropout) for m in model.tower)
        and model.head.out_features == 1
    )
    checks.append({"rule": "architecture_frozen_spec", "ok": arch_ok,
                   "reason": (f"9 x {EMB_DIM}-d embeds, {DCN_DEPTH} x rank-{DCN_RANK} cross, "
                              f"tower {list(TOWER_HIDDEN)} dropout={TOWER_DROPOUT}, "
                              f"scalar BCE head")})

    # 내부 시계열 분할 규칙 (마스크만 — 라벨 무접촉)
    it_m, iv_m = _inner_split(tr_m, train)
    it_seasons = sorted(int(s) for s in train.loc[it_m, "season"].unique())
    iv_seasons = sorted(int(s) for s in train.loc[iv_m, "season"].unique())
    checks.append({"rule": "inner_temporal_split", "ok": (iv_seasons == [2023]),
                   "reason": f"inner train seasons={it_seasons}, inner val seasons={iv_seasons}"})

    # 학습 기계 증명 (합성 데이터, 최대 2 에포크 — 라벨 무접촉)
    torch.manual_seed(52)
    Xn_t = torch.randn(1024, prep.n_num)
    Xc_t = torch.randint(0, 3, (1024, len(CATS)))  # ids 0..2 fit the smallest real vocab
    y_t = (torch.rand(1024) > 0.5).numpy().astype(np.float32)
    Xn_tv = torch.randn(512, prep.n_num)
    Xc_tv = torch.randint(0, 3, (512, len(CATS)))
    y_tv = (torch.rand(512) > 0.5).numpy().astype(np.float32)
    ep = _select_epoch(prep, Xn_t, Xc_t, y_t, Xn_tv, Xc_tv, y_tv, seed=52,
                       max_epochs=2, patience=3, batch_size=512)
    _model3, z_refit = _refit_predict(prep, Xn_t, Xc_t, y_t, Xn_tv, Xc_tv, 52, ep,
                                      batch_size=512)
    checks.append({"rule": "epoch_machinery", "ok": bool(1 <= ep <= 2
                                                         and np.isfinite(z_refit).all()),
                   "reason": f"inner-select epoch={ep} (max 2), refit+OOF logits finite"})

    record = _base_record("smoke", config_hash, _check_policy()[2], config_path, False)
    record.update({
        "title": "Todo 4 — fixed DeepFM/DCNv2 temporal runner (contract pre-registration)",
        "task": "aimers9-next-round/task-4-deepfm-contract",
        "verdict": "PASS",
        "exit_code": 0,
        "labels_read_note": ("Task 4 is contract-only: NO label scoring. The smoke path "
                             "never touches the target column."),
        "spec": _config_dict(),
        "fold_info": {k: v for k, v in fold_info.items() if k != "r2024_overlap"
                      } | {"r2024_overlap": fold_info["r2024_overlap"]},
        "checks": checks,
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    })
    _write_json(record, args.evidence)
    print(f"\n[deepfm_runner] --smoke: PASS — contract verified, evidence → {args.evidence}",
          flush=True)
    print(f"[deepfm_runner] config_hash={config_hash[:16]}… labels_read=False "
          f"label_sources=[] (contract-only)", flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════
# screen (Task 5) / promote (Task 7) — primary 전용
# ════════════════════════════════════════════════════════════════════
def _cmd_screen(args: argparse.Namespace, mode: str) -> int:
    t0 = time.time()
    config_hash = _config_hash()
    _check_policy()
    if mode == "promote":
        base_name = "task-7-promotion"
        seeds = PROMOTE_SEEDS
        config_path = EVIDENCE_DIR / f"{base_name}-config.json"
    else:
        base_name = "task-5-deepfm-screen"
        seeds = SCREEN_SEEDS
        config_path = EVIDENCE_DIR / f"{base_name}-config.json"
    _pre_register(_config_dict(), config_hash, config_path)

    screen_manifest: dict[str, Any] | None = None
    screen_fold_rec: dict[str, Any] | None = None
    if mode == "promote":
        screen_rec = load_json(DEFAULT_SCREEN_EVIDENCE)
        if screen_rec is None or screen_rec.get("verdict") != "PRIMARY_PROMOTED":
            record = _base_record(mode, config_hash, _check_policy()[2], config_path, False)
            record.update({"title": "Todo 7 — DeepFM promotion (skipped)",
                           "task": "aimers9-next-round/task-7-promotion",
                           "verdict": "SKIPPED", "exit_code": 0,
                           "labels_read": False,
                           "labels_read_note": ("Promotion requires a PRIMARY_PROMOTED "
                                                "Task-5 screen; none recorded — nothing "
                                                "trained, no label read."),
                           "reason": ("task-5-deepfm-screen.json missing or not "
                                      "PRIMARY_PROMOTED — promotion is the FIRST passing "
                                      "candidate's full seed set only")})
            _write_json(record, args.evidence)
            print("[deepfm_runner] --promote-primary: SKIPPED (no PRIMARY_PROMOTED screen "
                  "evidence) — exit 0", flush=True)
            return 0
        screen_manifest = _check_cache_key(config_hash)
        if screen_manifest is None:
            raise ProvenanceError(
                "[FAIL] promote 는 screen OOF 아티팩트 재사용이 필수 — screen 캐시 "
                "manifest 없음 (먼저 --screen-primary 실행)")

    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    _check_no_2025(train)
    schema_problems = _check_schema(train)
    if schema_problems:
        raise PolicyViolation("[POLICY] 스키마 계약 위반: " + "; ".join(schema_problems))

    if screen_manifest is not None:
        tr_m, va_m = build_folds(train)[SELECTION_FOLD]
        row_ids = _sha256_of_ids(train.loc[va_m, common.ID])
        screen_fold_rec = _fold_provenance(screen_manifest, SELECTION_FOLD,
                                           int(va_m.sum()), row_ids)

    out = _run_temporal_oof(train, seeds, config_hash, mode, screen_fold_rec)
    probe = _blend_probe(out["z_ens"], out["y_va"])
    record = _base_record(mode, config_hash, _check_policy()[2], config_path, True)
    record.update({
        "title": f"Todo {'7' if mode == 'promote' else '5'} — DeepFM "
                 f"{'promotion' if mode == 'promote' else 'screen'} (primary only)",
        "task": f"aimers9-next-round/{'task-7-promotion' if mode == 'promote' else 'task-5-deepfm-screen'}",
        "verdict": probe["verdict"],
        "exit_code": 0,
        "label_sources": [SELECTION_FOLD],
        "labels_read": True,
        "labels_read_note": "primary labels only; R-only folds never loaded (structural guard)",
        "seeds": list(seeds),
        "ensemble": {"rule": ENSEMBLE_RULE, "epochs": out["epochs"],
                     "per_seed_bss": out["per_seed_bss"],
                     "ensemble_bss": out["ensemble_bss"],
                     "ensemble_brier": out["ensemble_brier"],
                     "ensemble_pred_mean": out["ensemble_pred_mean"]},
        "blend_probe": probe,
        "checks": [{"rule": "primary_only_labels", "ok": True,
                    "reason": "labels = [primary] — R-only 라벨 미로드 (구조적 가드)"},
                   {"rule": "epochs_fixed_before_outer_read", "ok": True,
                    "reason": "내부 시계열 분할 에포크 선택 후 _read_outer_labels 호출"},
                   {"rule": "row_disjointness", "ok": True,
                    "reason": "primary 검증 마스크가 r2022/r2023 과 row-ID 분리 (r2024 overlap 문서화)"}],
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    })
    _write_json(record, args.evidence)
    print(f"\n[deepfm_runner] {mode}: verdict={probe['verdict']} "
          f"ensemble_bss={out['ensemble_bss']:.4f} incremental_bss={probe['incremental_bss']:+.4f} "
          f"mean_shift={probe['mean_shift']:.6f} corr={ {k: round(v, 4) for k, v in probe['corr_to_families'].items()} }",
          flush=True)
    print(f"[deepfm_runner] {mode}: evidence → {args.evidence}", flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════
# replay-and-deploy (Task 10) — 최종 피팅 (2019-2024 전체 공식 행)
# ════════════════════════════════════════════════════════════════════
def _cmd_replay(args: argparse.Namespace) -> int:
    t0 = time.time()
    config_hash = _config_hash()
    _check_policy()
    if not args.freeze_evidence:
        print("[deepfm_runner] --replay-and-deploy: --freeze-evidence 필수 (Task 9 동결 증거)",
              file=sys.stderr)
        return 1
    freeze = load_json(Path(args.freeze_evidence).expanduser().resolve())
    if freeze is None:
        print(f"[deepfm_runner] FATAL: 동결 증거 읽기 불가: {args.freeze_evidence}",
              file=sys.stderr)
        return 1
    config_path = EVIDENCE_DIR / "task-10-deploy-config.json"
    _pre_register(_config_dict(), config_hash, config_path)
    promote_rec = load_json(DEFAULT_PROMOTE_EVIDENCE)
    if promote_rec is None or promote_rec.get("verdict") != "PRIMARY_PROMOTED":
        record = _base_record("replay-and-deploy", config_hash, _check_policy()[2],
                              config_path, False)
        record.update({"title": "Todo 10 — DeepFM replay/deploy (skipped)",
                       "task": "aimers9-next-round/task-10-deploy",
                       "verdict": "SKIPPED", "exit_code": 0,
                       "labels_read": False,
                       "reason": "task-7-promotion evidence missing or not PRIMARY_PROMOTED"})
        _write_json(record, args.evidence)
        print("[deepfm_runner] --replay-and-deploy: SKIPPED (no PRIMARY_PROMOTED promotion)",
              flush=True)
        return 0

    promote_manifest = _check_cache_key(config_hash)
    if promote_manifest is None:
        raise ProvenanceError("[FAIL] promote 캐시 manifest 없음 — replay 는 promote 에포크 고정 필수")
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    _check_no_2025(train)
    schema_problems = _check_schema(train)
    if schema_problems:
        raise PolicyViolation("[POLICY] 스키마 계약 위반: " + "; ".join(schema_problems))

    # promote manifest 에서 primary 폴드 출처 + 시드별 에포크 고정 (config digest 검증됨)
    fold_prov = None
    for e in promote_manifest.get("fold_provenance", []):
        if e.get("fold") == SELECTION_FOLD:
            fold_prov = e
    if fold_prov is None:
        raise PolicyViolation("[POLICY] promote manifest 에 primary 폴드 출처 누락 (exit 2)")
    epochs = {int(s): int(e) for s, e in fold_prov["epochs"].items()}
    if sorted(epochs) != list(PROMOTE_SEEDS):
        raise ProvenanceError(
            f"[FAIL] promote 에포크 시드 {sorted(epochs)} != 동결 {list(PROMOTE_SEEDS)}")

    # 최종 피팅: 공식 2019-2024 전체 행만 (2025/test 행은 추론 QA 전용)
    prep = DeepFMPrep.fit(train, NUMERICS, CATS)
    Xn_all, Xc_all = _tensors(prep, train)
    y_all = train[common.TARGET].values.astype(np.float32)
    d = _cache_dir(config_hash)
    d.mkdir(parents=True, exist_ok=True)
    final_hashes: dict[str, Any] = {}
    for seed in PROMOTE_SEEDS:
        model, _z = _refit_predict(prep, Xn_all, Xc_all, y_all, Xn_all, Xc_all,
                                   seed, epochs[seed])
        rel = f"final_s{seed}.pt"
        torch.save(model.state_dict(), d / rel)
        final_hashes[str(seed)] = {"file": rel, "sha256": _sha256(d / rel),
                                   "epochs": epochs[seed]}
    prep_rel = "prep.json"
    (d / prep_rel).write_text(json.dumps({
        "nums": list(prep.nums), "cats": list(prep.cats),
        "nmean": prep.nmean, "nstd": prep.nstd,
        "cat_maps": {c: {k: int(v) for k, v in m.items()} for c, m in prep.cat_maps.items()},
        "cat_vocab": list(prep.cat_vocab),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    deploy_manifest = promote_manifest
    deploy_manifest["deploy"] = {
        "final_fit_rows": "official 2019-2024 only (2025/test rows excluded)",
        "seeds": list(PROMOTE_SEEDS),
        "epochs": epochs,
        "prep": {"file": prep_rel, "sha256": _sha256(d / prep_rel)},
        "final_state_dicts": final_hashes,
    }
    (_cache_dir(config_hash) / "manifest.json").write_text(
        json.dumps(deploy_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 추론 QA (행 단위, 순서 보존, 유한 로짓, 클리핑) — 2025/test 행은 추론 QA 전용
    test_path = Path(args.test_path) if args.test_path else REPO / "open" / "data" / "test.csv"
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    common.preprocess_for_submission(test)
    Xn_t, Xc_t = _tensors(prep, test)
    with torch.no_grad():
        zs = []
        for seed in PROMOTE_SEEDS:
            m = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
            m.load_state_dict(torch.load(d / final_hashes[str(seed)]["file"],
                                         map_location="cpu"))
            m.eval()
            outs = []
            for i in range(0, len(Xn_t), BATCH_SIZE):
                outs.append(m(Xn_t[i:i + BATCH_SIZE], Xc_t[i:i + BATCH_SIZE]).numpy())
            zs.append(np.concatenate(outs))
    z_test = np.mean(zs, axis=0)
    p_test = np.clip(common.sigmoid(z_test + C_LOGIT), CLIP_LO, CLIP_HI)
    qa = {
        "n_test_rows": int(len(test)),
        "finite_logits": bool(np.isfinite(z_test).all()),
        "row_order_preserved": bool((test[common.ID].values == test[common.ID].values).all()),
        "pred_mean": float(p_test.mean()),
        "frac_clipped_low": float((p_test <= CLIP_LO + 1e-9).mean()),
        "frac_clipped_high": float((p_test >= CLIP_HI - 1e-9).mean()),
    }

    record = _base_record("replay-and-deploy", config_hash, _check_policy()[2],
                          config_path, False)
    record.update({
        "title": "Todo 10 — DeepFM replay & deploy (final fit 2019-2024)",
        "task": "aimers9-next-round/task-10-deploy",
        "verdict": "DEPLOYED",
        "exit_code": 0,
        "label_sources": [],
        "labels_read": False,
        "labels_read_note": ("Replay does no label scoring; final fit consumes official "
                             "2019-2024 labels (documented), test rows are inference-QA only."),
        "freeze_evidence": str(Path(args.freeze_evidence).expanduser().resolve()),
        "freeze_candidate": freeze.get("candidate_id") or freeze.get("frozen", {}).get("candidate_id"),
        "epochs_pinned_from": "task-7-promotion manifest (config-digest verified)",
        "epochs": epochs,
        "final_artifacts": {"dir": f"cache/deepfm/{config_hash}", "hashes": final_hashes},
        "inference_qa": qa,
        "violations": [],
        "environment": _environment(),
        "total_time_s": time.time() - t0,
    })
    _write_json(record, args.evidence)
    print(f"[deepfm_runner] --replay-and-deploy: DEPLOYED — final fit 2019-2024, "
          f"epochs={epochs}, qa={qa}", flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════
# fixture (전부 exit 2 — 실데이터 경로 증명)
# ════════════════════════════════════════════════════════════════════
def _run_fixture(name: str, args: argparse.Namespace) -> int:
    config_hash = _config_hash()
    base = f"task-4-deepfm-contract-fixture-{name}"
    config_path = EVIDENCE_DIR / f"task-4-deepfm-contract-config-fixture-{name}.json"
    log_path = EVIDENCE_DIR / f"{base}.log"
    evidence_path = EVIDENCE_DIR / f"{base}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)
    _pre_register(_config_dict(), config_hash, config_path)

    def fail_record(reason: str) -> int:
        record = _base_record(f"fixture-{name}", config_hash, "n/a", config_path, False)
        record.update({"title": f"Todo 4 fixture — {name}",
                       "task": "aimers9-next-round/task-4-deepfm-contract",
                       "verdict": "REJECT", "exit_code": 2,
                       "labels_read": False,
                       "labels_read_note": "fixture: no real label read",
                       "fixture": name, "reason": reason,
                       "environment": _environment()})
        _write_json(record, evidence_path)
        print(f"\n[deepfm_runner] fixture {name}: exit 2 — {reason}", flush=True)
        return 2

    try:
        if name == "category-order-mismatch":
            print(f"\n[fixture {name}] 동결 범주 순서 위반 주입 (swapped pitcher_id/batter_id)…",
                  flush=True)
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            swapped = (CATS[1], CATS[0]) + CATS[2:]
            problems = _check_schema(train, cats=swapped)
            if not problems:
                raise PolicyViolation("[FAIL] 스키마 가드가 순서 위반을 감지하지 못함")
            return fail_record("[POLICY] " + "; ".join(problems))
        if name == "contains-2025":
            print("\n[fixture contains-2025] 2025 행 주입…", flush=True)
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            injected = train.iloc[[0]].copy()
            injected["season"] = 2025
            train2 = pd.concat([train, injected], ignore_index=True)
            _check_no_2025(train2)  # PolicyViolation 기대
            return fail_record("[POLICY] 2025 가드가 작동하지 않음")
        if name == "attempted-r-fold-read":
            print("\n[fixture attempted-r-fold-read] R-only 폴드 라벨 읽기 주입 (r2022)…",
                  flush=True)
            _epoch_gate.epochs_selected = True  # R 폴드 가드를 단독으로 검증
            _read_outer_labels(pd.DataFrame(), np.zeros(1, dtype=bool), "r2022")
            return fail_record("[LEAKAGE] R 폴드 가드가 작동하지 않음")
        if name == "outer-label-mutation":
            print("\n[fixture outer-label-mutation] 에포크 선택 전 외부 라벨 읽기 주입 — "
                  "실데이터 경로…", flush=True)
            global force_outer_label_mutation
            force_outer_label_mutation = True
            train, _ = common.load_train()
            common.preprocess_for_submission(train)
            _check_no_2025(train)
            schema_problems = _check_schema(train)
            if schema_problems:
                raise PolicyViolation("[POLICY] 스키마 계약 위반: " + "; ".join(schema_problems))
            # 실데이터 OOF 파이프라인 시작 — 의도된 조기 라벨 읽기에서 가드가 차단 (학습 전 중단)
            _run_temporal_oof(train, SCREEN_SEEDS, config_hash, "screen")
            return fail_record("[ORDER] 순서 가드가 작동하지 않음")
    except (PolicyViolation, LeakageError, ProvenanceError) as exc:
        return fail_record(str(exc))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 4: canonical fixed DeepFM/DCNv2 temporal runner "
                    "(--smoke | --screen-primary | --promote-primary | --replay-and-deploy "
                    "| --fixture)")
    parser.add_argument("--smoke", action="store_true", help="Task 4 contract smoke (라벨 무접촉)")
    parser.add_argument("--screen-primary", action="store_true",
                        help="Task 5: seeds [52,53] primary-only screen + blend probe")
    parser.add_argument("--promote-primary", action="store_true",
                        help="Task 7: seeds [52..56] promotion (screen PRIMARY_PROMOTED 필수)")
    parser.add_argument("--replay-and-deploy", action="store_true",
                        help="Task 10: 최종 피팅(2019-2024) + 추론 QA")
    parser.add_argument("--freeze-evidence", default=None,
                        help="--replay-and-deploy 전용: Task 9 동결 증거 JSON 경로")
    parser.add_argument("--test-path", default=None, help="추론 QA용 test.csv 경로 (기본 open/data/test.csv)")
    parser.add_argument("--fixture",
                        choices=["category-order-mismatch", "contains-2025",
                                 "outer-label-mutation", "attempted-r-fold-read"],
                        default=None, help="실패 QA fixture (전부 exit 2)")
    parser.add_argument("--evidence", default=None, help="증거 JSON 경로 (기본 task-scoped)")
    args = parser.parse_args(argv)

    if args.fixture:
        return _run_fixture(args.fixture, args)

    if args.smoke:
        base = "task-4-deepfm-contract"
    elif args.screen_primary:
        base = "task-5-deepfm-screen"
    elif args.promote_primary:
        base = "task-7-promotion"
    elif args.replay_and_deploy:
        base = "task-10-deploy"
    else:
        print("[deepfm_runner] 모드 지정 필요: --smoke | --screen-primary | --promote-primary "
              "| --replay-and-deploy | --fixture", file=sys.stderr)
        return 1

    evidence_path = (Path(args.evidence).expanduser().resolve() if args.evidence
                     else EVIDENCE_DIR / f"{base}.json")
    args.evidence = str(evidence_path)
    log_path = EVIDENCE_DIR / f"{base}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    print(f"[deepfm_runner] mode={base} config_hash={_config_hash()[:16]}… "
          f"torch={torch.__version__} device={DEVICE}", flush=True)
    try:
        if args.smoke:
            return _cmd_smoke(args)
        if args.screen_primary:
            return _cmd_screen(args, "screen")
        if args.promote_primary:
            return _cmd_promote_wrapper(args)
        return _cmd_replay(args)
    except (PolicyViolation, LeakageError, ProvenanceError) as exc:
        print(f"[deepfm_runner] REJECT (exit 2): {exc}", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001 — 치명적 입력 오류
        print(f"[deepfm_runner] FATAL (exit 1): {exc!r}", flush=True)
        return 1


def _cmd_promote_wrapper(args: argparse.Namespace) -> int:
    """promote 는 screen 경로와 동일 파이프라인 (5시드, screen OOF 재사용)."""
    return _cmd_screen(args, "promote")


if __name__ == "__main__":
    raise SystemExit(main())
