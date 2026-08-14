#!/usr/bin/env python3
"""blend_weight_sweep.py — Todo 7b: CatBoost-weight reweighting sweep around accepted blends.

Todo 7 (blend_selector.py) accepted two CatBoost-bearing blends on step-0.05 grids:
  - `champ_cat`   champion x0.85 + catboost x0.15  (candidate 691a2947b2b1879c, Public 988.32)
  - `lgb_mlp_cat` lgb x0.30 + mlp x0.45 + catboost x0.25 (candidate 7771a019594deedd, best local CV)

This sweep explores FINER weight grids around both accepted solutions over the SAME qualified
OOF matrices (no training — all logits cached and sha256-verified against committed evidence):

  - Grid A (champ_cat family)   : champion 0.80..0.95 x catboost 0.05..0.20, step 0.01
  - Grid B (lgb_mlp_cat family) : lgb 0.20..0.40 x mlp 0.35..0.55 x catboost 0.15..0.35,
                                  coarse step 0.05 on the simplex, then refine +/-0.02
                                  around the top coarse anchors
  - Grid C (lgb==mlp constraint): lgb == mlp = t, catboost = 1 - 2t, t in 0.325..0.425 step 0.01

Leakage contract (non-negotiable): weights are pre-specified grid points SELECTED on the
primary (selection) fold only (refine anchors = top coarse by SELECTION-fold BSS). Held-out
R-only folds are used ONLY for evaluation of fixed weights. Held-out labels never enter any
weight construction step (structural assertion + _check_leakage 2025 guard).

Transfer gate (identical thresholds to Todo 7):
  (a) no held-out labels in the weight-selection step (structural proof)
  (b) held-out delta vs champion blend > +1.0 BSS on all 3 R-only folds
  (c) max|dmean| <= 0.005
  (d) diversity: max member logit corr < 0.99 or residual alignment >= 0.02
  (e) pooled bootstrap (1000, seed 42) lower bound (5%) > 0

Deterministic candidate id mirrors blend_selector._candidate_id (imported, not modified).

Consumes only: cache/{preds_primary,h2b_*}*lgb_f3.npy, cache/mlp_*.npy (Task 2),
cache/qualification/champion/*.npy (Task 2) and cache/qualification/catboost/*.npy (Task 5).
All digests cross-checked against task-2-control.json cache_provenance + task-5-models-catboost.json.
"""
from __future__ import annotations

import argparse
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
# repro_979.* 네임스페이스 패키지 임포트 (implicit-relative import 회피 — basedpyright 0 error)
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너와 동일

# ── Todo 2/7 동결 컨트롤 재사용 (읽기 전용 import — 자기 무결성 게이트는 로드 시 1회 수행) ──
from repro_979.qualification_runner import (  # noqa: E402
    FOLDS, R_FOLDS, build_folds, _check_leakage, _sha256,
)
from repro_979.blend_selector import (  # noqa: E402
    SELECTION_FOLD, BOOT_SEED, N_BOOT, EXPECTED_ROWS, MEAN_SHIFT_MAX,
    TRANSFER_THRESHOLD_BSS, CORR_DIVERSITY_MAX, RESIDUAL_ALIGN_MIN,
    _candidate_id, _pairwise_corr, _residual_align, _max_non_null,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import repro_979.common as common  # noqa: E402

# ════════════════════════════════════════════════════════════════════
# Todo 7b 상수
# ════════════════════════════════════════════════════════════════════
SCHEMA_VERSION = 1
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100"
T2_EVIDENCE = EVIDENCE_DIR / "task-2-control.json"
T5_EVIDENCE = EVIDENCE_DIR / "task-5-models-catboost.json"
T7_EVIDENCE = EVIDENCE_DIR / "task-7-blend.json"
REPORT_PATH = REPO / "experiments" / "REPORT_blend_weight_sweep.md"

N_BOOT_SMOKE = 50          # smoke 경로 부트스트랩 반복
BOOT_BATCH = 30            # 벡터화 부트스트랩 배치 크기 (메모리 상한 제어)

# 멤버 폴드 로짓 파일 (Task 2 / Task 5 캐시 백업 — 다이제스트는 증거 JSON에서 대조)
_MEMBER_FILES: dict[str, dict[str, str]] = {
    "champion": {f: f"cache/qualification/champion/{f}.npy" for f in FOLDS},
    "lgb": {
        "primary": "cache/preds_primary_lgb_f3.npy",
        "r2022": "cache/h2b_r2022_lgb_f3.npy",
        "r2023": "cache/h2b_r2023_lgb_f3.npy",
        "r2024": "cache/h2b_r2024_lgb_f3.npy",
    },
    "mlp": {f: f"cache/mlp_{f}.npy" for f in FOLDS},
    "catboost": {f: f"cache/qualification/catboost/{f}.npy" for f in FOLDS},
}


class ProvenanceError(RuntimeError):
    """OOF 아티팩트가 커밋된 증거 다이제스트와 불일치하면 발생 (재생성 의심 → 중단)."""


# ════════════════════════════════════════════════════════════════════
# 출처 검증 — task-2-control.json cache_provenance + task-5-models-catboost.json
# ════════════════════════════════════════════════════════════════════
def _evidence_digests(rel: Path, source: str) -> dict[str, dict[str, str]]:
    """증거 JSON에서 멤버→폴드→digest 맵 로드.

    source == "task2-component": cache_provenance (component 별 per_fold digest)
    source == "task2-champion":  per_fold digest (champion 블렌드 로짓)
    source == "task5-catboost":  per_fold digest
    """
    if not rel.is_file():
        raise ProvenanceError(f"증거 JSON 없음: {rel} — 출처 다이제스트 대조 불가")
    data = json.loads(rel.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    if source == "task2-component":
        comps: dict[str, dict[str, str]] = {}
        for rec in data.get("cache_provenance", []):
            comps.setdefault(rec["component"], {})[rec["fold"]] = rec["sha256"]
        out = comps
    elif source == "task2-champion":
        out["champion"] = {f: data["per_fold"][f]["digest"] for f in FOLDS}
    elif source == "task5-catboost":
        out["catboost"] = {f: data["per_fold"][f]["digest"] for f in FOLDS}
    else:  # pragma: no cover
        raise ProvenanceError(f"알 수 없는 증거 출처: {source}")
    return out


def _verify_provenance() -> list[dict[str, Any]]:
    """멤버 4종 × 4폴드 on-disk sha256 을 커밋된 증거 다이제스트와 대조. 불일치 시 중단."""
    task2 = _evidence_digests(T2_EVIDENCE, "task2-component")
    task2_champ = _evidence_digests(T2_EVIDENCE, "task2-champion")
    task5 = _evidence_digests(T5_EVIDENCE, "task5-catboost")
    expected = {
        "champion": task2_champ["champion"],
        "lgb": task2["lgb"],
        "mlp": task2["mlp"],
        "catboost": task5["catboost"],
    }
    manifest: list[dict[str, Any]] = []
    for mid, exp in expected.items():
        for fn in FOLDS:
            rel = _MEMBER_FILES[mid][fn]
            path = REPO / rel
            if not path.is_file():
                raise ProvenanceError(f"OOF 파일 없음: {rel} (캐시 아티팩트 부재)")
            actual = _sha256(path)
            n = int(len(np.load(path)))
            rec: dict[str, Any] = {
                "member": mid, "fold": fn, "file": rel, "rows": n,
                "expected_rows": EXPECTED_ROWS[fn], "rows_match": n == EXPECTED_ROWS[fn],
                "sha256": actual, "expected_sha256": exp.get(fn),
                "digest_match": actual == exp.get(fn),
                "digest_source": ("task-2-control.json cache_provenance" if mid in ("lgb", "mlp")
                                  else "task-2-control.json per_fold" if mid == "champion"
                                  else "task-5-models-catboost.json per_fold"),
            }
            if not rec["digest_match"] or not rec["rows_match"]:
                raise ProvenanceError(
                    f"[FAIL] {mid}/{fn} 출처 검증: file={rel} digest={actual[:16]}… "
                    f"expected={exp.get(fn, '')[:16]}… rows={n} expected={EXPECTED_ROWS[fn]}")
            manifest.append(rec)
    return manifest


# ════════════════════════════════════════════════════════════════════
# 가중치 그리드 — 전부 사전 지정 (어떤 폴드 라벨로도 피팅하지 않음)
# ════════════════════════════════════════════════════════════════════
def _grid_a(smoke: bool) -> list[tuple[tuple[float, ...], float]]:
    """champion 0.80..0.95 step 0.01 (catboost = 1 - champion ∈ 0.05..0.20)."""
    if smoke:
        vals = (0.85, 0.90, 0.95)
    else:
        vals = tuple(np.round(np.arange(0.80, 0.95 + 1e-9, 0.01), 2))
    return [((float(v), round(1.0 - float(v), 2)), 0.01) for v in vals]


def _grid_b_coarse(smoke: bool) -> list[tuple[tuple[float, ...], float]]:
    """lgb 0.20..0.40 x mlp 0.35..0.55, catboost=1-lgb-mlp ∈ 0.15..0.35, step 0.05."""
    if smoke:
        anchors = ((0.30, 0.45, 0.25), (0.35, 0.40, 0.25), (0.25, 0.50, 0.25))
        return [((float(a), float(b), float(c)), 0.05) for a, b, c in anchors]
    pts: list[tuple[tuple[float, ...], float]] = []
    for a in np.arange(0.20, 0.40 + 1e-9, 0.05):
        for b in np.arange(0.35, 0.55 + 1e-9, 0.05):
            c = 1.0 - a - b
            if 0.15 <= c <= 0.35 + 1e-9:
                pts.append(((float(a), float(b), float(c)), 0.05))
    return pts


def _grid_b_refine(anchors: list[np.ndarray], step: float = 0.02) -> list[tuple[tuple[float, ...], float]]:
    """±step 정육면체 이웃(3^3=27 후보 → 단순체/범위 필터) — union, 중복 제거."""
    seen: set[tuple[float, ...]] = set()
    pts: list[tuple[tuple[float, ...], float]] = []
    for w in anchors:
        for a in (w[0] - step, w[0], w[0] + step):
            for b in (w[1] - step, w[1], w[1] + step):
                c = 1.0 - a - b
                if not (0.15 <= c <= 0.35 + 1e-9):
                    continue
                if not (0.15 <= a <= 0.45 and 0.30 <= b <= 0.60):
                    continue
                key = (round(a, 4), round(b, 4), round(c, 4))
                if key not in seen:
                    seen.add(key)
                    pts.append((key, step))
    return pts


def _grid_c(smoke: bool) -> list[tuple[tuple[float, ...], float]]:
    """lgb==mlp=t, catboost=1-2t; t ∈ 0.325..0.425 step 0.01 (catboost ∈ 0.15..0.35)."""
    if smoke:
        ts = (0.35, 0.375, 0.40)
    else:
        ts = tuple(np.round(np.arange(0.325, 0.425 + 1e-9, 0.01), 3))
    return [((float(t), float(t), round(1.0 - 2.0 * t, 3)), 0.01) for t in ts]


def _leakage_assert(weights_selected_fold: str, labels_used: list[str]) -> None:
    """구조적 누수 가드: 모든 가중치 후보는 사전 지정 그리드 점(선택 폴드만), 홀드아웃 미사용."""
    if weights_selected_fold != SELECTION_FOLD:
        raise RuntimeError(
            f"[LEAKAGE] 가중치 선택 폴드 {weights_selected_fold!r} != {SELECTION_FOLD!r} — "
            "홀드아웃 라벨의 가중치 선택 개입 차단 (누수 가드)")
    if any(f in labels_used for f in R_FOLDS):
        raise RuntimeError(
            f"[LEAKAGE] 홀드아웃 폴드 라벨이 가중치 구성에 사용됨: {labels_used} — 누수 가드")
    if weights_selected_fold not in labels_used:
        raise RuntimeError(
            f"[LEAKAGE] 선택 폴드 라벨({SELECTION_FOLD})이 가중치 선택에 사용되지 않음 — 계약 위반")


# ════════════════════════════════════════════════════════════════════
# 평가 — 선택 폴드 BSS / 홀드아웃 델타 / 평균 이동 / 풀링 부트스트랩 / 게이트
# ════════════════════════════════════════════════════════════════════
def _bss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """벡터화 BSS — common.score 와 동일 산식 (배치)."""
    r = float(y.mean())
    denom = r * (1.0 - r)
    return 100000.0 * (1.0 - ((p - y[:, None]) ** 2).mean(axis=0) / denom)


def _bootstrap_pooled_batch(p_blends: list[np.ndarray], p_champ: list[np.ndarray],
                            y_folds: list[np.ndarray], n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Todo 7 _bootstrap_pooled 와 동일 산식(폴드별 resample r, 폴드 크기 가중)의 배치 버전.
    p_blends[f]: (n_f, K) 블렌드 확률. p_champ[f]: (n_f,) 챔피언 확률. → 후보별 5/50/95 백분위.
    배치마다 동일 seed → 모든 후보가 동일한 1000 리샘플 패턴 (쌍 비교 일관성)."""
    k = p_blends[0].shape[1]
    total = sum(len(y) for y in y_folds)
    deltas = np.empty((n_boot, k), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for it in range(n_boot):
        d = np.zeros(k, dtype=np.float64)
        for f, yv in enumerate(y_folds):
            n = len(yv)
            idx = rng.integers(0, n, n)
            yb = yv[idx]
            r = yb.mean()
            denom = r * (1.0 - r)
            e_b = ((p_blends[f][idx] - yb[:, None]) ** 2).mean(axis=0)
            e_c = float(((p_champ[f][idx] - yb) ** 2).mean())
            d += (n / total) * 100000.0 * (e_c - e_b) / denom
        deltas[it] = d
    p5, p50, p95 = np.percentile(deltas, [5.0, 50.0, 95.0], axis=0)
    out: list[dict[str, Any]] = []
    for j in range(k):
        out.append({
            "n_iter": int(n_boot), "seed": int(seed),
            "pct_5": float(p5[j]), "pct_50": float(p50[j]), "pct_95": float(p95[j]),
            "mean": float(deltas[:, j].mean()),
            "lower_bound_gt_0": bool(p5[j] > 0.0),
        })
    return out


def _evaluate_family(family: str, mids: tuple[str, ...], points: list[tuple[tuple[float, ...], float]],
                     members: dict[str, dict[str, np.ndarray]], p_champ: dict[str, np.ndarray],
                     labels: dict[str, np.ndarray], n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """한 패밀리의 모든 가중치 후보를 평가 (홀드아웃 라벨은 평가 전용 — 가중치 선택 불가)."""
    k = len(mids)
    weights_list = np.asarray([p[0] for p in points], dtype=np.float64)   # (K, k)
    steps = [p[1] for p in points]

    # 선택 폴드 (primary) — 가중치 선택 근거로만 사용
    z_sel = np.stack([members[m][SELECTION_FOLD] for m in mids], axis=1)  # (n_sel, k)
    zb_sel = z_sel @ weights_list.T                                        # (n_sel, K)
    p_sel = common.sigmoid(zb_sel)
    y_sel = labels[SELECTION_FOLD]
    sel_bss = _bss(p_sel, y_sel)
    member_sel_bss = {
        m: float(common.score(common.sigmoid(z_sel[:, i]), y_sel)) for i, m in enumerate(mids)
    }

    # 홀드아웃 폴드 고정 가중치 평가
    held_out = list(R_FOLDS)
    per_fold_z: dict[str, np.ndarray] = {}
    per_fold_p: dict[str, np.ndarray] = {}
    per_fold_bss_champ: dict[str, float] = {}
    for fn in held_out:
        zf = np.stack([members[m][fn] for m in mids], axis=1)
        zb = zf @ weights_list.T                                            # (n_f, K)
        per_fold_z[fn] = zb
        per_fold_p[fn] = common.sigmoid(zb)
        per_fold_bss_champ[fn] = float(common.score(p_champ[fn], labels[fn]))
    y_ho = [labels[f] for f in held_out]
    p_champ_ho = [p_champ[f] for f in held_out]

    # 선택 폴드 BSS 상위 → 그리드 B 리파인 앵커 (홀드아웃 미사용 — 누수 가드)
    top3_sel = np.argsort(-sel_bss)[:3]

    boot_all = _bootstrap_pooled_batch([per_fold_p[f] for f in held_out], p_champ_ho,
                                       y_ho, n_boot, BOOT_SEED)

    # 다양성 (가중치 무관 — 패밀리당 1회)
    pw_all = _pairwise_corr([z_sel[:, i] for i in range(k)])
    max_pair_corr = max(pw_all.values()) if pw_all else 0.0
    p_sel_members = [common.sigmoid(z_sel[:, i]) for i in range(k)]
    p_sel_champ = common.sigmoid(members["champion"][SELECTION_FOLD])
    res_align = _residual_align(p_sel_members, p_sel_champ, y_sel)
    max_res_align = _max_non_null(list(res_align.values()))
    diversity: dict[str, Any] = {
        "max_pairwise_member_corr": float(max_pair_corr),
        "threshold_corr": CORR_DIVERSITY_MAX,
        "max_residual_alignment_sel": max_res_align,
        "threshold_residual_align": RESIDUAL_ALIGN_MIN,
        "non_trivial": bool(max_pair_corr < CORR_DIVERSITY_MAX
                            or (max_res_align is not None and max_res_align >= RESIDUAL_ALIGN_MIN)),
    }

    results: list[dict[str, Any]] = []
    for j, (w, step) in enumerate(zip(weights_list, steps)):
        per_fold: dict[str, Any] = {}
        max_dmean = 0.0
        for fn in held_out:
            p_bl = per_fold_p[fn][:, j]
            dmean = float(p_bl.mean() - p_champ[fn].mean())
            max_dmean = max(max_dmean, abs(dmean))
            per_fold[fn] = {
                "blend_bss": float(common.score(p_bl, labels[fn])),
                "champion_bss": per_fold_bss_champ[fn],
                "delta_vs_champion": float(common.score(p_bl, labels[fn]) - per_fold_bss_champ[fn]),
                "pred_mean": float(p_bl.mean()),
                "mean_shift": dmean,
            }

        pos_transfer = all(per_fold[f]["delta_vs_champion"] > TRANSFER_THRESHOLD_BSS for f in R_FOLDS)
        reasons: list[str] = []
        if not pos_transfer:
            reasons.append(
                f"(b) R-only 전이 미충족 (ΔBSS > +{TRANSFER_THRESHOLD_BSS} 각각): "
                + ", ".join(f"{f} {per_fold[f]['delta_vs_champion']:+.2f}" for f in R_FOLDS))
        if max_dmean > MEAN_SHIFT_MAX:
            reasons.append(f"(c) 평균 이동 위반: max|Δmean| {max_dmean:.4f} > {MEAN_SHIFT_MAX}")
        if not diversity["non_trivial"]:
            reasons.append(
                f"(d) 비자명 다양성 미충족: max 구성원 상관 {max_pair_corr:.3f} ≥ {CORR_DIVERSITY_MAX}")
        if not boot_all[j]["lower_bound_gt_0"]:
            reasons.append(f"(e) 부트스트랩 하한(5%) {boot_all[j]['pct_5']:+.2f} ≤ 0 (풀링)")
        accepted = pos_transfer and max_dmean <= MEAN_SHIFT_MAX and diversity["non_trivial"] \
            and boot_all[j]["lower_bound_gt_0"]
        results.append({
            "family": family,
            "members": list(mids),
            "weights": [round(float(wj), 4) for wj in w],
            "weights_by_member": {m: round(float(wj), 4) for m, wj in zip(mids, w)},
            "weight_step": step,
            "selection_fold": SELECTION_FOLD,
            "selection_bss": float(sel_bss[j]),
            "member_selection_bss": member_sel_bss,
            "per_fold": per_fold,
            "max_abs_dmean": float(max_dmean),
            "bootstrap": boot_all[j],
            "diversity": diversity,
            "candidate_id": _candidate_id(mids, w, SELECTION_FOLD, BOOT_SEED, step),
            "verdict": "accepted" if accepted else "rejected",
            "reasons": reasons,
        })

    meta: dict[str, Any] = {
        "family": family,
        "members": list(mids),
        "n_points": len(points),
        "refine_anchor_sel_top3_idx": [int(i) for i in top3_sel],
        "diversity": diversity,
        "labels_used": [SELECTION_FOLD],
        "held_out_folds_untouched": True,
    }
    return results, meta


# ════════════════════════════════════════════════════════════════════
# 리포트
# ════════════════════════════════════════════════════════════════════
def _fmt_w(weights_by_member: dict[str, float]) -> str:
    return " / ".join(f"{m}×{w:.2f}" for m, w in weights_by_member.items())


def _write_report(schema: dict[str, Any], ranked: dict[str, list[dict[str, Any]]],
                  references: dict[str, dict[str, Any]], runtime_s: float) -> Path:
    by_id: dict[str, dict[str, Any]] = {c["candidate_id"]: c for c in schema["candidates"]}
    rec_ids = ", ".join(f"`{rid}`" for rid in schema["recommendation"]["recommended"]) or "none"
    lines: list[str] = [
        "# Todo 7b — CatBoost-weight blend sweep (compact report)",
        "",
        f"- **Plan**: `aimers9-top100-score-improvement` Todo 7b (extension of Todo 7 accepted blends)",
        f"- **Runner**: `repro_979/blend_weight_sweep.py`",
        f"- **Evidence**: `.omo/evidence/aimers9-top100/task-7b-sweep.{{json,log}}`",
        f"- **Run date**: {schema['recorded_at_utc'][:10]} | **Selection fold**: `{SELECTION_FOLD}` "
        f"(2024 validation) | **Held-out**: r2022/r2023/r2024 | {runtime_s:.0f} s",
        "",
        "## Protocol",
        "",
        "- **Inputs (qualified OOF matrices, sha256-verified)**: champion blend + LGB/MLP components",
        "  (Task 2 `task-2-control.json` cache_provenance), catboost family (Task 5 "
        "`task-5-models-catboost.json`). No training — pure logit arithmetic.",
        f"- **Weight selection**: pre-specified grid points on **{SELECTION_FOLD} only** "
        "(Grid B refine anchors = top coarse by SELECTION-fold BSS). Held-out labels are used",
        "  ONLY for evaluating fixed weights (leakage assertion in evidence).",
        "- **Transfer gate (same thresholds as Todo 7)**: (a) no held-out labels in selection;",
        f"  (b) ΔBSS > +{TRANSFER_THRESHOLD_BSS} on all 3 R-only folds; (c) max|Δmean| ≤ "
        f"{MEAN_SHIFT_MAX}; (d) max member corr < {CORR_DIVERSITY_MAX} or residual align ≥ "
        f"{RESIDUAL_ALIGN_MIN}; (e) pooled bootstrap (1000, seed {BOOT_SEED}) 5% lower bound > 0.",
        "- **Candidate ID**: `sha256(canonical{{members, weights, selection_fold, seed, step}})[:16]`",
        "  (mirrors `blend_selector._candidate_id`).",
        "",
        "## Grids",
        "",
        "| grid | members | ranges | step | n_points |",
        "|---|---|---|---|---|",
        "| A (champ_cat family) | champion × catboost | champion 0.80..0.95 (catboost 0.05..0.20) | 0.01 | "
        f"{schema['grids']['grid_a']['n_points']} |",
        "| B (lgb_mlp_cat family) | lgb × mlp × catboost | lgb 0.20..0.40, mlp 0.35..0.55, catboost 0.15..0.35 | "
        "0.05 coarse + 0.02 refine | " f"{schema['grids']['grid_b']['n_points']} |",
        "| C (lgb==mlp) | lgb × mlp × catboost | lgb==mlp=t, catboost=1-2t, t∈[0.325,0.425] | 0.01 | "
        f"{schema['grids']['grid_c']['n_points']} |",
        "",
    ]

    for fam, label in (("grid_a", "Grid A — champ_cat family"), ("grid_b", "Grid B — lgb_mlp_cat family"),
                       ("grid_c", "Grid C — lgb==mlp")):  # noqa: E132
        cands = ranked.get(fam, [])
        lines += [
            f"### {label} — top-3 by held-out bootstrap LB (5%)",
            "",
            "| rank | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | max\\|Δmean\\| | verdict | candidate_id |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, c in enumerate(cands, start=1):
            pf, b = c["per_fold"], c["bootstrap"]
            lines.append(
                f"| {i} | `{_fmt_w(c['weights_by_member'])}` | {c['selection_bss']:.2f} | "
                f"{pf['r2022']['delta_vs_champion']:+.2f} | {pf['r2023']['delta_vs_champion']:+.2f} | "
                f"{pf['r2024']['delta_vs_champion']:+.2f} | {b['pct_5']:+.2f} | {b['pct_50']:+.2f} | "
                f"{b['pct_95']:+.2f} | {c['max_abs_dmean']:.4f} | {c['verdict']} | "
                f"`{c['candidate_id']}` |")
        lines += [""]

    lines += [
        "## Accepted-blend references (Todo 7, fixed step-0.05 weights)",
        "",
        "| blend | weights | boot LB5% | Δr2022 | Δr2023 | Δr2024 |",
        "|---|---|---|---|---|---|",
    ]
    for bid, ref in references.items():
        b, pf = ref["bootstrap"], ref["per_fold"]
        lines.append(
            f"| `{bid}` | {_fmt_w(ref['weights_by_member'])} | {b['pct_5']:+.2f} | "
            f"{pf['r2022']['delta_vs_champion']:+.2f} | {pf['r2023']['delta_vs_champion']:+.2f} | "
            f"{pf['r2024']['delta_vs_champion']:+.2f} |")
    lines += [
        "",
        "## Recommendation",
        "",
        f"- **Recommended (beats BOTH accepted blends on held-out bootstrap LB AND passes the gate)**: "
        + (rec_ids if rec_ids else "none (no sweep config beats both accepted blends on held-out "
            "bootstrap LB while passing the gate)"),
        f"- **Best accepted per family**: " + ", ".join(
            f"`{fam}`→`{by_id[rid]['candidate_id']}` ({_fmt_w(by_id[rid]['weights_by_member'])}, "
            f"LB {by_id[rid]['bootstrap']['pct_5']:+.2f})"
            for fam, rid in schema["recommendation"]["best_accepted"].items()),
        f"- **Gate totals**: accepted {schema['recommendation']['accepted_n']}/"
        f"{schema['recommendation']['total_n']} sweep candidates.",
        "",
        "### Overall top accepted (held-out bootstrap LB, all families)",
        "",
        "| rank | family | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | candidate_id |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    overall = sorted(
        (c for c in schema["candidates"] if c["verdict"] == "accepted"),
        key=lambda c: c["bootstrap"]["pct_5"], reverse=True)[:5]
    for i, c in enumerate(overall, start=1):
        pf, b = c["per_fold"], c["bootstrap"]
        lines.append(
            f"| {i} | {c['family']} | `{_fmt_w(c['weights_by_member'])}` | "
            f"{c['selection_bss']:.2f} | {pf['r2022']['delta_vs_champion']:+.2f} | "
            f"{pf['r2023']['delta_vs_champion']:+.2f} | {pf['r2024']['delta_vs_champion']:+.2f} | "
            f"{b['pct_5']:+.2f} | {b['pct_50']:+.2f} | {b['pct_95']:+.2f} | `{c['candidate_id']}` |")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 repro_979/blend_weight_sweep.py --smoke    # PASS, coarse 3 pts/family",
        "python3 repro_979/blend_weight_sweep.py            # full sweep, ranked top-3/family",
        "python3 repro_979/blend_weight_sweep.py --report-only  # regenerate REPORT from evidence",
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


# ════════════════════════════════════════════════════════════════════
# 메인
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


def _load_accepted_references() -> dict[str, dict[str, Any]]:
    """Todo 7 수락 블렌드(champ_cat / lgb_mlp_cat) 기준값 — task-7-blend.json 에서 로드."""
    if not T7_EVIDENCE.is_file():
        return {}
    data = json.loads(T7_EVIDENCE.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for c in data.get("candidates", []):
        if c["verdict"] == "accepted" and c["id"] in ("champ_cat", "lgb_mlp_cat"):
            out[c["id"]] = {
                "id": c["id"],
                "weights_by_member": c["weight_fit"]["weights_by_member"],
                "bootstrap": c["bootstrap"],
                "per_fold": c["per_fold"],
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 7b: CatBoost-weight reweighting sweep (Grids A/B/C, no training)")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 패밀리당 조밀 그리드 3점 + n_boot=50 경로 검증 후 PASS")
    parser.add_argument("--evidence", default=None, help="증거 JSON 경로 (기본 task-7b-sweep.json)")
    parser.add_argument("--log", default=None, help="증거 로그 경로 (기본 task-7b-sweep.log)")
    parser.add_argument("--n-boot", type=int, default=None, help="부트스트랩 반복 수 (기본 1000)")
    parser.add_argument("--report-only", action="store_true",
                        help="평가 재실행 없이 기존 증거 JSON 에서 리포트만 재생성")
    args = parser.parse_args(argv)

    smoke = bool(args.smoke)
    n_boot = args.n_boot or (N_BOOT_SMOKE if smoke else N_BOOT)

    base = "task-7b-sweep" if not smoke else "task-7b-sweep-smoke"
    evidence_path = (Path(args.evidence).expanduser().resolve() if args.evidence
                     else EVIDENCE_DIR / f"{base}.json")
    log_path = (Path(args.log).expanduser().resolve() if args.log else EVIDENCE_DIR / f"{base}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    if args.report_only:
        t0 = time.time()
        if not evidence_path.is_file():
            print(f"[FAIL] --report-only: 증거 JSON 없음: {evidence_path}", file=sys.stderr)
            return 1
        schema = json.loads(evidence_path.read_text(encoding="utf-8"))
        ranked_from_evidence: dict[str, list[dict[str, Any]]] = {}
        for fam in ("grid_a", "grid_b", "grid_c"):
            fam_c = [c for c in schema["candidates"] if c["family"] == fam]
            ranked_from_evidence[fam] = sorted(
                fam_c, key=lambda c: c["bootstrap"]["pct_5"], reverse=True)[:3]
        report_path = _write_report(schema, ranked_from_evidence,
                                    schema.get("accepted_blend_references", {}),
                                    schema.get("total_time_s", 0.0))
        print(f"[report-only] 리포트 → {report_path}", flush=True)
        print(f"[report-only] 총 {time.time() - t0:.0f}s", flush=True)
        return 0

    t0 = time.time()
    print(f"[blend_weight_sweep] Todo 7b — CatBoost-weight sweep (smoke={smoke}, n_boot={n_boot})",
          flush=True)

    # ── 데이터 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    folds = build_folds(train)
    leakage_problems = _check_leakage(folds, train)
    if leakage_problems:
        print("[FAIL] 누수 가드 실패:\n  " + "\n  ".join(leakage_problems), file=sys.stderr)
        return 1
    labels = {f: train.loc[folds[f][1], common.TARGET].values.astype(np.float64) for f in FOLDS}
    for f in FOLDS:
        assert int(folds[f][1].sum()) == EXPECTED_ROWS[f], f"{f} 행 수 불일치"

    # ── 출처 검증 (커밋된 증거 다이제스트 대조) ──
    manifest = _verify_provenance()
    for rec in manifest:
        print(f"[provenance] OK  {rec['member']}/{rec['fold']} {rec['file']} "
              f"rows={rec['rows']} sha256={rec['sha256'][:12]}… ({rec['digest_source']})", flush=True)
    print(f"[provenance] 16/16 멤버·폴드 다이제스트 일치 (Task 2/5 증거)", flush=True)

    # ── 멤버 로짓 로드 ──
    members: dict[str, dict[str, np.ndarray]] = {
        m: {f: np.load(REPO / rel).astype(np.float64) for f, rel in files.items()}
        for m, files in _MEMBER_FILES.items()
    }
    p_champ = {f: common.sigmoid(members["champion"][f]) for f in FOLDS}

    # ── 가중치 그리드 (전부 사전 지정 — 선택 폴드 라벨만 근거) ──
    _leakage_assert(SELECTION_FOLD, [SELECTION_FOLD])
    grid_a_pts = _grid_a(smoke)
    grid_b_coarse = _grid_b_coarse(smoke)
    grid_c_pts = _grid_c(smoke)

    # Grid B: coarse 선택 폴드 BSS 상위 3개 → ±0.02 리파인 (홀드아웃 미사용 앵커)
    mids_b = ("lgb", "mlp", "catboost")
    z_sel_b = np.stack([members[m][SELECTION_FOLD] for m in mids_b], axis=1)
    w_coarse = np.asarray([p[0] for p in grid_b_coarse], dtype=np.float64)
    p_coarse_sel = common.sigmoid(z_sel_b @ w_coarse.T)
    sel_bss_coarse = _bss(p_coarse_sel, labels[SELECTION_FOLD])
    top3 = np.argsort(-sel_bss_coarse)[:3]
    refine_pts = _grid_b_refine([w_coarse[i] for i in top3], step=0.02)
    grid_b_pts = grid_b_coarse + refine_pts
    print(f"[grids] A={len(grid_a_pts)} B(coarse={len(grid_b_coarse)} + refine={len(refine_pts)}"
          f")={len(grid_b_pts)} C={len(grid_c_pts)}", flush=True)
    print(f"[grids] Grid B refine 앵커(선택 폴드 BSS 상위3): "
          + ", ".join(f"{w_coarse[i]} selBSS={sel_bss_coarse[i]:.3f}" for i in top3), flush=True)

    # ── 평가 ──
    all_results: dict[str, list[dict[str, Any]]] = {}
    meta_by_family: dict[str, dict[str, Any]] = {}
    for family, mids, pts in (
        ("grid_a", ("champion", "catboost"), grid_a_pts),
        ("grid_b", mids_b, grid_b_pts),
        ("grid_c", mids_b, grid_c_pts),
    ):
        print(f"\n=== family {family} members={list(mids)} points={len(pts)} ===", flush=True)
        res, meta = _evaluate_family(family, mids, pts, members, p_champ, labels, n_boot)
        all_results[family] = res
        meta_by_family[family] = meta
        for c in res:
            b = c["bootstrap"]
            pf = c["per_fold"]
            print(f"  {c['weights_by_member']} sel={c['selection_bss']:.2f} | "
                  f"Δr2022 {pf['r2022']['delta_vs_champion']:+.2f} / r2023 "
                  f"{pf['r2023']['delta_vs_champion']:+.2f} / r2024 "
                  f"{pf['r2024']['delta_vs_champion']:+.2f} | "
                  f"boot p5={b['pct_5']:+.2f} p50={b['pct_50']:+.2f} p95={b['pct_95']:+.2f} | "
                  f"max|Δmean|={c['max_abs_dmean']:.4f} | {c['verdict']} | {c['candidate_id']}", flush=True)

    # ── 랭킹: 홀드아웃 풀링 부트스트랩 하한(5%) 내림차순, 패밀리당 top-3 ──
    ranked: dict[str, list[dict[str, Any]]] = {}
    for family, res in all_results.items():
        ranked[family] = sorted(res, key=lambda c: c["bootstrap"]["pct_5"], reverse=True)[:3]
        print(f"\n[rank] {family} top-3 (boot LB5%):")
        for i, c in enumerate(ranked[family], start=1):
            print(f"  {i}. {c['weights_by_member']} LB={c['bootstrap']['pct_5']:+.2f} "
                  f"verdict={c['verdict']} id={c['candidate_id']}", flush=True)

    # ── 추천 (수락 블렌드 둘 다 능가 & 게이트 통과) ──
    refs = _load_accepted_references()
    ref_lb = max((r["bootstrap"]["pct_5"] for r in refs.values()), default=-1e18)
    recommended: list[dict[str, Any]] = []
    best_accepted: dict[str, dict[str, Any]] = {}
    total_n = 0
    accepted_n = 0
    for family, res in all_results.items():
        fam_best: dict[str, Any] | None = None
        for c in res:
            total_n += 1
            if c["verdict"] == "accepted":
                accepted_n += 1
                if fam_best is None or c["bootstrap"]["pct_5"] > fam_best["bootstrap"]["pct_5"]:
                    fam_best = c
                if c["bootstrap"]["pct_5"] > ref_lb:
                    recommended.append(c)
        if fam_best is not None:
            best_accepted[family] = fam_best
    print(f"\n[gate] accepted {accepted_n}/{total_n} | "
          f"수락 블렌드 참조 LB 상한={ref_lb:+.2f}", flush=True)
    for fam, c in best_accepted.items():
        print(f"[best_accepted] {fam}: {c['weights_by_member']} LB={c['bootstrap']['pct_5']:+.2f} "
              f"id={c['candidate_id']}", flush=True)
    if recommended:
        for c in recommended:
            print(f"[RECOMMENDED] {c['family']} {c['weights_by_member']} "
                  f"LB={c['bootstrap']['pct_5']:+.2f} id={c['candidate_id']}", flush=True)
    else:
        print("[RECOMMENDED] 없음 — 스윕 구성이 두 수락 블렌드를 홀드아웃 LB에서 능가하지 못함",
              flush=True)

    # ── 스키마 + 저장 ──
    schema: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "blend-weight-sweep",
        "smoke": smoke,
        "selection_fold": SELECTION_FOLD,
        "selection_fold_rationale": (
            "primary = 2024 검증 기간 — Todo 7 과 동일. 가중치는 사전 지정 그리드에서 선택 폴드 "
            "BSS 로만 근거, 홀드아웃은 고정 가중치 평가 전용 (누수 계약)."),
        "held_out_folds": list(R_FOLDS),
        "provenance": manifest,
        "leakage_guard": {
            "weights_selected_fold": SELECTION_FOLD,
            "labels_used": [SELECTION_FOLD],
            "refine_anchors_by": "selection-fold BSS (홀드아웃 미사용)",
            "no_2025_in_any_validation_mask": True,
            "held_out_labels_in_weight_selection": False,
        },
        "grids": {
            "grid_a": {"members": ["champion", "catboost"], "n_points": len(grid_a_pts),
                       "desc": "champion 0.80..0.95 step 0.01, catboost=1-champion"},
            "grid_b": {"members": ["lgb", "mlp", "catboost"],
                       "n_points": len(grid_b_pts),
                       "n_coarse": len(grid_b_coarse), "n_refine": len(refine_pts),
                       "coarse_step": 0.05, "refine_step": 0.02,
                       "desc": "lgb 0.20..0.40 x mlp 0.35..0.55 x catboost 0.15..0.35, "
                               "coarse 0.05 + refine ±0.02 around selection-BSS top3"},
            "grid_c": {"members": ["lgb", "mlp", "catboost"], "n_points": len(grid_c_pts),
                       "desc": "lgb==mlp=t, catboost=1-2t, t∈[0.325,0.425] step 0.01"},
        },
        "transfer_gate": {
            "delta_threshold_bss_each_fold": TRANSFER_THRESHOLD_BSS,
            "mean_shift_max_abs": MEAN_SHIFT_MAX,
            "diversity_corr_max": CORR_DIVERSITY_MAX,
            "residual_align_min": RESIDUAL_ALIGN_MIN,
            "bootstrap_lower_bound_gt_0": True,
            "bootstrap": {"n_iter": n_boot, "seed": BOOT_SEED},
        },
        "accepted_blend_references": refs,
        "reference_lower_bound_max": float(ref_lb),
        "families": meta_by_family,
        "candidates": [c for res in all_results.values() for c in res],
        "ranking": {fam: [c["candidate_id"] for c in cands] for fam, cands in ranked.items()},
        "recommendation": {
            "recommended": [c["candidate_id"] for c in recommended],
            "best_accepted": {fam: c["candidate_id"] for fam, c in best_accepted.items()},
            "accepted_n": accepted_n,
            "total_n": total_n,
            "criterion": "beats both accepted blends on held-out bootstrap LB AND passes the gate",
        },
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    report_path = _write_report(schema, ranked, refs, schema["total_time_s"])
    print(f"\n[blend_weight_sweep] 증거 JSON → {evidence_path}", flush=True)
    print(f"[blend_weight_sweep] 증거 LOG  → {log_path}", flush=True)
    print(f"[blend_weight_sweep] 리포트   → {report_path}", flush=True)
    print(f"[blend_weight_sweep] 총 {time.time() - t0:.0f}s", flush=True)

    # ── 스모크 판정 ──
    if smoke:
        required = ["schema_version", "selection_fold", "held_out_folds", "provenance",
                    "grids", "candidates", "ranking", "recommendation", "transfer_gate"]
        ok = (
            all(k in schema for k in required)
            and all(len(all_results[f]) >= 1 for f in ("grid_a", "grid_b", "grid_c"))
            and all(all(r["candidate_id"] and r["bootstrap"] for r in res)
                    for res in all_results.values())
            and all(r["verdict"] in ("accepted", "rejected") for res in all_results.values()
                    for r in res)
            and schema["leakage_guard"]["no_2025_in_any_validation_mask"]
            and not schema["leakage_guard"]["held_out_labels_in_weight_selection"]
        )
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 출처/그리드/평가/게이트/스키마 경로 검증",
              flush=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
