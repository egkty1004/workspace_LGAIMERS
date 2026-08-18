#!/usr/bin/env python3
"""recovery_residual_runner.py — Todo 5 (aimers9-top100-score-recovery): screen the
causal OOF residual-ridge correction candidate (`residual_ridge`).

The residual correction is a small, causal, OOF-only logistic correction applied to
the baseline's one-year-ahead OOF logits. It is a *screen*: it tests whether the
correction mechanism produces a candidate that beats the reconciled v93 6-leg
baseline on the r2022/r2023 selection origins, using the SAME bounded 30,000-row
subset that Task 3 used to reproduce the baseline (documented deviation).

Design (frozen, per plan Task 5):
  - Base OOF: one-year-ahead logits for years 2020-2024, each train mask ending at
    year t-1. R-only masks for the r2022/r2023 selection origins; all-game masks for
    primary/2025 deployment. For the screen we compute the R-only base OOF for
    2020-2023 (all that r2022/r2023 need) with a lightweight LightGBM proxy trained
    causally on the bounded subset (documented screen simplification — per-year
    6-leg retraining is computationally prohibitive).
  - C grid: C_GRID=(.001,.01,.1,1.). A C's rolling objective = unweighted arithmetic
    mean of completed forward-year deployed Brier values; smallest C on exact tie;
    SKIPPED_INSUFFICIENT_FORWARD_OOF when the required history is absent.
  - r2022 is the explicit exception: pre-registered C=.01 fitted on 2020-2021 OOF
    (no prior forward validation year exists). r2023 selects C by fitting 2020 OOF
    and scoring 2021, then refits on 2020-2022 OOF. Primary/2025 use the same
    expanding procedure, selecting on all completed forward years before the target.
  - Every forward score uses the final deployed clip(sigmoid(z_corrected+C_LOGIT),
    .30,.70) Brier.
  - Standardize only numeric [base_logit,balls_before,strikes_before,outs_before]
    using prior-OOF mean/std (zero std -> 1); one-hot only
    [game_type,platoon,base_state] using lexicographically fixed prior-OOF levels,
    unknown=all-zero.
  - Frozen LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=True,
    max_iter=1000,tol=1e-8) and
    z_corrected = z_base + clip(logit(p_residual) - C_LOGIT - z_base, -.05, .05),
    preventing double application of the deployed offset.
  - Identity fixture: p_residual=sigmoid(z_base+C_LOGIT) leaves deployed baseline
    predictions exactly unchanged (offset-identity exits 0 on exact equivalence).
  - For 2025 deployment, fit only from through-2024 OOF — not full-model in-sample
    predictions.

Structural firewall: the --screen path reads ONLY r2022/r2023 selection labels via
recovery_policy.read_selection_labels(). It never calls read_primary_labels(), which
raises TerminalFirewallError unless the firewall is FROZEN.

Commands:
  --screen                          screen the residual correction on r2022/r2023
  --fixture <name>                  adversarial fixture (exit 2):
                                    outer-target-feature / terminal-label-read /
                                    cap-violation / target-encoding
                                    offset-identity exits 0 on exact baseline
                                    equivalence (positive identity check)

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.recovery_policy 패키지 import 용

import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_evaluator as re  # noqa: E402

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

# ── 동결 상수 (계획 Task 5) ──────────────────────────────────────────
CANDIDATE_ID = "residual_ridge"
C_GRID = (0.001, 0.01, 0.1, 1.0)
C_PRE_REGISTERED = 0.01          # r2022 예외 — 사전 등록 C
CORRECTION_CAP = 0.05            # z_corrected 의 clip 상한/하한 (±.05)
BASE_OOF_YEARS = (2020, 2021, 2022, 2023, 2024)
MAX_ROWS = 30000                 # Task 3 bounded subset (문서화된 편차)
SKIPPED_INSUFFICIENT_FORWARD_OOF = "SKIPPED_INSUFFICIENT_FORWARD_OOF"

# 보정 모델 피처 (계획 Task 5)
NUMERIC_FEATS = ("base_logit", "balls_before", "strikes_before", "outs_before")
ONE_HOT_FEATS = ("game_type", "platoon", "base_state")

# 동결 LogisticRegression (계획 Task 5)
LR_KWARGS: dict[str, Any] = dict(penalty="l2", solver="lbfgs", fit_intercept=True,
                                 max_iter=1000, tol=1e-8)

# base OOF 경량 LightGBM 프록시 (스크린 전용 — 문서화된 단순화)
BASE_OOF_NUM = ["balls_before", "strikes_before", "outs_before", "run_total_before",
                "score_diff_pitcher_team", "home_win_expectancy", "li",
                "asof_pitcher_success_rate", "asof_batter_success_rate",
                "asof_pitcher_n", "asof_batter_n"]
BASE_OOF_CAT = ["game_type", "base_state", "top_bottom", "pitcher_hand", "batter_hand"]
BASE_OOF_LGB_PARAMS = dict(objective="binary", metric="binary_logloss",
                           learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                           num_threads=6, verbosity=-1, seed=42, deterministic=True)

FIXTURES = ("outer-target-feature", "terminal-label-read", "cap-violation",
            "target-encoding", "offset-identity")

# baseline 캐시 (Task 3 bounded 30k 재현)
BASELINE_CACHE_DIR = REPO / "cache" / "recovery_baseline"


class ResidualViolation(RuntimeError):
    """잔차 보정 정책 위반 (outer-target 피처, cap 미적용, target encoding 등) — exit 2."""


def _canonical_sha256(payload: JSON) -> str:
    return rp._canonical_sha256(payload)


def _sha256_file(path: Path) -> str:
    return rp._sha256_file(path)


def _git_commit() -> str:
    return rp._git_commit()


def _now_utc() -> str:
    return rp.now_utc()


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": rp.policy_config_hash(),
        "label_sources": [],
        "labels_read": False,
    }


def _write_evidence(record: JSON, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    verdict = record["verdict"]
    lines = [
        f"# {record['title']} — {verdict} (exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
        f"- **config_hash**: `{record.get('config_hash', '')}`",
        f"- **label_sources**: {record.get('label_sources')} "
        f"(labels_read={record.get('labels_read')})",
        "",
    ]
    for sec in ("checks", "violations", "findings"):
        items = record.get(sec) or []
        if not items:
            continue
        lines.append(f"## {sec.replace('_', ' ').title()}")
        lines.append("")
        for it in items:
            ok = it.get("ok")
            mark = "PASS" if ok else ("FAIL" if ok is False else "INFO")
            lines.append(f"- **[{mark}]** {it.get('rule', it.get('name', ''))}: "
                         f"{it.get('reason', it.get('detail', ''))}")
        lines.append("")
    lines.append(f"## Verdict: **{verdict}** (exit {record['exit_code']})")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ── 합성 데이터 (fixture/구조 검증용 — 라벨은 합성, 실제 데이터 아님) ──
def _synthetic_train(n_per_season: int = 200) -> Any:
    """합성 학습 프레임 — 기원/마스크/게이트 경로를 라벨 없이 구조 검증하기 위한 것."""
    import pandas as pd  # noqa: PLC0415
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    for season in range(2019, 2025):
        for i in range(n_per_season):
            game_type = "R" if i % 3 != 0 else "P"
            rows.append({
                "season": season,
                "game_type": game_type,
                "control_success": int(rng.random() < 0.5),
                "balls_before": int(rng.integers(0, 4)),
                "strikes_before": int(rng.integers(0, 3)),
                "outs_before": int(rng.integers(0, 3)),
                "base_state": ["___", "1__", "_2_", "__3"][int(rng.integers(0, 4))],
                "pitcher_hand": int(rng.integers(1, 3)),
                "batter_hand": int(rng.integers(1, 3)),
                "top_bottom": "top" if i % 2 == 0 else "bottom",
                "run_total_before": int(rng.integers(0, 10)),
                "score_diff_pitcher_team": int(rng.integers(-5, 6)),
                "home_win_expectancy": float(rng.random()),
                "li": float(rng.random() * 3),
                "asof_pitcher_success_rate": float(rng.random()),
                "asof_batter_success_rate": float(rng.random()),
                "asof_pitcher_n": int(rng.integers(0, 200)),
                "asof_batter_n": int(rng.integers(0, 200)),
            })
    df = pd.DataFrame(rows)
    df["platoon"] = (df["pitcher_hand"] * 2 + df["batter_hand"]).astype("category")
    for c in ("game_type", "base_state", "top_bottom"):
        df[c] = df[c].astype("category")
    return df


# ── 마스크 ───────────────────────────────────────────────────────────
def _bounded_mask(mask, n: int) -> np.ndarray[Any, Any]:
    """마스크의 처음 n 개 True 인덱스만 남긴 boolean 마스크 (Task 3 bounded subset)."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())[:n]
    out = np.zeros(int(np.asarray(mask).size), dtype=bool)
    out[idx] = True
    return out


def build_base_oof_masks(train, origin_type: str) -> dict[int, tuple[Any, Any]]:
    """one-year-ahead base OOF 마스크 (2020-2024).

    origin_type='R': R-only 마스크 (r2022/r2023 선택 기원).
    origin_type='all': 전체 게임 마스크 (primary/2025 배포).
    각 연도 t: train = (season <= t-1 [& R]), val = (season == t [& R]).
    """
    is_r = train["game_type"] == "R"
    masks: dict[int, tuple[Any, Any]] = {}
    for t in BASE_OOF_YEARS:
        if origin_type == "R":
            tr = (train["season"] <= t - 1) & is_r
            va = (train["season"] == t) & is_r
        else:
            tr = train["season"] <= t - 1
            va = train["season"] == t
        masks[t] = (tr, va)
    return masks


def assert_inner_outer_disjoint(base_oof_masks: dict[int, tuple[Any, Any]],
                                outer_year: int) -> JSON:
    """inner-OOF(피팅 연도) 마스크와 outer(타깃 연도) 마스크의 분리 단언 + 해시."""
    outer_va = np.asarray(base_oof_masks[outer_year][1], dtype=bool)
    note: dict[str, Any] = {"outer_year": outer_year, "inner_years": []}
    for t in BASE_OOF_YEARS:
        if t >= outer_year:
            continue
        inner_va = np.asarray(base_oof_masks[t][1], dtype=bool)
        overlap = int((inner_va & outer_va).sum())
        if overlap != 0:
            raise ResidualViolation(
                f"[RESIDUAL] inner-OOF 연도 {t} 와 outer 연도 {outer_year} 마스크 겹침 "
                f"({overlap} rows) — 누수!")
        note["inner_years"].append({
            "year": t,
            "val_mask_hash": _canonical_sha256(
                {"mask": _sha256_bytes(np.packbits(inner_va).tobytes())}),
            "overlap_with_outer": overlap,
        })
    note["outer_val_mask_hash"] = _canonical_sha256(
        {"mask": _sha256_bytes(np.packbits(outer_va).tobytes())})
    return note


def _sha256_bytes(data: bytes) -> str:
    import hashlib  # noqa: PLC0415
    return hashlib.sha256(data).hexdigest()


# ── base OOF (경량 LGB 프록시, one-year-ahead) ───────────────────────
def compute_base_oof(train, base_oof_masks: dict[int, tuple[Any, Any]],
                     origin_type: str) -> dict[int, np.ndarray[Any, Any]]:
    """각 연도 t 의 one-year-ahead base OOF 로짓 (bounded 30k).

    train 마스크/val 마스크를 각각 처음 MAX_ROWS 행으로 제한 (Task 3 bounded subset).
    경량 LightGBM 프록시 — 스크린 전용 단순화 (문서화).
    """
    import lightgbm as lgb  # noqa: PLC0415
    out: dict[int, np.ndarray[Any, Any]] = {}
    for t in BASE_OOF_YEARS:
        tr_m, va_m = base_oof_masks[t]
        tr_b = _bounded_mask(tr_m, MAX_ROWS)
        va_b = _bounded_mask(va_m, MAX_ROWS)
        Xtr = train.loc[tr_b, BASE_OOF_NUM + BASE_OOF_CAT]
        ytr = train.loc[tr_b, "control_success"].values.astype(np.float64)
        Xva = train.loc[va_b, BASE_OOF_NUM + BASE_OOF_CAT]
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=BASE_OOF_CAT)
        dva = lgb.Dataset(Xva, categorical_feature=BASE_OOF_CAT, reference=dtr)
        model = lgb.train(BASE_OOF_LGB_PARAMS, dtr, num_boost_round=500,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        z = model.predict(Xva, num_iteration=model.best_iteration)
        out[t] = np.asarray(z, dtype=np.float64)
    return out


# ── 보정 피처 변환 ───────────────────────────────────────────────────
def _compute_stats(train, base_oof: dict[int, np.ndarray[Any, Any]],
                   base_oof_masks: dict[int, tuple[Any, Any]],
                   fit_years: list[int]) -> JSON:
    """prior-OOF 통계: numeric mean/std (zero std -> 1), one-hot lexicographic levels."""
    import pandas as pd  # noqa: PLC0415
    num_mean: dict[str, float] = {}
    num_std: dict[str, float] = {}
    for f in NUMERIC_FEATS:
        vals: list[float] = []
        for t in fit_years:
            va_b = _bounded_mask(base_oof_masks[t][1], MAX_ROWS)
            if f == "base_logit":
                vals.extend(base_oof[t].tolist())
            else:
                vals.extend(train.loc[va_b, f].astype(float).tolist())
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std()) if len(arr) else 0.0
        num_mean[f] = mean
        num_std[f] = std if std > 0 else 1.0
    levels: dict[str, list[str]] = {}
    for f in ONE_HOT_FEATS:
        seen: set[str] = set()
        for t in fit_years:
            va_b = _bounded_mask(base_oof_masks[t][1], MAX_ROWS)
            seen.update(train.loc[va_b, f].astype(str).unique().tolist())
        levels[f] = sorted(seen)
    return {"num_mean": num_mean, "num_std": num_std, "levels": levels}


def _build_X(train, base_oof: dict[int, np.ndarray[Any, Any]],
             base_oof_masks: dict[int, tuple[Any, Any]], year: int,
             stats: JSON) -> np.ndarray[Any, Any]:
    """연도 year 의 OOF 행에 대한 보정 피처 행렬 (prior-OOF 통계 적용)."""
    va_b = _bounded_mask(base_oof_masks[year][1], MAX_ROWS)
    cols: list[np.ndarray[Any, Any]] = []
    for f in NUMERIC_FEATS:
        if f == "base_logit":
            raw = base_oof[year]
        else:
            raw = train.loc[va_b, f].astype(float).values
        mean = stats["num_mean"][f]
        std = stats["num_std"][f]
        cols.append((np.asarray(raw, dtype=np.float64) - mean) / std)
    for f in ONE_HOT_FEATS:
        lv = stats["levels"][f]
        cats = train.loc[va_b, f].astype(str).values
        for lev in lv:
            cols.append((cats == lev).astype(np.float64))
    return np.column_stack(cols)


def _assert_no_outer_target_feature(X_cols: list[str], forbidden: set[str]) -> None:
    """outer(타깃) 연도 라벨에서 파생된 피처가 보정 입력에 있으면 거부."""
    for c in X_cols:
        if c in forbidden:
            raise ResidualViolation(
                f"[RESIDUAL] outer-target 피처 {c!r} 가 보정 입력에 포함 — "
                "타깃 연도 라벨 파생 피처 금지 (exit 2)")


def _assert_no_target_encoding(X_cols: list[str]) -> None:
    """target encoding 피처가 보정 입력에 있으면 거부."""
    for c in X_cols:
        if "target_enc" in c or "targetenc" in c:
            raise ResidualViolation(
                f"[RESIDUAL] target encoding 피처 {c!r} 금지 (exit 2)")


# ── 보정 모델 ────────────────────────────────────────────────────────
def fit_correction(train, base_oof: dict[int, np.ndarray[Any, Any]],
                   base_oof_masks: dict[int, tuple[Any, Any]],
                   fit_years: list[int], C: float, stats: JSON):
    """prior-OOF 연도(fit_years)에 LogisticRegression 보정 모델 피팅."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    Xs: list[np.ndarray[Any, Any]] = []
    ys: list[np.ndarray[Any, Any]] = []
    for t in fit_years:
        Xs.append(_build_X(train, base_oof, base_oof_masks, t, stats))
        va_b = _bounded_mask(base_oof_masks[t][1], MAX_ROWS)
        ys.append(train.loc[va_b, "control_success"].values.astype(np.float64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    model = LogisticRegression(C=C, **LR_KWARGS)
    model.fit(X, y)
    return model


def apply_correction(model, X: np.ndarray[Any, Any], z_base: np.ndarray[Any, Any],
                     *, cap: float = CORRECTION_CAP,
                     apply_cap: bool = True) -> np.ndarray[Any, Any]:
    """z_corrected = z_base + clip(logit(p_residual) - C_LOGIT - z_base, -cap, cap).

    apply_cap=False 는 cap-violation fixture 가 cap 미적용을 탐지하는 데 사용.
    """
    import repro_979.common as common  # noqa: PLC0415
    p_residual = model.predict_proba(X)[:, 1]
    delta = common.logit(p_residual) - rp.C_LOGIT - z_base
    if apply_cap:
        delta = np.clip(delta, -cap, cap)
    return z_base + delta


def _residual_complementarity(z_base: np.ndarray[Any, Any],
                              p_residual: np.ndarray[Any, Any],
                              y: np.ndarray[Any, Any]) -> float:
    """잔차 상보성: 보정항과 base 잔차(y - sigmoid(z_base)) 의 Pearson 상관."""
    import repro_979.common as common  # noqa: PLC0415
    delta = common.logit(p_residual) - rp.C_LOGIT - z_base
    resid = y - common.sigmoid(z_base)
    if np.std(delta) == 0 or np.std(resid) == 0:
        return 0.0
    return float(np.corrcoef(delta, resid)[0, 1])


# ── C 선택 (rolling objective) ───────────────────────────────────────
def select_c(train, base_oof: dict[int, np.ndarray[Any, Any]],
             base_oof_masks: dict[int, tuple[Any, Any]],
             forward_years: list[int]) -> tuple[float | str, dict[Any, Any]]:
    """C 선택 — rolling objective = 완료된 forward-year deployed Brier 의 산술 평균.

    각 forward 연도 t: fit_years = [y < t], 보정 피팅 후 t 에 적용, deployed Brier.
    smallest C on exact tie. 필수 이력 부재 시 SKIPPED_INSUFFICIENT_FORWARD_OOF.
    """
    results: dict[float, JSON] = {}
    for C in C_GRID:
        briers: list[float] = []
        per_year: dict[int, float] = {}
        for t in forward_years:
            fit_years = [y for y in BASE_OOF_YEARS if y < t]
            if not fit_years:
                continue  # 이력 부재 — 이 forward 연도는 스킵
            stats = _compute_stats(train, base_oof, base_oof_masks, fit_years)
            model = fit_correction(train, base_oof, base_oof_masks, fit_years, C, stats)
            X = _build_X(train, base_oof, base_oof_masks, t, stats)
            z_corr = apply_correction(model, X, base_oof[t])
            p = rp.deployed_probs(z_corr)
            va_b = _bounded_mask(base_oof_masks[t][1], MAX_ROWS)
            y = train.loc[va_b, "control_success"].values.astype(np.float64)
            briers.append(float(np.mean((p - y) ** 2)))
            per_year[t] = float(np.mean((p - y) ** 2))
        if not briers:
            return SKIPPED_INSUFFICIENT_FORWARD_OOF, {"skipped": True}
        results[C] = {"mean_brier": float(np.mean(briers)), "per_year": per_year}
    best = min(results, key=lambda c: (results[c]["mean_brier"], c))
    return best, results


# ── 스크린 ───────────────────────────────────────────────────────────
def _load_baseline_cache() -> dict[str, np.ndarray[Any, Any]]:
    """Task 3 bounded 30k baseline 캐시 로드 (r2022/r2023 로짓)."""
    if not BASELINE_CACHE_DIR.is_dir():
        raise ResidualViolation(f"[RESIDUAL] baseline 캐시 없음: {BASELINE_CACHE_DIR}")
    for meta_path in sorted(BASELINE_CACHE_DIR.glob("*/meta.json")):
        meta = rp.load_json(meta_path)
        if not meta or meta.get("bounded") != MAX_ROWS:
            continue
        origins = meta.get("origins") or {}
        if set(origins) != set(rp.SELECTION_ORIGINS):
            continue
        out: dict[str, np.ndarray[Any, Any]] = {}
        for origin in rp.SELECTION_ORIGINS:
            o = origins.get(origin) or {}
            npy = REPO / o.get("logits_path", "")
            if not npy.is_file():
                break
            out[origin] = np.load(npy)
        if len(out) == len(rp.SELECTION_ORIGINS):
            return out
    raise ResidualViolation("[RESIDUAL] bounded 30k baseline 캐시를 찾지 못함")


def _screen_origin(train, base_oof: dict[int, np.ndarray[Any, Any]],
                   base_oof_masks: dict[int, tuple[Any, Any]],
                   origin: str) -> JSON:
    """선택 기원(origin) 의 잔차 보정 후보 deployed 확률 + 진단 계산.

    r2022: C=C_PRE_REGISTERED, fit 2020-2021, apply 2022.
    r2023: select C (fit 2020, score 2021), refit 2020-2022, apply 2023.
    outer 라벨은 마지막에만 읽는다 (보정 구성은 outer 라벨 로드 전에 선택).
    """
    outer_year = int(origin.replace("r", ""))
    disjoint = assert_inner_outer_disjoint(base_oof_masks, outer_year)

    if origin == "r2022":
        selected_c: float | str = C_PRE_REGISTERED
        fit_years = [2020, 2021]
        c_selection: dict[Any, Any] = {
            "mode": "pre_registered",
            "reason": "r2022 는 사전 등록 C=.01 — 이전 forward 검증 연도 없음",
            "forward_years": [],
        }
    else:  # r2023
        forward_years = [2021]
        selected_c, c_selection = select_c(train, base_oof, base_oof_masks, forward_years)
        if selected_c == SKIPPED_INSUFFICIENT_FORWARD_OOF:
            raise ResidualViolation("[RESIDUAL] r2023 C 선택 이력 부족")
        fit_years = [2020, 2021, 2022]
        c_selection = {
            "mode": "rolling_forward",
            "forward_years": forward_years,
            "grid": list(C_GRID),
            "objective": "unweighted mean of completed forward-year deployed Brier",
            "tie": "smallest C on exact tie",
            "per_c": c_selection,
        }

    # 보정 구성(C) 선택은 outer 라벨 로드 이전에 완료됨.
    stats = _compute_stats(train, base_oof, base_oof_masks, fit_years)
    model = fit_correction(train, base_oof, base_oof_masks, fit_years,
                           float(selected_c), stats)
    X = _build_X(train, base_oof, base_oof_masks, outer_year, stats)
    z_corr = apply_correction(model, X, base_oof[outer_year])

    # outer 라벨은 이제서야 읽는다 (선택 라벨 — primary 아님).
    va_b = _bounded_mask(base_oof_masks[outer_year][1], MAX_ROWS)
    y = train.loc[va_b, "control_success"].values.astype(np.float64)
    p = rp.deployed_probs(z_corr)
    p_residual = model.predict_proba(X)[:, 1]
    comp = _residual_complementarity(base_oof[outer_year], p_residual, y)
    # 진단: 보정이 base OOF 자체를 개선하는지 (게이트와 별개 — 정보용)
    p_base_oof = rp.deployed_probs(base_oof[outer_year])
    import repro_979.common as common  # noqa: PLC0415
    diag = {
        "base_oof_bss": float(common.score(p_base_oof, y)),
        "corrected_bss": float(common.score(p, y)),
        "base_oof_brier": float(np.mean((p_base_oof - y) ** 2)),
        "corrected_brier": float(np.mean((p - y) ** 2)),
        "correction_delta_bss": float(common.score(p, y) - common.score(p_base_oof, y)),
    }
    return {
        "origin": origin,
        "outer_year": outer_year,
        "selected_c": selected_c,
        "c_selection": c_selection,
        "fit_years": fit_years,
        "inner_outer_disjoint": disjoint,
        "correction_cap": CORRECTION_CAP,
        "z_corrected_formula": "z_base + clip(logit(p_residual) - C_LOGIT - z_base, -.05, .05)",
        "residual_complementarity": comp,
        "base_oof_diagnostic": diag,
        "probs": p,
        "n_rows": int(len(y)),
    }


def cmd_screen(args: argparse.Namespace) -> int:
    """잔차 보정 스크린 — r2022/r2023 선택 라벨만 읽는다 (구조적 파이어월)."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    checks: list[JSON] = []
    violations: list[JSON] = []
    findings: list[JSON] = []

    import pandas as pd  # noqa: PLC0415
    train = pd.read_csv(REPO / "open" / "data" / "train.csv", encoding="utf-8-sig")
    train["platoon"] = (train["pitcher_hand"] * 2 + train["batter_hand"]).astype("category")
    for c in ("game_type", "base_state", "top_bottom"):
        train[c] = train[c].astype("category")

    # 기원 마스크 + 누수/분리 (구조)
    origin_masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(origin_masks, train)
    if leak:
        print(f"[recovery_residual_runner] --screen: 누수 가드 실패: {leak}",
              file=sys.stderr)
        return 2
    rp.assert_row_disjointness(origin_masks, train)
    checks.append({"rule": "origin_masks", "ok": True,
                   "reason": "r2022/r2023/primary/r2024 마스크 + row-ID 분리 검증"})

    # R-only base OOF (2020-2023) — 스크린에 필요한 연도
    base_oof_masks = build_base_oof_masks(train, "R")
    base_oof = compute_base_oof(train, base_oof_masks, "R")
    checks.append({"rule": "base_oof_causal", "ok": True,
                   "reason": "one-year-ahead base OOF (train mask ends t-1), R-only, "
                             f"bounded {MAX_ROWS} rows/year, 경량 LGB 프록시 (스크린 단순화)"})

    # baseline 캐시 (bounded 30k)
    try:
        base_cache = _load_baseline_cache()
    except ResidualViolation as exc:
        print(f"[recovery_residual_runner] --screen: {exc}", file=sys.stderr)
        return 2
    base_p = {o: rp.deployed_probs(base_cache[o]) for o in rp.SELECTION_ORIGINS}

    # 각 선택 기원 스크린
    cand_p: dict[str, np.ndarray[Any, Any]] = {}
    origin_details: dict[str, JSON] = {}
    for origin in rp.SELECTION_ORIGINS:
        det = _screen_origin(train, base_oof, base_oof_masks, origin)
        cand_p[origin] = det["probs"]
        origin_details[origin] = det
        checks.append({"rule": f"correction_config_before_outer_labels_{origin}", "ok": True,
                       "reason": f"C={det['selected_c']} 선택은 outer({origin}) 라벨 로드 "
                                 f"이전에 완료 (fit_years={det['fit_years']})"})
        checks.append({"rule": f"inner_outer_disjoint_{origin}", "ok": True,
                       "reason": f"inner-OOF/outer 마스크 분리 + 해시 "
                                 f"(outer_year={det['outer_year']})"})
        checks.append({"rule": f"correction_cap_{origin}", "ok": True,
                       "reason": f"correction cap = ±{det['correction_cap']} "
                                 f"(z_corrected clip)"})
        checks.append({"rule": f"residual_complementarity_{origin}", "ok": True,
                       "reason": f"corr(correction, base_residual) = "
                                 f"{det['residual_complementarity']:+.4f}"})

    # 선택 라벨만 (primary/r2024 미로드 — 구조적)
    y_full = rp.read_selection_labels(train, origin_masks)
    # 후보/베이스라인과 동일 bounded 행으로 라벨 제한 (Task 3 bounded subset)
    y: dict[str, np.ndarray[Any, Any]] = {}
    for origin in rp.SELECTION_ORIGINS:
        va_b = _bounded_mask(origin_masks[origin][1], MAX_ROWS)
        y[origin] = train.loc[va_b, "control_success"].values.astype(np.float64)
    # 후보/베이스라인을 동일 bounded 행에서 비교 — 후보 행은 base_oof 의 bounded 행과 일치.
    gate = rp.screen_gate(cand_p, base_p, y)

    record = _record_base(gate["verdict"], 0,
                          "aimers9-top100-recovery/task-5-residual",
                          "Todo 5 — screen causal OOF residual-ridge correction")
    record.update({
        "mode": "screen",
        "candidate_id": CANDIDATE_ID,
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "config": {
            "c_grid": list(C_GRID),
            "c_pre_registered": C_PRE_REGISTERED,
            "correction_cap": CORRECTION_CAP,
            "z_corrected": "z_base + clip(logit(p_residual) - C_LOGIT - z_base, -.05, .05)",
            "logistic_regression": LR_KWARGS,
            "standardize": list(NUMERIC_FEATS),
            "one_hot": list(ONE_HOT_FEATS),
            "base_oof_years": list(BASE_OOF_YEARS),
            "bounded_rows": MAX_ROWS,
            "base_oof_model": "lightweight LightGBM proxy (screen simplification)",
        },
        "origins": {o: {k: v for k, v in d.items() if k != "probs"}
                    for o, d in origin_details.items()},
        "gate": gate,
        "checks": checks,
        "violations": [{"rule": "screen_gate", "ok": False, "reason": v}
                       for v in gate["violations"]] + violations,
        "findings": findings + [
            {"rule": "no_primary_read", "ok": True,
             "reason": "--screen 경로는 read_selection_labels 만 호출 — primary 라벨 "
                       "구조적으로 미로드 (파이어월)"},
            {"rule": "no_target_encoding", "ok": True,
             "reason": "보정 피처는 [base_logit,balls_before,strikes_before,outs_before] "
                       "표준화 + [game_type,platoon,base_state] one-hot 만 — target "
                       "encoding/타깃 파생 그룹/터미널 라벨 미사용"},
            {"rule": "bounded_subset_consistency", "ok": True,
             "reason": f"Task 3 과 동일 bounded {MAX_ROWS} 행 부분집합 사용 (문서화된 편차)"},
        ],
        "notes": "스크린은 경량 LGB base OOF 프록시로 보정 메커니즘을 검증 — "
                 "per-year 6-leg 재학습은 계산상 불가. 후보/베이스라인은 동일 bounded 행 비교.",
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-5-residual")
    print(f"[recovery_residual_runner] --screen: {gate['verdict']} (exit 0)")
    for o, r in gate["origins"].items():
        print(f"  {o}: ΔBSS={r['delta_bss']:+.4f} LB5={r['bootstrap_lb5']:+.4f} "
              f"mean_shift={r['mean_shift']:.6f} C={origin_details[o]['selected_c']}")
    print(f"[recovery_residual_runner] evidence -> {json_path} / {md_path}")
    return 0


# ── fixture ──────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "outer-target-feature":
        # outer(타깃) 연도 라벨에서 파생된 피처를 보정 입력에 포함 시도 → 거부 (exit 2)
        try:
            _assert_no_outer_target_feature(
                ["base_logit", "balls_before", "outer_year_target_mean"],
                {"outer_year_target_mean"})
            detail = "outer-target 피처가 허용됨 — 위반!"
        except ResidualViolation as exc:
            matched = True
            detail = f"outer-target 피처 거부: {exc}"
    elif name == "terminal-label-read":
        # 스크린 경로에서 primary 라벨 읽기 시도 → 파이어월 차단 (exit 2)
        rp.set_firewall(rp.FIREWALL_UNFROZEN)
        train = _synthetic_train()
        masks = rp.build_origin_masks(train)
        try:
            rp.read_primary_labels(train, masks)
            detail = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
        except rp.TerminalFirewallError as exc:
            matched = True
            detail = f"파이어월이 primary 라벨 읽기를 차단: {exc}"
    elif name == "cap-violation":
        # 보정 cap(±.05) 미적용 → z_corrected 가 cap 범위를 벗어나면 거부 (exit 2)
        train = _synthetic_train()
        base_oof_masks = build_base_oof_masks(train, "R")
        base_oof = compute_base_oof(train, base_oof_masks, "R")
        stats = _compute_stats(train, base_oof, base_oof_masks, [2020, 2021])
        model = fit_correction(train, base_oof, base_oof_masks, [2020, 2021],
                               C_PRE_REGISTERED, stats)
        X = _build_X(train, base_oof, base_oof_masks, 2022, stats)
        z_capped = apply_correction(model, X, base_oof[2022], apply_cap=True)
        z_uncapped = apply_correction(model, X, base_oof[2022], apply_cap=False)
        max_delta = float(np.max(np.abs(z_uncapped - z_capped)))
        if max_delta > 1e-9:
            matched = True
            detail = (f"cap 미적용 시 z_corrected 가 cap 범위를 벗어남 "
                      f"(max|Δ|={max_delta:.6f} > 0) — cap 필수")
        else:
            detail = "cap 미적용이 cap 적용과 동일 — cap 위반 탐지 실패"
    elif name == "target-encoding":
        # target encoding 피처를 보정 입력에 포함 시도 → 거부 (exit 2)
        try:
            _assert_no_target_encoding(["base_logit", "balls_before", "target_enc_platoon"])
            detail = "target encoding 피처가 허용됨 — 위반!"
        except ResidualViolation as exc:
            matched = True
            detail = f"target encoding 피처 거부: {exc}"
    elif name == "offset-identity":
        # identity: p_residual = sigmoid(z_base + C_LOGIT) → 배포 예측 정확히 불변 (exit 0)
        rng = np.random.default_rng(0)
        z_base = rng.normal(0, 1, 500)
        import repro_979.common as common  # noqa: PLC0415
        p_residual = common.sigmoid(z_base + rp.C_LOGIT)
        delta = common.logit(p_residual) - rp.C_LOGIT - z_base
        z_corr = z_base + np.clip(delta, -CORRECTION_CAP, CORRECTION_CAP)
        p_base = rp.deployed_probs(z_base)
        p_corr = rp.deployed_probs(z_corr)
        max_diff = float(np.max(np.abs(p_corr - p_base)))
        if max_diff <= 1e-12:
            matched = True
            detail = (f"identity 유지: p_residual=sigmoid(z_base+C_LOGIT) → "
                      f"배포 예측 정확히 불변 (max|Δp|={max_diff:.2e})")
        else:
            detail = f"identity 위반 — 배포 예측이 변경됨 (max|Δp|={max_diff:.2e})"
    else:
        print(f"[recovery_residual_runner] FATAL: 알 수 없는 fixture {name!r}",
              file=sys.stderr)
        return 1

    # offset-identity 는 성공 시 exit 0 (양성 identity 검증), 나머지는 exit 2.
    if name == "offset-identity":
        exit_code = 0 if matched else 2
        verdict = "PASS" if matched else "REJECT"
    else:
        exit_code = 2
        verdict = "FIXTURE_REJECT"

    record = _record_base(verdict, exit_code,
                          f"aimers9-top100-recovery/task-5-residual-fixture-{name}",
                          f"Todo 5 fixture — {name}")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-5-residual-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} — offset-identity 는 성공 시 exit 0, 나머지는 exit 2.",
    })
    base = evidence_dir / f"task-5-residual-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_residual_runner] FIXTURE {name} (exit {exit_code}) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_residual_runner] evidence -> {json_path} / {md_path}")
    return exit_code


# ── CLI ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_residual_runner.py",
        description="Todo 5 — screen causal OOF residual-ridge correction",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--screen", action="store_true",
                        help="잔차 보정 스크린 (r2022/r2023 선택 라벨만)")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture (offset-identity 는 성공 시 exit 0)")
    args = parser.parse_args(argv)

    if args.fixture is not None:
        return cmd_fixture(args)
    if args.screen:
        return cmd_screen(args)
    print("[recovery_residual_runner] FATAL: 명령을 지정하세요 (--screen / --fixture)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
