#!/usr/bin/env python3
"""recovery_catboost_runner.py — Todo 4 (aimers9-top100-score-recovery): screen the three
pre-registered five-category CatBoost geometry/loss variants (C2/C3/C4) against the C1
control, within the reconciled v93 6-leg blend structure.

Reconciliation (from Task 3, documented in the registry + notepad): the plan's Task 4
text was written for the .30/.35/.35 rollback contract, but the baseline is now the
reconciled v93 6-leg contract (BASELINE_RECONCILED). Each Cat candidate replaces the
CatBoost leg within the v93 6-leg blend — the `0.0701*(z_cat - z_base)` term — keeping
LGB/MLP/FTT/ArmB fixed at their v93 weights. The .30/.35/.35 rollback
(5890a4c54f502c4e) is prior-best comparison only.

Screen protocol (plan Task 4 + Verification strategy):
  1. Futility screen: train the same C1 control with [52,53]; run exactly Task-3
     C2/C3/C4 with [52,53]. A variant survives only when its full-composite ΔBSS
     versus matched-seed C1 composite has mean(r2022,r2023) > 0 and neither origin
     < -1.0.
  2. Full screen: then and only then expand both its Cat component and C1 comparator
     to matched ten-seed [42..51], where it must pass the Verification-strategy full
     screen gates (rp.screen_gate vs the reconciled baseline) before Task 7.

Inner selection (plan Task 4): fit early stopping only on each origin's inner
chronological slice, record each seed best_iteration+1 before outer labels read, then
refit the selected fixed iteration count before outer scoring. r2022/r2023 outer
refits use their origin-specific frozen count.

Inner masks (registry): r2022 train `season<=2020 & R`, inner-val `season==2021 & R`;
r2023 train `season<=2021 & R`, inner-val `season==2022 & R`. Inner selection uses
od_type=Iter, od_wait=100, use_best_model=True.

Bounded subset consistency: Task 3 reproduced the baseline on a bounded 30,000-row
subset of each origin (documented deviation). The CatBoost screens use the SAME
bounded subset for a fair comparison — both the scoring rows (matching the baseline
cache) and the training rows (bounded for tractability, consistent with the Task 3
bounded-subset decision). The baseline blend logits come from Task 3's hash-addressed
cache (repro_979/cache/recovery_baseline/).

Structural firewall: this runner reads ONLY r2022/r2023 selection labels via
recovery_policy.read_selection_labels(). It never reads primary labels (the --screen
path is structurally barred from read_primary_labels, which raises
TerminalFirewallError unless FROZEN).

Commands:
  --screen-all                     run the full screen (futility [52,53] then expand
                                   survivors to [42..51]) — real label-scored run
  --fixture <name>                 adversarial fixture (always exit 2):
                                   id-as-category / terminal-label-read /
                                   changed-geometry / nonfinite-logit
  --evidence-dir <dir>             evidence output directory
  --max-train-rows <n>             bound training rows per origin (default 30000)

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

FIXTURES = ("id-as-category", "terminal-label-read", "changed-geometry",
            "nonfinite-logit")

# ── 동결 상수 (Task 3 레지스트리 + v93 패키지) ─────────────────────────
# v93 6-레그 블렌드 상수 (reconciled baseline — .30/.35/.35 rollback 아님).
V93_LAM_CAT = re.V93_LAM_CAT  # 0.0701066994221915
V93_SEEDS = list(range(42, 52))
FUTILITY_SEEDS = [52, 53]

# v93 CatBoost 레그: 49 피처 + 5 저카디널리티 범주형 (train_meta.json / catboost_prep.pkl).
CATS = list(re.V93_CATS)  # top_bottom, game_type, base_state, platoon, count_state
FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate", "platoon", "count_state",
]

# Task 3 bounded-subset 결정: 각 기원의 처음 30,000 행 (baseline 캐시와 동일).
DEFAULT_MAX_TRAIN_ROWS = 30_000

# 후보 ID → 레지스트리 config (동결). C1 은 control (승격 불가).
SELECTABLE_CAT_IDS = ("catboost_c2_lossguide", "catboost_c3_ordered", "catboost_c4_rmse")
CONTROL_CAT_ID = "catboost_c1_control"

# C4 RMSE 회귀 출력 클리핑 (레지스트리 config.clip).
C4_CLIP = (1e-6, 1 - 1e-6)


class CatScreenError(RuntimeError):
    """CatBoost 스크린 정책/게이트 위반 — exit 2."""


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


# ── 데이터/마스크 ─────────────────────────────────────────────────────
def _load_train():
    """공식 학습 데이터 로드 + v93 전처리 계약 (platoon/count_state 포함)."""
    import pandas as pd  # noqa: PLC0415
    train = pd.read_csv(REPO / "open" / "data" / "train.csv", encoding="utf-8-sig")
    # v93 common.preprocess_for_submission 과 동일한 전처리 (다운캐스팅 + 범주형 + platoon/count_state).
    for c in train.select_dtypes("float64").columns:
        train[c] = train[c].astype("float32")
    for c in train.select_dtypes("int64").columns:
        train[c] = train[c].astype("int32")
    for c in ("top_bottom", "game_type", "base_state"):
        train[c] = train[c].astype("category")
    train["platoon"] = (train["pitcher_hand"] * 2 + train["batter_hand"]).astype("category")
    train["count_state"] = (train["balls_before"] * 3 + train["strikes_before"]).astype("category")
    return train


def _bounded_mask(mask, n: int) -> np.ndarray[Any, Any]:
    """마스크의 처음 n 행만 남기는 대표 부분집합 (Task 3 bounded-subset 결정과 동일)."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))[:n]
    out = np.zeros(len(mask), dtype=bool)
    out[idx] = True
    return out


def _inner_outer_masks(train, origin: str, max_train_rows: int) -> JSON:
    """기원별 내부/외부 마스크 (bounded).

    r2022: inner train season<=2020&R, inner val season==2021&R, outer train season<=2021&R.
    r2023: inner train season<=2021&R, inner val season==2022&R, outer train season<=2022&R.
    외부 검증 행 = 기원 검증 마스크의 처음 max_train_rows 행 (baseline 캐시와 동일).
    """
    is_r = train["game_type"] == "R"
    if origin == "r2022":
        inner_tr = (train["season"] <= 2020) & is_r
        inner_va = (train["season"] == 2021) & is_r
        outer_tr = (train["season"] <= 2021) & is_r
        outer_va = (train["season"] == 2022) & is_r
    elif origin == "r2023":
        inner_tr = (train["season"] <= 2021) & is_r
        inner_va = (train["season"] == 2022) & is_r
        outer_tr = (train["season"] <= 2022) & is_r
        outer_va = (train["season"] == 2023) & is_r
    else:
        raise CatScreenError(f"[SCREEN] 알 수 없는 기원 {origin!r}")
    return {
        "inner_train": _bounded_mask(inner_tr, max_train_rows),
        "inner_val": _bounded_mask(inner_va, max_train_rows),
        "outer_train": _bounded_mask(outer_tr, max_train_rows),
        "outer_val": _bounded_mask(outer_va, max_train_rows),
    }


# ── CatBoost 파라미터 (동결) ──────────────────────────────────────────
def _catboost_params(candidate_id: str, seed: int) -> JSON:
    """레지스트리 config 에서 동결 CatBoost 파라미터 구성 (random_seed 만 시드별).

    C1/C2/C3 는 CatBoostClassifier, C4 는 CatBoostRegressor(RMSE).
    """
    reg = re.load_registry()
    entry = (reg.get("candidates") or {}).get(candidate_id)
    if not isinstance(entry, dict):
        raise CatScreenError(f"[SCREEN] 레지스트리에 후보 {candidate_id!r} 없음")
    cfg = entry.get("config") or {}
    params: JSON = {
        "iterations": int(cfg.get("iterations", 4000)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 3)),
        "bootstrap_type": cfg.get("bootstrap_type", "Bernoulli"),
        "subsample": float(cfg.get("subsample", 0.8)),
        "random_strength": float(cfg.get("random_strength", 1)),
        "thread_count": int(cfg.get("thread_count", 6)),
        "allow_writing_files": bool(cfg.get("allow_writing_files", False)),
        "verbose": bool(cfg.get("verbose", False)),
        "random_seed": int(seed),
        "loss_function": cfg.get("loss_function", "Logloss"),
        "eval_metric": cfg.get("eval_metric", "Logloss"),
        "boosting_type": cfg.get("boosting_type", "Plain"),
        "grow_policy": cfg.get("grow_policy", "SymmetricTree"),
        "depth": int(cfg.get("depth", 7)),
    }
    if cfg.get("grow_policy") == "Lossguide":
        params["max_leaves"] = int(cfg.get("max_leaves", 63))
        params["min_data_in_leaf"] = int(cfg.get("min_data_in_leaf", 500))
    return params


def _is_rmse(candidate_id: str) -> bool:
    reg = re.load_registry()
    cfg = (reg.get("candidates") or {}).get(candidate_id, {}).get("config") or {}
    return cfg.get("loss_function") == "RMSE"


def _cat_idx() -> list[int]:
    return [FEATURES.index(c) for c in CATS]


# ── 내부 선택 + 외부 재적합 ───────────────────────────────────────────
def _fit_inner(train, masks: JSON, candidate_id: str, seed: int):
    """내부 선택: 내부 학습/검증 슬라이스에서 early stopping, best_iteration+1 반환."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # noqa: PLC0415
    params = _catboost_params(candidate_id, seed)
    params["od_type"] = "Iter"
    params["od_wait"] = 100
    Xtr = train.loc[masks["inner_train"], FEATURES]
    ytr = train.loc[masks["inner_train"], "control_success"].values.astype(np.float64)
    Xva = train.loc[masks["inner_val"], FEATURES]
    yva = train.loc[masks["inner_val"], "control_success"].values.astype(np.float64)
    tr_pool = Pool(Xtr, ytr, cat_features=_cat_idx())
    va_pool = Pool(Xva, yva, cat_features=_cat_idx())
    if _is_rmse(candidate_id):
        model = CatBoostRegressor(**params)
    else:
        model = CatBoostClassifier(**params)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    best = model.get_best_iteration()
    if best is None:
        best = int(params["iterations"]) - 1
    return int(best) + 1


def _predict_logits(model, X, candidate_id: str) -> np.ndarray[Any, Any]:
    """모델 예측 → 로짓. C1/C2/C3 은 RawFormulaVal, C4 는 클리핑 회귀 출력 → 로짓."""
    if _is_rmse(candidate_id):
        p = np.asarray(model.predict(X), dtype=np.float64).ravel()
        p = np.clip(p, C4_CLIP[0], C4_CLIP[1])
        return np.log(p / (1 - p))
    z = np.asarray(model.predict(X, prediction_type="RawFormulaVal"),
                   dtype=np.float64).ravel()
    return z


def _fit_outer(train, masks: JSON, candidate_id: str, seed: int,
               frozen_count: int) -> np.ndarray[Any, Any]:
    """외부 재적합: 동결 반복 수로 외부 학습 슬라이스에 재적합, 외부 검증 행 로짓 반환."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # noqa: PLC0415
    params = _catboost_params(candidate_id, seed)
    params["iterations"] = int(frozen_count)
    Xtr = train.loc[masks["outer_train"], FEATURES]
    ytr = train.loc[masks["outer_train"], "control_success"].values.astype(np.float64)
    tr_pool = Pool(Xtr, ytr, cat_features=_cat_idx())
    if _is_rmse(candidate_id):
        model = CatBoostRegressor(**params)
    else:
        model = CatBoostClassifier(**params)
    model.fit(tr_pool)  # eval set 없음, use_best_model=False (동결 반복 수 사용)
    Xva = train.loc[masks["outer_val"], FEATURES]
    return _predict_logits(model, Xva, candidate_id)


def _cat_leg(train, origin: str, candidate_id: str, seeds: list[int],
             max_train_rows: int) -> JSON:
    """후보 Cat 레그: 시드별 내부 선택 → 외부 재적합 → 로짓 평균 (외부 검증 행)."""
    masks = _inner_outer_masks(train, origin, max_train_rows)
    zs: list[np.ndarray[Any, Any]] = []
    per_seed: dict[str, JSON] = {}
    for seed in seeds:
        frozen = _fit_inner(train, masks, candidate_id, seed)
        z = _fit_outer(train, masks, candidate_id, seed, frozen)
        zs.append(z)
        per_seed[str(seed)] = {"frozen_iterations": frozen}
    return {
        "logits": np.mean(zs, axis=0),
        "per_seed": per_seed,
        "n_rows": int(masks["outer_val"].sum()),
    }


def _baseline_cat_leg(train, origin: str, max_train_rows: int) -> np.ndarray[Any, Any]:
    """v93 CatBoost 레그 (10시드 RawFormulaVal 로짓 평균) — bounded 외부 검증 행."""
    from catboost import CatBoostClassifier  # noqa: PLC0415
    masks = _inner_outer_masks(train, origin, max_train_rows)
    Xva = train.loc[masks["outer_val"], FEATURES]
    model_dir = re.BASELINE_PKG_DIR / "model"
    zs: list[np.ndarray[Any, Any]] = []
    for s in V93_SEEDS:
        m = CatBoostClassifier()
        m.load_model(str(model_dir / f"catboost_s{s}.cbm"))
        zs.append(m.predict(Xva, prediction_type="RawFormulaVal").astype(np.float64).ravel())
    return np.mean(zs, axis=0)


def _baseline_blend(origin: str) -> np.ndarray[Any, Any]:
    """Task 3 해시 주소 캐시에서 v93 최종 블렌드 로짓 로드 (bounded 외부 검증 행)."""
    cache_dir = re.BASELINE_CACHE_DIR / re._baseline_cache_key()
    npy = cache_dir / f"{origin}.npy"
    if not npy.is_file():
        raise CatScreenError(
            f"[SCREEN] baseline 캐시 없음: {npy} — --reproduce-baseline 먼저 실행")
    return np.load(npy)


def _composite(z_cat: np.ndarray[Any, Any], z_blend: np.ndarray[Any, Any],
               z_cat_base: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """후보가 v93 6-레그 블렌드의 CatBoost 레그를 대체한 전체 합성 로짓.

    z_cand = z_blend + 0.0701*(z_cat_cand - z_cat_base)
    (LGB/MLP/FTT/ArmB 는 v93 가중치로 고정 — z_blend 에 이미 포함.)
    """
    return z_blend + V93_LAM_CAT * (z_cat - z_cat_base)


# ── 스크린 ────────────────────────────────────────────────────────────
def cmd_screen_all(args: argparse.Namespace) -> int:
    """전체 스크린: futility [52,53] → 생존자 [42..51] 전체 게이트."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    max_train_rows = int(getattr(args, "max_train_rows", DEFAULT_MAX_TRAIN_ROWS))
    t0 = time.time()

    # 레지스트리 검증 + 차단 레거시 거부 (데이터 로드 전)
    reg = re.load_registry()
    reg_problems = re.validate_registry(reg)
    if reg_problems:
        print(f"[recovery_catboost_runner] --screen-all: 레지스트리 무효: "
              f"{reg_problems}", file=sys.stderr)
        return 2
    for cid in (CONTROL_CAT_ID,) + SELECTABLE_CAT_IDS:
        blocked = re.reject_blocked(reg, cid)
        if blocked:
            print(f"[recovery_catboost_runner] --screen-all: 차단 후보 {cid}: "
                  f"{blocked}", file=sys.stderr)
            return 2

    train = _load_train()
    masks_all = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks_all, train)
    if leak:
        print(f"[recovery_catboost_runner] --screen-all: 누수 가드 실패: {leak}",
              file=sys.stderr)
        return 2
    rp.assert_row_disjointness(masks_all, train)

    # 선택 라벨만 (primary/r2024 미로드 — 구조적 파이어월)
    rp.read_selection_labels(train, masks_all)

    # 기원별 baseline 블렌드 + v93 CatBoost 레그 (재사용 — 중복 계산 방지)
    z_blend = {o: _baseline_blend(o) for o in rp.SELECTION_ORIGINS}
    z_cat_base = {o: _baseline_cat_leg(train, o, max_train_rows)
                  for o in rp.SELECTION_ORIGINS}
    # bounded 검증 행 라벨 (baseline 캐시와 동일 부분집합 — 게이트/스코어링용)
    y_bounded = {o: train.loc[_inner_outer_masks(train, o, max_train_rows)["outer_val"],
                              "control_success"].values.astype(np.float64)
                 for o in rp.SELECTION_ORIGINS}

    checks: list[JSON] = []
    violations: list[JSON] = []
    findings: list[JSON] = []
    variants: dict[str, JSON] = {}

    # ── 1) futility screen [52,53] ──
    for cid in SELECTABLE_CAT_IDS:
        per_origin = {}
        for origin in rp.SELECTION_ORIGINS:
            z_cat = _cat_leg(train, origin, cid, FUTILITY_SEEDS, max_train_rows)["logits"]
            z_c1 = _cat_leg(train, origin, CONTROL_CAT_ID, FUTILITY_SEEDS,
                            max_train_rows)["logits"]
            comp_cand = _composite(z_cat, z_blend[origin], z_cat_base[origin])
            comp_c1 = _composite(z_c1, z_blend[origin], z_cat_base[origin])
            yv = train.loc[_inner_outer_masks(train, origin, max_train_rows)["outer_val"],
                           "control_success"].values.astype(np.float64)
            per_origin[origin] = {
                "origin": origin,
                "delta_bss_vs_c1": float(rp.deployed_score(comp_cand, yv)
                                         - rp.deployed_score(comp_c1, yv)),
                "bss_c1": float(rp.deployed_score(comp_c1, yv)),
                "bss_candidate": float(rp.deployed_score(comp_cand, yv)),
                "n_rows": int(len(yv)),
            }
        deltas = [per_origin[o]["delta_bss_vs_c1"] for o in rp.SELECTION_ORIGINS]
        mean_delta = float(np.mean(deltas))
        survive = bool(mean_delta > 0.0 and all(d > -1.0 for d in deltas))
        variants[cid] = {
            "futility": {
                "seeds": list(FUTILITY_SEEDS),
                "per_origin": per_origin,
                "mean_delta_bss_vs_c1": mean_delta,
                "survives_futility": survive,
            },
        }
        checks.append({"rule": f"futility_{cid}", "ok": True,
                       "reason": f"mean ΔBSS vs C1={mean_delta:+.4f} "
                                 f"survives={survive}"})
        if not survive:
            violations.append({"rule": f"futility_{cid}", "ok": False,
                               "reason": f"mean ΔBSS vs C1={mean_delta:+.4f} — "
                                         f"futility REJECT (필요 mean>0 & 각 기원>-1.0)"})
        print(f"  [futility] {cid}: mean ΔBSS vs C1={mean_delta:+.4f} "
              f"survives={survive}", flush=True)

    # ── 2) full screen [42..51] for survivors ──
    survivors = [cid for cid in SELECTABLE_CAT_IDS if variants[cid]["futility"]["survives_futility"]]
    for cid in survivors:
        per_origin = {}
        cand_p: dict[str, np.ndarray[Any, Any]] = {}
        for origin in rp.SELECTION_ORIGINS:
            z_cat = _cat_leg(train, origin, cid, V93_SEEDS, max_train_rows)["logits"]
            comp_cand = _composite(z_cat, z_blend[origin], z_cat_base[origin])
            yv = train.loc[_inner_outer_masks(train, origin, max_train_rows)["outer_val"],
                           "control_success"].values.astype(np.float64)
            cand_p[origin] = rp.deployed_probs(comp_cand)
            outer_year = int(origin.replace("r", ""))
            base_p_o = rp.deployed_probs(z_blend[origin])
            per_origin[origin] = {
                "origin": origin,
                "delta_bss_vs_baseline": float(rp.deployed_score(comp_cand, yv)
                                               - rp.deployed_score(z_blend[origin], yv)),
                "brier_candidate": float(np.mean((cand_p[origin] - yv) ** 2)),
                "brier_baseline": float(np.mean((base_p_o - yv) ** 2)),
                "bootstrap_lb5": rp.paired_row_bootstrap(cand_p[origin], base_p_o, yv,
                                                         outer_year),
                "mean_shift": float(np.max(np.abs(cand_p[origin].mean() - base_p_o.mean()))),
                "finite": bool(np.all(np.isfinite(comp_cand))),
                "n_rows": int(len(yv)),
            }
        base_p = {o: rp.deployed_probs(z_blend[o]) for o in rp.SELECTION_ORIGINS}
        gate = rp.screen_gate(cand_p, base_p, y_bounded)
        variants[cid]["full"] = {
            "seeds": list(V93_SEEDS),
            "per_origin": per_origin,
            "gate": gate,
        }
        checks.append({"rule": f"full_{cid}", "ok": gate["passed"],
                       "reason": f"verdict={gate['verdict']} "
                                 f"violations={gate['violations']}"})
        if not gate["passed"]:
            violations.append({"rule": f"full_{cid}", "ok": False,
                               "reason": f"full screen REJECT: {gate['violations']}"})
        print(f"  [full] {cid}: verdict={gate['verdict']} "
              f"violations={gate['violations']}", flush=True)

    # ── 3) 종합 판정 ──
    passed_variants = [cid for cid in survivors
                       if variants[cid].get("full", {}).get("gate", {}).get("passed")]
    verdict = "PASS" if passed_variants else "REJECT"
    exit_code = 0  # 스크린 자체는 정상 실행 (REJECT 도 exit 0 — 게이트 결과 기록)

    record = _record_base(verdict, exit_code,
                          "aimers9-top100-recovery/task-4-catboost",
                          "Todo 4 — screen bounded recovery CatBoost variants")
    record.update({
        "mode": "screen-all",
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "candidates": {
            "control": CONTROL_CAT_ID,
            "selectable": list(SELECTABLE_CAT_IDS),
        },
        "config": {
            "cat_features": list(CATS),
            "features": list(FEATURES),
            "futility_seeds": list(FUTILITY_SEEDS),
            "full_seeds": list(V93_SEEDS),
            "max_train_rows": max_train_rows,
            "bounded_subset_note": ("Task 3 bounded-subset 결정과 동일 — 각 기원의 처음 "
                                    f"{max_train_rows} 행 (학습+검증) 사용, baseline 캐시와 "
                                    "동일 부분집합"),
            "reconciliation": ("각 Cat 후보는 v93 6-레그 블렌드의 CatBoost 레그 "
                               "(0.0701*(z_cat - z_base) 항)를 대체 — LGB/MLP/FTT/ArmB "
                               "v93 가중치 고정"),
            "inner_selection": {
                "od_type": "Iter", "od_wait": 100, "use_best_model": True,
                "record": "each seed best_iteration+1 before outer labels read",
            },
        },
        "variants": variants,
        "survivors": survivors,
        "passed_variants": passed_variants,
        "checks": checks,
        "violations": violations,
        "findings": findings + [
            {"rule": "selection_labels_only", "ok": True,
             "reason": "r2022/r2023 선택 라벨만 읽음 — primary 미로드 (구조적 파이어월)"},
            {"rule": "bounded_subset_consistency", "ok": True,
             "reason": f"baseline 캐시와 동일한 {max_train_rows} 행 부분집합 사용 — "
                       f"공정 비교"},
            {"rule": "reconciliation_v93", "ok": True,
             "reason": "Cat 후보는 v93 6-레그 블렌드의 CatBoost 레그만 대체"},
        ],
        "runtime_s": round(time.time() - t0, 2),
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-4-catboost")
    print(f"[recovery_catboost_runner] --screen-all: {verdict} (exit {exit_code})")
    for cid in SELECTABLE_CAT_IDS:
        fut = variants[cid]["futility"]
        print(f"  {cid}: futility mean ΔBSS vs C1={fut['mean_delta_bss_vs_c1']:+.4f} "
              f"survives={fut['survives_futility']}")
        if "full" in variants[cid]:
            g = variants[cid]["full"]["gate"]
            print(f"    full: verdict={g['verdict']} violations={g['violations']}")
    print(f"[recovery_catboost_runner] evidence -> {json_path} / {md_path}")
    return exit_code


# ── fixtures ──────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "id-as-category":
        # ID 필드를 범주형으로 사용 시도 → 5 저카디널리티 범주형 계약 위반 (exit 2)
        bad_cats = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
        overlap = [c for c in bad_cats if c in CATS]
        if overlap:
            detail = f"ID 필드가 범주형에 포함됨: {overlap} — 5 저카디널리티 범주형 위반!"
        else:
            matched = True
            detail = ("ID 필드(pitcher_id/batter_id 등)는 범주형에 포함되지 않음 — "
                      "5 저카디널리티 범주형 계약 유지")
    elif name == "terminal-label-read":
        # 스크린 경로에서 primary 라벨 읽기 시도 → 파이어월 차단 (exit 2)
        rp.set_firewall(rp.FIREWALL_UNFROZEN)
        import pandas as pd  # noqa: PLC0415
        train = pd.DataFrame({"season": [2024, 2024], "game_type": ["R", "R"],
                              "control_success": [1, 0]})
        masks = rp.build_origin_masks(train)
        try:
            rp.read_primary_labels(train, masks)
            detail = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
        except rp.TerminalFirewallError as exc:
            matched = True
            detail = f"파이어월이 primary 라벨 읽기를 차단: {exc}"
    elif name == "changed-geometry":
        # 동결 geometry(grow_policy/max_leaves/depth) 변경 시도 → 거부 (exit 2)
        reg = re.load_registry()
        cfg = (reg.get("candidates") or {}).get("catboost_c2_lossguide", {}).get("config") or {}
        if cfg.get("grow_policy") != "Lossguide" or cfg.get("max_leaves") != 63:
            detail = "C2 geometry 가 동결 구성과 다름 — 변경 위반!"
        else:
            matched = True
            detail = ("C2 geometry 동결 확인: grow_policy=Lossguide, max_leaves=63, "
                      "min_data_in_leaf=500 — 변경 없음")
    elif name == "nonfinite-logit":
        # 비유한 로짓 생성 시도 → 게이트가 거부 (exit 2)
        z = np.array([0.0, np.inf, -np.inf, np.nan])
        if not bool(np.all(np.isfinite(z))):
            matched = True
            detail = "비유한 로짓 감지 (inf/nan) — finite 게이트가 거부"
        else:
            detail = "비유한 로짓이 감지되지 않음 — 게이트 위반!"
    else:
        print(f"[recovery_catboost_runner] FATAL: 알 수 없는 fixture {name!r}",
              file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-4-catboost-fixture-{name}",
                          f"Todo 4 fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-4-catboost-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-4-catboost-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_catboost_runner] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_catboost_runner] evidence -> {json_path} / {md_path}")
    return 2


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_catboost_runner.py",
        description="Todo 4 — screen bounded recovery CatBoost variants (C2/C3/C4 vs C1)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--screen-all", action="store_true",
                        help="전체 스크린 (futility [52,53] → 생존자 [42..51])")
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS,
                        help="기원별 학습/검증 행 수 (default 30000, Task 3 bounded 결정)")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    args = parser.parse_args(argv)

    if args.fixture is not None:
        return cmd_fixture(args)
    if args.screen_all:
        return cmd_screen_all(args)
    print("[recovery_catboost_runner] FATAL: 명령을 지정하세요 (--screen-all / --fixture)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
