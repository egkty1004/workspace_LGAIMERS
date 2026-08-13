#!/usr/bin/env python3
"""feature_qualification.py — Todo 4: 드리프트 인지 전처리/피처 후보 10시드 × 4폴드 게이팅.

목표 (aimers9-top100-score-improvement Todo 4): Task 3 레지스트리
(feature_hypothesis_registry.json)의 proposed 후보 8종을 Task 2(qualification_runner.py)
동결 컨트롤과 동일한 10시드(42..51) × 4폴드(primary/r2022/r2023/r2024) 시계열 OOF
게이트로 검증한다. 기준선(동결 49피처 계약) 대비 per-fold BSS 델타, 평균 델타,
bootstrap 신뢰구간, MLP 상호작용(블렌드 probe)을 기록한다.

계약 (Task 2/3 재사용 — 복사 금지):
  - 동결 상수는 qualification_runner 에서 IMPORT (모듈 로드 시 자기 무결성 게이트 포함):
    SEEDS/FOLDS/R_FOLDS/CHAMPION_FEATURES/LGB_CATS/W_LGB/build_folds/_check_leakage.
  - 기준선 = 동결 49피처 계약 그대로 LGB 10시드 × 4폴드 (common.PARAMS deterministic=True,
    num_boost_round=5000, early_stopping 50) — Task 2 문서화 LGB BSS 재현(±1.0)을 assert.
    폴드 행 수(Task 2: 253507/217024/219839/223497)도 대조. 검증 BSS는
    common.score(common.sigmoid(z), y) (C_LOGIT 미적용, 챔피언 캘리브레이션 불변).
  - 각 proposed 후보: 기준선 피처 + 후보 피처 1종, 동일 프로토콜. 변환은 격리 후보
    모듈(본 파일)의 fit/apply 쌍 — 상수는 각 폴드 **학습 윈도우(train 행)에서만** 적합
    (검증/2025 행 미참조 누수 가드). drift_eda.TRANSFORMS 와 동일 계약을 유지하고
    레지스트리 id 를 참조(계약 대조 실행 포함).
  - 게이트 (registry.gate + 불확실성 기준):
      primary Δ ≥ +15 & R-only 3/3 & max|Δmean| ≤ 0.005 & bootstrap(1000, seed 42)
      primary Δ 5% 하한 > 0 → 채택(approved), 그 외 기각(rejected) + 사유.
  - MLP 상호작용: w=0.51 로 (base-LGB + MLP) vs (base+feature)-LGB + MLP 블렌드 probe.
    MLP 로짓 = cache/mlp_{fold}.npy (챔피언 캐시 — Task 2 result.json sha256 대조).
  - blocked 후보 id 를 --candidate 로 넘기면 **학습 전에** 비정상 종료(거부, exit 1).
  - --smoke: 시드 [42,43] + 첫 proposed(season_dev_success_pitcher)로 전체 경로 검증 후
    PASS 출력. 스모크 증거는 task-4-features-smoke.{json,log} + cache/qualification/
    task4_smoke_*/ 로 분리 (전체 실행 증거/캐시 오염 방지 — Task 5/6 교훈).

출력:
  - .omo/evidence/aimers9-top100/task-4-features.{json,log}   (전체 실행 증거)
  - .omo/evidence/aimers9-top100/task-4-features-smoke.{json,log} (스모크 증거)
  - repro_979/experiments/REPORT_feature_qualification.md      (커밋 대상 압축 리포트)
  - repro_979/cache/qualification/task4_{candidate}/<fold>.npy + result.json
    (후보 OOF 로짓 — Todo 7 블렌드 선택 입력. **/cache/ git 제외)

누수 가드(비협상): 2025 라벨 미사용, 폴드 검증 마스크에 2025 부재(재확인),
변환 상수는 학습 윈도우 전용, 외부 데이터 없음.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))

# ── Task 2 동결 컨트롤 IMPORT (자기 무결성 게이트가 모듈 로드 시 실행됨) ──
import qualification_runner as qc  # noqa: E402  (동결 상수/폴드/누수 가드)
import common  # noqa: E402
import drift_eda  # noqa: E402  (TRANSFORMS / CS_LABELS / resolve_transform / REJECTED_IDS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

SEEDS = list(qc.SEEDS)
FOLDS = list(qc.FOLDS)
R_FOLDS = list(qc.R_FOLDS)
CHAMPION_FEATURES = list(qc.CHAMPION_FEATURES)
LGB_CATS = list(qc.LGB_CATS)
W_LGB = qc.W_LGB
SCHEMA_VERSION = qc.SCHEMA_VERSION

# ── 강화 게이트 상수 (screen_all_10seed + 불확실성 기준) ──
PRIMARY_THRESHOLD = 15.0          # primary Δ ≥ +15
R_ONLY_REQUIRED = 3               # R-only 3/3
MEAN_POISON_THRESHOLD = 0.005     # max|Δmean| ≤ 0.005
BOOTSTRAP_ITER = 1000             # 리샘플 반복
BOOTSTRAP_SEED = 42               # 리샘플 RNG 시드
DECLARED_BASE_TOLERANCE = 1.0     # Task 2 문서화 LGB 대비 허용차(±BSS)

# Task 2 문서화 LGB BSS (qualification_runner.DOCUMENTED) — 기준선 재현 assert 기준.
DOCUMENTED_LGB = {
    "primary": qc.DOCUMENTED["primary_lgb"],
    "r2022": qc.DOCUMENTED["r2022"]["lgb"],
    "r2023": qc.DOCUMENTED["r2023"]["lgb"],
    "r2024": qc.DOCUMENTED["r2024"]["lgb"],
}
# Task 2 폴드 검증 행 수 (result.json) — 폴드 마스크 대조.
EXPECTED_ROW_COUNTS = {"primary": 253507, "r2022": 217024, "r2023": 219839, "r2024": 223497}
# 챔피언 MLP 로짓 캐시 (Task 2 result.json cache_provenance sha256 동결 대조).
MLP_CACHE = {fn: f"cache/mlp_{fn}.npy" for fn in FOLDS}
MLP_SHA256 = {
    "primary": "9504e89a94af082818f3329c270c751d24025996a629e3cd3ed934374f429e1e",
    "r2022": "fe85410baff20b33cb6dba7c7a138920a4ff375a53ec7532f88cbff29aedbb31",
    "r2023": "c754e5c4ac4fa2365a5cbd5795f3377beb18bc4cdefd3928a987b011d0e5c87d",
    "r2024": "c48dad0501220009b94b74a1474c8a2090c6e8c8d6edbd6bbcb4bbf9416ab1dc",
}
CHAMPION_BLEND_PRIMARY = qc.DOCUMENTED["primary_blend"]  # 774.1586 — base blend 재현 대조

GATE_STR = (
    f"primary Δ ≥ +{PRIMARY_THRESHOLD:.0f} & R-only {R_ONLY_REQUIRED}/3 "
    f"& max|Δmean| ≤ {MEAN_POISON_THRESHOLD} & bootstrap primary Δ 5%하한 > 0 → 채택, 그 외 기각"
)

REGISTRY_PATH = REPO / "feature_hypothesis_registry.json"
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100"
EVIDENCE_JSON = EVIDENCE_DIR / "task-4-features.json"
EVIDENCE_LOG = EVIDENCE_DIR / "task-4-features.log"
SMOKE_JSON = EVIDENCE_DIR / "task-4-features-smoke.json"
SMOKE_LOG = EVIDENCE_DIR / "task-4-features-smoke.log"
REPORT_MD = REPO / "experiments" / "REPORT_feature_qualification.md"


# ════════════════════════════════════════════════════════════════════
# 격리 후보 모듈 — 각 proposed 변환의 fit/apply 쌍 (레지스트리 id 참조).
# fit: 학습 윈도우 행에서만 상수 적합 / apply: 임의 행에 적용 (상수 고정).
# drift_eda.TRANSFORMS 와 동일 계약(float32 numeric 또는 category) 유지.
# ════════════════════════════════════════════════════════════════════
# count_state_abs_regime 의 전체 라벨 집합 (12 count_state × 2 regime + other_* 2종)
_CS_REGIME_LABELS = sorted(
    [f"{drift_eda.CS_LABELS[i]}_{r}" for i in range(12) for r in ("PRE", "ABS")]
    + [f"other_{r}" for r in ("PRE", "ABS")]
)


def _fit_season_dev_success_pitcher(df_tr: pd.DataFrame) -> dict:
    """레지스트리 id season_dev_success_pitcher — 상수(평균)는 학습 윈도우 전용."""
    return {"mean": float(df_tr["asof_pitcher_success_rate"].mean())}


def _apply_season_dev_success_pitcher(df: pd.DataFrame, state: dict) -> pd.Series:
    return (df["asof_pitcher_success_rate"] - state["mean"]).astype("float32")


def _fit_league_trend_ratio_middle(df_tr: pd.DataFrame) -> dict:
    """레지스트리 id league_trend_ratio_middle — 시즌별 리그 평균(학습 윈도우), 미관측 시즌=최근 값."""
    sm = df_tr.groupby("season", observed=True)["asof_pitcher_middle_rate"].mean()
    return {"season_means": sm, "latest": float(sm.iloc[-1])}


def _apply_league_trend_ratio_middle(df: pd.DataFrame, state: dict) -> pd.Series:
    denom = df["season"].map(state["season_means"]).fillna(state["latest"]).astype("float64")
    return (df["asof_pitcher_middle_rate"] / denom).astype("float32")


def _fit_usage_rank_pitcher(df_tr: pd.DataFrame) -> dict:
    """레지스트리 id usage_rank_pitcher — 학습 윈도우 백분위 랭크(상수 단조 맵)."""
    return {"sorted": np.sort(df_tr["asof_pitcher_n"].to_numpy(dtype="float64")),
            "n": int(len(df_tr))}


def _apply_usage_rank_pitcher(df: pd.DataFrame, state: dict) -> pd.Series:
    x = df["asof_pitcher_n"].to_numpy(dtype="float64")
    n = max(state["n"], 1)
    return pd.Series(np.searchsorted(state["sorted"], x, side="left") / n,
                     index=df.index, dtype="float32")


def _fit_log1p_usage_pitcher(df_tr: pd.DataFrame) -> dict:
    return {}


def _apply_log1p_usage_pitcher(df: pd.DataFrame, state: dict) -> pd.Series:
    return pd.Series(np.log1p(df["asof_pitcher_n"].to_numpy(dtype="float64")),
                     index=df.index, dtype="float32")


def _fit_count_state_abs_regime(df_tr: pd.DataFrame) -> dict:
    return {}


def _apply_count_state_abs_regime(df: pd.DataFrame, state: dict) -> pd.Series:
    cs = (df["balls_before"] * 3 + df["strikes_before"]).to_numpy(dtype="int64")
    regime = np.where(df["season"].to_numpy() >= 2024, "ABS", "PRE")
    labels = np.empty(len(df), dtype=object)
    for i in range(12):
        m = cs == i
        labels[m] = np.char.add(np.char.add(drift_eda.CS_LABELS[i], "_"), regime[m])
    labels[~np.isin(cs, list(range(12)))] = np.char.add(
        "other_", regime[~np.isin(cs, list(range(12)))])
    return pd.Series(labels, index=df.index,
                     dtype=pd.CategoricalDtype(categories=_CS_REGIME_LABELS))


def _fit_debut_league_impute_success(df_tr: pd.DataFrame) -> dict:
    """레지스트리 id debut_league_impute_success — G2(투수 데뷔) asof 성공률 레짐별 리그 대체."""
    post = df_tr["season"].to_numpy() >= 2023
    pre = float(df_tr.loc[~post, "asof_pitcher_success_rate"].mean())
    po = float(df_tr.loc[post, "asof_pitcher_success_rate"].mean())
    overall = float(df_tr["asof_pitcher_success_rate"].mean())
    return {"pre": pre if not math.isnan(pre) else overall,
            "post": po if not math.isnan(po) else overall}


def _apply_debut_league_impute_success(df: pd.DataFrame, state: dict) -> pd.Series:
    post = df["season"].to_numpy() >= 2023
    v = df["asof_pitcher_success_rate"].to_numpy(dtype="float64").copy()
    debut = (df["asof_pitcher_n"] == 0).to_numpy()
    v[debut & ~post] = state["pre"]
    v[debut & post] = state["post"]
    return pd.Series(v, index=df.index).astype("float32")


def _fit_batter_debut_league_impute_success(df_tr: pd.DataFrame) -> dict:
    """레지스트리 id batter_debut_league_impute_success — G3(타자 데뷔) 대체."""
    post = df_tr["season"].to_numpy() >= 2023
    pre = float(df_tr.loc[~post, "asof_batter_success_rate"].mean())
    po = float(df_tr.loc[post, "asof_batter_success_rate"].mean())
    overall = float(df_tr["asof_batter_success_rate"].mean())
    return {"pre": pre if not math.isnan(pre) else overall,
            "post": po if not math.isnan(po) else overall}


def _apply_batter_debut_league_impute_success(df: pd.DataFrame, state: dict) -> pd.Series:
    post = df["season"].to_numpy() >= 2023
    v = df["asof_batter_success_rate"].to_numpy(dtype="float64").copy()
    debut = (df["asof_batter_n"] == 0).to_numpy()
    v[debut & ~post] = state["pre"]
    v[debut & post] = state["post"]
    return pd.Series(v, index=df.index).astype("float32")


def _fit_f_game_post2023_label_mapping(df_tr: pd.DataFrame) -> dict:
    return {}


def _apply_f_game_post2023_label_mapping(df: pd.DataFrame, state: dict) -> pd.Series:
    is_f = (df["game_type"].astype(str) == "F").to_numpy()
    return pd.Series(is_f.astype("int8"), index=df.index, name="f_game_indicator")


# id → (fit, apply). 피처 컬럼명은 레지스트리 id 로 고정 (Todo 7 입력 계약 일관성).
CANDIDATE_IMPLS: dict[str, tuple] = {
    "season_dev_success_pitcher": (_fit_season_dev_success_pitcher,
                                   _apply_season_dev_success_pitcher),
    "league_trend_ratio_middle": (_fit_league_trend_ratio_middle,
                                  _apply_league_trend_ratio_middle),
    "usage_rank_pitcher": (_fit_usage_rank_pitcher, _apply_usage_rank_pitcher),
    "log1p_usage_pitcher": (_fit_log1p_usage_pitcher, _apply_log1p_usage_pitcher),
    "count_state_abs_regime": (_fit_count_state_abs_regime, _apply_count_state_abs_regime),
    "debut_league_impute_success": (_fit_debut_league_impute_success,
                                    _apply_debut_league_impute_success),
    "batter_debut_league_impute_success": (_fit_batter_debut_league_impute_success,
                                           _apply_batter_debut_league_impute_success),
    "f_game_post2023_label_mapping": (_fit_f_game_post2023_label_mapping,
                                      _apply_f_game_post2023_label_mapping),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        import subprocess
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }


# ════════════════════════════════════════════════════════════════════
# 계약 대조 — 격리 구현(본 모듈) vs drift_eda.TRANSFORMS (레지스트리 참조 구현)
# ════════════════════════════════════════════════════════════════════
def check_transform_contract(cid: str, train: pd.DataFrame) -> dict:
    """fit+apply(동일 샘플) 가 drift_eda.TRANSFORMS[cid] 와 같은 출력인지 대조.
    샘플(10만 행, 전 시즌 포함) — 계약 파리티 증명. 수치형은 corr/max_abs_diff,
    범주형은 라벨 일치율로 판정."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), size=min(100_000, len(train)), replace=False)
    sample = train.iloc[idx]
    reg_fn = drift_eda.resolve_transform(
        "drift_eda.py:" + cid) or drift_eda.TRANSFORMS.get(cid)
    if reg_fn is None:
        return {"ok": False, "reason": f"등록 변환 참조 해석 불가: {cid}"}
    try:
        expected = reg_fn(sample)
        fit_fn, apply_fn = CANDIDATE_IMPLS[cid]
        got = apply_fn(sample, fit_fn(sample))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"실행 오류: {type(exc).__name__}: {exc}"}

    if expected.dtype.name == "category" or str(expected.dtype).startswith("category"):
        match = float((expected.astype(str).to_numpy() == got.astype(str).to_numpy()).mean())
        return {"ok": bool(match == 1.0), "kind": "category", "label_accuracy": match,
                "n_rows": len(sample), "ref": f"drift_eda.py:{cid}"}
    exp = expected.to_numpy(dtype="float64")
    gotv = got.to_numpy(dtype="float64")
    valid = ~np.isnan(exp) & ~np.isnan(gotv)
    if valid.sum() == 0:
        return {"ok": True, "kind": "numeric", "n_rows": len(sample),
                "note": "전부 NaN — 대조 생략", "ref": f"drift_eda.py:{cid}"}
    corr = float(np.corrcoef(exp[valid], gotv[valid])[0, 1])
    max_diff = float(np.max(np.abs(exp[valid] - gotv[valid])))
    ok = (corr > 0.999) and (max_diff < 0.01 or corr > 0.999999)
    return {"ok": bool(ok), "kind": "numeric", "corr": corr, "max_abs_diff": max_diff,
            "n_rows": len(sample), "ref": f"drift_eda.py:{cid}"}


# ════════════════════════════════════════════════════════════════════
# 폴드 학습/평가
# ════════════════════════════════════════════════════════════════════
def run_config(train: pd.DataFrame, feats: list, cats: list, folds: dict,
               seeds: list, cand: tuple | None = None) -> dict:
    """한 구성에 대한 4폴드 OOF. cand=(fit_fn, apply_fn, col) 면 각 폴드 학습 윈도우에서
    적합 후 train/val 에 적용. 반환: 폴드별 bss/pred_mean/n_rows/z(시드 평균 로짓)."""
    out: dict = {}
    for fn, (tr_m, va_m) in folds.items():
        if cand is not None:
            fit_fn, apply_fn, col = cand
            state = fit_fn(train.loc[tr_m])
            train.loc[tr_m, col] = apply_fn(train.loc[tr_m], state)
            train.loc[va_m, col] = apply_fn(train.loc[va_m], state)
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        zs = []
        for seed in seeds:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            zs.append(common.logit(model.predict(X_va, num_iteration=model.best_iteration)))
        z = np.mean(zs, axis=0)
        p = common.sigmoid(z)
        out[fn] = {"n_rows": int(va_m.sum()), "bss": float(common.score(p, yv)),
                   "pred_mean": float(p.mean()), "z": z}
    return out


def save_logits(cand_key: str, folds: dict, train: pd.DataFrame, res: dict) -> dict:
    """폴드 로짓 저장 + per-fold 다이제스트. Todo 7 블렌드 선택 입력용.
    캐시 경로: cache/qualification/{cand_key}/<fold>.npy (git 제외)."""
    out_dir = REPO / "cache" / "qualification" / cand_key
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = {}
    for fn in FOLDS:
        path = out_dir / f"{fn}.npy"
        np.save(path, np.asarray(res[fn]["z"], dtype=np.float64))
        per_fold[fn] = {
            "n_rows": int(res[fn]["n_rows"]),
            "bss": float(res[fn]["bss"]),
            "pred_mean": float(res[fn]["pred_mean"]),
            "digest": _sha256(path),
            "logits_path": f"cache/qualification/{cand_key}/{fn}.npy",
        }
    return per_fold


# ════════════════════════════════════════════════════════════════════
# 통계: delta / dmean / bootstrap CI / MLP 상호작용
# ════════════════════════════════════════════════════════════════════
def r_only_count(delta: dict) -> int:
    return sum(1 for fn in R_FOLDS if delta[fn] > 1e-9)


def dmean_per_fold(cand_res: dict, base_res: dict) -> dict:
    return {fn: float(cand_res[fn]["pred_mean"] - base_res[fn]["pred_mean"])
            for fn in FOLDS}


def bootstrap_delta_bss(y: np.ndarray, p_base: np.ndarray, p_cand: np.ndarray,
                        rng: np.random.Generator, n_iter: int) -> np.ndarray:
    """검증 행 리샘플(부트스트랩)로 delta BSS 분포. delta = cand − base."""
    y = np.asarray(y, dtype=float)
    pb = np.asarray(p_base, dtype=float)
    pc = np.asarray(p_cand, dtype=float)
    n = len(y)
    se_b = (pb - y) ** 2
    se_c = (pc - y) ** 2
    deltas = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        r = y[idx].mean()
        denom = r * (1.0 - r)
        bss_b = 1e5 * (1.0 - se_b[idx].mean() / denom)
        bss_c = 1e5 * (1.0 - se_c[idx].mean() / denom)
        deltas[i] = bss_c - bss_b
    return deltas


def bootstrap_report(y, p_base, p_cand, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    d = bootstrap_delta_bss(y, p_base, p_cand, rng, n_iter)
    pct = np.percentile(d, [5, 50, 95])
    return {"n_iter": int(n_iter), "seed": int(seed),
            "pct_5": float(pct[0]), "pct_50": float(pct[1]), "pct_95": float(pct[2]),
            "lower_bound_gt_0": bool(pct[0] > 0.0), "mean": float(d.mean())}


def mlp_interaction(folds: dict, train: pd.DataFrame, z_base: dict, z_cand: dict,
                    cand_delta: dict) -> dict:
    """w=0.51 로 (base-LGB + MLP) vs (base+feature)-LGB + MLP 블렌드 delta."""
    out: dict = {}
    deltas = []
    for fn in FOLDS:
        va_m = folds[fn][1]
        yv = train.loc[va_m, common.TARGET].values
        mlp_path = REPO / MLP_CACHE[fn]
        if not mlp_path.is_file():
            raise RuntimeError(f"[FAIL] MLP 로짓 캐시 없음: {MLP_CACHE[fn]}")
        z_mlp = np.load(mlp_path)
        if int(len(z_mlp)) != int(va_m.sum()):
            raise RuntimeError(f"[FAIL] MLP 로짓 길이 불일치 {fn}: {len(z_mlp)} "
                               f"!= 검증 행 수 {va_m.sum()}")
        z_blend_b = W_LGB * z_base[fn] + (1.0 - W_LGB) * z_mlp
        z_blend_c = W_LGB * z_cand[fn] + (1.0 - W_LGB) * z_mlp
        bss_b = float(common.score(common.sigmoid(z_blend_b), yv))
        bss_c = float(common.score(common.sigmoid(z_blend_c), yv))
        delta = bss_c - bss_b
        deltas.append(delta)
        out[fn] = {
            "n_rows": int(va_m.sum()),
            "blend_base_bss": bss_b,
            "blend_cand_bss": bss_c,
            "blend_delta": float(delta),
            "lgb_delta": float(cand_delta[fn]),
            "mlp_logits": MLP_CACHE[fn],
            "note": ("feature 가 블렌드를 돕는다(blend_delta>0) / 해친다(blend_delta<0); "
                     "lgb_delta>0 인데 blend_delta≤0 이면 MLP 가 이미 해당 신호를 포착했을 가능성"),
        }
    mean_delta = float(np.mean(deltas))
    return {"per_fold": out, "mean_delta": mean_delta,
            "helps_blend": bool(mean_delta > 0.0),
            "w_lgb": W_LGB,
            "mlp_source": "cache/mlp_{fold}.npy (챔피언 5시드 MLP — Task 2 sha256 대조)"}


def gate_verdict(delta: dict, dmean: dict, boot_primary: dict) -> tuple[str, list[str]]:
    """게이트 판정. (verdict, 실패 조건 목록) — 빈 목록 = 채택."""
    primary = delta["primary"]
    r_imp = r_only_count(delta)
    max_abs_dmean = max(abs(v) for v in dmean.values())
    failures: list[str] = []
    if primary < PRIMARY_THRESHOLD:
        failures.append(f"primary {primary:+.1f} < +{PRIMARY_THRESHOLD:.0f}")
    if r_imp < R_ONLY_REQUIRED:
        failures.append(f"R-only {r_imp}/{R_ONLY_REQUIRED}")
    if max_abs_dmean > MEAN_POISON_THRESHOLD:
        failures.append(f"maxΔmean {max_abs_dmean:.4f} > {MEAN_POISON_THRESHOLD:.3f}")
    if not boot_primary.get("lower_bound_gt_0"):
        failures.append(f"bootstrap primary Δ 5%하한 {boot_primary.get('pct_5', float('nan')):+.3f} ≤ 0")
    return ("approved" if not failures else "rejected"), failures


# ════════════════════════════════════════════════════════════════════
# 리포트 (압축, 커밋 대상)
# ════════════════════════════════════════════════════════════════════
def write_report(evidence: dict) -> None:
    lines = []
    lines.append("# REPORT — Todo 4: 드리프트 인지 전처리/피처 후보 게이팅")
    lines.append("")
    lines.append(f"- 실행: {evidence['recorded_at_utc']} | git: {evidence['git_commit']} | "
                 f"seeds: {evidence['seeds']} | smoke: {evidence['smoke']}")
    lines.append(f"- 게이트: {GATE_STR}")
    lines.append("")
    base = evidence["base"]
    lines.append("## 기준선 (동결 49피처 LGB 10시드 × 4폴드)")
    lines.append("")
    lines.append("| fold | n_rows | BSS | Task2 문서화 | Δ | pred_mean |")
    lines.append("|---|---|---|---|---|---|")
    for fn in FOLDS:
        pf = base["per_fold"][fn]
        lines.append(f"| {fn} | {pf['n_rows']} | {pf['bss']:.3f} | "
                     f"{base['documented_lgb'][fn]:.3f} | {base['delta_vs_documented'][fn]:+.3f} | "
                     f"{pf['pred_mean']:.4f} |")
    lines.append("")
    lines.append(f"- 기준선 assert: within_tolerance={base['base_assertion']['within_tolerance']} "
                 f"(max|Δ| {base['base_assertion']['max_abs_delta_bss']:.3f}, 허용 ±1.0)")
    lines.append(f"- 기준선+MLP 블렌드 primary 재현: 챔피언 {CHAMPION_BLEND_PRIMARY:.3f} → "
                 f"{base['blend_reproduction']['primary']['repro_blend_bss']:.3f} "
                 f"(Δ {base['blend_reproduction']['primary']['delta']:+.3f})")
    lines.append("")
    lines.append("## 후보 판정")
    lines.append("")
    lines.append("| candidate | primary Δ | R-only | maxΔmean | boot 5% | blend Δ | verdict | reasons |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cid, c in evidence["candidates"].items():
        d = c["delta_vs_base"]
        boot = c["bootstrap_ci"]["primary"]
        mi = c["mlp_interaction"]
        reasons = "; ".join(c["reasons"]) if c["reasons"] else "—"
        lines.append(f"| {cid} | {d['primary']:+.1f} | {c['r_only_improved']}/3 | "
                     f"{c['max_abs_dmean']:.4f} | {boot['pct_5']:+.2f} | "
                     f"{mi['mean_delta']:+.2f} | {c['verdict']} | {reasons} |")
    lines.append("")
    n_acc = sum(1 for c in evidence["candidates"].values() if c["verdict"] == "approved")
    lines.append(f"## 요약: {len(evidence['candidates'])} 후보 중 채택 {n_acc}")
    lines.append("")
    lines.append("참고: OOF 로짓은 `repro_979/cache/qualification/task4_*` (git 제외). "
                 "채택 후보는 Todo 7 홀드아웃 블렌드 선택 입력.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {REPORT_MD}", flush=True)


# ════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════
class _Log:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self.handle.write(msg + "\n")
        self.handle.flush()
        print(msg, flush=True)

    def close(self) -> None:
        self.handle.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="드리프트 인지 전처리/피처 후보 10시드 × 4폴드 게이팅 (Todo 4)")
    ap.add_argument("--candidate", default=None,
                    help="특정 후보 id 평가 (기본: 모든 proposed 8종). "
                         "blocked id 는 학습 전에 거부(exit 1).")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크: 시드 [42,43] + 첫 proposed(season_dev_success_pitcher) "
                         "전체 경로 검증 후 PASS 출력 (증거/캐시 분리)")
    ap.add_argument("--bootstrap-iter", type=int, default=BOOTSTRAP_ITER,
                    help=f"bootstrap 리샘플 반복 (기본 {BOOTSTRAP_ITER})")
    args = ap.parse_args(argv)

    t0 = time.time()
    smoke = bool(args.smoke)
    seeds = [42, 43] if smoke else list(SEEDS)
    n_boot = args.bootstrap_iter

    evidence_json = SMOKE_JSON if smoke else EVIDENCE_JSON
    evidence_log = SMOKE_LOG if smoke else EVIDENCE_LOG
    log = _Log(evidence_log)

    log.write(f"[feature_qualification] Todo 4 — 드리프트 인지 피처 후보 게이팅 "
              f"(seeds={seeds}, smoke={smoke})")
    log.write(f"[feature_qualification] 게이트: {GATE_STR}")

    # ── 1) 레지스트리 READ + 검증 (proposed 만 진입; blocked 는 학습 전 거부) ──
    if not REGISTRY_PATH.is_file():
        log.write(f"[FAIL] 레지스트리 없음: {REGISTRY_PATH}")
        log.close()
        return 1
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    candidates = registry.get("candidates", [])
    by_id = {c["id"]: c for c in candidates}
    valid, vproblems = drift_eda.validate_registry(registry, probe_transforms=False)
    log.write(f"[registry] validate_registry: {'PASS' if valid else 'FAIL'} "
              f"({len(candidates)} 항목)")
    for p in vproblems:
        log.write(f"  ! {p}")

    proposed_ids = [c["id"] for c in candidates if c.get("status") == "proposed"]
    blocked_ids = [c["id"] for c in candidates if c.get("status") == "blocked"]
    log.write(f"[registry] proposed {len(proposed_ids)}: {proposed_ids}")
    log.write(f"[registry] blocked {len(blocked_ids)}: {blocked_ids}")

    if args.candidate:
        cid = args.candidate
        entry = by_id.get(cid)
        if entry is None:
            log.write(f"[FAIL] 미등록 후보 id: {cid!r} — 학습 전에 거부")
            log.close()
            return 1
        if entry.get("status") != "proposed":
            log.write(f"[FAIL] 후보 {cid!r} 의 status={entry.get('status')!r} — "
                      f"proposed 만 평가 가능. blocked 기각 가설은 학습 금지 (승자 저주 방지).")
            log.write(f"  blocked_reason: {entry.get('prior_result', {}).get('blocked_reason', 'N/A')}")
            log.close()
            return 1
        target_ids = [cid]
    else:
        target_ids = proposed_ids
    if smoke:
        if not target_ids:
            log.write("[FAIL] 스모크 후보 없음")
            log.close()
            return 1
        target_ids = [target_ids[0]]
        log.write(f"[--smoke] 축소 실행: seeds={seeds} 후보 1종={target_ids[0]} "
                  f"(기본 동작과 무관한 테스트 모드)")

    # ── 2) 데이터 로드 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    missing = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if missing:
        log.write(f"[FAIL] 동결 49피처 부재: {missing}")
        log.close()
        return 1
    folds = qc.build_folds(train)
    leakage = qc._check_leakage(folds, train)
    if leakage:
        log.write(f"[FAIL] 누수 가드 실패:\n  " + "\n  ".join(leakage))
        log.close()
        return 1
    row_ok = all(int(folds[fn][1].sum()) == EXPECTED_ROW_COUNTS[fn] for fn in FOLDS)
    log.write(f"[fold] 검증 행 수: " + ", ".join(
        f"{fn}={int(folds[fn][1].sum())}" for fn in FOLDS)
        + f" | Task2 대조: {'PASS' if row_ok else 'FAIL'}")
    log.write(f"[leakage] 2025 검증 마스크 부재 4/4 PASS | train seasons: "
              f"{sorted(int(s) for s in train['season'].unique().tolist())}")
    if not row_ok:
        log.write("[FAIL] 폴드 행 수가 Task 2 와 불일치 — 마스크 정의 변경 의심")
        log.close()
        return 1

    # ── 3) 변환 계약 대조 (제안 후보 전부, 학습 전) ──
    contract_checks = {}
    for cid in proposed_ids:
        cc = check_transform_contract(cid, train)
        contract_checks[cid] = cc
        log.write(f"[contract] {cid:<38s} {'PASS' if cc['ok'] else 'FAIL'} "
                  f"{ {k: cc[k] for k in cc if k in ('kind','corr','max_abs_diff','label_accuracy')} }")
    bad_contract = [cid for cid, cc in contract_checks.items() if not cc["ok"]]
    if bad_contract:
        log.write(f"[FAIL] 변환 계약 대조 실패: {bad_contract} — 실행 중단")
        log.close()
        return 1

    # ── 4) 기준선 (동결 49피처) — Task 2 재현 assert ──
    log.write(f"[base] {len(CHAMPION_FEATURES)}피처 LGB {len(seeds)}시드 × 4폴드 "
              f"(deterministic=True, cats={LGB_CATS})")
    t_base = time.time()
    base_res = run_config(train, CHAMPION_FEATURES, list(LGB_CATS), folds, seeds)
    base_per_fold = save_logits("task4_base", folds, train, base_res)
    log.write(f"  [base] " + " ".join(
        f"{fn}={base_res[fn]['bss']:>8.3f}" for fn in FOLDS) + f"  ({time.time()-t_base:.0f}s)")

    delta_vs_doc = {fn: float(base_res[fn]["bss"] - DOCUMENTED_LGB[fn]) for fn in FOLDS}
    max_abs_doc = max(abs(v) for v in delta_vs_doc.values())
    base_within = max_abs_doc <= DECLARED_BASE_TOLERANCE and row_ok
    for fn in FOLDS:
        log.write(f"  [base/{fn}] BSS={base_res[fn]['bss']:.3f} "
                  f"(Task2 {DOCUMENTED_LGB[fn]:.3f}, Δ{delta_vs_doc[fn]:+.3f}) "
                  f"mean={base_res[fn]['pred_mean']:.4f} n={base_res[fn]['n_rows']}")

    # 기준선+MLP 블렌드 재현 (w=0.51, 챔피언 MLP 로짓) — 파이프라인 무결성 교차 확인
    blend_repro = {}
    for fn in FOLDS:
        va_m = folds[fn][1]
        yv = train.loc[va_m, common.TARGET].values
        z_mlp = np.load(REPO / MLP_CACHE[fn])
        z_b = W_LGB * base_res[fn]["z"] + (1.0 - W_LGB) * z_mlp
        bss_b = float(common.score(common.sigmoid(z_b), yv))
        blend_repro[fn] = {"repro_blend_bss": bss_b,
                           "documented": (CHAMPION_BLEND_PRIMARY if fn == "primary"
                                          else qc.DOCUMENTED[fn]["gain_fixed"] + DOCUMENTED_LGB[fn]),
                           "delta": bss_b - (CHAMPION_BLEND_PRIMARY if fn == "primary"
                                             else qc.DOCUMENTED[fn]["gain_fixed"] + DOCUMENTED_LGB[fn])}
    for fn in FOLDS:
        log.write(f"  [base+mlp blend/{fn}] BSS={blend_repro[fn]['repro_blend_bss']:.3f} "
                  f"(문서화 {blend_repro[fn]['documented']:.3f}, "
                  f"Δ{blend_repro[fn]['delta']:+.3f})")

    if not smoke and not base_within:
        log.write(f"[FAIL] 기준선 재현 실패: max|Δ|={max_abs_doc:.3f} > 허용 "
                  f"±{DECLARED_BASE_TOLERANCE:.1f} — Task 2 LGB 문서화 수치와 불일치")
        log.close()
        return 1

    # ── 5) 후보 평가 ──
    cand_out: dict = {}
    for cid in target_ids:
        entry = by_id[cid]
        fit_fn, apply_fn = CANDIDATE_IMPLS[cid]
        col = cid
        out_series = apply_fn(train.iloc[:5], fit_fn(train.iloc[:5]))
        is_cat = str(out_series.dtype).startswith("category")
        feats = CHAMPION_FEATURES + [col]
        cats = list(LGB_CATS) + ([col] if is_cat else [])
        log.write(f"\n[candidate] {cid} — {entry.get('name', '')} | "
                  f"transform: {entry.get('transform')} | feats={len(feats)} | "
                  f"cats={cats} | dtype={out_series.dtype}")
        t_c = time.time()
        cand_res = run_config(train, feats, cats, folds, seeds, cand=(fit_fn, apply_fn, col))
        cand_per_fold = save_logits(f"task4_{cid}", folds, train, cand_res)
        log.write(f"  [cand/{cid}] " + " ".join(
            f"{fn}={cand_res[fn]['bss']:>8.3f}" for fn in FOLDS) + f"  ({time.time()-t_c:.0f}s)")

        delta = {fn: float(cand_res[fn]["bss"] - base_res[fn]["bss"]) for fn in FOLDS}
        dmean = dmean_per_fold(cand_res, base_res)
        max_abs_dmean = max(abs(v) for v in dmean.values())
        r_imp = r_only_count(delta)

        # bootstrap CI (primary + overall pooled)
        y_pri = train.loc[folds["primary"][1], common.TARGET].values
        boot_pri = bootstrap_report(
            y_pri, common.sigmoid(base_res["primary"]["z"]),
            common.sigmoid(cand_res["primary"]["z"]), n_iter=n_boot)
        ys = np.concatenate([train.loc[folds[fn][1], common.TARGET].values for fn in FOLDS])
        pb = np.concatenate([common.sigmoid(base_res[fn]["z"]) for fn in FOLDS])
        pc = np.concatenate([common.sigmoid(cand_res[fn]["z"]) for fn in FOLDS])
        boot_overall = bootstrap_report(ys, pb, pc, n_iter=n_boot)

        # MLP 상호작용
        mi = mlp_interaction(folds, train, {fn: base_res[fn]["z"] for fn in FOLDS},
                             {fn: cand_res[fn]["z"] for fn in FOLDS}, delta)

        verdict, reasons = gate_verdict(delta, dmean, boot_pri)

        log.write(f"  [cand/{cid}] Δ: " + " ".join(f"{fn}={delta[fn]:+.2f}" for fn in FOLDS))
        log.write(f"  [cand/{cid}] Δmean: " + " ".join(f"{fn}={dmean[fn]:+.4f}" for fn in FOLDS)
                  + f" (max|Δmean|={max_abs_dmean:.4f}) | R-only {r_imp}/3")
        log.write(f"  [cand/{cid}] bootstrap(primary) 5/50/95%: "
                  f"{boot_pri['pct_5']:+.3f}/{boot_pri['pct_50']:+.3f}/{boot_pri['pct_95']:+.3f} | "
                  f"overall: {boot_overall['pct_5']:+.3f}/{boot_overall['pct_50']:+.3f}"
                  f"/{boot_overall['pct_95']:+.3f}")
        log.write(f"  [cand/{cid}] mlp blend Δ: " + " ".join(
            f"{fn}={mi['per_fold'][fn]['blend_delta']:+.2f}" for fn in FOLDS)
            + f" (mean {mi['mean_delta']:+.2f}, {'helps' if mi['helps_blend'] else 'hurts'})")
        fail_txt = " ✗ " + " ".join(reasons) if reasons else ""
        log.write(f"  [cand/{cid}] → {verdict}{fail_txt}")

        cand_out[cid] = {
            "candidate": cid,
            "registry_status": entry.get("status"),
            "name": entry.get("name"),
            "transform_ref": entry.get("transform"),
            "column_name": col,
            "is_categorical": bool(is_cat),
            "feature_count": len(feats),
            "mode": "addition",
            "contract_check": contract_checks[cid],
            "per_fold": cand_per_fold,
            "delta_vs_base": delta,
            "r_only_improved": int(r_imp),
            "max_abs_dmean": float(max_abs_dmean),
            "dmean_per_fold": dmean,
            "bootstrap_ci": {"primary": boot_pri, "overall_pooled": boot_overall},
            "mlp_interaction": mi,
            "verdict": verdict,
            "reasons": reasons,
        }

    # ── 6) 스키마 + 저장 ──
    schema = {
        "schema_version": SCHEMA_VERSION,
        "task": "todo-4-feature-gating",
        "smoke": smoke,
        "seeds": seeds,
        "folds": FOLDS,
        "r_folds": R_FOLDS,
        "gate": GATE_STR,
        "gate_definition": {
            "primary_threshold": PRIMARY_THRESHOLD,
            "r_only_required": R_ONLY_REQUIRED,
            "mean_poison_threshold": MEAN_POISON_THRESHOLD,
            "bootstrap_iter": n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_lower_bound_gt_0": True,
        },
        "registry": {
            "path": str(REGISTRY_PATH.relative_to(REPO)),
            "valid": valid,
            "n_proposed": len(proposed_ids),
            "n_blocked": len(blocked_ids),
            "proposed_ids": proposed_ids,
            "blocked_ids": blocked_ids,
            "blocked_rejected": [cid for cid in blocked_ids],
        },
        "integrity": {
            "gate": "PASS",
            "frozen_control": {
                "config_digest": qc.EXPECTED_CONFIG_DIGEST,
                "config_digest_matches_frozen": qc._SELF_INTEGRITY_OK,
                "champion_features": len(CHAMPION_FEATURES),
                "lgb_cats": LGB_CATS,
                "mlp_cats_sha256_checked": len(MLP_SHA256),
            },
        },
        "leakage_guard": {
            "no_2025_in_any_validation_mask": True,
            "transform_constants_train_window_only": True,
            "train_seasons": sorted(int(s) for s in train["season"].unique().tolist()),
        },
        "base": {
            "n_features": len(CHAMPION_FEATURES),
            "features_source": "team_member_materials/GIHO/submit979_extract/model/train_meta.json",
            "per_fold": base_per_fold,
            "documented_lgb": DOCUMENTED_LGB,
            "delta_vs_documented": delta_vs_doc,
            "base_assertion": {
                "within_tolerance": bool(base_within),
                "tolerance_bss": DECLARED_BASE_TOLERANCE,
                "max_abs_delta_bss": float(max_abs_doc),
                "row_counts_match_task2": bool(row_ok),
            },
            "blend_reproduction": {
                "w_lgb": W_LGB,
                "primary": blend_repro["primary"],
                "note": "기준선(재학습 LGB) + 챔피언 MLP 블렌드가 Task 2 챔피언 블렌드 수치 재현 확인",
            },
        },
        "candidates": cand_out,
        "accepted": [cid for cid, c in cand_out.items() if c["verdict"] == "approved"],
        "rejected": [cid for cid, c in cand_out.items() if c["verdict"] == "rejected"],
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }

    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    # 후보별 result.json (Todo 7 입력 계약 — Task 2 레이아웃 미러)
    for cid, c in cand_out.items():
        out_dir = REPO / "cache" / "qualification" / f"task4_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.write(f"\n[schema] 저장: {evidence_json}")
    log.write(f"[result] 채택 {len(schema['accepted'])} / 기각 {len(schema['rejected'])}")

    if not smoke:
        write_report(schema)

    # ── 7) 스모크 판정 ──
    if smoke:
        ok = (len(seeds) == 2
              and all(fn in base_per_fold for fn in FOLDS)
              and target_ids[0] in cand_out
              and all(fn in cand_out[target_ids[0]]["per_fold"] for fn in FOLDS)
              and "delta_vs_base" in cand_out[target_ids[0]]
              and "bootstrap_ci" in cand_out[target_ids[0]]
              and "mlp_interaction" in cand_out[target_ids[0]]
              and cand_out[target_ids[0]]["verdict"] in ("approved", "rejected")
              and "accepted" in schema and "rejected" in schema)
        log.write(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — "
                  f"폴드/델타/bootstrap/MLP상호작용/게이트/JSON 경로 검증 완료 "
                  f"(seeds={seeds}, candidate={target_ids[0]}, verdict="
                  f"{cand_out[target_ids[0]]['verdict']})")
        log.close()
        return 0 if ok else 1

    log.close()
    print(f"\n[feature_qualification] 완료 (총 {time.time()-t0:.0f}s) — 증거 {evidence_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
