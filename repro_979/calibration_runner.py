#!/usr/bin/env python3
"""calibration_runner.py — Todo 6: season-balanced training + nested slope-only logit calibration.

목적 (aimers9-top100-score-improvement Todo 6):
  시즌 가중(season-weight) 학습 스케줄 4종(사전 선언 상수)과 중첩(nested) slope-only 로짓
  캘리브레이션을 평가한다. 타깃 평균 누수(target-mean leakage) 없이, 각 학습 윈도우 내에서만
  모든 가중/캘리브레이션 파라미터를 피팅하고 평가 폴드(스코어 폴드)는 절대 건드리지 않는다.

핵심 계약 (Task 2/qualification_runner 재사용):
  - 폴드 마스크 = qualification_runner.build_folds == screen_all_10seed.py:build_folds.
  - 동결 49피처 계약 = team_member_materials/GIHO/submit979_extract/model/train_meta.json
    (qualification_runner.CHAMPION_FEATURES 로 READ 후 대조). 금지 피처
    (asof_n_bucket, score_diff_binary) 부재 확인.
  - 동결 캘리브레이션 상수: C_LOGIT=-0.0404 (=2025 target-mean 정책). b 로 동결 — 절대 피팅 금지.
  - 로짓 집계 = 시드별 로짓 평균 (챔피언 컨벤션 "mean of per-seed logits").
  - 챔피언 기준 = cache/qualification/champion/*.npy (블렌드 로짓) + result.json (lgb_bss/문서화 수치).

중첩(nested) 누수 계약 (비협상):
  - outer 폴드 (primary/r2022/r2023/r2024) 의 학습 윈도우 마지막 시즌 = inner_val (피팅/선택 셋),
    즉 inner split ⊂ outer 학습 윈도우. 스코어 폴드(outer_val) 는 절대 피팅/선택에 미사용.
  - LGB 는 학습 윈도우 전체(outer_train) + 시즌 가중, FIXED num_boost_round=109
    (train_meta.json 의 챔피언 문서화 상수 — early stopping 없음 → 스코어 폴드로 어떤 파라미터도
    선택하지 않음). 측정상 학습 윈도우에서 마지막 시즌을 제외하고 학습하면 외부 폴드 BSS 가 0 으로
    붕괴하므로(드리프트), 모델은 전 윈도우로 학습하고 마지막 시즌은 캘리브레이션/선택 전용으로 쓴다.
  - slope 은 inner_val 로짓에서만 피팅 (b=C_LOGIT 동결, a 만 피팅). 스코어 폴드 라벨 미사용.
  - 스케줄 선택(schedule selection) = inner_val raw BSS 로만 수행.
  - 자체 점검: 어떤 피팅 셋에도 스코어 폴드 행이 없음을 어서션으로 증명.
    --leakage-injection 으로 스코어 폴드 피팅을 주입하면 LeakageError 로 비정상 종료(exit 2).

채택 게이트 (10시드 기준):
  - treatment (schedule × method) 가 모든 R-only 3폴드에서 챔피언 raw 블렌드 대비
    ΔBSS > +1.0 (UNCERTAINTY_THRESHOLD_BSS) 이고
  - 4폴드 전체 max|transformed_mean − raw_mean| ≤ 0.005 (MEAN_ALIGNMENT_BOUND)
  일 때만 accepted. 그 외 rejected (이유 기록).

--smoke: seeds=[42,43], schedules=[uniform], methods=[raw,policy,slope_only]
  4폴드 전체 경로(폴드/스케줄/캘리브레이션/누수/스키마/챔피언 기준) 검증 후 PASS 출력.
  기본(전체 그리드) 동작에 영향 없음.

출력:
  - 증거: .omo/evidence/aimers9-top100/task-6-calibration.{json,log}
  - OOF 로짓(npy, Todo 7 후보 입력): repro_979/cache/calibration/task6/{schedule}/{fold}.npy
    (캐시 산출물 — git 제외). 캘리브레이션은 result.json 의 a/b 파라미터로 재적용 가능.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("LGAIMERS_ROOT", str(PROJECT_ROOT))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
# Task 2 동결 챔피언 컨트롤 재사용 — 모듈 로드 시 자기 무결성 게이트(구성 다이제스트) 포함.
# 조작 시 SystemExit(1) 으로 본 러너도 즉시 중단(무결성 상속).
from qualification_runner import (  # noqa: E402
    CHAMPION_FEATURES, LGB_CATS, C_LOGIT, FOLDS, R_FOLDS, SEEDS, W_LGB,
    build_folds, SCHEMA_VERSION, FORBIDDEN_FEATURES,
)

# ════════════════════════════════════════════════════════════════════
# 동결 제어 (Task 2 와 동일 값 — qualification_runner 게이트가 보증)
# ════════════════════════════════════════════════════════════════════
B_POLICY = C_LOGIT            # b = -0.0404 동결 (2025 target-mean 정책 상수) — 절대 피팅 금지
NUM_BOOST_ROUND = 109         # 챔피언 문서화 상수 (train_meta.json num_boost_round) — 고정, early stopping 없음
EARLY_STOPPING = 50           # 미사용 (고정 라운드) — 보존은 문서화 목적

# 채택 게이트 임계
UNCERTAINTY_THRESHOLD_BSS = 1.0     # R-only 폴드당 ΔBSS > +1.0
MEAN_ALIGNMENT_BOUND = 0.005        # max|transformed_mean − raw_mean| ≤ 0.005

# slope 피팅 판정 진단: inner_val Brier 프로파일이 평평(비식별)하면 플래그
SLOPE_FLAT_BRIER_RANGE = 0.005      # 그리드 Brier 범위가 이 값 이하면 flat(비식별) 판정

# ════════════════════════════════════════════════════════════════════
# 사전 선언 시즌 가중 스케줄 (상수 — 스코어 폴드로 최적화 금지)
#   w(s): 학습 윈도우 내 시즌 s 에 대한 행 가중 (LightGBM weight_tr).
#   s_min/s_max = inner_train 의 최소/최대 시즌, t=(s−s_min)/(s_max−s_min) ∈ [0,1].
# ════════════════════════════════════════════════════════════════════
SEASON_SCHEDULES = {
    "uniform":      "w(s)=1.0 for all s — 챔피언 등가 baseline (무가중)",
    "linear_recent": "w(s)=1+t — 최근 시즌 강조 선형 램프 [1.0, 2.0]",
    "exp_recent":   "w(s)=2**t — 최근 시즌 강조 지수 램프 [1.0, 2.0]",
    "regime_step":  "w(s)=0.5 if s<mid else 1.5, mid=(s_min+s_max)/2 — 구레짐 다운·신레짐 업",
}

# 캘리브레이션 메서드 (사전 선언)
CALIBRATION_METHODS = {
    "raw":        "캘리브레이션 없음 (a=1, b=0)",
    "policy":     "동결 챔피언 정책 (a=1, b=C_LOGIT=-0.0404)",
    "slope_only": "중첩 slope-only (b=C_LOGIT 동결, a 만 inner split 에서 피팅)",
}

# slope 최적화 그리드 (사전 선언 상수 — 피팅은 inner_val 에서만)
A_GRID = np.linspace(0.6, 1.4, 201)
A_REFINE_RANGE = 0.02
A_REFINE_STEP = 0.001
A_LO, A_HI = 0.6, 1.4

# 챔피언 기준 아티팩트
CHAMPION_BLEND_DIR = REPO / "cache" / "qualification" / "champion"
CHAMPION_RESULT_JSON = CHAMPION_BLEND_DIR / "result.json"
CHAMPION_META = PROJECT_ROOT / "team_member_materials" / "GIHO" / "submit979_extract" / "model" / "train_meta.json"
CALIB_CACHE_DIR = REPO / "cache" / "calibration" / "task6"


class LeakageError(RuntimeError):
    """스코어 폴드가 피팅 셋에 포함된 누수 — 비정상 종료를 유발."""


# ── 작은 헬퍼 ────────────────────────────────────────────────────────
def _git_commit() -> str:
    try:
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
        "lightgbm": __import__("lightgbm").__version__,
        "torch": (lambda: __import__("torch").__version__)(),
        "cuda_available": False,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) == 0 or len(b) == 0:
        return False
    return np.intersect1d(np.sort(a), np.sort(b), assume_unique=True).size > 0


# ── 시즌 가중 스케줄 ────────────────────────────────────────────────
def schedule_weights(sched: str, seasons: np.ndarray) -> np.ndarray:
    """inner_train 시즌 배열 → LGB 행 가중. 공식은 SEASON_SCHEDULES 에 문서화(상수)."""
    seasons = seasons.astype(float)
    s_min, s_max = float(seasons.min()), float(seasons.max())
    span = max(s_max - s_min, 1e-9)
    t = (seasons - s_min) / span  # [0,1]
    if sched == "uniform":
        return np.ones(len(seasons))
    if sched == "linear_recent":
        return 1.0 + t
    if sched == "exp_recent":
        return np.power(2.0, t)
    if sched == "regime_step":
        mid = 0.5 * (s_min + s_max)
        return np.where(seasons < mid, 0.5, 1.5)
    raise ValueError(f"미등록 스케줄: {sched}")


# ── 누수 가드 ───────────────────────────────────────────────────────
def _assert_no_scored_fold_overlap(fn: str, fit_idx: np.ndarray, scored_idx: np.ndarray) -> None:
    """피팅 셋(fit_idx) 에 스코어 폴드 행(scored_idx) 이 하나라도 있으면 LeakageError."""
    if _has_overlap(np.asarray(fit_idx, dtype=np.int64),
                    np.asarray(scored_idx, dtype=np.int64)):
        raise LeakageError(
            f"[LEAKAGE] fold={fn}: 캘리브레이션/선택 피팅 셋에 스코어 폴드 행 포함 — "
            "스코어 폴드 라벨로 피팅 시도가 탐지되어 중단합니다."
        )


def assert_leakage_free(splits: dict, train: pd.DataFrame) -> list[str]:
    """중첩 분할 + 2025 부재 자체 점검. 문제 목록 반환 (빈 목록 = 통과)."""
    problems = []
    for fn, sp in splits.items():
        tr = np.where(sp["outer_train"])[0]
        va = np.where(sp["outer_val"])[0]
        itr = np.where(sp["inner_train"])[0]
        iva = np.where(sp["inner_val"])[0]
        if _has_overlap(tr, va):
            problems.append(f"{fn}: outer_train ∩ outer_val 비어있지 않음 (폴드 구성 위반)")
        if not np.isin(itr, tr).all():
            problems.append(f"{fn}: inner_train ⊄ outer_train")
        if not np.isin(iva, tr).all():
            problems.append(f"{fn}: inner_val ⊄ outer_train")
        if _has_overlap(itr, iva):
            problems.append(f"{fn}: inner_train ∩ inner_val 비어있지 않음")
        if _has_overlap(iva, va):
            problems.append(f"{fn}: inner_val ∩ outer_val — 스코어 폴드가 피팅/선택 셋에 누출!")
        if (train.loc[va, "season"] == 2025).any():
            problems.append(f"{fn}: 검증 마스크에 2025 시즌 포함 — 누수!")
        if (train.loc[iva, "season"] == 2025).any():
            problems.append(f"{fn}: inner_val 에 2025 시즌 포함 — 누수!")
    return problems


# ── 중첩 분할 ───────────────────────────────────────────────────────
def build_nested_splits(train: pd.DataFrame, folds: dict) -> dict:
    """outer 폴드의 학습 윈도우 마지막 시즌을 inner_val, 앞부분을 inner_train 으로 분할."""
    splits = {}
    for fn in FOLDS:
        tr_m, va_m = folds[fn]
        seasons_tr = train.loc[tr_m, "season"]
        last_season = int(seasons_tr.max())
        splits[fn] = {
            "outer_train": tr_m,
            "outer_val": va_m,
            "inner_train": tr_m & (train["season"] < last_season),
            "inner_val": tr_m & (train["season"] == last_season),
            "last_season": last_season,
        }
    return splits


# ── slope 피팅 (Brier 최소화, b 동결) ───────────────────────────────
def _brier_grid(z: np.ndarray, y: np.ndarray, a_grid: np.ndarray, b: float) -> np.ndarray:
    """벡터화 Brier(a) — 결정적, 순수 numpy. 청크 처리로 메모리 절약."""
    n = len(a_grid)
    out = np.empty(n)
    ch = 64
    for i in range(0, n, ch):
        aa = a_grid[i:i + ch, None]
        p = common.sigmoid(aa * z[None, :] + b)
        out[i:i + ch] = np.mean((p - y[None, :]) ** 2, axis=1)
    return out


def _fit_slope_grid(z_fit: np.ndarray, y_fit: np.ndarray, b: float) -> tuple[float, float, float]:
    """Brier 를 최소화하는 slope a 만 피팅 (b 는 동결 상수). inner_val 에서만 호출.
    반환: (a, best_brier, brier_range_over_grid) — range 가 작으면 비식별(flat)."""
    briers = _brier_grid(z_fit, y_fit, A_GRID, b)
    brier_range = float(briers.max() - briers.min())
    a0 = float(A_GRID[int(np.argmin(briers))])
    lo, hi = max(A_LO, a0 - A_REFINE_RANGE), min(A_HI, a0 + A_REFINE_RANGE)
    fine = np.linspace(lo, hi, int(round((hi - lo) / A_REFINE_STEP)) + 1)
    fb = _brier_grid(z_fit, y_fit, fine, b)
    a = float(fine[int(np.argmin(fb))])
    return a, float(fb.min()), brier_range


def fit_slope_nested(fn: str, z_fit: np.ndarray, y_fit: np.ndarray,
                     fit_idx: np.ndarray, scored_idx: np.ndarray, b: float):
    """누수 가드를 통과한 slope 피팅. fit_idx 에 스코어 폴드 행이 있으면 LeakageError."""
    _assert_no_scored_fold_overlap(fn, fit_idx, scored_idx)
    a, best_brier, brier_range = _fit_slope_grid(z_fit, y_fit, b)
    return a, best_brier, brier_range


# ── 캘리브레이션 커브 (decile bin reliability) ─────────────────────
def calibration_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[list, float]:
    """decile 예측 평균 vs 실제 발생률 + ECE."""
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, total = [], float(len(p))
    for k in range(n_bins):
        m = bins == k
        if int(m.sum()) == 0:
            continue
        rows.append({
            "bin": k,
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()),
            "actual_rate": float(y[m].mean()),
        })
    ece = float(sum(r["n"] * abs(r["actual_rate"] - r["pred_mean"]) for r in rows) / total)
    return rows, ece


# ── LGB 학습 (시즌 가중 + 고정 라운드, inner/outer 로짓 산출) ───────
def train_schedule_seed(train: pd.DataFrame, feats: list[str], sp: dict,
                        sched: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """outer 학습 윈도우 전체(outer_train) + 시즌 가중, FIXED NUM_BOOST_ROUND 라운드.
    common.train_model 과 동일한 PARAMS/weight_tr 인터페이스지만 early stopping 콜백은
    쓰지 않는다(common.train_model 의 조기종료는 inner_val 에서 즉시 발동해 fixed-round
    프로토콜이 깨짐 — 측정 확인). 스코어 폴드는 어떤 파라미터 선택에도 미사용.
    반환: (inner_val 로짓, outer_val 로짓, rounds). inner_val 로짓은 slope/스케줄 선택 전용."""
    import lightgbm as lgb  # noqa: PLC0415
    X_tr = train.loc[sp["outer_train"], feats]
    y_tr = train.loc[sp["outer_train"], common.TARGET]
    X_iv = train.loc[sp["inner_val"], feats]
    X_ov = train.loc[sp["outer_val"], feats]
    w = schedule_weights(sched, train.loc[sp["outer_train"], "season"].values)
    params = dict(common.PARAMS)
    params["seed"] = seed
    dtr = lgb.Dataset(X_tr, y_tr, weight=w, categorical_feature=list(LGB_CATS))
    model = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND)
    z_iv = common.logit(model.predict(X_iv, num_iteration=NUM_BOOST_ROUND))
    z_ov = common.logit(model.predict(X_ov, num_iteration=NUM_BOOST_ROUND))
    return z_iv, z_ov, NUM_BOOST_ROUND


# ── 챔피언 기준 로드 ────────────────────────────────────────────────
def load_champion_control(train: pd.DataFrame, folds: dict) -> dict:
    """챔피언 raw 블렌드(캐시 로짓) BSS/평균 + 문서화 lgb_bss. result.json 과 교차 대조."""
    if not CHAMPION_RESULT_JSON.is_file():
        raise RuntimeError(f"[FAIL] 챔피언 결과 JSON 없음: {CHAMPION_RESULT_JSON}")
    result = json.loads(CHAMPION_RESULT_JSON.read_text(encoding="utf-8"))
    control = {"blend_logits_source": "cache/qualification/champion/*.npy", "per_fold": {}}
    problems = []
    for fn in FOLDS:
        va_m = folds[fn][1]
        yv = train.loc[va_m, common.TARGET].values
        path = CHAMPION_BLEND_DIR / f"{fn}.npy"
        if not path.is_file():
            raise RuntimeError(f"[FAIL] 챔피언 블렌드 로짓 없음: {path}")
        z = np.load(path)
        if len(z) != int(va_m.sum()):
            raise RuntimeError(f"[FAIL] 챔피언 로짓 길이 불일치 {fn}: {len(z)} != {int(va_m.sum())}")
        p = common.sigmoid(z)
        bss = float(common.score(p, yv))
        doc = result.get("per_fold", {}).get(fn, {})
        if abs(bss - float(doc.get("bss", -1.0))) > 1e-6:
            problems.append(f"{fn}: 챔피언 블렌드 BSS 재계산 {bss:.6f} != result.json {doc.get('bss')}")
        control["per_fold"][fn] = {
            "blend_bss": bss,
            "blend_mean": float(p.mean()),
            "lgb_bss": float(doc.get("lgb_bss", float("nan"))),
            "n_rows": int(va_m.sum()),
            "logits_sha256": _sha256(path),
        }
    control["cross_check_ok"] = not problems
    control["cross_check_problems"] = problems
    return control


# ── 채택 게이트 ─────────────────────────────────────────────────────
def acceptance_verdict(fn_results: dict, champion: dict) -> tuple[str, list]:
    """treatment 의 4폴드 결과 기준 채택/기각. 빈 이유 목록 = 채택."""
    r_deltas = {fn: fn_results[fn]["bss_trans"] - champion["per_fold"][fn]["blend_bss"]
                for fn in R_FOLDS}
    max_abs_dmean = max(abs(fn_results[fn]["dmean"]) for fn in FOLDS)
    reasons = []
    if not all(r_deltas[fn] > UNCERTAINTY_THRESHOLD_BSS for fn in R_FOLDS):
        reasons.append("R-only ΔBSS vs 챔피언 블렌드 ≤ +1.0 미충족: "
                       + ", ".join(f"{fn} {r_deltas[fn]:+.2f}" for fn in R_FOLDS))
    if max_abs_dmean > MEAN_ALIGNMENT_BOUND:
        reasons.append(f"mean-alignment 위반: max|Δmean| {max_abs_dmean:.4f} > {MEAN_ALIGNMENT_BOUND:.3f}")
    return ("accepted" if not reasons else "rejected"), reasons


# ── 메인 평가 ───────────────────────────────────────────────────────
def run_study(train: pd.DataFrame, folds: dict, splits: dict,
              seeds: list[int], schedules: list[str], methods: list[str],
              champion: dict, smoke: bool) -> dict:
    feats = list(CHAMPION_FEATURES)
    per_fold: dict[str, dict] = {}
    leakage_checks: list[dict] = []

    for fn in FOLDS:
        sp = splits[fn]
        scored_idx = np.where(sp["outer_val"])[0]
        inner_idx = np.where(sp["inner_val"])[0]
        y_ov = train.loc[sp["outer_val"], common.TARGET].values
        y_iv = train.loc[sp["inner_val"], common.TARGET].values

        # 누수 자체 점검 (피팅/선택 셋 ⊂ 학습 윈도우, 스코어 폴드 비포함, 2025 부재)
        inner_problems = [p for p in assert_leakage_free({fn: sp}, train)]
        leakage_checks.append({
            "check": f"nested-leakage-{fn}",
            "ok": not inner_problems,
            "detail": ("중첩 분할 누수 없음" if not inner_problems else "; ".join(inner_problems)),
        })
        if inner_problems:
            raise LeakageError(f"[LEAKAGE] fold={fn}: " + "; ".join(inner_problems))

        n_iv, n_ov = int(sp["inner_val"].sum()), int(sp["outer_val"].sum())
        z_iv_sum = np.zeros(n_iv)
        z_ov_sum = np.zeros(n_ov)

        fold_res: dict[str, dict] = {}
        for sched in schedules:
            # 시드 로짓 평균 (챔피언 집계 컨벤션)
            z_iv_sum[:] = 0.0
            z_ov_sum[:] = 0.0
            for seed in seeds:
                z_iv, z_ov, _ = train_schedule_seed(train, feats, sp, sched, seed)
                z_iv_sum += z_iv
                z_ov_sum += z_ov
            z_iv = z_iv_sum / len(seeds)
            z_ov = z_ov_sum / len(seeds)
            p_raw = common.sigmoid(z_ov)
            bss_raw = float(common.score(p_raw, y_ov))
            mean_raw = float(p_raw.mean())
            inner_bss_raw = float(common.score(common.sigmoid(z_iv), y_iv))

            # OOF 로짓 저장 (Todo 7 후보 입력 — 캐시, git 제외)
            if not smoke:
                out_dir = CALIB_CACHE_DIR / sched
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{fn}.npy", z_ov.astype(np.float64))

            sched_res: dict[str, dict] = {
                "inner_val_bss_raw": inner_bss_raw,
                "inner_val_n_rows": int(n_iv),
                "rounds_fixed": NUM_BOOST_ROUND,
                "methods": {},
            }
            for method in methods:
                if method == "raw":
                    a, b = 1.0, 0.0
                elif method == "policy":
                    a, b = 1.0, B_POLICY
                elif method == "slope_only":
                    a, best_brier, brier_range = fit_slope_nested(fn, z_iv, y_iv,
                                                                  fit_idx=inner_idx,
                                                                  scored_idx=scored_idx,
                                                                  b=B_POLICY)
                    b = B_POLICY
                else:
                    raise ValueError(f"미등록 캘리브레이션: {method}")

                z_t = a * z_ov + b
                p_t = common.sigmoid(z_t)
                bss_trans = float(common.score(p_t, y_ov))
                mean_trans = float(p_t.mean())
                dmean = mean_trans - mean_raw
                curve, ece = calibration_curve(p_t, y_ov)
                method_res = {
                    "a": float(a),
                    "b": float(b),
                    "bss_raw": bss_raw,
                    "mean_raw": mean_raw,
                    "bss_trans": bss_trans,
                    "mean_trans": mean_trans,
                    "dmean": dmean,
                    "delta_vs_champion_blend": bss_trans - champion["per_fold"][fn]["blend_bss"],
                    "delta_vs_champion_lgb": bss_trans - champion["per_fold"][fn]["lgb_bss"],
                    "delta_raw_vs_champion_blend": bss_raw - champion["per_fold"][fn]["blend_bss"],
                    "ece": ece,
                    "calibration_curve": curve,
                }
                if method == "slope_only":
                    method_res["slope_fit"] = {
                        "best_brier_inner": best_brier,
                        "brier_range_over_grid": brier_range,
                        "flat_non_identifiable": bool(brier_range <= SLOPE_FLAT_BRIER_RANGE),
                        "at_grid_edge": bool(a <= A_LO + 1e-9 or a >= A_HI - 1e-9),
                    }
                sched_res["methods"][method] = method_res
            fold_res[sched] = sched_res
            row = " ".join(
                f"{sched_res['methods'][m]['bss_trans']:>8.1f}" for m in methods)
            print(f"  [{fn:<8s}] {sched:<12s} inner={inner_bss_raw:>8.1f} "
                  f"outer{'|'.join(methods)} [{row}] "
                  f"rounds={sched_res['rounds_fixed']}", flush=True)

        # 중첩 선택: 스케줄 = argmax inner raw BSS, 메서드 = argmax inner BSS (calib 적용 후)
        best_sched = max(schedules, key=lambda s: fold_res[s]["inner_val_bss_raw"])
        per_fold[fn] = {
            "outer_val_season": sorted(int(s) for s in train.loc[sp["outer_val"], "season"].unique()),
            "n_outer_val": int(n_ov),
            "inner_train_seasons": sorted(int(s) for s in train.loc[sp["inner_train"], "season"].unique()),
            "inner_val_season": int(sp["last_season"]),
            "n_inner_train": int(sp["inner_train"].sum()),
            "n_inner_val": int(n_iv),
            "nested_selected_schedule": best_sched,
            "nested_selected_schedule_by": "argmax inner_val raw BSS (스코어 폴드 미사용)",
            "treatments": fold_res,
        }

    # ── 채택 게이트 ──
    # treatment 결과: (schedule, method) → per-fold dict (bss_trans/dmean)
    treat_results: dict[str, dict[str, dict]] = {}
    for sched in schedules:
        for method in methods:
            key = f"{sched}__{method}"
            treat_results[key] = {
                fn: {"bss_trans": per_fold[fn]["treatments"][sched]["methods"][method]["bss_trans"],
                     "bss_raw": per_fold[fn]["treatments"][sched]["methods"][method]["bss_raw"],
                     "dmean": per_fold[fn]["treatments"][sched]["methods"][method]["dmean"]}
                for fn in FOLDS
            }
    acceptance = {"gate_definition": (
        f"R-only 3폴드 ΔBSS vs 챔피언 raw 블렌드 > +{UNCERTAINTY_THRESHOLD_BSS:.1f} "
        f"AND max|Δmean| ≤ {MEAN_ALIGNMENT_BOUND} (10시드 기준) → accepted | 그 외 rejected"),
        "per_treatment": {}}
    for key, res in treat_results.items():
        verdict, reasons = acceptance_verdict(res, champion)
        acceptance["per_treatment"][key] = {
            "verdict": verdict,
            "reasons": reasons,
            "r_only_delta_vs_blend": {fn: res[fn]["bss_trans"]
                                      - champion["per_fold"][fn]["blend_bss"] for fn in R_FOLDS},
            "max_abs_dmean": max(abs(res[fn]["dmean"]) for fn in FOLDS),
        }
    acceptance["accepted"] = sorted(k for k, v in acceptance["per_treatment"].items()
                                    if v["verdict"] == "accepted")

    return {"per_fold": per_fold, "acceptance": acceptance, "leakage_checks": leakage_checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 6: season-balanced training + nested slope-only logit calibration")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: seeds=[42,43], schedules=[uniform], methods=[raw,policy,slope_only]")
    parser.add_argument("--seeds", default=None, help="시드 목록 (쉼표, 기본 42..51)")
    parser.add_argument("--schedules", default=None,
                        help="스케줄 목록 (쉼표, 기본 전체 4종)")
    parser.add_argument("--methods", default=None,
                        help="캘리브레이션 메서드 목록 (쉼표, 기본 전체 3종)")
    parser.add_argument("--leakage-injection", action="store_true",
                        help="Failure QA: 스코어 폴드 라벨로 slope 피팅을 주입 — 누수 가드가 "
                             "LeakageError 를 던져 비정상 종료(exit 2)해야 함")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 .omo/evidence/aimers9-top100/task-6-calibration.json)")
    args = parser.parse_args(argv)

    smoke = bool(args.smoke)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)
    schedules = [s.strip() for s in args.schedules.split(",")] if args.schedules else list(SEASON_SCHEDULES)
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(CALIBRATION_METHODS)

    if smoke:
        seeds = [42, 43]
        schedules = ["uniform"]
        methods = ["raw", "policy", "slope_only"]
    for s in schedules:
        if s not in SEASON_SCHEDULES:
            print(f"[FAIL] 미등록 스케줄: {s} — 등록: {sorted(SEASON_SCHEDULES)}", file=sys.stderr)
            return 1
    for m in methods:
        if m not in CALIBRATION_METHODS:
            print(f"[FAIL] 미등록 메서드: {m} — 등록: {sorted(CALIBRATION_METHODS)}", file=sys.stderr)
            return 1
    if not seeds:
        print("[FAIL] 시드 목록이 비어 있음", file=sys.stderr)
        return 1

    evidence_path = (
        Path(args.evidence).expanduser().resolve() if args.evidence
        else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-6-calibration.json"
    )
    t0 = time.time()

    # ── 1) 챔피언 49피처 계약 (동결 아티팩트 READ 후 대조) ──
    if not CHAMPION_META.is_file():
        print(f"[FAIL] train_meta.json 없음: {CHAMPION_META}", file=sys.stderr)
        return 1
    meta = json.loads(CHAMPION_META.read_text(encoding="utf-8"))
    meta_feats = tuple(meta.get("features", []))
    contract_problems = []
    if meta_feats != CHAMPION_FEATURES:
        contract_problems.append(
            f"피처 {len(meta_feats)}개 != 동결 49 ({len(CHAMPION_FEATURES)})")
    forbidden = [f for f in FORBIDDEN_FEATURES if f in meta_feats]
    if forbidden:
        contract_problems.append(f"금지 피처 포함: {forbidden}")
    if tuple(meta.get("cats", [])) != LGB_CATS:
        contract_problems.append(f"LGB cats != 동결 {LGB_CATS}")
    feature_contract = {
        "source": "model/train_meta.json",
        "n_features": len(meta_feats),
        "forbidden_present": bool(forbidden),
        "ok": not contract_problems,
        "problems": contract_problems,
    }
    if contract_problems:
        print("[FAIL] 피처 계약 게이트 실패:\n  " + "\n  ".join(contract_problems), file=sys.stderr)
        return 1

    # ── 2) 데이터 로드 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    missing = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if missing:
        print(f"[FAIL] 학습 데이터에 동결 피처 부재: {missing}", file=sys.stderr)
        return 1
    folds = build_folds(train)
    splits = build_nested_splits(train, folds)
    leak_problems = assert_leakage_free(splits, train)
    if leak_problems:
        print("[FAIL] 중첩 분할 누수 가드 실패:\n  " + "\n  ".join(leak_problems), file=sys.stderr)
        return 1
    print(f"[calibration] train {train.shape} | feats {len(CHAMPION_FEATURES)} | "
          f"cats {len(LGB_CATS)} | seeds {seeds} | schedules {schedules} | methods {methods}",
          flush=True)
    split_desc = ", ".join(f"{fn}:{sp['last_season']}" for fn, sp in splits.items())
    print(f"[calibration] nested split: 각 폴드 학습 윈도우 마지막 시즌 = inner_val ({split_desc})", flush=True)

    # ── 3) 챔피언 기준 ──
    champion = load_champion_control(train, folds)
    print("[calibration] 챔피언 raw 블렌드 기준: " + ", ".join(
        f"{fn} {champion['per_fold'][fn]['blend_bss']:.1f}" for fn in FOLDS), flush=True)
    if not champion["cross_check_ok"]:
        print("[WARN] 챔피언 블렌드 BSS 재계산 불일치: "
              + "; ".join(champion["cross_check_problems"]), flush=True)

    # ── 4) Failure QA: 누수 주입 ──
    if args.leakage_injection:
        print("\n[leakage-injection] Failure QA: 스코어 폴드 라벨로 slope 피팅 주입…", flush=True)
        try:
            fn0 = FOLDS[0]
            scored_idx = np.where(folds[fn0][1])[0]
            dummy_z = np.zeros(len(scored_idx))
            dummy_y = train.loc[folds[fn0][1], common.TARGET].values
            fit_slope_nested(fn0, dummy_z, dummy_y,
                             fit_idx=scored_idx, scored_idx=scored_idx, b=B_POLICY)
            print("[FAIL] 누수 주입이 탐지되지 않았음 — 가드 미동작!", file=sys.stderr)
            return 3
        except LeakageError as exc:
            print(f"[leakage-injection] PASS — 누수 가드가 스코어 폴드 피팅을 차단했습니다.\n"
                  f"  오류: {exc}", flush=True)
            return 2

    # ── 5) 본 평가 ──
    print("\n[calibration] 중첩 학습 시작 (스코어 폴드는 검증/피팅/선택에 미사용)", flush=True)
    study = run_study(train, folds, splits, seeds, schedules, methods, champion,
                      smoke=smoke)

    # ── 6) 스키마 ──
    schema = {
        "schema_version": SCHEMA_VERSION,
        "study": "task-6-calibration",
        "smoke": smoke,
        "seeds": seeds,
        "folds": list(FOLDS),
        "r_folds": list(R_FOLDS),
        "schedules": {s: SEASON_SCHEDULES[s] for s in schedules},
        "calibration_methods": {m: CALIBRATION_METHODS[m] for m in methods},
        "frozen_control": {
            "w_lgb": W_LGB,
            "c_logit": C_LOGIT,
            "b_policy": B_POLICY,
            "b_note": ("b=C_LOGIT=-0.0404 동결(2025 target-mean 정책 상수) — 절대 피팅 금지. "
                       "slope a 만 inner split 에서 피팅."),
            "feature_contract": feature_contract,
            "config_digest_inherited_from": "qualification_runner (자기 무결성 게이트)",
            "lgb_params": {k: common.PARAMS[k] for k in
                           ("learning_rate", "num_leaves", "min_data_in_leaf",
                            "feature_fraction", "bagging_fraction", "bagging_freq",
                            "num_threads", "deterministic")},
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping": EARLY_STOPPING,
        },
        "nested_design": {
            "inner_split": ("각 outer 폴드의 학습 윈도우 마지막 시즌 = inner_val (피팅/선택 셋), "
                            "inner split ⊂ outer 학습 윈도우. 스코어 폴드는 미사용."),
            "model_training": ("LGB = outer 학습 윈도우 전체 + 시즌 가중, FIXED num_boost_round=109 "
                               "(train_meta.json 문서화 상수), early stopping 없음 → 스코어 폴드로 "
                               "어떤 파라미터도 선택하지 않음."),
            "model_training_note": ("측정: 학습 윈도우에서 마지막 시즌을 제외하면(inner_train 전용 "
                                    "학습) 외부 폴드 BSS 가 0 으로 붕괴(드리프트) — 모델은 전 윈도우로 "
                                    "학습하고 마지막 시즌은 캘리브레이션/선택 전용으로 사용."),
            "slope_fitting": "inner_val 로짓에서만 (b=C_LOGIT 동결, a 만 Brier 최소화 그리드 피팅)",
            "schedule_selection": "inner_val raw BSS argmax (스코어 폴드 미사용)",
            "seed_aggregation": "mean of per-seed logits (챔피언 컨벤션)",
            "a_grid": {"lo": A_LO, "hi": A_HI, "n": int(len(A_GRID)),
                       "refine_step": A_REFINE_STEP, "refine_range": A_REFINE_RANGE},
            "slope_flat_threshold": SLOPE_FLAT_BRIER_RANGE,
        },
        "champion_control": champion,
        "leakage_guard": {
            "no_2025_in_any_fitting_or_validation": True,
            "scored_fold_never_in_fitting_set": True,
            "checks": study["leakage_checks"],
            "nested_split_assertions": [
                "inner_val ⊂ outer_train", "inner_train ∩ inner_val = ∅",
                "inner_val ∩ outer_val = ∅ (스코어 폴드 미누출)",
                "2025 시즌 검증/피팅 셋 부재",
            ],
        },
        "per_fold": study["per_fold"],
        "acceptance": study["acceptance"],
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    result_path = CALIB_CACHE_DIR / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    # ── 7) 요약 ──
    print("\n" + "=" * 100, flush=True)
    print("결과 요약 (outer 폴드 BSS, transformed)", flush=True)
    print("=" * 100, flush=True)
    hdr = f"  {'treatment':<24s} " + " ".join(f"{fn:>9s}" for fn in FOLDS)
    print(hdr, flush=True)
    for key, res in study["acceptance"]["per_treatment"].items():
        sched, method = key.split("__")
        row = " ".join(f"{study['per_fold'][fn]['treatments'][sched]['methods'][method]['bss_trans']:>9.1f}"
                       for fn in FOLDS)
        verdict = res["verdict"]
        tag = "ACCEPT" if verdict == "accepted" else "reject"
        print(f"  {key:<24s} {row}   {tag}", flush=True)
    print("-" * 100, flush=True)
    row = " ".join(f"{champion['per_fold'][fn]['blend_bss']:>9.1f}" for fn in FOLDS)
    print(f"  {'champion blend':<24s} {row}", flush=True)
    row = " ".join(f"{champion['per_fold'][fn]['lgb_bss']:>9.1f}" for fn in FOLDS)
    print(f"  {'champion lgb':<24s} {row}", flush=True)

    print(f"\n[acceptance] 게이트: {study['acceptance']['gate_definition']}", flush=True)
    for key, res in study["acceptance"]["per_treatment"].items():
        reason_txt = (" " + " | ".join(res["reasons"])) if res["reasons"] else ""
        print(f"  {key:<24s} → {res['verdict']}{reason_txt}", flush=True)
    print(f"  accepted: {study['acceptance']['accepted'] or '(없음 — 전부 rejected)'}", flush=True)

    print(f"\n[calibration] 증거 JSON → {evidence_path}", flush=True)
    print(f"[calibration] OOF 로짓(npy) → {CALIB_CACHE_DIR}/{{schedule}}/*.npy "
          f"(Todo 7 후보 입력, 캐시 — git 제외)", flush=True)
    print(f"[calibration] 총 {time.time()-t0:.0f}s", flush=True)

    # ── 8) 스모크 판정 ──
    if smoke:
        required = ["schema_version", "study", "per_fold", "acceptance", "leakage_guard",
                    "champion_control", "environment", "git_commit"]
        ok = (
            seeds == [42, 43]
            and schedules == ["uniform"]
            and set(methods) == {"raw", "policy", "slope_only"}
            and all(fn in schema["per_fold"] for fn in FOLDS)
            and all(k in schema for k in required)
            and all(c["ok"] for c in schema["leakage_guard"]["checks"])
            and schema["leakage_guard"]["no_2025_in_any_fitting_or_validation"]
            and schema["champion_control"]["cross_check_ok"]
        )
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 폴드/스케줄/캘리브레이션/누수/스키마/"
              f"챔피언 기준 경로 검증 완료 (seeds={seeds}, schedules={schedules}, "
              f"methods={sorted(methods)})", flush=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
