#!/usr/bin/env python3
"""model_family_runner.py — 독립 모델 패밀리 자격(qualification) 러너 (Todo 5).

목표(aimers9-top100-score-improvement Todo 5): 챔피언 LGB/MLP 블렌드와 독립적인
모델 패밀리 후보를 빌드/자격 평가한다. XGBoost(GPU-hist 우선, CUDA 부재 시 CPU-hist
폴백 + 명시적 가속기 판정)와 범주형-네이티브 대안 최대 1종(catboost)을 등록하되,
catboost 는 오프라인 패키지 시뮬레이션(현재 인터프리터에서 import + 버전 확인)이
평가 환경과 호환됨을 증명한 경우에만 학습한다.

계약(모두 Task 2 qualification_runner.py 의 동결 컨트롤을 그대로 재사용):
  - 동결 피처 49 (CHAMPION_FEATURES, train_meta.json 에서 READ) — 신규 피처 없음,
    금지 피처(asof_n_bucket, score_diff_binary) 미사용.
  - 폴드 마스크 = qualification_runner.build_folds == screen_all_10seed.py:69-76
    (primary/r2022/r2023/r2024). 2025 라벨/테스트 행 미사용 (누수 가드).
  - 시드 10종(42..51), 폴드 로짓은 시드별 로짓 평균 (Task 2 seed_aggregation 동일).
  - BSS = common.score(common.sigmoid(z), y) — raw 시그모이드, C_LOGIT 무적용.
  - OOF 로짓 저장: repro_979/cache/qualification/<family>/<fold>.npy (gitignore).
  - 챔피언 폴드 로짓(cache/qualification/champion/*.npy, Task 2 산출물) 대비
    correlation / residual complementarity / blend probe / calibration / runtime 보고.

가속기/패키지 게이트(필수 실패 경로 QA):
  - --require-gpu: CUDA 미가용이면 [FAIL] accelerator gate, 비정상 종료(nonzero).
    기본 모드는 CUDA 미가용 시 CPU-hist 폴백으로 OOF 를 생성하되
    verdict.accelerator = cpu-fallback 을 기록. 평가 환경은 L4 GPU(CUDA 12.8)라
    GPU-hist 가 배포 경로 — 로컬에서는 코드 경로 + CPU 수치를 검증한다.
  - --check-env: 각 패밀리 import 상태 + 버전 보고. --simulate-missing-dep:
    패키지 게이트 실패 주입(비정상 종료) — 의존성 미import 시 게이트가 실패함을 증명.
  - --smoke: 시드 [42,43] + 패밀리 1종으로 폴드/스키마/판정 전체 경로 검증 후 PASS.
    기본 동작에 영향 없음.

패밀리 승격 게이트:
  PROMOTED iff (R-only 전이: R-only 3폴드 모두 챔피언 블렌드 참조 대비 개선)
           or  (패키징 입증 + 상보성: 패키지 import/버전 통과 & corr < 0.96 &
                R-only 중 blend probe 이득 > 0)
  그 외 rejected + 사유. 낮은 correlation 단독은 품질로 인정하지 않는다.

출처: qualification_runner.py(Task 2), common.py, screen_all_10seed.py, e6c_blend_folds.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
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

# ════════════════════════════════════════════════════════════════════
# Task 2 동결 컨트롤 재사용 (import 시 자기 무결성 게이트 1회 실행 —
# 동결 상수 조작 시 SystemExit 로 fail-closed).
# ════════════════════════════════════════════════════════════════════
import qualification_runner as qr  # noqa: E402
from qualification_runner import (  # noqa: E402,F401
    SEEDS, FOLDS, R_FOLDS, CHAMPION_FEATURES, LGB_CATS, build_folds,
    _check_leakage, _check_feature_contract, _sha256, _git_commit,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

SCHEMA_VERSION = 2  # Task 2(v1) 스키마 확장: family / accelerator / package / acceptance 블록

CHAMPION_REFERENCE_JSON = REPO / "cache" / "qualification" / "champion" / "result.json"
CHAMPION_DIR = REPO / "cache" / "qualification" / "champion"

# R-only 전이 판정 허용오차(BSS) — 캐시 재현 등 미세 변동 대비.
R_TRANSFER_EPS = 0.05
# 상보성 임계: 챔피언 블렌드 대비 평균 로짓 corr 가 이보다 낮아야 함.
COMPLEMENTARITY_CORR_MAX = 0.96
# blend probe 가중치 격자 (챔피언 가중치 w, 진단 전용 — in-sample, 승격 기준 아님).
BLEND_W_GRID = tuple(round(0.05 * i, 2) for i in range(21))  # 0.00 .. 1.00

# ════════════════════════════════════════════════════════════════════
# 패밀리 레지스트리
# ════════════════════════════════════════════════════════════════════
FAMILIES: dict[str, dict] = {
    "xgboost": {
        "id": "xgboost",
        "kind": "gradient-boosted-trees-hist",
        "description": ("XGBoost hist GBDT — GPU-hist(CUDA 가용 시, 배포 경로 L4) / "
                        "CPU-hist 폴백(로컬 CUDA 미가용 → accelerator verdict 기록)"),
        "package": "xgboost",
        "cat_features": list(LGB_CATS),
        "categorical_native": False,
        "deployment": {
            "rating": "viable",
            "note": ("pip 휠 — 챔피언 lightgbm 과 동일한 requirements.txt 설치 경로, "
                     "패키지 크기 보통(수십 MB). offline 설치 시뮬레이션(import+버전) 통과."),
        },
        "params_base": dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_depth=7,           # LGB num_leaves=63 ≈ max_depth 6-7
            eta=0.05,
            min_child_weight=100,  # ≈ min_data_in_leaf 500 (이진 로그로스 헤시안 0.25×)
            subsample=0.8,
            colsample_bytree=0.8,
            nthread=32,
            deterministic_histogram=True,
        ),
        "rounds": 4000,
        "early_stopping_rounds": 100,
    },
    "catboost": {
        "id": "catboost",
        "kind": "categorical-native-gbdt",
        "description": ("CatBoost 범주형-네이티브 GBDT — 순수 범주형 처리(one_hot+target stats) "
                        "로 LGB/MLP 와 독립적인 패밀리. 오프라인 패키지 시뮬레이션 통과 시에만 학습"),
        "package": "catboost",
        "cat_features": list(LGB_CATS),
        "categorical_native": True,
        "deployment": {
            "rating": "install_risk",
            "note": ("평가환경 기본 설치 패키지 목록(docs 166-184행)에 없음. requirements.txt 추가 시 "
                     "10분 설치 한도 내 미검증이며 휠이 큼(~60-90MB). offline import/버전 시뮬레이션은 통과 "
                     "(로컬 1.2.10). 설치 검증 전까지 배포 등급은 install_risk 로 기록."),
        },
        "params_base": dict(
            loss_function="Logloss",
            eval_metric="Logloss",
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            random_strength=1.0,
            thread_count=32,
            allow_writing_files=False,
            verbose=False,
        ),
        "rounds": 4000,
        "early_stopping_rounds": 100,
    },
}
FAMILY_IDS = tuple(FAMILIES)


def _package_check(package: str) -> dict:
    """오프라인 패키지 시뮬레이션: 현재 인터프리터에서 import + 버전 확인."""
    entry = {"package": package}
    try:
        mod = importlib.import_module(package)
    except Exception as exc:  # noqa: BLE001 — 시뮬레이션은 모든 import 실패를 잡아야 함
        entry.update(importable=False, version=None,
                     error=f"{type(exc).__name__}: {exc}", status="missing")
        return entry
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    entry.update(importable=True, version=str(version), status="available")
    return entry


def _cuda_probe() -> dict:
    """CUDA 가용성 probe — torch 신호 + XGBoost 자체 CUDA 런타임 probe.

    XGBoost 는 CUDA 런타임 미가용 시 조용히 CPU 로 폴백하므로(common.assert_gpu
    패턴), device=cuda 1라운드 학습 후 save_config 의 device 가 cuda 인지 확인한다.
    """
    import torch  # noqa: PLC0415
    torch_cuda = bool(torch.cuda.is_available())
    xgb_probe = {"ok": False, "detail": "not probed"}
    try:
        import xgboost as xgb  # noqa: PLC0415
        X = np.random.RandomState(0).rand(1000, 10).astype(np.float32)
        y = (np.random.RandomState(0).rand(1000) > 0.5).astype(int)
        bst = xgb.train({"device": "cuda", "tree_method": "hist", "nthread": 4},
                        xgb.DMatrix(X, y), num_boost_round=1)
        cfg = bst.save_config().replace(" ", "")
        xgb_probe = {
            "ok": '"device":"cuda"' in cfg,
            "detail": ("xgboost CUDA probe config device=cuda 일치"
                       if '"device":"cuda"' in cfg
                       else f"xgboost 가 CUDA 미가용으로 CPU 폴백 (config: {cfg[:120]}...)"),
        }
    except Exception as exc:  # noqa: BLE001
        xgb_probe = {"ok": False, "detail": f"xgboost CUDA probe 예외: {type(exc).__name__}: {exc}"}
    available = bool(torch_cuda and xgb_probe["ok"])
    return {
        "available": available,
        "torch_cuda_available": torch_cuda,
        "torch": torch.__version__,
        "xgboost_probe": xgb_probe,
        "verdict": ("gpu-available" if available
                    else "cpu-fallback (로컬 CUDA 미가용 — 평가 환경 L4 GPU 는 GPU-hist 배포 경로)"),
    }


def _load_champion_reference() -> tuple[dict, list[str]]:
    """챔피언 참조: cache/qualification/champion/result.json + 폴드 로짓 무결성 대조."""
    problems: list[str] = []
    if not CHAMPION_REFERENCE_JSON.is_file():
        return {}, ["champion 참조 없음: task-2 러너를 먼저 실행 (cache/qualification/champion/result.json)"]
    ref = json.loads(CHAMPION_REFERENCE_JSON.read_text(encoding="utf-8"))
    for fn in FOLDS:
        entry = ref.get("per_fold", {}).get(fn)
        if entry is None:
            problems.append(f"champion 참조에 fold {fn} 없음")
            continue
        npy = CHAMPION_DIR / f"{fn}.npy"
        if not npy.is_file():
            problems.append(f"champion 로짓 없음: {npy}")
            continue
        if _sha256(npy) != entry.get("digest"):
            problems.append(f"champion 로짓 다이제스트 불일치 (변조 의심): {fn}.npy")
    return ref, problems


def _calibration_report(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """신뢰도(reliability) 진단 — 재보정 없이 통계만. 분위수 기반 10구간."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    bounds = np.floor(np.linspace(0, len(ps), n_bins + 1)).astype(int)
    bins, empty = [], 0
    for b in range(n_bins):
        lo_i, hi_i = int(bounds[b]), int(bounds[b + 1])
        if hi_i <= lo_i:
            empty += 1
            continue
        seg_p = ps[lo_i:hi_i]
        seg_y = ys[lo_i:hi_i]
        bins.append({
            "bin": b,
            "range": [round(float(seg_p[0]), 4), round(float(seg_p[-1]), 4)],
            "pred_mean": float(seg_p.mean()),
            "actual_rate": float(seg_y.mean()),
            "n": int(len(seg_p)),
            "deviation": float(seg_y.mean() - seg_p.mean()),
        })
    if len(bins) >= 2:
        slope, intercept = np.polyfit(
            [b["pred_mean"] for b in bins], [b["actual_rate"] for b in bins], 1)
    else:
        slope, intercept = float("nan"), float("nan")
    devs = [b["deviation"] for b in bins]
    overall_rate = float(y.mean())
    return {
        "method": "quantile-bins-10",
        "n_bins_used": len(bins),
        "n_bins_empty": empty,
        "bins": bins,
        "max_abs_cal_dev": float(max(abs(d) for d in devs)) if devs else float("nan"),
        "mean_abs_cal_dev": float(np.mean(np.abs(devs))) if devs else float("nan"),
        "reliability_slope": float(slope),
        "reliability_intercept": float(intercept),
        "pred_mean_overall": float(p.mean()),
        "actual_rate_overall": overall_rate,
        "mean_alignment_dev": float(p.mean() - overall_rate),
    }


def _blend_probe(z_family: np.ndarray, z_champion: np.ndarray, yv: np.ndarray) -> dict:
    """챔피언 블렌드 로짓과 패밀리 로짓의 in-sample 블렌드 이득 프로브 (진단 전용).
    승격 기준 아님 — Todo 7 홀드아웃 블렌드 선택의 상한 참조."""
    best_w, best_bss, gains = None, -1e9, []
    for w in BLEND_W_GRID:
        zb = w * z_champion + (1.0 - w) * z_family
        bss = float(common.score(common.sigmoid(zb), yv))
        gains.append(bss)
        if bss > best_bss:
            best_bss, best_w = bss, w
    champ_bss = float(common.score(common.sigmoid(z_champion), yv))
    return {
        "best_w_champion": best_w,
        "best_blend_bss": best_bss,
        "gain_vs_champion": best_bss - champ_bss,
        "champion_alone_bss": champ_bss,
        "note": "in-sample 진단(홀드아웃 아님) — 승격 기준으로 사용하지 않음",
    }


def _train_xgboost_fold(train: pd.DataFrame, tr_m: pd.Series, va_m: pd.Series,
                        seeds: list[int], device: str) -> tuple[np.ndarray, dict]:
    """XGBoost hist — 패밀리 단일 폴드, 시드별 로짓 평균."""
    import xgboost as xgb  # noqa: PLC0415
    feats = list(CHAMPION_FEATURES)
    cats = list(LGB_CATS)
    X_tr = train.loc[tr_m, feats]
    y_tr = train.loc[tr_m, common.TARGET].values
    X_va = train.loc[va_m, feats]
    y_va = train.loc[va_m, common.TARGET].values
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    zs, per_seed = [], {}
    for seed in seeds:
        params = dict(FAMILIES["xgboost"]["params_base"])
        params["seed"] = seed
        params["device"] = device
        bst = xgb.train(params, dtr, num_boost_round=FAMILIES["xgboost"]["rounds"],
                        evals=[(dva, "va")],
                        early_stopping_rounds=FAMILIES["xgboost"]["early_stopping_rounds"],
                        verbose_eval=False)
        p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        z = common.logit(p)
        zs.append(z)
        per_seed[seed] = {"best_iteration": int(bst.best_iteration),
                          "bss": float(common.score(common.sigmoid(z),
                                                    train.loc[va_m, common.TARGET].values))}
    return np.mean(zs, axis=0), per_seed


def _train_catboost_fold(train: pd.DataFrame, tr_m: pd.Series, va_m: pd.Series,
                         seeds: list[int]) -> tuple[np.ndarray, dict]:
    """CatBoost 범주형-네이티브 — 패밀리 단일 폴드, 시드별 로짓 평균."""
    from catboost import CatBoostClassifier, Pool  # noqa: PLC0415
    feats = list(CHAMPION_FEATURES)
    cats = list(LGB_CATS)
    X_tr = train.loc[tr_m, feats]
    y_tr = train.loc[tr_m, common.TARGET].values
    X_va = train.loc[va_m, feats]
    y_va = train.loc[va_m, common.TARGET].values
    tr_pool = Pool(X_tr, y_tr, cat_features=cats)
    va_pool = Pool(X_va, y_va, cat_features=cats)
    zs, per_seed = [], {}
    for seed in seeds:
        params = dict(FAMILIES["catboost"]["params_base"])
        params["random_seed"] = seed
        params["iterations"] = FAMILIES["catboost"]["rounds"]
        params["od_type"] = "Iter"
        params["od_wait"] = FAMILIES["catboost"]["early_stopping_rounds"]
        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        p = np.asarray(model.predict(va_pool, prediction_type="Probability"), dtype=np.float64)
        if p.ndim == 2:
            p = p[:, 1]  # class-1 확률 열
        z = common.logit(p)
        zs.append(z)
        per_seed[seed] = {
            "best_iteration": int(model.get_best_iteration() or -1),
            "bss": float(common.score(common.sigmoid(z), y_va)),
        }
    return np.mean(zs, axis=0), per_seed


def _evaluate_fold(fn: str, z: np.ndarray, yv: np.ndarray, z_champ: np.ndarray,
                   out_dir: Path, runtime_s: float, per_seed: dict) -> dict:
    """폴드 평가: 저장 + BSS/평균/다이제스트 + 챔피언 대비 상관/잔차/캘리브레이션."""
    if len(z) != len(yv):
        raise RuntimeError(f"{fn}: 패밀리 로짓 길이 {len(z)} != 검증 행 수 {len(yv)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fn}.npy"
    np.save(out_path, np.asarray(z, dtype=np.float64))
    p = common.sigmoid(z)
    p_champ = common.sigmoid(z_champ)
    resid_family = yv - p
    resid_champ = yv - p_champ
    return {
        "n_rows": int(len(yv)),
        "bss": float(common.score(p, yv)),
        "pred_mean": float(p.mean()),
        "digest": _sha256(out_path),
        "logits_path": str(out_path.relative_to(REPO)),
        "runtime_s": float(runtime_s),
        "per_seed": per_seed,
        "vs_champion": {
            "corr_logits": float(np.corrcoef(z, z_champ)[0, 1]),
            "corr_probs": float(np.corrcoef(p, p_champ)[0, 1]),
            # 잔차 상관: 두 모델의 오차(prob 공간) 상관. 라벨 분산이 지배해 1.0 근처로
            # 수렴하는 축소성 있음 — 아래 residual_alignment 가 더 판별적인 상보성 지표.
            "residual_corr_errors": float(np.corrcoef(resid_family, resid_champ)[0, 1]),
            # 상보성 정렬: 패밀리 예측이 챔피언 예측에서 벗어난 방향이 챔피언 잔차와
            # 정렬되는가(양수 = 챔피언이 놓친 곳을 패밀리가 보정하는 방향 = 상보적).
            "residual_alignment": float(np.corrcoef(p - p_champ, resid_champ)[0, 1]),
            "champion_bss": float(common.score(p_champ, yv)),
            "delta_bss_vs_champion": float(common.score(p, yv) - common.score(p_champ, yv)),
        },
        "blend_probe": _blend_probe(z, z_champ, yv),
        "calibration": _calibration_report(p, yv),
    }


def _acceptance_gate(family_id: str, per_fold: dict, champ_ref: dict,
                     package: dict, accelerator: dict) -> dict:
    """패밀리 승격 게이트.

    PROMOTED iff R-only 전이(3폴드 모두 참조 개선) or 패키징 입증 + 상보성.
    낮은 correlation 단독은 품질로 인정하지 않음.
    """
    fam = FAMILIES[family_id]
    ref_bss = {fn: champ_ref["per_fold"][fn]["bss"] for fn in FOLDS}
    fam_bss = {fn: per_fold[fn]["bss"] for fn in FOLDS}
    r_only_delta = {fn: fam_bss[fn] - ref_bss[fn] for fn in R_FOLDS}
    r_only_transfer = all(d >= -R_TRANSFER_EPS for d in r_only_delta.values())

    mean_corr = float(np.mean([per_fold[fn]["vs_champion"]["corr_logits"] for fn in FOLDS]))
    any_r_blend_gain = any(per_fold[fn]["blend_probe"]["gain_vs_champion"] > 0 for fn in R_FOLDS)
    packaging_ok = bool(package.get("importable"))
    deployment_ok = fam["deployment"]["rating"] == "viable"
    complementarity = bool(mean_corr < COMPLEMENTARITY_CORR_MAX and any_r_blend_gain)

    reasons: list[str] = []
    if r_only_transfer:
        verdict = "promoted"
        reasons.append("R-only 전이: R-only 3폴드 모두 챔피언 블렌드 참조 대비 개선")
    elif packaging_ok and deployment_ok and complementarity:
        verdict = "promoted"
        reasons.append("패키징 입증(import+버전, 배포 viable) + 상보성(평균 corr "
                       f"{mean_corr:.3f} < {COMPLEMENTARITY_CORR_MAX} & R-only blend probe 이득 > 0)")
    else:
        verdict = "rejected"
        if not r_only_transfer:
            reasons.append("R-only 전이 미달 (모든 R-only 폴드에서 챔피언 블렌드 대비 개선 없음)")
        if not packaging_ok:
            reasons.append(f"패키지 미import ({package.get('error')})")
        if not deployment_ok:
            reasons.append(f"배포 등급 {fam['deployment']['rating']} — {fam['deployment']['note']}")
        if not complementarity:
            reasons.append(f"상보성 미충족 (평균 corr {mean_corr:.3f} ≥ {COMPLEMENTARITY_CORR_MAX} "
                           f"또는 R-only blend probe 이득 ≤ 0) — corr 단독은 품질 아님")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "r_only_delta_bss_vs_champion": {fn: round(r_only_delta[fn], 3) for fn in R_FOLDS},
        "mean_corr_champion": round(mean_corr, 4),
        "complementarity_condition": {
            "packaging_ok": packaging_ok,
            "deployment_ok": deployment_ok,
            "corr_below_max": bool(mean_corr < COMPLEMENTARITY_CORR_MAX),
            "any_r_blend_probe_gain": any_r_blend_gain,
        },
        "transfer_condition": {"r_only_improved_all": r_only_transfer, "eps_bss": R_TRANSFER_EPS},
        "gate_definition": ("PROMOTED iff R-only 전이(all 3 R-only folds 개선) or "
                            "패키징 입증 + 상보성(corr<0.96 & R-only blend probe gain>0)"),
    }


def _environment() -> dict:
    env = dict(qr._environment())
    for fam in FAMILY_IDS:
        env[FAMILIES[fam]["package"]] = _package_check(FAMILIES[fam]["package"])["version"]
    return env


def _check_env_all(require_gpu: bool, simulate_missing: bool) -> tuple[bool, dict]:
    """--check-env: 패밀리 import/버전 + 가속기 probe. 게이트 실패 시 False."""
    report = {"schema_version": SCHEMA_VERSION,
              "check_env_mode": True,
              "cuda": _cuda_probe(),
              "packages": {fam: _package_check(FAMILIES[fam]["package"]) for fam in FAMILY_IDS},
              "environment": _environment(),
              "git_commit": _git_commit()}
    ok = True
    print(f"[check-env] CUDA: available={report['cuda']['available']} "
          f"(torch={report['cuda']['torch_cuda_available']}, "
          f"xgb_probe={report['cuda']['xgboost_probe']['ok']})", flush=True)
    print(f"[check-env] verdict: {report['cuda']['verdict']}", flush=True)
    for fam in FAMILY_IDS:
        pk = report["packages"][fam]
        tag = "available" if pk["importable"] else "MISSING"
        print(f"[check-env] {fam:<9s} {pk['package']}=={pk.get('version')} → {tag}", flush=True)
        if not pk["importable"]:
            ok = False
    if simulate_missing:
        print("[check-env] [FAIL] --simulate-missing-dep: 패키지 게이트 실패 주입 "
              "(의존성 미import 시 게이트가 비정상 종료함을 증명)", flush=True)
        ok = False
    if require_gpu and not report["cuda"]["available"]:
        print("[FAIL] accelerator gate — --require-gpu 요청이지만 CUDA 미가용 "
              "(로컬 검증 환경은 CPU-only; 평가 환경은 L4 GPU).", file=sys.stderr, flush=True)
        ok = False
    return ok, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="독립 모델 패밀리 자격 러너 (XGBoost GPU/CPU + 범주형 네이티브 대안)")
    parser.add_argument("--family", default=None,
                        help="패밀리 id (기본: 모든 등록 패밀리) — 등록: " + ", ".join(FAMILY_IDS))
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 시드 [42,43] + 패밀리 1종으로 전체 경로 검증 후 PASS")
    parser.add_argument("--check-env", action="store_true",
                        help="오프라인 패키지/가속기 시뮬레이션만 수행 (학습 없음)")
    parser.add_argument("--require-gpu", action="store_true",
                        help="가속기 게이트 강제: CUDA 미가용이면 [FAIL] + 비정상 종료")
    parser.add_argument("--simulate-missing-dep", action="store_true",
                        help="패키지 게이트 실패 주입 (의존성 미import 시나리오, QA 전용)")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 .omo/evidence/aimers9-top100/task-5-models.json)")
    parser.add_argument("--seeds", default=None, help="컴마 구분 시드 (기본 42..51; --smoke 는 42,43)")
    args = parser.parse_args(argv)

    t0 = time.time()
    smoke = bool(args.smoke)
    require_gpu = bool(args.require_gpu)
    simulate = bool(args.simulate_missing_dep)

    evidence_path = (
        Path(args.evidence).expanduser().resolve() if args.evidence
        else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-5-models.json"
    )
    if smoke:  # 스모크가 전체 실행 증거를 덮어쓰지 않도록 별도 경로 사용
        evidence_path = evidence_path.with_name(
            f"{evidence_path.stem}-smoke{evidence_path.suffix}")

    # ── 0) --check-env 단독 모드 ──
    if args.check_env:
        ok, report = _check_env_all(require_gpu, simulate)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"[check-env] 증거 JSON → {evidence_path}", flush=True)
        print(f"[check-env] {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1

    # ── 1) 패밀리 선택 + 등록 검증 ──
    if smoke and args.family is None:
        args.family = "xgboost"  # 스모크 기본 패밀리 1종
    if args.family is None:
        selected = list(FAMILY_IDS)
    else:
        selected = [args.family]
    unknown = [f for f in selected if f not in FAMILIES]
    if unknown:
        print(f"[FAIL] 미등록 패밀리 id: {unknown} — 등록됨: {FAMILY_IDS}", file=sys.stderr)
        return 1

    # ── 2) 전역 가속기/패키지 게이트 (선택 패밀리 전부 시뮬레이션) ──
    env_ok, env_report = _check_env_all(require_gpu, simulate)
    if require_gpu and not env_report["cuda"]["available"]:
        return 1  # [FAIL] accelerator gate — 이미 위에서 출력
    if simulate:
        return 1  # [FAIL] 패키지 게이트 주입 실패 — 학습 진행 안 함

    # ── 3) 데이터 로드 + 동결 피처 계약 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feat_ok, feat_detail = _check_feature_contract("family-runner")
    missing_feats = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if not feat_ok or missing_feats:
        print(f"[FAIL] 동결 피처 계약 불일치: {feat_detail} / 부재: {missing_feats}",
              file=sys.stderr)
        return 1
    folds = build_folds(train)
    leakage_problems = _check_leakage(folds, train)
    if leakage_problems:
        print("[FAIL] 누수 가드:\n  " + "\n  ".join(leakage_problems), file=sys.stderr)
        return 1

    # ── 4) 챔피언 참조 로드 (task-2 산출물; 다이제스트 무결성 대조) ──
    champ_ref, champ_problems = _load_champion_reference()
    if champ_problems:
        print("[FAIL] 챔피언 참조 게이트:\n  " + "\n  ".join(champ_problems), file=sys.stderr)
        return 1
    z_champ_all = {fn: np.load(CHAMPION_DIR / f"{fn}.npy") for fn in FOLDS}

    # ── 5) 패밀리별 학습/평가 ──
    seeds = list(SEEDS[:2]) if smoke else (list(SEEDS) if args.seeds is None
                                           else [int(s) for s in args.seeds.split(",")])
    accelerator = env_report["cuda"]
    device = "cuda" if accelerator["available"] else "cpu"
    families_out: dict[str, dict] = {}
    n_va = {fn: int(folds[fn][1].sum()) for fn in FOLDS}

    for family_id in selected:
        fam = FAMILIES[family_id]
        pk = _package_check(fam["package"])
        print(f"\n[{family_id}] package={pk['package']}=={pk.get('version')} "
              f"importable={pk['importable']} | seeds={seeds} | device={device} "
              f"| accelerator={accelerator['verdict']}", flush=True)
        if not pk["importable"]:
            entry = {
                "schema_version": SCHEMA_VERSION,
                "task": "aimers9-top100/task-5-models",
                "family": family_id,
                "mode": "rejected",
                "package": pk,
                "accelerator": accelerator,
                "acceptance": {"verdict": "rejected",
                               "reasons": [f"패키지 미import: {pk.get('error')} — 학습 생략"],
                               "gate_definition": FAMILIES[family_id]["deployment"]},
                "environment": env_report["environment"],
                "git_commit": env_report["git_commit"],
            }
            families_out[family_id] = entry
            print(f"[{family_id}] [REJECTED] 패키지 게이트 실패 — 학습하지 않음", flush=True)
            continue

        out_dir = REPO / "cache" / "qualification" / family_id
        per_fold: dict[str, dict] = {}
        for fn in FOLDS:
            tr_m, va_m = folds[fn]
            yv = train.loc[va_m, common.TARGET].values
            if int(len(yv)) != n_va[fn]:
                raise RuntimeError(f"{fn}: 검증 행 수 불일치 {len(yv)} != {n_va[fn]}")
            t_fold = time.time()
            if family_id == "xgboost":
                z, per_seed = _train_xgboost_fold(train, tr_m, va_m, seeds, device)
            elif family_id == "catboost":
                z, per_seed = _train_catboost_fold(train, tr_m, va_m, seeds)
            else:  # pragma: no cover — 등록 검증에서 이미 차단
                raise RuntimeError(f"미구현 패밀리: {family_id}")
            runtime_s = time.time() - t_fold
            per_fold[fn] = _evaluate_fold(fn, z, yv, z_champ_all[fn], out_dir,
                                          runtime_s, per_seed)
            row = per_fold[fn]
            print(f"  [{fn:<8s}] bss={row['bss']:>8.1f} (champ {row['vs_champion']['champion_bss']:>7.1f}, "
                  f"Δ{row['vs_champion']['delta_bss_vs_champion']:>+7.1f}) corr={row['vs_champion']['corr_logits']:.4f} "
                  f"resid={row['vs_champion']['residual_corr_errors']:+.4f} mean={row['pred_mean']:.4f} "
                  f"n={row['n_rows']} t={runtime_s:.0f}s", flush=True)

        acceptance = _acceptance_gate(family_id, per_fold, champ_ref, pk, accelerator)
        print(f"[{family_id}] 판정: {acceptance['verdict']} — " + "; ".join(acceptance["reasons"]),
              flush=True)

        families_out[family_id] = {
            "schema_version": SCHEMA_VERSION,
            "task": "aimers9-top100/task-5-models",
            "family": family_id,
            "family_description": fam["description"],
            "kind": fam["kind"],
            "categorical_native": fam["categorical_native"],
            "mode": "smoke" if smoke else "full",
            "smoke": smoke,
            "seeds_requested": list(seeds),
            "folds": list(FOLDS),
            "r_folds": list(R_FOLDS),
            "feature_contract": {
                "n_features": len(CHAMPION_FEATURES),
                "source": "model/train_meta.json (frozen)",
                "forbidden_present": False,
                "cat_features": fam["cat_features"],
                "categorical_handling": ("enable_categorical(DMatrix)" if family_id == "xgboost"
                                         else "Pool(cat_features)"),
            },
            "package": pk,
            "deployment": fam["deployment"],
            "accelerator": accelerator,
            "integrity": {
                "gate": "PASS",
                "feature_contract": {"ok": feat_ok, "detail": feat_detail},
                "leakage_guard": {"no_2025_in_any_validation_mask": True},
                "champion_reference": {"gate": "PASS", "digests_matched": True},
            },
            "per_fold": per_fold,
            "seed_aggregation": {
                "seeds": list(seeds),
                "aggregation": "mean of per-seed logits",
                "note": "시드별 로짓 평균 후 sigmoid → BSS (Task 2 와 동일 규약)",
            },
            "acceptance": acceptance,
            "environment": env_report["environment"],
            "git_commit": env_report["git_commit"],
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_time_s": time.time() - t0,
        }

        # 폴드/판정 직후 패밀리별 증거 즉시 저장 — 중간 실패 시에도 완료 패밀리 보존.
        per_family_path = evidence_path.with_name(
            f"{evidence_path.stem}-{family_id}{evidence_path.suffix}")
        per_family_path.parent.mkdir(parents=True, exist_ok=True)
        per_family_path.write_text(
            json.dumps(families_out[family_id], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"[{family_id}] 패밀리 증거 → {per_family_path}", flush=True)

    # ── 6) 통합 스키마 저장 ──
    schema = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-top100/task-5-models",
        "smoke": smoke,
        "mode": "smoke" if smoke else "full",
        "families": families_out,
        "champion_reference": {
            "source": "cache/qualification/champion/result.json (task-2)",
            "per_fold_bss": {fn: champ_ref["per_fold"][fn]["bss"] for fn in FOLDS},
            "digests_matched": True,
        },
        "cuda_local_available": bool(accelerator["available"]),
        "deployment_note": ("평가 환경은 L4 GPU(CUDA 12.8) — GPU-hist 가 배포 경로. "
                            "로컬(학생 계정)은 CUDA 미가용이라 CPU-hist 폴백 + 가속기 판정 기록으로 "
                            "코드 경로와 CPU 수치를 검증한다."),
        "environment": env_report["environment"],
        "git_commit": env_report["git_commit"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\n[model-family] 통합 스키마 → {evidence_path}", flush=True)

    # ── 7) 스모크 판정 ──
    if smoke:
        required = ["schema_version", "families", "champion_reference", "cuda_local_available",
                    "environment", "git_commit"]
        fam_entry = families_out.get(selected[0])
        ok = (
            len(seeds) == 2
            and len(selected) == 1
            and all(fn in (fam_entry or {}).get("per_fold", {}) for fn in FOLDS)
            and all(k in schema for k in required)
            and (fam_entry or {}).get("acceptance", {}).get("verdict") in ("promoted", "rejected")
        )
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 패밀리 {selected[0]} 폴드/패키지/"
              f"가속기/스키마/판정 경로 검증 완료 (seeds={seeds}, device={device})", flush=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
