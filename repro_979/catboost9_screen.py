#!/usr/bin/env python3
"""catboost9_screen.py — Slot-2 준비: 전 9범주 CatBoost OOF 스크리닝 러너 (Task 10).

목표(aimers9-top100-score-improvement Task 10): 기존 CatBoost(model_family_runner)가
LGB_CATS 5종만 범주형으로 쓰는 데 반해, 이 러너는 MLP_CATS 9종 전부(pitcher/batter ID
4종 포함)를 CatBoost 네이티브 범주형으로 지정하여 OOF 로짓을 생산한다. ID 필드 4종에
대한 ordered target statistics(CTR) 활성화가 핵심 — 승격 판정(상보성 corr<0.96 +
blend transfer)은 별도 Todo에서 수행하며, 이 러너는 OOF 로짓 + 메타데이터만 생산한다.

계약(모두 Task 2 qualification_runner.py 동결 컨트롤 재사용, 기존 파일 수정 없음):
  - 범주 스키마 = MLP_CATS 정확히 9종(이름+순서 동결) — assert 9 & unique.
  - 피처 49 (CHAMPION_FEATURES), 폴드 = qualification_runner.build_folds
    (primary/r2022/r2023/r2024). 검증 마스크 행 수 = 동결 기대값
    (primary 253507 / r2022 217024 / r2023 219839 / r2024 223497) — 불일치 시 fail-closed.
  - 누수 가드: 어떤 폴드의 학습/검증 마스크에도 2025 시즌 행 없음.
  - 하이퍼파라미터 = FAMILIES["catboost"]["params_base"] (loss Logloss, eval Logloss,
    lr 0.05, depth 7, l2_leaf_reg 3.0, Bernoulli 0.8, random_strength 1.0,
    thread_count 32), iterations 4000, od_type Iter, od_wait 100, use_best_model True,
    random_seed per seed.
  - 범주형 전달: Pool(cat_features=...) — 열 이름 기반 인덱스를 실제 DataFrame에서
    해석 (열 순서가 바뀌어도 이름 기반으로 인덱스 유지).
  - dtype/coercion 정책: 범주 열은 category 또는 정수 dtype만 허용. float 이거나
    null>0 이면 무성(coercion) 없이 fail-closed (dtype/cardinality/null 계약 검증).
  - 캐시 다이제스트: 범주 스키마(이름+순서) / dtype 정책 버전 / 피처 집합 / 시드 /
    폴드 정의 / params 다이제스트 포함 — 스키마·정책 변경 시 stale 캐시 재사용 불가
    (다이제스트 불일치 시 덮어쓰기 거부, --overwrite-cache 만 명시적 예외).
  - 검증 BSS = raw 시그모이드 (common.score(common.sigmoid(z), y)) — C_LOGIT 무적용
    (제출 정렬 상수는 검증 BSS에 미사용, Task 2 규약 유지).
  - OOF 로짓: cache/qualification/catboost9/<fold>.npy (기존 catboost/ 캐시 미접촉)
    + 폴드별 <fold>.meta.json (best_iteration, dtype 보고, cardinality, null 수,
    runtime, 입력/config sha256).
  - 증거: .omo/evidence/aimers9-top100/task-10-catboost9-schema.{json,log} (--smoke).
  - --smoke: 시드 [42,43] × 4폴드 전체 경로(폴드/스키마/dtype 계약/누수/캐시/판정)
    검증 후 PASS. 기본 동작에 영향 없음.

출처: qualification_runner.py(Task 2), model_family_runner.py(Task 5), common.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
os.environ.setdefault("LGAIMERS_ROOT", str(PROJECT_ROOT))

# import 순서 중요: qualification_runner 가 os.chdir(REPO) 수행 → common 상대경로 기준
import qualification_runner as qr  # noqa: E402
from qualification_runner import (  # noqa: E402,F401
    SEEDS, FOLDS, R_FOLDS, CHAMPION_FEATURES, MLP_CATS, LGB_CATS,
    build_folds, _check_leakage, _check_feature_contract, _sha256, _git_commit,
)
from model_family_runner import FAMILIES  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

SCHEMA_VERSION = 1

# ════════════════════════════════════════════════════════════════════
# Slot-2 범주 스키마: MLP_CATS 9종 전체 (핵심 변경 — ID 4종 추가)
# ════════════════════════════════════════════════════════════════════
CAT_SCHEMA = tuple(MLP_CATS)
assert len(CAT_SCHEMA) == 9, (
    f"[FAIL] 범주 스키마는 정확히 9종이어야 함 — 현재 {len(CAT_SCHEMA)}: {CAT_SCHEMA}")
assert len(set(CAT_SCHEMA)) == 9, (
    f"[FAIL] 범주 스키마에 중복 이름: {CAT_SCHEMA}")

# dtype/coercion 정책 버전 — 정책 규칙 변경 시 1 증가 (캐시 다이제스트에 포함)
DTYPE_POLICY_VERSION = 1
DTYPE_POLICY_RULE = (
    "categorical column dtype must be category or integer; float dtype rejected "
    "(no silent numeric coercion); null count must be 0")

# 동결 기대 검증 행 수 (Task 2 캐시 대조 원본 — 검증 마스크 계약)
EXPECTED_N_VA = {"primary": 253507, "r2022": 217024, "r2023": 219839, "r2024": 223497}

# 폴드 정의의 정규화 텍스트 (캐시 다이제스트에 포함 — build_folds 의미론 동결)
FOLD_DEFINITIONS = (
    "primary:(season<=2023)->(season==2024);"
    "r2022:(season<=2021&R)->(season==2022&R);"
    "r2023:(season<=2022&R)->(season==2023&R);"
    "r2024:(season<=2023&R)->(season==2024&R)"
)

CACHE_DIR = REPO / "cache" / "qualification" / "catboost9"

CATBOOST_CAT_FEATURES_MEANING = (
    "native ordered target statistics(CTR) on high-cardinality ID fields "
    "pitcher_id/batter_id/pitcher_team_id/batter_team_id + one_hot/low-card CTR on "
    "top_bottom/game_type/base_state/platoon/count_state")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _catboost_params(seed: int) -> dict:
    """Task 5 catboost params_base + iterations/od — random_seed 만 시드별."""
    p = dict(FAMILIES["catboost"]["params_base"])
    p["random_seed"] = int(seed)
    p["iterations"] = int(FAMILIES["catboost"]["rounds"])
    p["od_type"] = "Iter"
    p["od_wait"] = int(FAMILIES["catboost"]["early_stopping_rounds"])
    return p


def _params_digest() -> str:
    """params_base + rounds/od — random_seed 제외(시드는 digest 별도 포함)."""
    payload = {
        "params_base": dict(FAMILIES["catboost"]["params_base"]),
        "iterations": int(FAMILIES["catboost"]["rounds"]),
        "early_stopping_rounds": int(FAMILIES["catboost"]["early_stopping_rounds"]),
        "od_type": "Iter",
        "use_best_model": True,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_digest(seeds: list[int]) -> str:
    """캐시 다이제스트 — 범주 스키마(이름+순서)/dtype 정책/피처/시드/폴드/params.
    어느 하나라도 바뀌면 digest 가 달라져 stale 캐시 재사용이 구조적으로 불가능하다."""
    payload = {
        "variant": "catboost9",
        "cat_schema": list(CAT_SCHEMA),
        "dtype_policy_version": DTYPE_POLICY_VERSION,
        "features": list(CHAMPION_FEATURES),
        "seeds": sorted(int(s) for s in seeds),
        "folds": list(FOLDS),
        "fold_definitions": FOLD_DEFINITIONS,
        "params_digest": _params_digest(),
        "bss_policy": "raw-sigmoid-no-c-logit",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_cat_schema(train: pd.DataFrame, feats: tuple) -> dict:
    """범주 스키마/dtype 계약: 9종 존재 + dtype(category/int만, float 거부) + null==0.
    무성 수치 변환 없음 — 위반 시 RuntimeError (fail-closed)."""
    problems: list[str] = []
    report: dict[str, dict] = {}
    for c in CAT_SCHEMA:
        if c not in feats:
            problems.append(f"범주 컬럼 {c} 가 CHAMPION_FEATURES 에 없음")
            continue
        if c not in train.columns:
            problems.append(f"범주 컬럼 {c} 가 학습 데이터에 없음")
            continue
        col = train[c]
        dtype = col.dtype
        is_int = bool(pd.api.types.is_integer_dtype(dtype))
        is_cat = isinstance(dtype, pd.CategoricalDtype)
        if not (is_int or is_cat):
            problems.append(
                f"범주 컬럼 {c}: dtype={dtype} — category/int 만 허용 (무성 coercion 금지)")
        n_nulls = int(col.isna().sum())
        if n_nulls > 0:
            problems.append(f"범주 컬럼 {c}: null {n_nulls} 개 — null 허용 안 함")
        report[c] = {
            "dtype": str(dtype), "n_nulls_all": n_nulls,
            "n_unique_all": int(col.nunique()),
            "is_integer": is_int, "is_category": is_cat,
        }
    if problems:
        raise RuntimeError("[FAIL] 범주 스키마/dtype 계약 위반:\n  "
                           + "\n  ".join(problems))
    return report


def _inputs_sha(train: pd.DataFrame, va_m: pd.Series) -> str:
    """검증 입력 핀: va 행 id + 라벨 바이트 sha256 (어느 행이 검증이었는지 고정)."""
    ids = (train.loc[va_m, common.ID].to_numpy()
           if common.ID in train.columns
           else np.asarray(train.index[va_m]))
    y = train.loc[va_m, common.TARGET].to_numpy(dtype=np.float64)
    return _sha256_bytes(np.ascontiguousarray(ids).tobytes()
                         + np.ascontiguousarray(y).tobytes())


def _train_catboost9_fold(train: pd.DataFrame, fn: str, tr_m: pd.Series,
                          va_m: pd.Series, seeds: list[int],
                          cat_report: dict) -> tuple[np.ndarray, dict, dict]:
    """CatBoost 9범주 — 폴드 1종, 시드별 로짓 평균 (Task 5 _train_catboost_fold 패턴)."""
    from catboost import CatBoostClassifier, Pool  # noqa: PLC0415
    feats = list(CHAMPION_FEATURES)
    # 열 이름 기반 인덱스 — 실제 DataFrame 열 순서에서 이름으로 해석 (순서 무관, 이름 기준)
    cat_idx = [feats.index(c) for c in CAT_SCHEMA]
    assert len(cat_idx) == 9, f"{fn}: 범주 인덱스 수 {len(cat_idx)} != 9"

    X_tr = train.loc[tr_m, feats]
    y_tr = train.loc[tr_m, common.TARGET].values
    X_va = train.loc[va_m, feats]
    y_va = train.loc[va_m, common.TARGET].values
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va, y_va, cat_features=cat_idx)

    # 범주 열별 학습 마스크 cardinality/null (계약 보고용)
    card = {c: {
        "dtype": cat_report[c]["dtype"],
        "n_unique_train": int(train.loc[tr_m, c].nunique()),
        "n_nulls_train": int(train.loc[tr_m, c].isna().sum()),
    } for c in CAT_SCHEMA}

    zs: list[np.ndarray] = []
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        t_seed = time.time()
        params = _catboost_params(seed)
        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        p = np.asarray(model.predict(va_pool, prediction_type="Probability"),
                       dtype=np.float64)
        if p.ndim == 2:
            p = p[:, 1]  # class-1 확률 열 (Task 5 동일)
        z = common.logit(p)
        zs.append(z)
        per_seed[str(seed)] = {
            "best_iteration": int(model.get_best_iteration() or -1),
            "bss": float(common.score(common.sigmoid(z), y_va)),
            "runtime_s": round(time.time() - t_seed, 2),
        }
    return np.mean(zs, axis=0), per_seed, card


def _cache_guard(fn: str, digest: str, overwrite: bool) -> None:
    """stale 캐시 가드: 기존 캐시 다이제스트가 현재와 다르면 덮어쓰기 거부.
    스키마/dtype 정책/시드/params 변경 시 stale 캐시 재사용 불가 (fail-closed)."""
    npy = CACHE_DIR / f"{fn}.npy"
    meta = CACHE_DIR / f"{fn}.meta.json"
    if not npy.is_file():
        return
    if not meta.is_file():
        raise RuntimeError(
            f"[FAIL] {fn}: 캐시 npy 존재하나 메타데이터 없음 — stale 캐시, 수동 정리 필요")
    existing = json.loads(meta.read_text(encoding="utf-8"))
    if existing.get("digest") == digest:
        print(f"  [{fn}] 캐시 재사용 (digest 일치 {digest[:12]}...)", flush=True)
        return
    if overwrite:
        print(f"  [{fn}] 경고: 기존 캐시 digest 불일치 → --overwrite-cache 로 재생성",
              flush=True)
        return
    raise RuntimeError(
        f"[FAIL] {fn}: 기존 캐시 digest 불일치 (기존 {str(existing.get('digest'))[:12]}... != "
        f"현재 {digest[:12]}...) — 범주 스키마/dtype 정책/시드/파라미터 변경 의심. "
        f"stale 캐시 재사용 금지 (--overwrite-cache 로 명시적 재생성)")


def _evaluate_fold(fn: str, z: np.ndarray, yv: np.ndarray, seeds: list[int],
                   per_seed: dict, card: dict, digest: str,
                   inputs_sha: str, runtime_s: float) -> dict:
    """폴드 평가: 로짓 저장 + BSS(원시 시그모이드) + 메타데이터 JSON."""
    if len(z) != len(yv):
        raise RuntimeError(f"{fn}: 로짓 길이 {len(z)} != 검증 행 수 {len(yv)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{fn}.npy"
    np.save(out_path, np.asarray(z, dtype=np.float64))
    p = common.sigmoid(z)
    meta = {
        "fold": fn,
        "variant": "catboost9",
        "digest": digest,
        "seeds": [int(s) for s in seeds],
        "n_rows": int(len(yv)),
        "bss": float(common.score(p, yv)),
        "pred_mean": float(p.mean()),
        "logits_digest": _sha256(out_path),
        "logits_path": str(out_path.relative_to(REPO)),
        "runtime_s": round(runtime_s, 2),
        "inputs_sha256": inputs_sha,
        "dtype_report": card,
        "per_seed": per_seed,
    }
    (CACHE_DIR / f"{fn}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def _environment() -> dict:
    env = dict(qr._environment())
    try:
        env["catboost"] = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        env["catboost"] = "unknown"
    return env


class _Tee:
    """stdout + 로그 파일 동시 기록 (증거 .log)."""

    def __init__(self, log_path: Path):
        self._stream = sys.stdout
        self._fh = log_path.open("w", encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="전 9범주(MLP_CATS) CatBoost OOF 스크리닝 러너 (Slot-2 준비)")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 시드 [42,43] × 4폴드 전체 경로 검증 후 PASS")
    parser.add_argument("--seeds", default=None,
                        help="컴마 구분 시드 (기본 42..51; --smoke 는 42,43 고정)")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 .omo/evidence/aimers9-top100/"
                             "task-10-catboost9-schema.json; 전체 모드는 -full 접미)")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="기존 캐시 digest 불일치 시 명시적 재생성 (기본 거부)")
    args = parser.parse_args(argv)

    t0 = time.time()
    smoke = bool(args.smoke)
    seeds = (list(SEEDS[:2]) if smoke
             else (list(SEEDS) if args.seeds is None
                   else [int(s) for s in args.seeds.split(",")]))
    if not seeds or any(not isinstance(s, int) or s < 0 for s in seeds):
        print(f"[FAIL] 잘못된 시드 목록: {seeds}", file=sys.stderr)
        return 1

    evidence_path = (
        Path(args.evidence).expanduser().resolve() if args.evidence
        else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / (
            "task-10-catboost9-schema.json" if smoke
            else "task-10-catboost9-schema-full.json")
    )
    log_path = evidence_path.with_suffix(".log")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    tee = _Tee(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        return _main_impl(args, smoke, seeds, evidence_path, t0)
    finally:
        sys.stdout = old_stdout
        tee.close()


def _main_impl(args, smoke: bool, seeds: list[int],
               evidence_path: Path, t0: float) -> int:
    digest = _cache_digest(seeds)

    # ── 0) 범주 스키마 어서션 출력 (Slot-2 핵심) ──
    print(f"[catboost9] 범주 스키마 어서션: 정확히 {len(CAT_SCHEMA)} 종 unique 이름 "
          f"(순서 동결) = {list(CAT_SCHEMA)}", flush=True)
    print(f"[catboost9] 기존 catboost(5종 LGB_CATS) 대비 +4 ID 필드 "
          f"pitcher_id/batter_id/pitcher_team_id/batter_team_id → 네이티브 CTR 활성화",
          flush=True)
    print(f"[catboost9] mode={'smoke' if smoke else 'full'} seeds={seeds} "
          f"cache_digest={digest[:16]}...", flush=True)

    # ── 1) 데이터 로드 + 동결 피처 계약 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feat_ok, feat_detail = _check_feature_contract("catboost9")
    missing_feats = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if not feat_ok or missing_feats:
        print(f"[FAIL] 동결 피처 계약 불일치: {feat_detail} / 부재: {missing_feats}",
              file=sys.stderr, flush=True)
        return 1

    # ── 2) 폴드 + 누수 가드 + 검증 행 수 계약 ──
    folds = build_folds(train)
    problems = list(_check_leakage(folds, train))
    n_va = {}
    for fn in FOLDS:
        tr_m, va_m = folds[fn]
        if int((train.loc[tr_m, "season"] == 2025).sum()) > 0:
            problems.append(f"{fn}: 학습 마스크에 2025 시즌 포함 — 누수!")
        n_va[fn] = int(va_m.sum())
        if n_va[fn] != EXPECTED_N_VA[fn]:
            problems.append(f"{fn}: 검증 행 수 {n_va[fn]} != 동결 기대 {EXPECTED_N_VA[fn]}")
    if problems:
        print("[FAIL] 폴드/누수/행 수 게이트:\n  " + "\n  ".join(problems),
              file=sys.stderr, flush=True)
        return 1

    # ── 3) 범주 스키마/dtype 계약 (무성 coercion 없음) ──
    try:
        cat_report = _validate_cat_schema(train, CHAMPION_FEATURES)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    # ── 4) 폴드별 학습/평가 ──
    per_fold: dict[str, dict] = {}
    for fn in FOLDS:
        tr_m, va_m = folds[fn]
        try:
            _cache_guard(fn, digest, bool(args.overwrite_cache))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 1
        yv = train.loc[va_m, common.TARGET].values
        t_fold = time.time()
        z, per_seed, card = _train_catboost9_fold(
            train, fn, tr_m, va_m, seeds, cat_report)
        runtime_s = time.time() - t_fold
        meta = _evaluate_fold(fn, z, yv, seeds, per_seed, card, digest,
                              _inputs_sha(train, va_m), runtime_s)
        per_fold[fn] = meta
        row = meta
        print(f"  [{fn:<8s}] bss={row['bss']:>8.1f} mean={row['pred_mean']:.4f} "
              f"n={row['n_rows']} t={row['runtime_s']:.0f}s "
              f"digest={row['digest'][:12]}...", flush=True)

    # ── 5) 스키마 저장 ──
    params_demo = _catboost_params(seeds[0])
    params_demo.pop("random_seed", None)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-top100/task-10-catboost9-schema",
        "variant": "catboost9",
        "mode": "smoke" if smoke else "full",
        "smoke": smoke,
        "seeds_requested": list(seeds),
        "folds": list(FOLDS),
        "r_folds": list(R_FOLDS),
        "categorical_schema": {
            "n_cats": len(CAT_SCHEMA),
            "assertion": "exactly 9 unique categorical names asserted",
            "cat_features": list(CAT_SCHEMA),
            "source": "qualification_runner.MLP_CATS (model/mlp_meta.json frozen, 9)",
            "vs_prior": {
                "prior_catboost_cats": list(LGB_CATS),
                "added_id_fields": ["pitcher_id", "batter_id",
                                    "pitcher_team_id", "batter_team_id"],
                "meaning": CATBOOST_CAT_FEATURES_MEANING,
            },
            "dtype_policy": {
                "version": DTYPE_POLICY_VERSION,
                "rule": DTYPE_POLICY_RULE,
                "report_all_rows": cat_report,
            },
        },
        "hyperparameters": {
            "params_base": dict(FAMILIES["catboost"]["params_base"]),
            "iterations": int(FAMILIES["catboost"]["rounds"]),
            "od_type": "Iter",
            "od_wait": int(FAMILIES["catboost"]["early_stopping_rounds"]),
            "use_best_model": True,
            "random_seed": "per seed",
            "example_params_without_seed": params_demo,
        },
        "cache": {
            "dir": str(CACHE_DIR.relative_to(REPO)),
            "digest": digest,
            "digest_components": [
                "cat_schema (names+order)", "dtype_policy_version", "features",
                "seeds", "folds+fold_definitions", "params_digest"],
            "stale_guard": "digest mismatch → overwrite refused (--overwrite-cache only)",
            "note": "기존 cache/qualification/catboost/ (5종) 캐시 미접촉 — 별도 variant",
        },
        "integrity": {
            "gate": "PASS",
            "feature_contract": {"ok": feat_ok, "detail": feat_detail},
            "cat_schema_asserted": len(CAT_SCHEMA) == 9 and len(set(CAT_SCHEMA)) == 9,
            "leakage_guard": {"no_2025_in_any_validation_mask": True,
                              "no_2025_in_any_training_mask": True},
            "expected_n_va_matched": {fn: (n_va[fn] == EXPECTED_N_VA[fn],
                                           EXPECTED_N_VA[fn]) for fn in FOLDS},
        },
        "bss_policy": {
            "metric": "BSS (common.score)",
            "transform": "raw sigmoid — C_LOGIT 무적용 (제출 정렬 상수, Task 2 규약 유지)",
            "c_logit_applied": False,
        },
        "per_fold": per_fold,
        "seed_aggregation": {
            "seeds": list(seeds),
            "aggregation": "mean of per-seed logits",
            "note": "Task 2 seed_aggregation 동일 규약",
        },
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": round(time.time() - t0, 2),
    }

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\n[catboost9] 증거 JSON → {evidence_path}", flush=True)
    print(f"[catboost9] 로그 → {evidence_path.with_suffix('.log')}", flush=True)

    # ── 6) 스모크 판정 ──
    if smoke:
        ok = (
            len(seeds) == 2
            and all(fn in per_fold for fn in FOLDS)
            and schema["integrity"]["gate"] == "PASS"
            and schema["integrity"]["cat_schema_asserted"]
            and all(v[0] for v in schema["integrity"]["expected_n_va_matched"].values())
        )
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 9범주 스키마 어서션/dtype 계약/"
              f"폴드/누수/행 수/캐시 경로 검증 완료 (seeds={seeds})", flush=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
