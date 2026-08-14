#!/usr/bin/env python3
"""
[Todo 8 배포] CatBoost FULL-DATA 배포 모델 학습 (2026-08-14)

Todo 7 수락 블렌드 champ_cat(champion×0.85 + catboost×0.15) 의 catboost 구성원을
배포하기 위한 full-data 모델. Task 5 의 폴드별 OOF 모델은 학습 중 인메모리에서만
존재해 디스크 아티팩트(.cbm)가 없으므로, 평가 서버에서 직접 사용할 full-data 모델을
여기서 재학습한다.

프로토콜 (Task 5 model_family_runner.FAMILIES["catboost"] 와 동일):
  - loss_function=Logloss, eval_metric=Logloss, learning_rate=0.05, depth=7,
    l2_leaf_reg=3.0, bootstrap_type=Bernoulli, subsample=0.8,
    random_strength=1.0, thread_count=32, allow_writing_files=False
  - random_seed = 시드, iterations = --iterations (고정 — full-data 는 eval set 없음)
  - 피처 49 (CHAMPION_FEATURES), cat_features 5 (LGB_CATS), Pool(cat_features)

누수 가드:
  - 학습 데이터 = train.csv 전체 (시즌 2019-2024). 2025 행 존재 시 즉시 중단.
  - test 행/2025 라벨 미사용. 외부 데이터 미사용.

사용법:
  python3 repro_979/deploy_train_catboost_full.py --iterations 300 \
      --model-dir repro_979/submit_champ_cat_<ts>/model \
      [--seeds 42,43,...] [--thread-count 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import common  # noqa: E402
import numpy as np  # noqa: E402

# Task 2 동결 계약 (qualification_runner 에서 READ — 재계산 아님)
from qualification_runner import CHAMPION_FEATURES, LGB_CATS, SEEDS  # noqa: E402

# Task 5 catboost params (model_family_runner.FAMILIES["catboost"]["params_base"])
CATBOOST_PARAMS_BASE = dict(
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
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, required=True,
                        help="고정 반복 수 (full-data, eval set 없음)")
    parser.add_argument("--model-dir", required=True,
                        help="저장 디렉토리 (submit dir 의 model/ 예정)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="시드 목록 (기본 42..51)")
    parser.add_argument("--thread-count", type=int, default=32)
    parser.add_argument("--probe-rows", type=int, default=0,
                        help=">0 이면 전 데이터 대신 랜덤 n행으로 학습 (속도 프로브용)")
    args = parser.parse_args(argv)

    # --model-dir 는 워크스페이스 루트 기준 (os.chdir(REPO) 후에도 안정적)
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    # ── 1) 데이터 로드 + 전처리 (학습/추론 동일 모듈) ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feats = list(CHAMPION_FEATURES)
    cats = list(LGB_CATS)
    missing = [f for f in feats if f not in train.columns]
    if missing:
        print(f"[FAIL] 피처 부재: {missing}", file=sys.stderr)
        return 1

    # ── 2) 누수 가드: 2025 행/라벨 부재 확인 ──
    max_season = int(train["season"].max())
    n_2025 = int((train["season"] == 2025).sum())
    if max_season > 2024 or n_2025 > 0:
        print(f"[FAIL] 누수 가드: train 에 2025 라벨 {n_2025}행, max_season={max_season}",
              file=sys.stderr)
        return 1
    print(f"[leakage] train rows={len(train)} seasons={int(train['season'].min())}-{max_season} "
          f"2025행={n_2025} ✓", flush=True)

    if args.probe_rows > 0:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train), size=args.probe_rows, replace=False)
        train = train.iloc[idx]
        print(f"[probe] {args.probe_rows}행 부분 학습", flush=True)

    X = train[feats].copy()
    y = train[common.TARGET].astype(int).values
    print(f"X={X.shape} y_mean={y.mean():.4f}", flush=True)

    from catboost import CatBoostClassifier, Pool  # noqa: PLC0415
    pool = Pool(X, y, cat_features=cats)

    # ── 3) 시드별 학습 + 저장 ──
    params_base = dict(CATBOOST_PARAMS_BASE)
    params_base["thread_count"] = args.thread_count
    per_seed: dict[str, dict] = {}
    t0 = time.time()
    for seed in seeds:
        params = dict(params_base)
        params["random_seed"] = seed
        params["iterations"] = args.iterations
        model = CatBoostClassifier(**params)
        ts = time.time()
        model.fit(pool)
        dt = time.time() - ts
        out_path = model_dir / f"catboost_s{seed}.cbm"
        model.save_model(str(out_path))
        per_seed[str(seed)] = {"runtime_s": round(dt, 1), "model_size_mb": round(out_path.stat().st_size / 1e6, 2)}
        print(f"[seed {seed}] {dt:.1f}s → {out_path.name} "
              f"({per_seed[str(seed)]['model_size_mb']}MB)", flush=True)

    total_s = time.time() - t0
    print(f"[done] seeds={seeds} iterations={args.iterations} total={total_s:.1f}s "
          f"({total_s / 60:.1f}min) dir={model_dir}", flush=True)

    manifest = {
        "schema_version": 1,
        "task": "aimers9-top100/task-8-catboost-full",
        "params": {k: v for k, v in params_base.items() if k != "verbose"},
        "iterations": args.iterations,
        "seeds": seeds,
        "features": {"n": len(feats), "list": feats, "cat_features": cats},
        "n_train_rows": int(len(train)),
        "max_season": int(max_season),
        "leakage": {"no_2025": True, "no_test_rows": True, "no_external": True},
        "per_seed": per_seed,
        "total_runtime_s": round(total_s, 1),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = REPO / "cache" / "t8_install_sim" / "catboost_full_train_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
