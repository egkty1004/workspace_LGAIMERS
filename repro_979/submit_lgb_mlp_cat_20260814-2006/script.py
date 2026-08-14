#!/usr/bin/env python3
"""
[V6 lgb_mlp_cat] 제출용 추론 스크립트 — LGB × MLP × CatBoost 직접 블렌드 (2026-08-14)

Todo 7 수락 블렌드 `lgb_mlp_cat` (candidate_id 7771a019594deedd):
      z = 0.30 * z_lgb + 0.45 * z_mlp + 0.25 * z_catboost        (로짓 공간)
  구성원:
    z_lgb   = LGB F3 10시드 로짓 평균   (model/f3_s{seed}.txt — 챔피언 GIHO byte-copy)
    z_mlp   = MLP 10시드 로짓 평균      (model/mlp_s{seed}.pt — 챔피언 GIHO byte-copy, mlp_model.py)
    z_catboost = CatBoost full-data 10시드 로짓 평균 (model/catboost_s{seed}.cbm,
      Task 8 full-data 재학습, 시즌 2019-2024, 피처 49 / cat_features 5)
  최종: p = clip(sigmoid(z + C_LOGIT), 0.30, 0.70)

C_LOGIT = -0.0404 는 챔피언 정책 상수 (r_2025 추정치 0.477 정렬). Todo 7 측정상
lgb_mlp_cat 의 챔피언 대비 mean_shift 는 최대 3.2e-4 로 재정렬 불필요 (챔피언 정책 유지).

파이프라인:
  1) test 로드 (utf-8-sig)
  2) feats = common.get_feature_cols(test.columns) — 전처리 전 원본 컬럼 기준
  3) common.preprocess_for_submission(test) — 다운캐스팅 + 범주형 + platoon + count_state
  4) feature parity assert
  5) LGB 10시드 → 로짓 평균 / MLP 10시드 → 로짓 평균 / CatBoost 10시드 → 로짓 평균
  6) 로짓 공간 직접 블렌드 (lgb 0.30 / mlp 0.45 / catboost 0.25)
  7) C_LOGIT → 시그모이드 → clip(0.30, 0.70)
  8) output/submission.csv 저장 (sample_submission 기준 row_id 순서)

전처리: LGB/MLP/CatBoost 모두 common.py(전처리) + mlp_model.py(MLP) 공유 —
       이 스크립트가 있는 디렉토리의 common.py 를 사용 (재현 디렉토리 미참조).
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common
import mlp_model

# ── Todo 7 수락 블렌드 상수 (고정 가중치, test 미참조) ──
W_LGB = 0.30            # lgb_mlp_cat: LGB 가중치 (Todo 7 격자 피팅)
W_MLP = 0.45            # lgb_mlp_cat: MLP 가중치 (Todo 7 격자 피팅)
W_CAT = 0.25            # lgb_mlp_cat: catboost 가중치 (Todo 7 격자 피팅)
C_LOGIT = -0.0404       # 챔피언 정렬 상수 (r_2025 ≈ 0.477)

SEEDS = list(range(42, 52))
CLIP_LO, CLIP_HI = 0.30, 0.70
CAT_FEATURES = ["top_bottom", "game_type", "base_state", "platoon", "count_state"]
CATBOOST_THREADS = 6    # 평가 환경 6 vCPU 대응 (predict 전용)

# 모델 경로는 script.py 위치 기준 (평가 서버 zip 루트, 로컬 submit/ 모두 동작)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# 검증용 경로 오버라이드 (제출 환경에서는 미설정 → 기본값)
TEST_PATH = os.environ.get("LGA_TEST_PATH", "data/test.csv")
SAMPLE_PATH = os.environ.get("LGA_SAMPLE_PATH", "data/sample_submission.csv")
OUT_PATH = os.environ.get("LGA_OUT_PATH", "output/submission.csv")


def predict_z_catboost(X: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    """CatBoost 10시드 로짓 평균. catboost 미import 시 명확한 실패 (챔피언 롤백 안내)."""
    try:
        from catboost import CatBoostClassifier, Pool  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — 배포 게이트 상 미발생
        raise RuntimeError(
            "catboost import 실패 — lgb_mlp_cat 패키지 오류. 챔피언 롤백 사용: "
            "team_member_materials/GIHO/submit979_extract (979.31, requirements: lightgbm==4.7.0)"
        ) from exc
    pool = Pool(X, cat_features=cat_features)
    z = np.zeros(len(X), dtype=np.float64)
    for seed in SEEDS:
        path = os.path.join(MODEL_DIR, f"catboost_s{seed}.cbm")
        if not os.path.exists(path):
            raise RuntimeError(f"catboost 모델 없음: {path} — 챔피언 롤백 사용")
        model = CatBoostClassifier()
        model.load_model(path)
        p = np.asarray(model.predict(pool, prediction_type="Probability",
                                     thread_count=CATBOOST_THREADS), dtype=np.float64)
        if p.ndim == 2:
            p = p[:, 1]  # class-1 확률 열 (catboost 1.2.x)
        z += common.logit(p)
    z /= len(SEEDS)
    return z


def main():
    t0 = time.time()
    # 1) 데이터 로드 (utf-8-sig 필수 — BOM)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    print(f"test: {test.shape}", flush=True)
    # 피처 목록은 전처리 전 원본 컬럼 기준 (platoon/count_state 이중 추가 방지)
    feats = common.get_feature_cols(test.columns)

    # 2) 공통 전처리 (학습과 동일 모듈 — 이 디렉토리의 common.py)
    common.preprocess_for_submission(test)

    # 3) feature parity assert — feats 전부 test에 존재해야 함 (KeyError 방지)
    missing = [c for c in feats if c not in test.columns]
    assert not missing, f"feature parity 실패 — test에 없는 피처: {missing}"
    X = test[feats].copy()
    print(f"features: {len(feats)}", flush=True)

    # 4) LGB 모델 로드 + 예측 (로짓 평균)
    t1 = time.time()
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for seed in SEEDS:
        path = os.path.join(MODEL_DIR, f"f3_s{seed}.txt")
        bst = lgb.Booster(model_file=path)
        z_lgb += common.logit(bst.predict(X))
    z_lgb /= len(SEEDS)
    print(f"LGB 10시드: {time.time()-t1:.1f}s", flush=True)

    # 5) MLP 모델 로드 + 예측 (로짓 평균)
    t1 = time.time()
    prep, mlp_models = mlp_model.load(MODEL_DIR, SEEDS)
    z_mlp = mlp_model.predict_z(test, prep, mlp_models)
    print(f"MLP 10시드: {time.time()-t1:.1f}s", flush=True)

    # 6) CatBoost 모델 로드 + 예측 (로짓 평균)
    t1 = time.time()
    z_cat = predict_z_catboost(X, CAT_FEATURES)
    print(f"CatBoost 10시드: {time.time()-t1:.1f}s", flush=True)

    # 7) 로짓 공간 직접 블렌드 (lgb_mlp_cat)
    z = W_LGB * z_lgb + W_MLP * z_mlp + W_CAT * z_cat

    # 8) 로짓 공간 C_LOGIT 적용 → sigmoid → clip (고정 상수, test 미참조)
    p = np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)

    # 9) submission 저장 (sample_submission 기준 row_id 순서)
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    pred = pd.DataFrame({"row_id": test[common.ID], "control_success": p})
    sub = sample[["row_id"]].merge(pred, on="row_id", how="left")
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"submission: {OUT_PATH} ({len(sub)}행, "
          f"mean={p.mean():.4f})  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
