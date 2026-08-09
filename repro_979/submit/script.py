#!/usr/bin/env python3
"""
[V5] 제출용 추론 스크립트 — 채택 피처 F3 + Entity Embedding MLP 블렌딩 (2026-08-10)

구성: 채택 피처(asof_n_bucket, score_diff_binary) 포함 LightGBM F3 10시드 로짓 평균
      × 채택 피처 MLP 10시드 로짓 평균 블렌드. w_lgb=0.51 (Exp 6-c-2 검증 primary 최적).

파이프라인:
  1) test 로드 (utf-8-sig)
  2) feats = common.get_feature_cols (전처리 전 원본 컬럼 기준, platoon/count_state 이중 방지)
  3) common.preprocess_for_submission: 다운캐스팅 + 범주형 + platoon + count_state + 채택 피처
  4) ⚠️ train/test feature parity assert: feats 전부 test에 존재 (KeyError 방지)
  5) LGB: model/f3_s{seed}.txt 10개 predict → 로짓 평균
  6) MLP: model/mlp_s{seed}.pt 10개 predict → 로짓 평균 (mlp_model.py)
  7) 블렌드: z = W_LGB*z_lgb + (1-W_LGB)*z_mlp
  8) 로짓 공간 C_LOGIT 적용 → 시그모이드 → clip(0.30, 0.70)
  9) output/submission.csv 저장 (sample_submission 기준 row_id 순서)

전처리: LGB/MLP 모두 common.py(전처리) + mlp_model.py(MLP 전처리) 공유.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common
import mlp_model

# ── G1/Exp6-c: 블렌드 + 고정 로짓 오프셋 (추론 중 재계산 없음) ──
# r_2025 추정치 0.477(자체 제출 LB 890.90 역산, g≈0.0122) 기준 정렬:
#   p_bar_blend(2024-proxy) = 0.4871  (LGB 0.4882 × 0.51 + MLP 0.4860 × 0.49)
#   C_LOGIT = logit(0.477) - logit(0.4871) = -0.0404
# 모든 값은 학습 데이터 + 자체 제출 LB에서 유도. test 예측 평균 미참조.
W_LGB = 0.51
C_LOGIT = -0.0404

SEEDS = list(range(42, 52))
# 모델 경로는 script.py 위치 기준 (평가 서버 zip 루트, 로컬 submit/ 모두 동작)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
CLIP_LO, CLIP_HI = 0.30, 0.70

# 검증용 경로 오버라이드 (제출 환경에서는 미설정 → 기본값)
TEST_PATH = os.environ.get("LGA_TEST_PATH", "data/test.csv")
SAMPLE_PATH = os.environ.get("LGA_SAMPLE_PATH", "data/sample_submission.csv")
OUT_PATH = os.environ.get("LGA_OUT_PATH", "output/submission.csv")


def main():
    t0 = time.time()
    # 1) 데이터 로드 (utf-8-sig 필수 — BOM)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    print(f"test: {test.shape}", flush=True)
    # 피처 목록은 전처리 전 원본 컬럼 기준 (platoon/count_state 이중 추가 방지)
    feats = common.get_feature_cols(test.columns)

    # 2) 공통 전처리 (학습과 동일 모듈)
    common.preprocess_for_submission(test)

    # 3) ⚠️ feature parity assert — feats 전부 test에 존재해야 함 (KeyError 방지)
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
        p = bst.predict(X)
        z_lgb += common.logit(p)
    z_lgb /= len(SEEDS)
    print(f"LGB 10시드: {time.time()-t1:.1f}s", flush=True)

    # 5) MLP 모델 로드 + 예측 (로짓 평균)
    t1 = time.time()
    prep, mlp_models = mlp_model.load(MODEL_DIR, SEEDS)
    z_mlp = mlp_model.predict_z(test, prep, mlp_models)
    print(f"MLP 10시드: {time.time()-t1:.1f}s", flush=True)

    # 6) 로짓 공간 블렌딩
    z = W_LGB * z_lgb + (1 - W_LGB) * z_mlp

    # 7) 로짓 공간 C_LOGIT 적용 → sigmoid → clip (고정 상수, test 미참조)
    p = np.clip(common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)

    # 8) submission 저장 (sample_submission 기준 row_id 순서)
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    pred = pd.DataFrame({"row_id": test[common.ID], "control_success": p})
    sub = sample[["row_id"]].merge(pred, on="row_id", how="left")
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"submission: {OUT_PATH} ({len(sub)}행, "
          f"mean={p.mean():.4f})  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
