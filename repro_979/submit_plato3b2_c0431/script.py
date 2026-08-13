#!/usr/bin/env python3
"""
[plato3b2] 제출용 추론 스크립트 — count_platoon_3b2_same 포함 F3 + Entity Embedding MLP 블렌딩 (2026-08-11)

구성: count_platoon_3b2_same(3-2 & 동손) 추가 LightGBM F3 10시드 로짓 평균
      × 동일 피처 챔피언 아키텍처 MLP 10시드 로짓 평균 블렌드. w_lgb=0.51.

이 실험은 2-stage gate 프로토콜의 "submit-to-test" 1회 live test:
  - 통과: 10시드 R-only 3/3, Δmean 안전(|Δmean|≤0.00034), corr(lgb,mlp)<0.95,
          blend 시너지(primary +28.0, R-only 3/3)
  - 미달: 절대 primary ≥ +15 bar (실제 +4.9) → 강화 게이트 1번 조건 실패로 기각됐으나,
          R-only/blend/상관/평균 안전성은 전부 충족 → 2025 리더보드 1-submission live test로 승격

파이프라인:
  1) test 로드 (utf-8-sig)
  2) feats = common.get_feature_cols + ["count_platoon_3b2_same"] (58)
     — common.preprocess_for_submission(add_cross_cell_features)가 이미 이 컬럼을 계산
  3) feature parity assert (KeyError 방지)
  4) LGB: model/f3_s{seed}.txt 10개 predict → 로짓 평균
  5) MLP: model/mlp_s{seed}.pt 10개 predict → 로짓 평균 (mlp_model.py)
  6) 블렌드: z = W_LGB*z_lgb + (1-W_LGB)*z_mlp
  7) 로짓 공간 C_LOGIT 적용 → 시그모이드 → clip(0.30, 0.70)
  8) output/submission.csv 저장 (sample_submission 기준 row_id 순서)

전처리: LGB/MLP 모두 common.py(전처리) + mlp_model.py(MLP 전처리) 공유.

── C_LOGIT 재정렬 판정 v2 (2026-08-13, submit_plato3b2 재제출) ──────────────────
판정: C_LOGIT = -0.043134 (신 파이프라인 평균 정렬 재적용)

근거 — v1(2026-08-11)은 |p_new − 0.4871| = 0.00067 ≤ 0.001 을 이유로
"재정렬 불필요, 챔피언 -0.0404 유지"로 판정했으나, 실제 제출이 **-2.17**
(챔피언 979.31 대비)을 기록. δ-sweep 실험(±0.003 → ±2.5~4.5점)과 교차 검증:
  챔피언 -0.0404 = logit(0.477) − logit(0.4871)   (챔피언 파이프라인 2024 측정 평균)
  신 파이프라인 2024 측정 평균 p_new = 0.487770   (plato3b2_mean_check.json)
  C_LOGIT_measured = logit(0.477) − logit(0.487770) = -0.043134
  0.0027 차이 ≈ δ-sweep +0.003 효과(-2.52)와 동일 부호·크기 → **-2.17의 원인은
  피처 자체가 아니라 C_LOGIT 미스매치로 판정**, 재정렬 후 재제출.

측정값 (experiments/plato3b2_mean_check.json):
  p_lgb(2024) = 0.488201  p_mlp(2024) = 0.487321
  p_new = 0.51·0.488201 + 0.49·0.487321 = 0.487770
  C_LOGIT = logit(0.477) − logit(0.487770) = -0.043134

캐시 fold-OOF 평균(blend 0.4926) 기반 재정렬은 채택하지 않음 (v1과 동일 — 배포 모델
자연 평균이 아님, -0.0626으로 과보정). 모든 값은 학습 데이터/배포 모델 측정 +
자체 제출 LB에서 유도. test 예측 평균 미참조.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common
import mlp_model

W_LGB = 0.51
C_LOGIT = -0.043134

SEEDS = list(range(42, 52))
# 모델 경로는 script.py 위치 기준 (평가 서버 zip 루트, 로컬 submit/ 모두 동작)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
CLIP_LO, CLIP_HI = 0.30, 0.70
FEATURE = "count_platoon_3b2_same"

# 검증용 경로 오버라이드 (제출 환경에서는 미설정 → 기본값)
TEST_PATH = os.environ.get("LGA_TEST_PATH", "data/test.csv")
SAMPLE_PATH = os.environ.get("LGA_SAMPLE_PATH", "data/sample_submission.csv")
OUT_PATH = os.environ.get("LGA_OUT_PATH", "output/submission.csv")


def main():
    t0 = time.time()
    # 1) 데이터 로드 (utf-8-sig 필수 — BOM)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    print(f"test: {test.shape}", flush=True)
    # 피처 목록은 전처리 전 원본 컬럼 기준 + count_platoon_3b2_same
    # (preprocess의 add_cross_cell_features가 이미 계산 → 목록만 확장)
    feats = common.get_feature_cols(test.columns) + [FEATURE]

    # 2) 공통 전처리 (학습과 동일 모듈)
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
