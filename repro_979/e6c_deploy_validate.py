#!/usr/bin/env python3
"""
[Exp 6-c 배포 검증] LGB F3 + MLP 블렌드 오프셋 산정 (2026-08-08)

전체 데이터 학습 모델(LGB f3 10시드 + MLP 10시드)로 2024-as-test 예측:
  z_blend = W_LGB*z_lgb + (1-W_LGB)*z_mlp
p_bar(2024-proxy) 측정 → r_2025≈0.477 정렬용 c_logit 도출 → clip 접촉/추론 시간 확인.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "submit"))
import common  # noqa: E402
import mlp_model  # noqa: E402
import lightgbm as lgb  # noqa: E402

W_LGB = 0.51
SEEDS_LGB = list(range(42, 52))
SEEDS_MLP = list(range(42, 52))
MODEL_DIR = "submit/model"
CLIP_LO, CLIP_HI = 0.30, 0.70
R_2025_EST = 0.477


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    t0 = time.time()
    raw_cols = pd.read_csv("data/test_2024_245k.csv", encoding="utf-8-sig", nrows=0).columns
    feats = common.get_feature_cols(raw_cols)
    test = pd.read_csv("data/test_2024_245k.csv", encoding="utf-8-sig")
    common.preprocess_for_submission(test)
    X = test[feats].copy()
    print(f"test: {test.shape}", flush=True)

    # LGB 로짓
    t1 = time.time()
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in SEEDS_LGB:
        bst = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"f3_s{s}.txt"))
        z_lgb += common.logit(bst.predict(X))
    z_lgb /= len(SEEDS_LGB)
    print(f"LGB 10시드: {time.time()-t1:.1f}s  mean_z={z_lgb.mean():.4f}", flush=True)

    # MLP 로짓
    t1 = time.time()
    prep, mlp_models = mlp_model.load(MODEL_DIR, SEEDS_MLP)
    z_mlp = mlp_model.predict_z(test, prep, mlp_models)
    print(f"MLP 10시드: {time.time()-t1:.1f}s  mean_z={z_mlp.mean():.4f}", flush=True)

    z_blend = W_LGB * z_lgb + (1 - W_LGB) * z_mlp
    p_raw = common.sigmoid(z_blend)
    p_lgb = common.sigmoid(z_lgb)
    p_mlp = common.sigmoid(z_mlp)
    print(f"\np_bar(2024-proxy): LGB={p_lgb.mean():.4f}  MLP={p_mlp.mean():.4f}  "
          f"blend={p_raw.mean():.4f}", flush=True)

    # 오프셋: r_2025≈0.477 정렬 (LGB 파이프라인의 검증된 목표)
    c_logit = logit(np.array(R_2025_EST))[()] - logit(np.array(p_raw.mean()))[()]
    print(f"c_logit(r_2025={R_2025_EST} 정렬) = {c_logit:+.4f}", flush=True)
    p_final = np.clip(common.sigmoid(z_blend + c_logit), CLIP_LO, CLIP_HI)
    print(f"보정 후 평균 = {p_final.mean():.4f}  (목표 r_2025 {R_2025_EST})", flush=True)
    frac = float(np.mean((p_raw < CLIP_LO) | (p_raw > CLIP_HI)))
    print(f"clip 접촉 비율 = {frac*100:.4f}%", flush=True)
    print(f"총 소요 = {time.time()-t0:.1f}s", flush=True)

    np.save("cache/deploy_blend_z.npy", z_blend)
    print("저장: cache/deploy_blend_z.npy", flush=True)


if __name__ == "__main__":
    main()
