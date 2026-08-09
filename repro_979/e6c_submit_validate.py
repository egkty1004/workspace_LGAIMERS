#!/usr/bin/env python3
"""
[Exp 6-c 배포] 제출 파이프라인 검증 — F3+MLP 블렌드 (2026-08-08)

1) 2024-as-test 출력 평균 ≈ 0.477 (r_2025 정렬 목표, ±0.002)
2) clip(0.30,0.70) 접촉 비율 (<0.1%)
3) script.py vs 참조 파이프라인 최대 절대차 < 1e-6
4) 245,789행 추론 시간
5) 5행 end-to-end
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
import common  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "submit"))
import mlp_model  # noqa: E402
import lightgbm as lgb  # noqa: E402

SUBMIT = os.path.join(REPO, "submit")
W_LGB = 0.51
C_LOGIT = -0.0404
CLIP_LO, CLIP_HI = 0.30, 0.70
SEEDS = list(range(42, 52))
TEST_245K = os.path.join(REPO, "data/test_2024_245k.csv")
SAMPLE_245K = os.path.join(REPO, "data/sample_submission_2024_245k.csv")
OUT_245K = os.path.join(REPO, "tmp_blend_245k.csv")


def reference_predict(test_path):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    feats = common.get_feature_cols(test.columns)
    common.preprocess_for_submission(test)
    X = test[feats].copy()
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in SEEDS:
        bst = lgb.Booster(model_file=os.path.join(SUBMIT, "model", f"f3_s{s}.txt"))
        z_lgb += common.logit(bst.predict(X))
    z_lgb /= len(SEEDS)
    prep, models = mlp_model.load(os.path.join(SUBMIT, "model"), SEEDS)
    z_mlp = mlp_model.predict_z(test, prep, models)
    z = W_LGB * z_lgb + (1 - W_LGB) * z_mlp
    p_pre = common.sigmoid(z)
    p_post = common.sigmoid(z + C_LOGIT)
    return np.clip(p_post, CLIP_LO, CLIP_HI), p_post, p_pre


def run_script(test_path, sample_path, out_path):
    env = dict(os.environ)
    env["LGA_TEST_PATH"] = test_path
    env["LGA_SAMPLE_PATH"] = sample_path
    env["LGA_OUT_PATH"] = out_path
    t0 = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=SUBMIT, env=env,
                       capture_output=True, text=True)
    dt = time.time() - t0
    print(r.stdout.strip())
    if r.stderr:
        print("STDERR:", r.stderr[-1500:])
    assert r.returncode == 0, "script.py 실패"
    return dt


def main():
    print("=" * 70)
    print("[배포 검증] 245,789행 2024-as-test (F3+MLP 블렌드)")
    print("=" * 70)
    dt = run_script(TEST_245K, SAMPLE_245K, OUT_245K)
    out = pd.read_csv(OUT_245K, encoding="utf-8-sig")
    p_script = out["control_success"].values
    p_ref, p_post, p_pre = reference_predict(TEST_245K)
    assert len(p_ref) == 245789

    m_raw = float(p_pre.mean())
    m = float(p_script.mean())
    print(f"\n[1] blend raw 평균(p_bar) = {m_raw:.4f} | 출력 평균 = {m:.4f} "
          f"(목표 0.477±0.002)")
    ok1 = abs(m - 0.477) <= 0.002
    print(f"    {'✅' if ok1 else '⚠'}")

    frac = float(np.mean((p_post < CLIP_LO) | (p_post > CLIP_HI)))
    print(f"[2] clip 접촉 비율 = {frac*100:.4f}% (임계 0.1%)")
    ok2 = frac <= 0.001
    print(f"    {'✅' if ok2 else '⚠ 초과'}")

    md = float(np.max(np.abs(p_script - p_ref)))
    print(f"[3] script.py vs 참조 최대 절대차 = {md:.3e} (목표 < 1e-6)")
    ok3 = md < 1e-6
    print(f"    {'✅' if ok3 else '❌'}")

    print(f"[4] 245,789행 wall time = {dt:.2f}s (제한 600s)")

    print("\n" + "=" * 70)
    print("[배포 검증] 5행 end-to-end (기본 경로)")
    print("=" * 70)
    if not os.path.islink(os.path.join(SUBMIT, "data")):
        os.symlink(os.path.join(REPO, "data"), os.path.join(SUBMIT, "data"))
    r = subprocess.run([sys.executable, "script.py"], cwd=SUBMIT,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr:
        print("STDERR:", r.stderr[-1500:])
    assert r.returncode == 0, "5행 실패"
    sub5 = pd.read_csv(os.path.join(SUBMIT, "output/submission.csv"), encoding="utf-8-sig")
    assert len(sub5) == 5 and sub5["control_success"].notna().all()
    p5_ref, _, _ = reference_predict(os.path.join(REPO, "data/test.csv"))
    md5 = float(np.max(np.abs(sub5["control_success"].values - p5_ref)))
    print(f"[5] 5행 출력 {len(sub5)}행, max|diff|={md5:.3e}")
    ok5 = md5 < 1e-6
    print(f"    {'✅' if ok5 else '❌'}")

    print(f"\n요약: mean {m:.4f} / clip {frac*100:.4f}% / diff {md:.3e} / time {dt:.2f}s / 5행 {'✅' if ok5 else '❌'}")
    print("전체:", "✅ 통과" if (ok1 and ok2 and ok3 and ok5) else "⚠ 확인 필요")


if __name__ == "__main__":
    main()
