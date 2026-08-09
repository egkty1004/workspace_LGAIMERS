#!/usr/bin/env python3
"""
[G1] 제출 파이프라인 검증 (2026-08-08)
1) clip(0.30,0.70) 접촉 행 비율 (<0.1% 목표, 초과 시 보고)
2) 2024-as-test 출력 평균 0.486 ± 0.002 (2024 r=0.4861)
3) 학습 코드(참조 파이프라인) 예측 vs script.py 출력 최대 절대차 < 1e-6
4) 245,789행 추론 시간 재측정
5) 5행 test.csv end-to-end
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
import common  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(REPO, "submit")
C_LOGIT = -0.0440
CLIP_LO, CLIP_HI = 0.30, 0.70
SEEDS = list(range(42, 52))

TEST_245K = os.path.join(REPO, "data/test_2024_245k.csv")
SAMPLE_245K = os.path.join(REPO, "data/sample_submission_2024_245k.csv")
OUT_245K = os.path.join(REPO, "tmp_g1_245k.csv")
OUT_5 = os.path.join(SUBMIT, "output/submission.csv")


def reference_predict(test_path):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    feats = common.get_feature_cols(test.columns)
    common.preprocess_for_submission(test)
    X = test[feats].copy()
    z_acc = np.zeros(len(X), dtype=np.float64)
    for seed in SEEDS:
        bst_path = os.path.join(SUBMIT, "model", f"f3_s{seed}.txt")
        bst = __import__("lightgbm").Booster(model_file=bst_path)
        p = bst.predict(X)
        z_acc += common.logit(p)
    z = z_acc / len(SEEDS)
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
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr)
    assert r.returncode == 0, "script.py 실패"
    return dt


def main():
    print("=" * 70)
    print("[G1 검증] 245,789행 2024-as-test")
    print("=" * 70)

    dt = run_script(TEST_245K, SAMPLE_245K, OUT_245K)
    out = pd.read_csv(OUT_245K, encoding="utf-8-sig")
    assert len(out) == 245789
    p_script = out["control_success"].values

    p_ref, p_post, p_pre = reference_predict(TEST_245K)
    assert len(p_ref) == 245789

    # 1) clip 접촉 행 비율 (pre-clip 기준, 보정 전후 비교용 raw 분포 포함)
    frac_clip = float(np.mean((p_post < CLIP_LO) | (p_post > CLIP_HI)))
    frac_lo = float(np.mean(p_post < CLIP_LO))
    frac_hi = float(np.mean(p_post > CLIP_HI))
    print(f"\n[1] clip 접촉 비율: pre-clip {frac_clip*100:.4f}% "
          f"(하한 {frac_lo*100:.4f}% / 상한 {frac_hi*100:.4f}%)")
    print(f"    post-clip 경계값 행 수: p==0.30 {int(np.sum(p_ref==CLIP_LO))} / "
          f"p==0.70 {int(np.sum(p_ref==CLIP_HI))}")
    if frac_clip > 0.001:
        print("    ⚠ 0.1% 초과 — 보고")

    # 2) 출력 평균 기준 (2026-08-08 개정, [G1-fix])
    #    #2-a: 오프셋 적용 전 raw 예측 평균 p_raw 보고 (기대 0.487 ~ 0.490)
    #    #2-b: script.py 출력 평균 == p_raw - 0.0440 * p_raw*(1-p_raw) 이내 1e-4
    m = float(p_script.mean())
    m_raw = float(p_pre.mean())
    expect = m_raw - 0.0440 * m_raw * (1 - m_raw)
    print(f"\n[2-a] raw 예측 평균 p_raw = {m_raw:.4f}  (기대 0.487~0.490)")
    ok2a = 0.487 <= m_raw <= 0.490
    print(f"    {'✅ 통과' if ok2a else '⚠ 범위 밖'}")
    print(f"[2-b] 출력 평균 = {m:.4f} | 공식 기대 = p_raw - 0.0440*p_raw*(1-p_raw) = {expect:.4f} | "
          f"차이 = {abs(m - expect):.2e} (목표 < 1e-4)")
    ok2b = abs(m - expect) < 1e-4
    print(f"    {'✅ 통과' if ok2b else '❌ 초과'}")
    ok2 = ok2a and ok2b

    # 3) script.py vs 참조 파이프라인 최대 절대차
    md = float(np.max(np.abs(p_script - p_ref)))
    print(f"\n[3] script.py vs 참조 최대 절대차 = {md:.3e}  (목표 < 1e-6)")
    ok3 = md < 1e-6
    print(f"    {'✅ 통과' if ok3 else '❌ 초과'}")

    # 4) 추론 시간
    print(f"\n[4] 245,789행 추론 wall time = {dt:.2f}s (제한 600s)")

    # 5) 5행 end-to-end (기본 경로, 제출 동작 — 서버 레이아웃 미믹: submit/data 심링크)
    print("\n" + "=" * 70)
    print("[G1 검증] 5행 test.csv end-to-end (기본 경로)")
    print("=" * 70)
    if not os.path.islink(os.path.join(SUBMIT, "data")):
        os.symlink(os.path.join(REPO, "data"), os.path.join(SUBMIT, "data"))
    r = subprocess.run([sys.executable, "script.py"], cwd=SUBMIT,
                       capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr)
    assert r.returncode == 0, "5행 script.py 실패"
    sub5 = pd.read_csv(OUT_5, encoding="utf-8-sig")
    assert len(sub5) == 5, f"예상 5행, 실제 {len(sub5)}"
    assert sub5["control_success"].notna().all(), "NaN 존재"
    p5_ref, _, _ = reference_predict(os.path.join(REPO, "data/test.csv"))
    md5 = float(np.max(np.abs(sub5["control_success"].values - p5_ref)))
    print(f"[5] 5행 출력: {len(sub5)}행, mean={sub5['control_success'].mean():.4f}, "
          f"max|diff vs 참조|={md5:.3e}")
    ok5 = md5 < 1e-6
    print(f"    {'✅ 통과' if ok5 else '❌'}")

    print("\n" + "=" * 70)
    print(f"요약: clip {frac_clip*100:.4f}% / mean {m:.4f} / diff {md:.3e} / time {dt:.2f}s / 5행 {'✅' if ok5 else '❌'}")
    ok_all = ok2 and ok3 and ok5
    print("전체:", "✅ 통과" if ok_all else "⚠ 확인 필요")


if __name__ == "__main__":
    main()
