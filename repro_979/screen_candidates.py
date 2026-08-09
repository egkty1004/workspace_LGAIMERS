#!/usr/bin/env python3
"""screen_candidates.py — 후보 피처 LGB-only 스크리닝 (계획 Todo 2, 2026-08-09)

기준선(F3) + 후보 피처 각각을 LGB F3에 단독 추가하여 5시드 × 4폴드 로짓 평균으로
primary + R-only(r2022/r2023/r2024) BSS를 측정하고 기준선 대비 delta를 계산한다.

후보:
  F1: F_post2023 = (game_type=='F') & (season>=2023)  이진
  F2: three_balls = (balls_before==3)                  이진
  (F3: platoon_detail — common.add_platoon_feature가 이미 4범주(3,4,5,6) 생성 → 기각)

주의: R-only 폴드에서 F_post2023은 항상 0(상수) → LGB가 무시, delta≈0 예상.
      F1은 primary 폴드에서만 의미가 있다.

출력: experiments/screen_results.json + 요약 테이블 (stdout)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEEDS = list(range(42, 47))  # 5시드
N_FOLD = 4


def add_candidate(t, name):
    """후보 피처를 df에 추가. 이름 그대로 컬럼 생성 (누수 없음, 행 단위 변환)."""
    if name == "F_post2023":
        t["F_post2023"] = ((t["game_type"] == "F") & (t["season"] >= 2023)).astype("int8")
    elif name == "three_balls":
        t["three_balls"] = (t["balls_before"] == 3).astype("int8")
    else:
        raise ValueError(name)


def run_config(train, feats, cats, folds, extra):
    """한 구성(기준선 또는 후보+1)에 대해 4폴드 BSS. 5시드 로짓 평균."""
    out = {}
    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        zs = []
        for seed in SEEDS:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            zs.append(common.logit(m.predict(X_va, num_iteration=m.best_iteration)))
        z = np.mean(zs, axis=0)
        p = common.sigmoid(z)
        out[fn] = dict(bss=float(common.score(p, yv)), pred_mean=float(p.mean()))
    return out


def main():
    t0 = time.time()
    print("[screen] 후보 피처 LGB-only 스크리닝 (5시드 × 4폴드)", flush=True)
    train, base_feats = common.load_train()
    common.preprocess_for_submission(train)
    base_feats = common.get_feature_cols(
        pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = common.CAT_COLS + common.F3_EXTRA
    print(f"train: {train.shape} | base_features: {len(base_feats)} | seeds: {SEEDS}",
          flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    configs = {"baseline": (list(base_feats), [])}
    for cand in ["F_post2023", "three_balls"]:
        configs[cand] = (list(base_feats) + [cand], [cand])

    results = {}
    for cfg_name, (feats, extra) in configs.items():
        t1 = time.time()
        t = train.copy()
        for c in extra:
            add_candidate(t, c)
        results[cfg_name] = run_config(t, feats, cats, folds, extra)
        for fn, r in results[cfg_name].items():
            print(f"  [{cfg_name}] {fn}: BSS={r['bss']:.1f} mean={r['pred_mean']:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    # ── delta 표 ──
    base = results["baseline"]
    order = ["primary", "r2022", "r2023", "r2024"]
    print("\n" + "=" * 78, flush=True)
    print("스크리닝 결과 (BSS, delta vs baseline)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'config':<12s} " + " ".join(f"{fn:>9s}" for fn in order), flush=True)
    for cfg_name in ["baseline", "F_post2023", "three_balls"]:
        row = " ".join(f"{results[cfg_name][fn]['bss']:>9.1f}" for fn in order)
        print(f"  {cfg_name:<12s} {row}", flush=True)
    print("-" * 78, flush=True)
    for cfg_name in ["F_post2023", "three_balls"]:
        deltas = " ".join(
            f"{results[cfg_name][fn]['bss'] - base[fn]['bss']:>+9.1f}" for fn in order)
        print(f"  {'Δ'+cfg_name:<12s} {deltas}", flush=True)

    # R-only 개선 카운트 (r2022/r2023/r2024)
    for cfg_name in ["F_post2023", "three_balls"]:
        improved = sum(
            1 for fn in ["r2022", "r2023", "r2024"]
            if results[cfg_name][fn]["bss"] > base[fn]["bss"] + 1e-9)
        print(f"  R-only 개선: {cfg_name} = {improved}/3", flush=True)

    results["_meta"] = dict(seeds=SEEDS, folds=order,
                            gate="primary +20 & R-only 2/3 (F1: primary 단독)",
                            total_time=time.time() - t0)
    with open("experiments/screen_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/screen_results.json (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
