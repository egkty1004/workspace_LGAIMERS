#!/usr/bin/env python3
"""
[F1] 오프셋 공간 불일치 수정 — delta_prob 기반 로짓 보정 스윕 (2026-08-08)

문제: 기존 보정 p_final = sigmoid(logit(p) - 0.0077)는 로짓 공간 상수라
      확률 공간 실현 평균 shift가 ≈ -0.0019 (p(1-p)≈0.25 가 곱해짐)에 불과.
      의도한 확률 공간 shift delta_prob를 얻으려면
          c_logit = delta_prob / (p_mean * (1 - p_mean))
      로 환산해 로짓 공간에 적용해야 한다.

방법: F3 구성(V0+G3+G1, seed 42~51 로짓 평균), 4폴드(primary/r2022/r2023/r2024).
      폴드별 홀드아웃 예측 평균 p_mean으로 c_logit 계산 (라벨 미사용).
      delta_prob ∈ {-0.0050, -0.0077, -0.0100, -0.0112} 스윕.
      참조: raw / 기존 로짓 상수(-0.0077) 방식.
      합산 최적 선택 시 primary/r2024(2025와 동일 구조)에 높은 가중.

결과: experiments/f1_results.json
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common

SEEDS = list(range(42, 52))
DELTA_PROB_SWEEP = [-0.0050, -0.0077, -0.0100, -0.0112]
OLD_LOGIT_DELTA = -0.0077  # 기존 방식 참조


def main():
    t0 = time.time()
    print("[F1] delta_prob 기반 오프셋 스윕", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feats = common.get_feature_cols(
        pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = common.CAT_COLS + common.F3_EXTRA
    print(f"train: {train.shape}  features: {len(feats)}  cats: {cats}", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }
    order = ["primary", "r2022", "r2023", "r2024"]

    results = {}
    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        preds = []
        for seed in SEEDS:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            preds.append(m.predict(X_va, num_iteration=m.best_iteration))
            print(f"  [{fn}] seed={seed} best_iter={m.best_iteration}", flush=True)
        z = np.mean([common.logit(p) for p in preds], axis=0)
        p = common.sigmoid(z)
        p_mean = float(p.mean())

        r = dict(
            raw=float(common.score(p, yv)),
            old_logit_delta=float(common.score(common.sigmoid(z + OLD_LOGIT_DELTA), yv)),
            p_mean=p_mean,
            delta_prob={},
        )
        for dp in DELTA_PROB_SWEEP:
            c_logit = dp / (p_mean * (1 - p_mean))
            p_shift = common.sigmoid(z + c_logit)
            r["delta_prob"][str(dp)] = dict(
                c_logit=float(c_logit),
                score=float(common.score(p_shift, yv)),
                pred_mean=float(p_shift.mean()),
            )
        results[fn] = r
        print(f"  [{fn}] raw={r['raw']:.1f} old_logit(-0.0077)={r['old_logit_delta']:.1f} "
              f"p_mean={p_mean:.4f}", flush=True)
        for dp in DELTA_PROB_SWEEP:
            rr = r["delta_prob"][str(dp)]
            print(f"      dp={dp:+.4f} -> c_logit={rr['c_logit']:+.4f} "
                  f"score={rr['score']:.1f} mean={rr['pred_mean']:.4f}", flush=True)

    print("\n=== 요약 (4폴드 합산/평균) ===", flush=True)
    summary = {}
    for key in ["raw", "old_logit_delta"] + [f"dp_{dp}" for dp in DELTA_PROB_SWEEP]:
        if key.startswith("dp_"):
            dp = float(key[3:])
            vals = [results[fn]["delta_prob"][str(dp)]["score"] for fn in order]
        else:
            vals = [results[fn][key] for fn in order]
        summary[key] = vals
        print(f"  {key:>20s}  sum={sum(vals):8.1f}  mean={np.mean(vals):7.1f}  "
              f"per={['%.1f' % v for v in vals]}", flush=True)

    results["_meta"] = dict(seeds=SEEDS, sweep=DELTA_PROB_SWEEP,
                            old_logit_delta=OLD_LOGIT_DELTA,
                            summary=summary, total_time=time.time() - t0)
    with open("experiments/f1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/f1_results.json  (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
