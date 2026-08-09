#!/usr/bin/env python3
"""
[E3] 최종 구성 확정 실험 (2026-08-07)

구성:
  F0 = V0 (기준선, seed 42)
  F1 = V0 + G3 (platoon, seed 42)
  F2 = V0 + G3 + 10시드 앙상블 (로짓 평균)
  F3 = F2 + G1 (count_state)  [G1은 +10 수준이나 방향 일관]
  F4 = F2 + E2 채택 후보 → E2 채택 후보 없음 → F4 = F2 (재사용)

폴드: primary(≤2023→2024 전체) + R-only 3폴드(r2022/r2023/r2024)
각 구성에 고정 오프셋 delta=-0.0077(로짓) 적용 값 병기.
결과: experiments/e3_results.json
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import common

DELTA = -0.0077  # 고정 로짓 오프셋 (test 미참조 상수)
SEEDS = list(range(42, 52))


def shift_logit(p, delta):
    return common.sigmoid(common.logit(p) + delta)


def main():
    t0 = time.time()
    print("[E3] 최종 구성 확정 (F0~F3 + delta 병기)", flush=True)
    train, base_feats = common.load_train()
    print(f"train: {train.shape}  base_features: {len(base_feats)}", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }
    fold_order = ["primary", "r2022", "r2023", "r2024"]

    # 피처 구성 준비
    t_g3 = train.copy()
    common.add_platoon_feature(t_g3)
    t_g3["count_state"] = (t_g3["balls_before"] * 3 + t_g3["strikes_before"]).astype("category")

    CFG = {
        "F0": dict(t=train, feats=list(base_feats), cats=list(common.CAT_COLS), seeds=[42]),
        "F1": dict(t=t_g3, feats=list(base_feats) + ["platoon"],
                   cats=list(common.CAT_COLS) + ["platoon"], seeds=[42]),
        "F2": dict(t=t_g3, feats=list(base_feats) + ["platoon"],
                   cats=list(common.CAT_COLS) + ["platoon"], seeds=SEEDS),
        "F3": dict(t=t_g3, feats=list(base_feats) + ["platoon", "count_state"],
                   cats=list(common.CAT_COLS) + ["platoon", "count_state"], seeds=SEEDS),
    }
    # F4 = F2 (E2 채택 후보 없음)
    CFG["F4"] = CFG["F2"]

    results = {}
    for cname in ["F0", "F1", "F2", "F3", "F4"]:
        cfg = CFG[cname]
        results[cname] = {}
        print("\n" + "=" * 70, flush=True)
        print(f"[{cname}] features={len(cfg['feats'])} seeds={cfg['seeds'][0]}.."
              f"{cfg['seeds'][-1]} ({len(cfg['seeds'])}개)", flush=True)
        print("=" * 70, flush=True)
        for fn in fold_order:
            tr_m, va_m = folds[fn]
            X_tr, y_tr = cfg["t"].loc[tr_m, cfg["feats"]], cfg["t"].loc[tr_m, common.TARGET]
            X_va, y_va = cfg["t"].loc[va_m, cfg["feats"]], cfg["t"].loc[va_m, common.TARGET]
            yv = y_va.values
            preds = []
            for seed in cfg["seeds"]:
                params = dict(common.PARAMS)
                params["seed"] = seed
                dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cfg["cats"])
                dva = lgb.Dataset(X_va, y_va, categorical_feature=cfg["cats"], reference=dtr)
                model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                preds.append(model.predict(X_va, num_iteration=model.best_iteration))
            if len(preds) == 1:
                p = preds[0]
            else:
                p = common.sigmoid(np.mean([common.logit(x) for x in preds], axis=0))
            s_raw = common.score(p, yv)
            s_delta = common.score(shift_logit(p, DELTA), yv)
            results[cname][fn] = dict(raw=float(s_raw), delta=float(s_delta),
                                      pred_mean_raw=float(p.mean()),
                                      pred_mean_delta=float(shift_logit(p, DELTA).mean()),
                                      n_train=int(len(X_tr)), n_val=int(len(X_va)))
            print(f"  {fn:<8s} raw={s_raw:7.1f}  delta(-0.0077)={s_delta:7.1f}  "
                  f"Δ={s_delta - s_raw:+6.1f}  mean={p.mean():.4f}", flush=True)

    # ── 요약 ──
    print("\n" + "=" * 90, flush=True)
    print("E3 요약 (raw / delta=-0.0077)", flush=True)
    print("=" * 90, flush=True)
    hdr = f"  {'cfg':<4s}" + "".join(f"{fn:>22s}" for fn in fold_order)
    print(hdr, flush=True)
    for cname in ["F0", "F1", "F2", "F3", "F4"]:
        row = f"  {cname:<4s}"
        for fn in fold_order:
            r = results[cname][fn]
            row += f"  {r['raw']:6.1f}/{r['delta']:6.1f}"
        print(row, flush=True)

    results["_meta"] = dict(delta=DELTA, seeds=SEEDS, fold_order=fold_order,
                            total_time=time.time() - t0)
    with open("experiments/e3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/e3_results.json  (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
