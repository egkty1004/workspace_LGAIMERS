#!/usr/bin/env python3
"""screen_platoon_3b2.py — count_platoon_3b2_same 피처 LGB-only 빠른 스크리닝 (2026-08-09)

기준선(F3: base + platoon + count_state) vs 후보(F3 + count_platoon_3b2_same)를
5시드 × 4폴드 로짓 평균으로 primary + R-only(r2022/r2023/r2024) BSS를 측정하고
기준선 대비 delta를 계산한다. 제출 가능성 판단용.

후보 피처: count_platoon_3b2_same =
    (balls_before==3) & (strikes_before==2) & (pitcher_hand==batter_hand)  → int8 이진
  (EDA: recent(23-24) 3-2+동손 0.4365 vs 0-1+이손 0.5205, Δ-8.4pp,
   버킷 BSS 25.7~39.3 — 단, F3가 이미 count_state(12범주)+platoon(4범주)를
   포함하므로 실제 추가 이득은 더 작을 수 있음)

게이트: primary delta ≥ +20 & R-only 2/3 개선 → "채택 후보" / 미달 → "기각"
출력: experiments/screen_platoon_3b2.json + 요약 테이블 (stdout)
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
    """후보 피처를 df에 추가. 행 단위 변환, 누수 없음."""
    if name == "count_platoon_3b2_same":
        t[name] = ((t["balls_before"] == 3) & (t["strikes_before"] == 2)
                   & (t["pitcher_hand"] == t["batter_hand"])).astype("int8")
    else:
        raise ValueError(name)


def run_config(train, feats, cats, folds):
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
    print("[screen] count_platoon_3b2_same LGB-only 스크리닝 (5시드 × 4폴드)", flush=True)
    train, _ = common.load_train()
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

    CAND = "count_platoon_3b2_same"
    configs = {"baseline": (list(base_feats), []),
               "candidate": (list(base_feats) + [CAND], [CAND])}

    results = {}
    for cfg_name, (feats, extra) in configs.items():
        t1 = time.time()
        t = train.copy()
        for c in extra:
            add_candidate(t, c)
        results[cfg_name] = run_config(t, feats, cats, folds)
        for fn, r in results[cfg_name].items():
            print(f"  [{cfg_name}] {fn}: BSS={r['bss']:.1f} mean={r['pred_mean']:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    # ── delta 표 ──
    base = results["baseline"]
    cand = results["candidate"]
    order = ["primary", "r2022", "r2023", "r2024"]
    deltas = {fn: cand[fn]["bss"] - base[fn]["bss"] for fn in order}

    print("\n" + "=" * 78, flush=True)
    print("스크리닝 결과 (BSS, delta vs baseline)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'config':<12s} " + " ".join(f"{fn:>9s}" for fn in order), flush=True)
    for cfg_name in ["baseline", "candidate"]:
        row = " ".join(f"{results[cfg_name][fn]['bss']:>9.1f}" for fn in order)
        print(f"  {cfg_name:<12s} {row}", flush=True)
    print("-" * 78, flush=True)
    drow = " ".join(f"{deltas[fn]:>+9.1f}" for fn in order)
    print(f"  {'Δcandidate':<12s} {drow}", flush=True)

    # R-only 개선 카운트 (r2022/r2023/r2024)
    improved = sum(1 for fn in ["r2022", "r2023", "r2024"]
                   if cand[fn]["bss"] > base[fn]["bss"] + 1e-9)
    print(f"  R-only 개선: candidate = {improved}/3", flush=True)

    # ── 게이트 판정 ──
    gate_primary = deltas["primary"] >= 20.0
    gate_r = improved >= 2
    verdict = "채택 후보" if (gate_primary and gate_r) else "기각"
    print("-" * 78, flush=True)
    print(f"  게이트: primary delta {deltas['primary']:+.1f} "
          f"({'PASS' if gate_primary else 'FAIL'} ≥ +20) | "
          f"R-only {improved}/3 ({'PASS' if gate_r else 'FAIL'} ≥ 2/3)", flush=True)
    print(f"  판정: {verdict}", flush=True)

    results["_meta"] = dict(
        seeds=SEEDS, folds=order, candidate=CAND,
        delta={fn: deltas[fn] for fn in order},
        gate=dict(primary_ge_20=bool(gate_primary), r_only_ge_2of3=bool(gate_r),
                  verdict=verdict),
        total_time=time.time() - t0)
    out_path = "experiments/screen_platoon_3b2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: {out_path} (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
