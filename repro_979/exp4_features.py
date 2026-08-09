#!/usr/bin/env python3
"""
[Exp 4] 피처 엔지니어링 ablation (AGENTS.md §6 Exp 4, 2026-08-07)

V0 기준선에 G1~G6 각 그룹을 단독 적용하여
  primary(≤2023 전체 → 2024 전체) / r2024(R-only ≤2023 → 2024)
두 폴드에서 raw Score를 측정한다.

그룹:
  G1 count_state:  balls_before*3 + strikes_before  (12범주 카테고리)
  G2 폼 차분:      prev1/prev3/prev5 - overall (success_rate, middle_rate 각각, 총 6개)
  G3 플래툰:       pitcher_hand*2 + batter_hand (4범주 카테고리)
  G4 로짓 변환:    asof_*_rate 전체 → log(p/(1-p)), clip(1e-3, 1-1e-3)  [원본 대체]
  G5 베이지안 스무딩: (n*p_raw + m*p_prior)/(n+m), m=50/200/500  [원본 대체]
      p_prior = 폴드 학습 행의 rate 평균 (test/검증 미참조)
  G6 pitcher_id OOF target encoding: 학습 행 내부 5-fold OOF, 검증행은 전체 학습 매핑,
      미등장 id는 전역 평균

판정: primary raw가 V0 대비 +20 이상 → 채택 후보.
채택 후보 전체를 동시 적용한 조합(COMBO)도 1회 측정.
결과: experiments/exp4_results.json
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import common

RATE_SUFFIX = "_rate"

# G5 스무딩 대상: (rate 컬럼, n 컬럼) — "주요 rate"만 (prev game rate는 n 불명확 → 제외)
SMOOTH_SPECS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]


def add_g1(t, feats, cats):
    """G1: count_state 12범주 카테고리 (추가)."""
    t["count_state"] = (t["balls_before"] * 3 + t["strikes_before"]).astype("category")
    return feats + ["count_state"], cats + ["count_state"]


def add_g2(t, feats):
    """G2: 폼 차분 6개 (추가): prev{k} - overall, success_rate/middle_rate."""
    for suffix, overall in [
        ("success_rate", "asof_pitcher_success_rate"),
        ("middle_rate", "asof_pitcher_middle_rate"),
    ]:
        for k in (1, 3, 5):
            name = f"form_diff_prev{k}_{suffix}"
            t[name] = (t[f"asof_pitcher_prev{k}_game_{suffix}"] - t[overall]).astype("float32")
            feats = feats + [name]
    return feats


def add_g3(t, feats, cats):
    """G3: 플래툰 pitcher_hand*2 + batter_hand → 4범주 카테고리 (추가)."""
    t["platoon"] = (t["pitcher_hand"] * 2 + t["batter_hand"]).astype("category")
    return feats + ["platoon"], cats + ["platoon"]


def add_g4(t, feats):
    """G4: asof_*_rate 전체 로짓 변환 (원본 대체). clip(1e-3, 1-1e-3)."""
    rate_cols = sorted(
        c for c in t.columns
        if c.startswith("asof_") and c.endswith(RATE_SUFFIX)
    )
    for c in rate_cols:
        p = t[c].clip(1e-3, 1 - 1e-3)
        t[f"logit_{c}"] = np.log(p / (1 - p)).astype("float32")
        t.drop(columns=[c], inplace=True)
    feats = [f"logit_{c}" if c in rate_cols else c for c in feats]
    return feats


def add_g5(t, feats, m, tr_mask):
    """G5: 베이지안 스무딩 (원본 대체). p_prior는 학습 행(tr_mask) 평균."""
    for rate_col, n_col in SMOOTH_SPECS:
        n = t[n_col].astype("float64")
        p_raw = t[rate_col].astype("float64")
        p_prior = float(t.loc[tr_mask, rate_col].mean())
        smoothed = (n * p_raw + m * p_prior) / (n + m)
        smoothed = smoothed.where(n.notna(), p_prior).astype("float32")
        t[f"smooth_{rate_col}"] = smoothed
        t.drop(columns=[rate_col], inplace=True)
    repl = {c: f"smooth_{c}" for c, _ in SMOOTH_SPECS}
    feats = [repl.get(c, c) for c in feats]
    return feats


def add_g6(t, feats, tr_mask, va_mask, seed=42):
    """G6: pitcher_id OOF target encoding.
    학습 행: 내부 5-fold OOF. 검증 행: 전체 학습 매핑. 미등장 id: 전역 평균."""
    ids, y = t["pitcher_id"], t[common.TARGET]
    tr_idx = np.where(tr_mask)[0]
    oof = np.full(len(t), np.nan, dtype="float64")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i_tr, i_va in kf.split(tr_idx):
        g, v = tr_idx[i_tr], tr_idx[i_va]
        m = y.iloc[g].groupby(ids.iloc[g]).mean()
        oof[v] = ids.iloc[v].map(m).values
    global_mean = float(y.iloc[tr_idx].mean())
    # 검증 행: 전체 학습 매핑
    va_idx = np.where(va_mask)[0]
    if len(va_idx):
        full_map = y.iloc[tr_idx].groupby(ids.iloc[tr_idx]).mean()
        oof[va_idx] = ids.iloc[va_idx].map(full_map).fillna(global_mean).values
    oof[np.isnan(oof)] = global_mean
    t["pitcher_id_oof"] = oof.astype("float32")
    return feats + ["pitcher_id_oof"]


def apply_group(t, feats, cats, group, tr_mask, va_mask):
    """그룹 적용. 반환: (feats, cats)."""
    if group == "G1":
        return add_g1(t, feats, cats)
    if group == "G2":
        return add_g2(t, feats), cats
    if group == "G3":
        return add_g3(t, feats, cats)
    if group == "G4":
        return add_g4(t, feats), cats
    if group.startswith("G5"):
        m = int(group.split("_m")[1])
        return add_g5(t, feats, m, tr_mask), cats
    if group == "G6":
        return add_g6(t, feats, tr_mask, va_mask), cats
    raise ValueError(group)


def eval_fold(t, feats, cats, tr_mask, va_mask, fold_name, group):
    """한 폴드 학습 + raw Score."""
    print(f"  [{fold_name}] 학습 {group}: n_train={int(tr_mask.sum())} "
          f"n_val={int(va_mask.sum())} n_feat={len(feats)}", flush=True)
    X_tr, y_tr = t.loc[tr_mask, feats], t.loc[tr_mask, common.TARGET]
    X_va, y_va = t.loc[va_mask, feats], t.loc[va_mask, common.TARGET]
    t0 = time.time()
    model = common.train_model(X_tr, y_tr, X_va, y_va, cat_cols=cats)
    train_time = time.time() - t0
    p = model.predict(X_va, num_iteration=model.best_iteration)
    s = common.score(p, y_va.values)
    res = dict(raw=float(s), best_iter=model.best_iteration,
               pred_mean=float(p.mean()), train_time=float(train_time),
               n_train=int(len(X_tr)), n_val=int(len(X_va)), n_feat=len(feats))
    print(f"    raw={s:7.1f}  best_iter={model.best_iteration}  "
          f"mean={p.mean():.4f}  ({train_time:.1f}s)", flush=True)
    return res


def main():
    t0 = time.time()
    print("[Exp 4] 피처 ablation (V0 + G1~G6 단독, primary/r2024)", flush=True)
    train, base_feats = common.load_train()
    print(f"train: {train.shape}  base_features: {len(base_feats)}", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    groups = ["V0", "G1", "G2", "G3", "G4",
              "G5_m50", "G5_m200", "G5_m500", "G6"]
    results = {g: {} for g in groups}

    # ── V0 기준선 ──
    for fn, (tr_m, va_m) in folds.items():
        results["V0"][fn] = eval_fold(train, list(base_feats), list(common.CAT_COLS),
                                      tr_m, va_m, fn, "V0")

    # ── 그룹 단독 ──
    for g in groups[1:]:
        for fn, (tr_m, va_m) in folds.items():
            t = train.copy()
            feats = list(base_feats)
            cats = list(common.CAT_COLS)
            feats, cats = apply_group(t, feats, cats, g, tr_m, va_m)
            results[g][fn] = eval_fold(t, feats, cats, tr_m, va_m, fn, g)

    # ── 채택 후보 (primary raw ≥ V0 + 20) ──
    v0_p = results["V0"]["primary"]["raw"]
    candidates = []
    for g in groups[1:]:
        gain = results[g]["primary"]["raw"] - v0_p
        results[g]["primary"]["gain_vs_V0"] = gain
        if gain >= 20.0:
            candidates.append(g)
    print(f"\n채택 후보 (primary raw ≥ V0+20): {candidates}", flush=True)

    # ── COMBO: 채택 후보 전체 동시 적용 ──
    results["COMBO"] = {}
    if candidates:
        for fn, (tr_m, va_m) in folds.items():
            t = train.copy()
            feats = list(base_feats)
            cats = list(common.CAT_COLS)
            for g in candidates:
                feats, cats = apply_group(t, feats, cats, g, tr_m, va_m)
            results["COMBO"][fn] = eval_fold(t, feats, cats, tr_m, va_m, fn,
                                             f"COMBO({'+'.join(candidates)})")
    else:
        print("채택 후보 없음 → COMBO 미실행", flush=True)

    # ── 요약 표 ──
    print("\n" + "=" * 78, flush=True)
    print("Exp 4 요약 (raw Score)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'group':<10s} {'primary':>9s} {'Δpri':>7s} {'r2024':>9s} {'Δr24':>7s}  후보", flush=True)
    for g in groups:
        r = results.get(g, {})
        if not r:
            continue
        pri = r["primary"]["raw"]
        r24 = r["r2024"]["raw"]
        d_pri = pri - v0_p
        d_r24 = r24 - results["V0"]["r2024"]["raw"]
        cand = "★" if (g in candidates) else ""
        print(f"  {g:<10s} {pri:>9.1f} {d_pri:>+7.1f} {r24:>9.1f} {d_r24:>+7.1f}  {cand}", flush=True)
    if candidates:
        c = results["COMBO"]
        print(f"  {'COMBO':<10s} {c['primary']['raw']:>9.1f} "
              f"{c['primary']['raw'] - v0_p:>+7.1f} {c['r2024']['raw']:>9.1f} "
              f"{c['r2024']['raw'] - results['V0']['r2024']['raw']:>+7.1f}  ★", flush=True)

    results["_meta"] = dict(candidates=candidates, v0_primary=v0_p,
                            total_time=time.time() - t0)
    with open("experiments/exp4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/exp4_results.json  (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
