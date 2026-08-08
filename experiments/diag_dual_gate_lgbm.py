"""diag_dual_gate_lgbm.py — LGBM 기반 이중 게이트(2023+2024) 후보 재검증 하네스.

배경: HGB 시드 노이즈 ±100으로 FE 판정 불가 → LGBM(±12)으로 재검증.
모든 후보를 3시드 중앙값 기준으로 평가. G23은 break year(BSS 0 클램프)라
G24 중앙값을 주 판정으로, G23 Brier는 참고용.

조합: base / interact(참조) / feat_eng 후보 8종 (B1,D1D2,A2,E2,E3,C2,C1,A1c)

실행: python diag_dual_gate_lgbm.py [--seeds 42,7,123] 2>&1 | tee backup/diag_dual_gate_lgbm.log
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import _to_category, InteractionAdder
from feat_eng import (
    DropDupCols, SeasonProgress, InningNorm, FormTrend, HandMatch,
    ScoreAbs, ReverseCount, CountCondition, SeasonCat,
)
from train_lgbm import LGBM_KWARGS, compute_bss, load_data

RESULTS_CSV = "backup/dual_gate_lgbm_results.csv"


def build_pipeline(variant, random_state=42):
    steps = []
    if variant == "interact":
        steps.append(("interact", InteractionAdder()))
    elif variant == "B1":
        steps.append(("b1", DropDupCols()))
    elif variant == "D1D2":
        steps.append(("d1", SeasonProgress()))
        steps.append(("d2", InningNorm()))
    elif variant == "A2":
        steps.append(("a2", FormTrend()))
    elif variant == "E2":
        steps.append(("e2", HandMatch()))
    elif variant == "E3":
        steps.append(("e3", ScoreAbs()))
    elif variant == "C2":
        steps.append(("c2", ReverseCount()))
    elif variant == "C1":
        steps.append(("c1", CountCondition()))
    elif variant == "A1c":
        steps.append(("a1c", SeasonCat()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", lgb.LGBMClassifier(**LGBM_KWARGS, random_state=random_state)))
    return Pipeline(steps)


VARIANTS = ["base", "interact", "B1", "D1D2", "A2", "E2", "E3", "C2", "C1", "A1c"]


def run_variant(variant, train, features, seed):
    """단일 변형 × 단일 시드 × 이중 게이트 → dict."""
    out = {"variant": variant, "seed": seed}
    for gate_name, tr_years, val_year in [("G23", (2019, 2022), 2023),
                                          ("G24", (2019, 2023), 2024)]:
        tr_mask = train["season"].between(*tr_years)
        va_mask = train["season"] == val_year
        X_tr, y_tr = train.loc[tr_mask, features], train.loc[tr_mask, "control_success"]
        X_va, y_va = train.loc[va_mask, features], train.loc[va_mask, "control_success"]

        pipe = build_pipeline(variant, random_state=seed)
        X_tr_t = pipe[:-1].fit_transform(X_tr)
        X_va_t = pipe[:-1].transform(X_va)
        X_tr_c = _to_category(X_tr_t)
        X_va_c = _to_category(X_va_t)
        # fit은 90%, eval은 10% (early stopping 올바르게 동작)
        Xt, Xv_, yt, yv_ = train_test_split(X_tr_c, y_tr, test_size=0.1, random_state=seed)
        t0 = time.time()
        pipe.fit(Xt, yt, clf__eval_set=[(Xv_, yv_)],
                 clf__eval_metric="binary_logloss")
        ft = time.time() - t0
        p = pipe.predict_proba(X_va_c)[:, 1]
        bss, r, brier = compute_bss(y_va, p)
        n_est = pipe.named_steps["clf"].best_iteration_ or pipe.named_steps["clf"].n_estimators
        out[f"{gate_name}_bss"] = bss
        out[f"{gate_name}_brier"] = brier
        out[f"{gate_name}_r"] = r
        out[f"{gate_name}_niter"] = n_est
        out[f"{gate_name}_fit"] = round(ft, 1)
    out["n_feat"] = pipe.named_steps["clf"].n_features_in_
    return out


def summarize(train, features, seeds):
    """모든 변형 × 시드 실행 후 중앙값 요약."""
    results = []
    for v in VARIANTS:
        for s in seeds:
            row = run_variant(v, train, features, s)
            results.append(row)
            print(f"[{v:>8}|seed={s:>3}] n_feat={row['n_feat']:2d} n_est={row['G24_niter']:>3} "
                  f"| G23 BSS={row['G23_bss']:6.2f} Brier={row['G23_brier']:.5f} "
                  f"| G24 BSS={row['G24_bss']:7.2f} Brier={row['G24_brier']:.5f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7,123")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    train, features = load_data()
    df = summarize(train, features, seeds)

    base = df[df.variant == "base"]
    base_g24 = base.groupby("variant")["G24_bss"].median().iloc[0]
    print("\n" + "=" * 92)
    print(f"LGBM 이중 게이트 요약 (base G24 중앙값 = {base_g24:.2f})")
    print("=" * 92)
    print(f"{'변형':>8} | {'G24 med':>8} | {'Δ vs base':>8} | {'G23 Brier med':>12} | "
          f"{'G24 시드별':<24} | 판정")
    for v in VARIANTS:
        sub = df[df.variant == v]
        g24 = sub["G24_bss"].median()
        g23_b = sub["G23_brier"].median()
        seeds_val = ",".join(f"{x:.0f}" for x in sub["G24_bss"].values)
        d = g24 - base_g24
        if v == "base":
            verdict = "기준"
        elif d >= 8:
            verdict = "후보 ⭐"
        elif d >= 3:
            verdict = "약간 개선"
        elif d > -3:
            verdict = "무변화"
        else:
            verdict = "하락"
        print(f"{v:>8} | {g24:>8.2f} | {d:>+8.1f} | {g23_b:>12.5f} | "
              f"{seeds_val:<24} | {verdict}")


if __name__ == "__main__":
    main()
