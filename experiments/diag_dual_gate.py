"""diag_dual_gate.py — 이중 게이트(2023+2024) 검증 하네스.

G23 = 2019~2022 학습 → 2023 홀드아웃 BSS (r = y_2023.mean())
G24 = 2019~2023 학습 → 2024 홀드아웃 BSS (r = y_2024.mean())

TDD 코어: base(47, interact 없음)는 G24 = 439.00 ± 0.5 재현해야 함 — 아니면 exit 2.

모드:
  --variant <name>  : 단일 변형 실행
  --all             : 고정 순서 전체 실행
  --sanity          : 데이터 방향성 검사만 (feat_eng.run_sanity)
  --assert          : 결과 CSV 기준 채택 판정 (마진 M=8)

변형 목록 (우선순위순): base, interact(참조 전용), B1, D1D2, A2, E2, E3, C2, C1, A1c
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import _to_category, InteractionAdder
from feat_eng import (
    DropDupCols, SeasonProgress, InningNorm, FormTrend, HandMatch,
    ScoreAbs, ReverseCount, CountCondition, SeasonCat, run_sanity,
)
from train_hgb import HGB_KWARGS, compute_bss, load_data

RESULTS_CSV = "backup/dual_gate_results.csv"
BASE_G24_REF = 439.00
MARGIN = 8


def build_pipeline(variant):
    """변형별 파이프라인 (base/interact/후보)."""
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
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


VARIANTS = ["base", "interact", "B1", "D1D2", "A2", "E2", "E3", "C2", "C1", "A1c"]


def run_variant(variant, train, features, seed=None):
    """단일 변형 이중 게이트 실행 → dict 결과."""
    kw = dict(HGB_KWARGS)
    if seed is not None:
        kw["random_state"] = seed
    model = build_pipeline(variant)
    model.set_params(clf__random_state=kw["random_state"])

    out = {"variant": variant, "seed": seed if seed is not None else 42}
    for gate_name, tr_years, val_year in [("G23", (2019, 2022), 2023), ("G24", (2019, 2023), 2024)]:
        tr_mask = train["season"].between(*tr_years)
        va_mask = train["season"] == val_year
        X_tr, y_tr = train.loc[tr_mask, features], train.loc[tr_mask, "control_success"]
        X_va, y_va = train.loc[va_mask, features], train.loc[va_mask, "control_success"]
        t0 = time.time()
        model.fit(X_tr, y_tr)
        ft = time.time() - t0
        pred = model.predict_proba(X_va)[:, 1]
        bss, r, brier = compute_bss(y_va, pred)
        n_iter = model.named_steps["clf"].n_iter_
        out[f"{gate_name}_bss"] = bss
        out[f"{gate_name}_brier"] = brier
        out[f"{gate_name}_r"] = r
        out[f"{gate_name}_niter"] = n_iter
        out[f"{gate_name}_fit"] = round(ft, 1)
    out["n_feat"] = model.named_steps["clf"].n_features_in_
    return out


def load_results():
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    return pd.DataFrame()


def save_result(row):
    df = load_results()
    df = df[df["variant"] != row["variant"]] if "variant" in df else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="이중 게이트(2023+2024) 검증 하네스")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--assert", action="store_true", dest="do_assert")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train, features = load_data()

    if args.sanity:
        run_sanity(train)
        return

    if args.do_assert:
        df = load_results()
        base = df[df["variant"] == "base"].iloc[0]
        print(f"base: G23={base['G23_bss']:.2f} G24={base['G24_bss']:.2f} (마진 M={MARGIN})")
        adopted, rejected = [], []
        for _, r in df[df["variant"] != "base"].iterrows():
            d23 = r["G23_bss"] - base["G23_bss"]
            d24 = r["G24_bss"] - base["G24_bss"]
            ok = d23 >= MARGIN and d24 >= MARGIN
            verdict = "ADOPT" if ok else "reject"
            if ok:
                adopted.append(r["variant"])
            else:
                rejected.append(r["variant"])
            print(f"  {r['variant']:>8}: ΔG23={d23:+.1f} ΔG24={d24:+.1f} -> {verdict}")
        print(f"\n채택: {adopted or '없음'} | 기각: {rejected or '없음'}")
        sys.exit(0 if adopted else 1)

    variants = [args.variant] if args.variant else (VARIANTS if args.all else ["base"])
    for v in variants:
        print(f"\n===== {v} (seed={args.seed or 42}) =====")
        row = run_variant(v, train, features, seed=args.seed)
        print(f"  n_feat={row['n_feat']} | G23 BSS={row['G23_bss']:.2f} (niter {row['G23_niter']}) | "
              f"G24 BSS={row['G24_bss']:.2f} (niter {row['G24_niter']})")
        save_result(row)
        # base 하드 게이트
        if v == "base" and args.variant == "base":
            if abs(row["G24_bss"] - BASE_G24_REF) > 0.5:
                print(f"[FATAL] base G24 = {row['G24_bss']:.2f} != {BASE_G24_REF}±0.5 — 하네스 오류", file=sys.stderr)
                sys.exit(2)


if __name__ == "__main__":
    main()
