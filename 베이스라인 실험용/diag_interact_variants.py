"""diag_interact_variants.py — interact 변형 4조합 게이트 판정 (제출 후보 선정).

조합 (기본 HGB_KWARGS, 2019~2023 → 2024 홀드아웃 BSS):
  1. base+interact          : 기준 (494.36 재현)
  2. base+interact+missing  : missing 지시자 8종 결합 (무죄 확인됨 — 추가 개선?)
  3. base+interact-base_state_li : 비안정 bin 제거 (count_cat + runner_risk만)
  4. base+interact+F_flag    : game_type=F(+0.08 신호)를 명시적 이진 피처로 강화

실행: python diag_interact_variants.py 2>&1 | tee backup/diag_interact_variants.log
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import InteractionAdder, _to_category
from train_hgb import HGB_KWARGS, compute_bss, load_data

MISSING_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
]


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        missing = [c for c in MISSING_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"컬럼 부재: {missing}")
        return self

    def transform(self, X):
        out = X.copy()
        for c in MISSING_COLS:
            out[f"{c}_missing"] = out[c].isna()
        return out


class CountRunnerOnly(BaseEstimator, TransformerMixin):
    """base_state_li 제외 — count_cat + runner_risk만 (비안정 bin 제거 시도)."""

    RUNNER_RISK_BINS = np.array([-1.0, 1.0, 2.0, 1e9])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["count_cat"] = (
            out["balls_before"].astype(int) * 3 + out["strikes_before"].astype(int)
        ).astype(int)
        li_codes = pd.cut(
            out["li"].astype(float), bins=self.RUNNER_RISK_BINS, include_lowest=True
        ).cat.codes
        li_codes = li_codes.where(li_codes >= 0, 0)
        out["runner_risk"] = (
            out["num_runners_on"].astype(int).fillna(0) * 3 + li_codes
        ).astype(int)
        return out


class FFlagAdder(BaseEstimator, TransformerMixin):
    """game_type=='F' → 이진 플래그 (기존 범주형 처리 외 명시 강화)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["is_postseason"] = (out["game_type"] == "F").astype(int)
        return out


def make_pipeline(variant):
    steps = []
    if variant == "interact":
        steps.append(("add_interact", InteractionAdder()))
    elif variant == "interact+missing":
        steps.append(("add_interact", InteractionAdder()))
        steps.append(("add_missing", MissingIndicatorAdder()))
    elif variant == "interact_no_basestate":
        steps.append(("add_interact", CountRunnerOnly()))
    elif variant == "interact+Fflag":
        steps.append(("add_interact", InteractionAdder()))
        steps.append(("add_fflag", FFlagAdder()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


VARIANTS = ["interact", "interact+missing", "interact_no_basestate", "interact+Fflag"]


def main():
    train, features = load_data()
    is_val = train["season"] == 2024
    is_2023 = train["season"] == 2023
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, "control_success"]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, "control_success"]
    X_23, y_23 = train.loc[is_2023, features], train.loc[is_2023, "control_success"]

    results = []
    for name in VARIANTS:
        model = make_pipeline(name)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        ft = time.time() - t0
        n_iter = model.named_steps["clf"].n_iter_
        n_feat = model.named_steps["clf"].n_features_in_
        pred = model.predict_proba(X_val)[:, 1]
        bss24, _, brier24 = compute_bss(y_val, pred)
        pred23 = model.predict_proba(X_23)[:, 1]
        bss23, _, _ = compute_bss(y_23, pred23)
        results.append(dict(name=name, n_feat=n_feat, n_iter=n_iter, fit=ft,
                            brier24=brier24, bss24=bss24, bss23=bss23))
        print(f"[{name:>20}] feat={n_feat:2d} n_iter={n_iter:4d} fit={ft:5.1f}s | "
              f"Brier24={brier24:.6f} BSS24={bss24:8.2f} | BSS23(in)={bss23:8.2f}")

    print("\n요약:")
    for r in sorted(results, key=lambda x: -x["bss24"]):
        print(f"  {r['name']:>20} | feat={r['n_feat']:2d} n_iter={r['n_iter']:4d} | "
              f"BSS24={r['bss24']:8.2f}")
    pd.DataFrame(results).to_csv("backup/diag_interact_variants.csv", index=False)
    print("저장: backup/diag_interact_variants.csv")


if __name__ == "__main__":
    main()
