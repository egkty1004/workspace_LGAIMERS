"""diag_cv_diagnosis.py — 로컬 CV 방식 검증.

문제: 로컬 CV(2024 홀드아웃) 순위가 Public(2025)과 반비례. 어떤 CV 방식이
Public 순위를 가장 잘 예측하는지 4개 제출 모델로 비교한다.

모델 (제출 이력 재현):
  RF (공식), HGB base, HGB+interact, LGBM base, LGBM+E2

CV 방식 (각 모델 × 각 홀드아웃 연도, 3시드 중앙값):
  - 2024 홀드아웃 (현재 방식)
  - 2023 홀드아웃 (break year)
  - 2022 홀드아웃

결과: 각 CV 연도의 모델 순위 vs Public 순위(spearman) 비교
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from bss_preprocess import _to_category, InteractionAdder
from feat_eng import HandMatch
from train_hgb import compute_bss

SEEDS = [42, 7, 123]
FEATURES_GLOBAL = []

PUBLIC = {"RF": 549.51, "HGB": 770.45, "HGB_interact": 667.72,
          "LGBM": None, "LGBM_E2": 535.96}
# LGBM base는 제출 안 했으므로 Public 없음 — 순위 비교에서 제외


def make_pipeline(model, seed=42):
    if model == "RF":
        from sklearn.compose import ColumnTransformer
        cat_cols = ["top_bottom", "game_type", "base_state"]
        num_cols = [c for c in FEATURES_GLOBAL if c not in cat_cols]
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ])
        clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     min_samples_leaf=200, n_jobs=-1, random_state=42)
        return Pipeline([("pre", pre), ("clf", clf)])
    if model == "HGB":
        return Pipeline([("to_cat", FunctionTransformer(_to_category)),
                         ("clf", HistGradientBoostingClassifier(
                             max_iter=1000, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=10, random_state=seed))])
    if model == "HGB_interact":
        return Pipeline([("interact", InteractionAdder()),
                         ("to_cat", FunctionTransformer(_to_category)),
                         ("clf", HistGradientBoostingClassifier(
                             max_iter=1000, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=10, random_state=seed))])
    if model == "LGBM":
        return Pipeline([("to_cat", FunctionTransformer(_to_category)),
                         ("clf", lgb.LGBMClassifier(
                             n_estimators=1000, learning_rate=0.1, num_leaves=31,
                             early_stopping_rounds=10, verbosity=-1, random_state=seed))])
    if model == "LGBM_E2":
        return Pipeline([("hand", HandMatch()),
                         ("to_cat", FunctionTransformer(_to_category)),
                         ("clf", lgb.LGBMClassifier(
                             n_estimators=1000, learning_rate=0.1, num_leaves=31,
                             early_stopping_rounds=10, verbosity=-1, random_state=seed))])


def fit_and_predict(model, X_tr, y_tr, X_va, seed):
    pipe = make_pipeline(model, seed)
    if model in ("LGBM", "LGBM_E2"):
        X_pre = pipe[:-1].fit_transform(X_tr)
        X_va_pre = pipe[:-1].transform(X_va)
        X_pre_c = _to_category(X_pre)
        X_va_c = _to_category(X_va_pre)
        Xt, Xv_, yt, yv_ = train_test_split(X_pre_c, y_tr, test_size=0.1, random_state=seed)
        pipe.fit(Xt, yt, clf__eval_set=[(Xv_, yv_)], clf__eval_metric="binary_logloss")
        return pipe.predict_proba(X_va_c)[:, 1]
    else:
        pipe.fit(X_tr, y_tr)
        return pipe.predict_proba(X_va)[:, 1]


def run_gate(model, train, features, val_year, seed):
    tr_mask = train["season"] < val_year
    va_mask = train["season"] == val_year
    X_tr, y_tr = train.loc[tr_mask, features], train.loc[tr_mask, "control_success"]
    X_va, y_va = train.loc[va_mask, features], train.loc[va_mask, "control_success"]
    p = fit_and_predict(model, X_tr, y_tr, X_va, seed)
    bss, r, brier = compute_bss(y_va, p)
    return bss


def main():
    train, features = None, None
    # 직접 로드
    import pandas as pd
    test_cols = pd.read_csv("./data/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != "row_id"]
    global FEATURES_GLOBAL
    FEATURES_GLOBAL = features
    train = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                        usecols=features + ["control_success"])
    print("train:", train.shape)

    models = ["RF", "HGB", "HGB_interact", "LGBM", "LGBM_E2"]
    val_years = [2022, 2023, 2024]

    results = {}
    for model in models:
        for vy in val_years:
            bss_list = []
            for seed in SEEDS:
                bss_list.append(run_gate(model, train, features, vy, seed))
            results[(model, vy)] = float(np.median(bss_list))
            print(f"{model:>12} | {vy} 홀드아웃 | 중앙값 BSS={np.median(bss_list):8.2f} "
                  f"(시드 {','.join(f'{x:.0f}' for x in bss_list)})")

    print("\n" + "=" * 88)
    print("Public 순위와 각 CV 연도 순위 비교")
    print("=" * 88)
    # Public 있는 모델만
    pub_models = [m for m in models if PUBLIC[m] is not None]
    pub_rank = {m: r for r, m in enumerate(sorted(pub_models, key=lambda x: -PUBLIC[x]))}
    print("Public 순위 (높을수록 좋음):", {m: PUBLIC[m] for m in pub_models})

    for vy in val_years:
        ranked = sorted(models, key=lambda m: -results[(m, vy)])
        print(f"\n{vy} CV 순위: " + " > ".join(f"{m}({results[(m,vy)]:.0f})" for m in ranked))
        # spearman 상관 (Public 있는 모델 한정)
        sub = [m for m in pub_models]
        pub_vals = [PUBLIC[m] for m in sub]
        cv_vals = [results[(m, vy)] for m in sub]
        from scipy.stats import spearmanr
        rho, pval = spearmanr(pub_vals, cv_vals)
        print(f"  Public vs CV{vy} spearman: rho={rho:+.3f} (p={pval:.3f})")
    print("\nrho > 0 이면 해당 CV가 Public을 올바르게 예측 / rho < 0 이면 반비례")


if __name__ == "__main__":
    main()
