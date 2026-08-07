"""diag_stage_ablation.py — Wave B 전처리 3단계 조합별 2024 게이트 (FAIL 범인 pinpoint).

배경: 코드가 Wave A로 롤백되어 bss_preprocess.py에 전처리 클래스가 없다.
Wave B 3단계를 스크립트 내부에 모듈 레벨로 재정의(원본 구현과 동일)해
조합별 2024 홀드아웃 BSS를 추적한다.

조합 (기본 HGB_KWARGS, best_params 미적용):
  base / +enc / +missing / +interact / +enc+missing / +enc+interact /
  +missing+interact / +all(60컬럼)

실행: python diag_stage_ablation.py 2>&1 | tee backup/diag_stage_ablation.log
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from bss_preprocess import _to_category
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


class IDTargetEncoder(BaseEstimator, TransformerMixin):
    """pitcher_id/batter_id → TargetEncoder (cv=5, smooth=auto, shuffle=False)."""

    def __init__(self):
        self.encoder_ = None
        self.global_mean_ = None

    def _make_encoder(self):
        return TargetEncoder(
            cv=5, smooth="auto", target_type="binary", shuffle=False
        ).set_output(transform="pandas")

    def fit(self, X, y=None):
        y = np.asarray(y, dtype=float)
        self.encoder_ = self._make_encoder()
        self.encoder_.fit(X[["pitcher_id", "batter_id"]], y)
        self.global_mean_ = float(y.mean())
        return self

    def fit_transform(self, X, y=None):
        y = np.asarray(y, dtype=float)
        self.encoder_ = self._make_encoder()
        enc = self.encoder_.fit_transform(X[["pitcher_id", "batter_id"]], y)
        self.global_mean_ = float(y.mean())
        return self._attach(X, enc)

    def transform(self, X):
        enc = self.encoder_.transform(X[["pitcher_id", "batter_id"]])
        return self._attach(X, enc)

    def _attach(self, X, enc):
        if not isinstance(enc, pd.DataFrame):
            enc = pd.DataFrame(
                np.asarray(enc), columns=["pitcher_id", "batter_id"], index=X.index
            )
        out = X.copy()
        out["pitcher_enc"] = enc["pitcher_id"].fillna(self.global_mean_)
        out["batter_enc"] = enc["batter_id"].fillna(self.global_mean_)
        return out


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


class InteractionAdder(BaseEstimator, TransformerMixin):
    RUNNER_RISK_BINS = np.array([-1.0, 1.0, 2.0, 1e9])

    def __init__(self):
        self.li_edges_ = None
        self.base_state_map_ = {}

    def fit(self, X, y=None):
        need = {"base_state", "li", "balls_before", "strikes_before", "num_runners_on"}
        missing = need - set(X.columns)
        if missing:
            raise ValueError(f"컬럼 부재: {sorted(missing)}")
        li = X["li"].astype(float)
        _, self.li_edges_ = pd.qcut(li, 5, duplicates="drop", retbins=True)
        med = X.groupby("base_state", observed=True)["li"].median()
        codes = pd.cut(
            med, bins=self.li_edges_, include_lowest=True, duplicates="drop"
        ).cat.codes
        self.base_state_map_ = {
            k: int(v) for k, v in codes.items() if not pd.isna(v)
        }
        return self

    def transform(self, X):
        out = X.copy()
        out["base_state_li"] = (
            out["base_state"].map(self.base_state_map_).fillna(-1).astype(int)
        )
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


def make_pipeline(stages):
    steps = []
    if "enc" in stages:
        steps.append(("enc_ids", IDTargetEncoder()))
    if "missing" in stages:
        steps.append(("add_missing", MissingIndicatorAdder()))
    if "interact" in stages:
        steps.append(("add_interact", InteractionAdder()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


VARIANTS = [
    ("base", set()),
    ("enc", {"enc"}),
    ("missing", {"missing"}),
    ("interact", {"interact"}),
    ("enc+missing", {"enc", "missing"}),
    ("enc+interact", {"enc", "interact"}),
    ("missing+interact", {"missing", "interact"}),
    ("all", {"enc", "missing", "interact"}),
]


def main():
    train, features = load_data()
    is_val = train["season"] == 2024
    is_2023 = train["season"] == 2023
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, "control_success"]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, "control_success"]
    X_23, y_23 = train.loc[is_2023, features], train.loc[is_2023, "control_success"]
    print("train(2019~2023):", len(X_tr), "| val(2024):", len(X_val))

    results = []
    for name, stages in VARIANTS:
        model = make_pipeline(stages)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        ft = time.time() - t0
        n_iter = model.named_steps["clf"].n_iter_
        n_feat = model.named_steps["clf"].n_features_in_
        pred = model.predict_proba(X_val)[:, 1]
        bss24, r24, brier24 = compute_bss(y_val, pred)
        pred23 = model.predict_proba(X_23)[:, 1]
        bss23, _, _ = compute_bss(y_23, pred23)
        results.append(dict(name=name, n_feat=n_feat, n_iter=n_iter, fit=ft,
                            brier24=brier24, bss24=bss24, bss23=bss23))
        print(f"[{name:>15}] feat={n_feat:2d} n_iter={n_iter:4d} fit={ft:5.1f}s | "
              f"Brier24={brier24:.6f} BSS24={bss24:8.2f} | BSS23(in)={bss23:8.2f}")

    print("\n" + "=" * 88)
    print("요약")
    print("=" * 88)
    print(f"{'조합':>15} | {'feat':>4} | {'n_iter':>5} | {'Brier24':>8} | {'BSS24':>8} | {'BSS23(in)':>9}")
    for r in results:
        print(f"{r['name']:>15} | {r['n_feat']:>4} | {r['n_iter']:>5} | "
              f"{r['brier24']:>8.6f} | {r['bss24']:>8.2f} | {r['bss23']:>9.2f}")
    pd.DataFrame(results).to_csv("backup/diag_stage_ablation.csv", index=False)
    print("\n저장: backup/diag_stage_ablation.csv")


if __name__ == "__main__":
    main()
