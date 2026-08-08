"""diag_enc_split.py — enc 범인 pinpoint: pitcher_enc vs batter_enc 분리.

단계별 검증에서 IDTargetEncoder(enc)가 FAIL 범인으로 확정됨.
이제 pitcher_enc/batter_enc를 분리해 어느 쪽이 2024를 망치는지 확인.
+ interact(개선 확인됨)와의 조합도 확인.

조합: base / +pitcher_enc / +batter_enc / +both(49=기존 enc) / +pitcher_enc+interact / +batter_enc+interact

실행: python diag_enc_split.py 2>&1 | tee backup/diag_enc_split.log
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


class EncAdder(BaseEstimator, TransformerMixin):
    """ID → TargetEncoder (pitcher/batter 중 선택), 신규 컬럼 추가."""

    def __init__(self, cols, out_names):
        self.cols = cols
        self.out_names = out_names
        self.encoder_ = None
        self.global_mean_ = None

    def fit(self, X, y=None):
        y = np.asarray(y, dtype=float)
        self.encoder_ = TargetEncoder(
            cv=5, smooth="auto", target_type="binary", shuffle=False
        ).set_output(transform="pandas")
        self.encoder_.fit(X[self.cols], y)
        self.global_mean_ = float(y.mean())
        return self

    def fit_transform(self, X, y=None):
        y = np.asarray(y, dtype=float)
        self.encoder_ = TargetEncoder(
            cv=5, smooth="auto", target_type="binary", shuffle=False
        ).set_output(transform="pandas")
        enc = self.encoder_.fit_transform(X[self.cols], y)
        self.global_mean_ = float(y.mean())
        return self._attach(X, enc)

    def transform(self, X):
        enc = self.encoder_.transform(X[self.cols])
        return self._attach(X, enc)

    def _attach(self, X, enc):
        if not isinstance(enc, pd.DataFrame):
            enc = pd.DataFrame(np.asarray(enc), columns=self.cols, index=X.index)
        out = X.copy()
        for c, on in zip(self.cols, self.out_names):
            out[on] = enc[c].fillna(self.global_mean_)
        return out


class InteractionAdder(BaseEstimator, TransformerMixin):
    RUNNER_RISK_BINS = np.array([-1.0, 1.0, 2.0, 1e9])

    def __init__(self):
        self.li_edges_ = None
        self.base_state_map_ = {}

    def fit(self, X, y=None):
        li = X["li"].astype(float)
        _, self.li_edges_ = pd.qcut(li, 5, duplicates="drop", retbins=True)
        med = X.groupby("base_state", observed=True)["li"].median()
        codes = pd.cut(
            med, bins=self.li_edges_, include_lowest=True, duplicates="drop"
        ).cat.codes
        self.base_state_map_ = {k: int(v) for k, v in codes.items() if not pd.isna(v)}
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


PIT = ["pitcher_id"]
BAT = ["batter_id"]


def make_pipeline(enc_cols=None, enc_outs=None, with_interact=False):
    steps = []
    if enc_cols:
        steps.append(("enc", EncAdder(enc_cols, enc_outs)))
    if with_interact:
        steps.append(("interact", InteractionAdder()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


VARIANTS = [
    ("base", None, None, False),
    ("pitcher_enc", PIT, ["pitcher_enc"], False),
    ("batter_enc", BAT, ["batter_enc"], False),
    ("both_enc", PIT + BAT, ["pitcher_enc", "batter_enc"], False),
    ("pitcher_enc+interact", PIT, ["pitcher_enc"], True),
    ("batter_enc+interact", BAT, ["batter_enc"], True),
]


def main():
    train, features = load_data()
    is_val = train["season"] == 2024
    is_2023 = train["season"] == 2023
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, "control_success"]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, "control_success"]
    X_23, y_23 = train.loc[is_2023, features], train.loc[is_2023, "control_success"]

    results = []
    for name, cols, outs, wi in VARIANTS:
        model = make_pipeline(cols, outs, wi)
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
    for r in results:
        print(f"  {r['name']:>20} | feat={r['n_feat']:2d} n_iter={r['n_iter']:4d} | "
              f"BSS24={r['bss24']:8.2f} | BSS23(in)={r['bss23']:8.2f}")
    pd.DataFrame(results).to_csv("backup/diag_enc_split.csv", index=False)
    print("저장: backup/diag_enc_split.csv")


if __name__ == "__main__":
    main()
