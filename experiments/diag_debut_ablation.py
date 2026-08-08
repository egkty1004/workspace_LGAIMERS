"""diag_debut_ablation.py — prev_game 결측(데뷔) vs asof_n 소표본 신호 대체 판정.

질문: Oracle 권고 [High] — Wave B FAIL 원인인 prev_game missing 지시자(2024 역전)를
결측과 1:1 구조인 `asof_pitcher_n`(데뷔=0/소표본) 기반 신호로 대체할 수 있는가?

실험 (모두 기본 HGB_KWARGS, 2019~2023 학습 → 2024 홀드아웃 BSS):
  base    : 47컬럼 (Wave A — 439.00 기대)
  +prev   : prev1_game_success_rate isna 지시자 1종 (6종 동일 패턴이므로 대표 1)
  +n0     : asof_pitcher_n==0 / asof_batter_n==0 boolean 2종 (첫 투구 = 데뷔)
  +nle10  : asof_pitcher_n<=10 / asof_batter_n<=10 boolean 2종 (소표본)
  +all    : prev + n0 + nle10 전부 (13컬럼 신규)

추가: 데이터 레벨에서 prev 결측 행의 asof_n 분포 vs 비결측 행 교차(완전 대체 가능성).

실행: python diag_debut_ablation.py 2>&1 | tee backup/diag_debut_ablation.log
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import _to_category
from train_hgb import HGB_KWARGS, compute_bss, load_data

PREV1_COL = "asof_pitcher_prev1_game_success_rate"


class PrevMissingFlag(BaseEstimator, TransformerMixin):
    """prev1_game_success_rate 결측 지시자 (6종 동일 패턴 — 대표 1종)."""

    def fit(self, X, y=None):
        if PREV1_COL not in X.columns:
            raise ValueError(f"컬럼 부재: {PREV1_COL}")
        return self

    def transform(self, X):
        out = X.copy()
        out["prev1_missing"] = out[PREV1_COL].isna()
        return out


class AsOfNZeroFlag(BaseEstimator, TransformerMixin):
    """asof_pitcher_n==0 / asof_batter_n==0 (첫 투구 = 데이터셋 데뷔 첫 행)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["pitcher_n0"] = out["asof_pitcher_n"] == 0
        out["batter_n0"] = out["asof_batter_n"] == 0
        return out


class AsOfNSmallFlag(BaseEstimator, TransformerMixin):
    """asof_pitcher_n<=10 / asof_batter_n<=10 (소표본 = 이력 부족 근사)."""

    THRESH = 10

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["pitcher_nle10"] = out["asof_pitcher_n"] <= self.THRESH
        out["batter_nle10"] = out["asof_batter_n"] <= self.THRESH
        return out


def make_pipeline(flags):
    steps = []
    if "prev" in flags:
        steps.append(("prev_flag", PrevMissingFlag()))
    if "n0" in flags:
        steps.append(("n0_flag", AsOfNZeroFlag()))
    if "nle10" in flags:
        steps.append(("nle10_flag", AsOfNSmallFlag()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


VARIANTS = [
    ("base", set()),
    ("+prev", {"prev"}),
    ("+n0", {"n0"}),
    ("+nle10", {"nle10"}),
    ("+all", {"prev", "n0", "nle10"}),
]


def data_level_check(X_train, X_val):
    """prev 결측 행 vs 비결측 행의 asof_n 분포 교차 (완전 대체 가능성)."""
    print("\n" + "=" * 88)
    print("[데이터 레벨] prev 결측 행의 asof_n 분포 vs 비결측 행")
    print("=" * 88)
    tr_miss = X_train[PREV1_COL].isna()
    tr_pres = ~tr_miss
    print(f"train(2019~2023) prev 결측: {tr_miss.sum():,} ({tr_miss.mean()*100:.3f}%)")
    miss_n = X_train.loc[tr_miss, "asof_pitcher_n"]
    pres_n = X_train.loc[tr_pres, "asof_pitcher_n"]
    print("결측 행  asof_pitcher_n: min {:.0f} | p25 {:.0f} | med {:.0f} | p75 {:.0f} | max {:.0f}".format(
        miss_n.min(), miss_n.quantile(.25), miss_n.median(), miss_n.quantile(.75), miss_n.max()))
    print("비결측 행 asof_pitcher_n: min {:.0f} | p25 {:.0f} | med {:.0f} | p75 {:.0f} | max {:.0f}".format(
        pres_n.min(), pres_n.quantile(.25), pres_n.median(), pres_n.quantile(.75), pres_n.max()))
    # 겹침: 결측 행 중 asof_n > 10 (n0/nle10으로 안 잡히는 결측) 비율
    big_miss = miss_n[miss_n > 10]
    print(f"\n결측 행 중 asof_pitcher_n > 10 (소표본 신호로 미포착): {len(big_miss):,} "
          f"({len(big_miss)/len(miss_n)*100:.1f}%)")
    # 반대: 비결측 행 중 asof_n <= 10 (소표본인데 prev 존재 = 교체 투수 등)
    small_pres = pres_n[pres_n <= 10]
    print(f"비결측 행 중 asof_pitcher_n <= 10: {len(small_pres):,} "
          f"({len(small_pres)/len(pres_n)*100:.1f}%)")
    # batter
    b_miss = X_train["asof_batter_success_rate"].isna()
    b_miss_n = X_train.loc[b_miss, "asof_batter_n"]
    b_pres_n = X_train.loc[~b_miss, "asof_batter_n"]
    print(f"\nbatter 결측 행 asof_batter_n: med {b_miss_n.median():.0f} | max {b_miss_n.max():.0f}")
    print(f"batter 비결측 행 asof_batter_n: min {b_pres_n.min():.0f} | med {b_pres_n.median():.0f}")
    b_big = b_miss_n[b_miss_n > 10]
    print(f"batter 결측 행 중 asof_batter_n > 10: {len(b_big):,} ({len(b_big)/len(b_miss_n)*100:.1f}%)")


def main():
    train, features = load_data()
    is_val = train["season"] == 2024
    is_2023 = train["season"] == 2023
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, "control_success"]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, "control_success"]
    X_23, y_23 = train.loc[is_2023, features], train.loc[is_2023, "control_success"]

    data_level_check(X_tr, X_val)

    print("\n" + "=" * 88)
    print("게이트 ablation (기본 HGB_KWARGS, best_params 미적용)")
    print("=" * 88)
    results = []
    for name, flags in VARIANTS:
        model = make_pipeline(flags)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        ft = time.time() - t0
        n_iter = model.named_steps["clf"].n_iter_
        pred = model.predict_proba(X_val)[:, 1]
        bss24, r24, brier24 = compute_bss(y_val, pred)
        pred23 = model.predict_proba(X_23)[:, 1]
        bss23, _, _ = compute_bss(y_23, pred23)
        n_feat = model.named_steps["clf"].n_features_in_
        results.append(dict(name=name, n_feat=n_feat, n_iter=n_iter, fit=ft,
                            brier24=brier24, bss24=bss24, bss23=bss23))
        print(f"[{name:>6}] feat={n_feat:2d} n_iter={n_iter:4d} fit={ft:5.1f}s | "
              f"Brier24={brier24:.6f} BSS24={bss24:8.2f} | BSS23(in)={bss23:8.2f}")

    print("\n" + "=" * 88)
    print("요약")
    print("=" * 88)
    print(f"{'조합':>6} | {'feat':>4} | {'n_iter':>5} | {'Brier24':>8} | {'BSS24':>8} | {'BSS23(in)':>9}")
    for r in results:
        print(f"{r['name']:>6} | {r['n_feat']:>4} | {r['n_iter']:>5} | "
              f"{r['brier24']:>8.6f} | {r['bss24']:>8.2f} | {r['bss23']:>9.2f}")
    pd.DataFrame(results).to_csv("backup/diag_debut_ablation.csv", index=False)
    print("\n저장: backup/diag_debut_ablation.csv")


if __name__ == "__main__":
    main()
