"""diag_calibration.py — BSS 확률 보정(calibration) 실험 (2026-08-08)

Tabular ML 강의 분석에서 나온 채택 후보 1순위: BSS는 확률의 정확도를 보므로
시드 앙상블 평균화(로짓 평균)와 직교하는 보정(Platt/isotonic) 이득을 확인한다.

프로토콜 (누수 없이):
  1) GIHO 구성(LGBM + platoon + count_state)을 2019~2023으로 학습
  2) 2024 검증 예측 (5시드 로짓 평균) → raw 예측
  3) 2024를 3-fold로 나눠 cross-fitting: 한 fold에서 calibrator 적합,
     다른 fold에 적용 → 전체 BSS
  4) raw vs platt vs isotonic BSS 비교 (5시드 중앙값)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data"
TARGET = "control_success"
SEEDS = [42, 7, 123, 2024, 2025]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PARAMS = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, num_threads=16, verbosity=-1,
)


def score(p, y):
    r = y.mean()
    return 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    feats_all = [c for c in test_cols if c != "row_id"]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=feats_all + [TARGET])
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
    train["platoon"] = (train["pitcher_hand"] * 2 + train["batter_hand"]).astype("category")
    train["count_state"] = (train["balls_before"] * 3 + train["strikes_before"]).astype("category")
    feats = feats_all + ["platoon", "count_state"]
    cats = CAT_COLS + ["platoon", "count_state"]
    print(f"train: {train.shape}", flush=True)

    tr = train[train["season"] < 2024]
    va = train[train["season"] == 2024]
    X_tr, y_tr = tr[feats], tr[TARGET]
    X_va, y_va = va[feats], va[TARGET].values
    n = len(va)

    # 5시드 로짓 평균 예측
    z_sum = np.zeros(n)
    for seed in SEEDS:
        p = dict(PARAMS); p["seed"] = seed
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
        model = lgb.train(p, dtr, num_boost_round=109)  # GIHO: best_iter 95 x 1.15
        z_sum += logit(model.predict(X_va))
    z = z_sum / len(SEEDS)
    p_raw = sigmoid(z)
    bss_raw = score(p_raw, y_va)
    print(f"raw  : BSS={bss_raw:8.1f} (pred_mean={p_raw.mean():.4f}, r={y_va.mean():.4f})", flush=True)

    # 3-fold cross-fitting calibration
    rng = np.random.RandomState(42)
    fold = rng.permutation(n) % 3
    p_platt = np.zeros(n); p_iso = np.zeros(n)
    for k in range(3):
        fit_idx = fold != k
        cal_idx = fold == k
        # Platt: 로짓 공간 선형 (z_shift = a*z + b)
        lr = LogisticRegression(C=1e6)
        lr.fit(z[fit_idx].reshape(-1, 1), y_va[fit_idx])
        a = lr.coef_[0, 0]; b = lr.intercept_[0]
        p_platt[cal_idx] = sigmoid(a * z[cal_idx] + b)
        # Isotonic: 확률 공간 단조 보정
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_raw[fit_idx], y_va[fit_idx])
        p_iso[cal_idx] = iso.predict(p_raw[cal_idx])

    bss_platt = score(p_platt, y_va)
    bss_iso = score(p_iso, y_va)
    print(f"platt : BSS={bss_platt:8.1f} (pred_mean={p_platt.mean():.4f})  Δ={bss_platt-bss_raw:+.1f}", flush=True)
    print(f"isot : BSS={bss_iso:8.1f} (pred_mean={p_iso.mean():.4f})  Δ={bss_iso-bss_raw:+.1f}", flush=True)


if __name__ == "__main__":
    main()
