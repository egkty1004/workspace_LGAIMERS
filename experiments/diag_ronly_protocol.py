"""diag_ronly_protocol.py — GIHO R-only 검증 프로토콜 (2026-08-08)

GIHO 849점 방법론 §2의 검증 프로토콜을 우리 실험 파이프라인으로 재현한다.

    primary : <=2023 -> 2024  (전체)
    r2022   : <=2021 -> 2022  (R만)
    r2023   : <=2022 -> 2023  (R만)
    r2024   : <=2023 -> 2024  (R만)

    채택 기준: primary +20점 AND R-only 3개 중 2개 이상 개선

용도:
  1. 기준선(GIHO 구성: 원본 47 + platoon + count_state)을 4폴드로 측정
  2. `--extra-feat` 로 후보 피처를 추가해 기준선과 비교
  3. 이론 상한(셀 평균 기반)도 함께 계산해 누수/버그 판별

사용법:
  python diag_ronly_protocol.py                    # 기준선만
  python diag_ronly_protocol.py --extra-feat xxx   # 후보 피처 추가 비교
"""
import argparse
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = "data"
TARGET = "control_success"
SEEDS = [42, 7, 123]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

# GIHO 최종 구성 (GIHO_849.3688.txt §5)
PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=500,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    num_threads=16,
    verbosity=-1,
)

# 폴드 정의: (이름, val_year, R_only)
FOLDS = [
    ("primary", 2024, False),
    ("r2022", 2022, True),
    ("r2023", 2023, True),
    ("r2024", 2024, True),
]


def score(p, y):
    """BSS — AGENTS.md §2 산식 (GIHO common.score와 동일)."""
    r = y.mean()
    return 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def add_giho_features(t):
    """GIHO 구성: platoon(4셀) + count_state(12범주)."""
    t["platoon"] = (t["pitcher_hand"] * 2 + t["batter_hand"]).astype("category")
    t["count_state"] = (t["balls_before"] * 3 + t["strikes_before"]).astype("category")
    return t


def add_extra_feature(t, extra):
    """후보 피처 추가. extra 이름에 따라 생성. 미지원이면 RuntimeError."""
    if extra == "hand_x_count":
        # platoon x count_state (48셀) — platoon 주효과 0이라 명시 조합 필요
        t["hand_x_count"] = (
            (t["platoon"].astype(int) * 12 + t["count_state"].astype(int))
        ).astype("category")
    elif extra == "count_x_li":
        # count_state x li_bin (60셀) — 승부처 x 볼카운트
        t["li_bin"] = pd.cut(t["li"], bins=[0, 0.5, 1.0, 1.5, 2.0, 99],
                             labels=False).fillna(0).astype(int)
        t["count_x_li"] = (
            t["count_state"].astype(int) * 5 + t["li_bin"]
        ).astype("category")
    elif extra == "cond_x_pitcher":
        # 투수 컨디션(성공률 구간) x pitcher_hand — pitcher_hand 주효과 0
        t["cond_bin"] = pd.cut(t["asof_pitcher_success_rate"],
                               bins=[0, 0.45, 0.475, 0.50, 0.525, 0.55, 1.0],
                               labels=False).fillna(0).astype(int)
        t["cond_x_pitcher"] = (t["cond_bin"] * 2 + t["pitcher_hand"]).astype("category")
    elif extra == "li_x_batter":
        # li_bin x batter_hand — batter_hand 주효과 0 (platoon 구조)
        t["li_bin"] = pd.cut(t["li"], bins=[0, 0.5, 1.0, 1.5, 2.0, 99],
                             labels=False).fillna(0).astype(int)
        t["li_x_batter"] = (t["li_bin"] * 2 + t["batter_hand"]).astype("category")
    elif extra == "sdiff_x_month":
        # score_diff_bin x month_bin — 둘 다 주효과 약한 변수
        t["score_diff_bin"] = pd.cut(t["score_diff_pitcher_team"],
                                     bins=[-99, -3, -1, 1, 3, 99], labels=False)
        t["month_bin"] = (t["game_month"] % 4).astype(int)
        t["sdiff_x_month"] = (t["score_diff_bin"].astype(int) * 4
                              + t["month_bin"]).astype("category")
    else:
        raise RuntimeError(f"미지원 후보 피처: {extra}")
    return t


def theoretical_max(train, val_year, r_only, feat_cols):
    """이론 상한: 검증셋 셀 평균을 완벽히 알 때의 BSS 상한.
    실측 BSS가 이를 초과하면 누수/구현 오류. (GIHO §3-2)"""
    va = train[train["season"] == val_year]
    if r_only:
        va = va[va["game_type"] == "R"]
    y = va[TARGET].values
    r = y.mean()
    # 셀 평균 (feat 조합) 기반 예측
    cells = va.groupby(feat_cols)[TARGET].agg(["mean", "size"])
    w = cells["size"] / cells["size"].sum()
    o_bar = r
    d_res = float((w * (cells["mean"] - o_bar) ** 2).sum())
    return d_res / (r * (1 - r)) * 1e5, cells


def run_fold(train, features, cats, val_year, r_only, seed, extra=None):
    tr_mask = (train["season"] < val_year) & (train["game_type"].isin(["R", "F"]))
    va_mask = train["season"] == val_year
    if r_only:
        tr_mask = tr_mask & (train["game_type"] == "R")
        va_mask = va_mask & (train["game_type"] == "R")

    X_tr = train.loc[tr_mask, features]
    y_tr = train.loc[tr_mask, TARGET]
    X_va = train.loc[va_mask, features]
    y_va = train.loc[va_mask, TARGET]

    params = dict(PARAMS)
    params["seed"] = seed
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=5000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p = model.predict(X_va, num_iteration=model.best_iteration)
    bss = score(p, y_va)
    return bss, model.best_iteration, p.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-feat", default=None, help="후보 피처 이름 (선택)")
    args = ap.parse_args()

    t0 = time.time()
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != "row_id"]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=features + [TARGET])
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
    add_giho_features(train)
    cats = list(CAT_COLS) + ["platoon", "count_state"]
    feats = features + ["platoon", "count_state"]
    extra = args.extra_feat
    if extra:
        add_extra_feature(train, extra)
        feats = feats + [extra]
        cats = cats + [extra]
    print(f"train: {train.shape} | features: {len(feats)} | extra: {extra or '기준선'} "
          f"| 로드 {time.time()-t0:.1f}s", flush=True)

    # 이론 상한 (feat 조합 기준 — 후보별로 별도 계산은 groupby가 무거워 1개 대표 조합만)
    print("=" * 78)
    results = {}
    for name, vy, r_only in FOLDS:
        vals = []
        for seed in SEEDS:
            bss, bi, pm = run_fold(train, feats, cats, vy, r_only, seed, extra)
            vals.append(bss)
            print(f"[{name:8s}] seed {seed:3d} BSS={bss:7.1f} best_iter={bi:3d} "
                  f"pred_mean={pm:.4f} ({time.time()-t0:.0f}s)", flush=True)
        med = float(np.median(vals))
        results[name] = med
        print(f"  -> {name}: 중앙값 {med:7.1f} (시드 {','.join(f'{v:.1f}' for v in vals)})",
              flush=True)

    print("\n" + "=" * 78)
    print("프로토콜 요약 (중앙값)")
    print("=" * 78)
    for name, vy, r_only in FOLDS:
        print(f"  {name:8s}: {results[name]:8.1f}")
    print(f"  소요: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
