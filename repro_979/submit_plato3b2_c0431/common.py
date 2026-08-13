#!/usr/bin/env python3
"""
[공용] Exp 1~3 공유 유틸: 데이터 로드, BSS 점수, 로짓 변환, season 외삽, LightGBM 학습.
학습/추론 간 전처리 불일치 방지를 위해 단일 모듈로 분리 (AGENTS.md §8 권장 구조).
"""
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = "./open/data"
CACHE_DIR = "./cache"
FEATHER_PATH = os.path.join(CACHE_DIR, "train_cache.feather")
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

# Exp 1 재현 구성 (AGENTS.md 4-3)
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
    seed=42,
    deterministic=True,
)


def assert_gpu():
    """GPU 가용성 확인 (torch 무의존 — XGBoost 자체 CUDA 런타임 사용)."""
    import xgboost as xgb, numpy as np
    X = np.random.rand(1000, 10)
    y = (np.random.rand(1000) > 0.5).astype(int)
    bst = xgb.train({"device": "cuda", "tree_method": "hist"},
                    xgb.DMatrix(X, y), num_boost_round=1)
    cfg = bst.save_config().replace(" ", "")
    assert '"device":"cuda' in cfg, f"XGBoost CPU 폴백 발생! {cfg[:300]}"
    print("[GPU OK] XGBoost CUDA 확인")


def score(p, y):
    """Brier Skill Score — AGENTS.md §2 산식 그대로."""
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_platoon_feature(t):
    """G3: 플래툰 pitcher_hand*2 + batter_hand → 4범주 카테고리 (행 단위 변환, 누수 없음)."""
    t["platoon"] = (t["pitcher_hand"] * 2 + t["batter_hand"]).astype("category")
    return t


def add_count_state_feature(t):
    """G1: count_state balls_before*3 + strikes_before → 12범주 카테고리."""
    t["count_state"] = (t["balls_before"] * 3 + t["strikes_before"]).astype("category")
    return t


def add_recent_gap_feature(t):
    """Wave C: recent_gap_success = prev1 경기 성공률 − 누적 성공률.
    NaN(prev1 결측, 1.98%) → 0.0 fill, float32."""
    t["recent_gap_success"] = (
        t["asof_pitcher_prev1_game_success_rate"] - t["asof_pitcher_success_rate"]
    ).fillna(0.0).astype("float32")
    return t


def add_return_gap_feature(t):
    """Wave C: return_gap = prev1_game 결측 & asof_pitcher_n < 100 (복귀/공백 소이력) → int8."""
    t["return_gap"] = (
        t["asof_pitcher_prev1_game_success_rate"].isna()
        & (t["asof_pitcher_n"] < 100)
    ).astype("int8")
    return t


def add_debut_feature(t):
    """Wave C: pitcher_debut = asof_pitcher_n==0, batter_debut = asof_batter_n==0 → int8."""
    t["pitcher_debut"] = (t["asof_pitcher_n"] == 0).astype("int8")
    t["batter_debut"] = (t["asof_batter_n"] == 0).astype("int8")
    return t


def add_asof_n_bucket_feature(t):
    """Wave C: asof_n_bucket = asof_pitcher_n 구간화 [0,1)/[1,10)/[10,50)/[50,100)/100+
    → int8 (0~4). NaN은 0(bucket 0) 처리 (cut 코드 -1 → 0)."""
    codes = pd.cut(t["asof_pitcher_n"], [0, 1, 10, 50, 100, np.inf],
                   labels=[0, 1, 2, 3, 4]).cat.codes
    t["asof_n_bucket"] = codes.replace(-1, 0).astype("int8")
    return t


def add_cross_cell_features(t):
    """Wave C 교차 셀 이진 플래그 4종 (행 단위 변환, 누수 없음):
    score_diff_binary(접전 |diff|<=1), li_risp_flag(2b/3b & li>=1.0),
    outs_count_3b2_2out(3-2 & 2아웃), count_platoon_3b2_same(3-2 & 동손).
    count_platoon_3b2_same은 스크리닝 기각(primary Δ-13.4)으로 F3_EXTRA에서 제외 —
    screen_all.py 개별 판정 시 add_cross_cell_features 호출로 직접 계산."""
    t["score_diff_binary"] = (t["score_diff_pitcher_team"].abs() <= 1).astype("int8")
    t["li_risp_flag"] = (
        ((t["runner_on_2b"] == 1) | (t["runner_on_3b"] == 1))
        & (t["li"] >= 1.0)
    ).astype("int8")
    t["outs_count_3b2_2out"] = (
        (t["balls_before"] == 3) & (t["strikes_before"] == 2)
        & (t["outs_before"] == 2)
    ).astype("int8")
    t["count_platoon_3b2_same"] = (
        (t["balls_before"] == 3) & (t["strikes_before"] == 2)
        & (t["pitcher_hand"] == t["batter_hand"])
    ).astype("int8")
    return t


F3_EXTRA = ["platoon", "count_state", "recent_gap_success", "return_gap",
            "pitcher_debut", "batter_debut", "asof_n_bucket",
            "score_diff_binary", "li_risp_flag", "outs_count_3b2_2out"]


def get_feature_cols(test_cols):
    """test 컬럼 기준 피처 + F3 추가 피처. 학습/추론 동일."""
    return [c for c in test_cols if c != ID] + F3_EXTRA


def preprocess_for_submission(df):
    """제출용 공통 전처리: 다운캐스팅 + 범주형 + G3(platoon) + G1(count_state)
    + Wave C 유도 피처(학습/추론 동일 모듈 사용, AGENTS.md §8).
    신규 피처는 기존 피처 뒤에 추가 (count_platoon_3b2_same은 F3_EXTRA 제외라
    모델 입력엔 안 들어가지만 screen_all.py 개별 판정용으로 컬럼은 생성됨)."""
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = df[c].astype("int32")
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    add_platoon_feature(df)
    add_count_state_feature(df)
    add_recent_gap_feature(df)
    add_return_gap_feature(df)
    add_debut_feature(df)
    add_asof_n_bucket_feature(df)
    add_cross_cell_features(df)
    return df


def extrapolate_logit(season_means, target_season):
    """2019~T 시즌 평균 로짓의 선형 회귀로 target_season 로짓 외삽.
    target_season의 실제 r은 사용하지 않음 (공정 평가)."""
    years = np.array(sorted(season_means), dtype=float)
    x = years - years[0]
    z = logit(np.array([season_means[int(y)] for y in years], dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, z, rcond=None)[0]
    return slope * (target_season - years[0]) + intercept


def load_train():
    """feather 캐시 우선 로드, 없으면 csv 폴백.
    utf-8-sig 필수(BOM), float32/int32 다운캐스팅, 범주형 변환."""
    if os.path.exists(FEATHER_PATH):
        train = pd.read_feather(FEATHER_PATH)
    else:
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
        for c in train.select_dtypes("float64").columns:
            train[c] = train[c].astype("float32")
        for c in train.select_dtypes("int64").columns:
            train[c] = train[c].astype("int32")
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
    test_cols = pd.read_csv(
        os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0
    ).columns
    features = [c for c in test_cols if c != ID]
    return train, features


def train_model(X_tr, y_tr, X_va, y_va, weight_tr=None, num_boost_round=5000,
                log_eval=100, cat_cols=None):
    """LightGBM 학습 (early stopping 50). weight_tr로 다운웨이트 지원.
    cat_cols: 카테고리 컬럼 목록 (기본 CAT_COLS, ablation에서 확장 가능)."""
    cat_cols = CAT_COLS if cat_cols is None else cat_cols
    dtr = lgb.Dataset(X_tr, y_tr, weight=weight_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(
        PARAMS, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(log_eval)],
    )
    return model


def evaluate_fold(model, X_va, y_va, season_means, val_year, last_tr_year,
                  X_last=None):
    """폴드 평가: raw / extrap-shift(제안, r 미사용) / oracle-shift(상한).
    X_last: 마지막 학습 시즌 행 (모델 레벨 측정용). None이면 X_va 사용 불가 → 호출부가 전달.
    """
    p_raw = model.predict(X_va, num_iteration=model.best_iteration)
    yv = y_va.values
    s_raw = score(p_raw, yv)

    # extrap-shift
    L_extrap = extrapolate_logit(season_means, val_year)
    assert X_last is not None, "X_last(마지막 학습 시즌 행) 필요"
    p_last = model.predict(X_last, num_iteration=model.best_iteration)
    L_model = logit(p_last.mean())
    offset = L_extrap - L_model
    p_extrap = sigmoid(logit(p_raw) + offset)
    s_extrap = score(p_extrap, yv)

    # oracle-shift (상한 — 실제 r 사용, 비교용만)
    r_val = yv.mean()
    p_oracle = sigmoid(logit(p_raw) + (logit(np.full(1, r_val))[0] - logit(p_raw.mean())))
    s_oracle = score(p_oracle, yv)

    return dict(
        raw=s_raw, extrap=s_extrap, oracle=s_oracle, offset=offset,
        pred_mean_raw=p_raw.mean(), pred_mean_extrap=p_extrap.mean(),
        L_extrap=L_extrap, L_model=L_model, best_iter=model.best_iteration,
    )
