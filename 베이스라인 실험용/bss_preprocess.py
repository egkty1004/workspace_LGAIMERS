"""bss_preprocess.py — HGB 학습 공용 전처리 모듈.

모든 정의는 모듈 레벨에 둔다 (N1):
- joblib 1.5.3은 함수를 by-reference(`bss_preprocess._to_category`)로 직렬화하므로
  __main__ 블록이나 람다 안에서 정의하면 새 프로세스/평가 서버 로드 시
  AttributeError가 발생한다.
- 반드시 모듈 레벨 함수로만 정의하고, train_hgb.py 등에서 import 하여 사용한다.
- 9개 후보 중 pitcher_id/batter_id는 sklearn HGB 범주형 카디널리티 255 제한(실측 711/742 > 255)으로 수치형(int) 처리 — 사용자 승인 옵션 A.
- 신규 (Todo 4/5): IDTargetEncoder / MissingIndicatorAdder / InteractionAdder — 모두 모듈 레벨 클래스,
  BaseEstimator+TransformerMixin 상속, fit/transform 구현 (joblib by-reference 직렬화 대응).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


def _to_category(df):
    """지정한 7개 범주형 컬럼을 pandas category dtype으로 변환.

    HistGradientBoostingClassifier(1.8.0 기본 categorical_features=None)은
    category dtype을 보고 자동으로 범주형으로 처리한다.
    """
    return df.astype({c: "category" for c in CAT_COLS})


# Todo 5: 결측 지시자 대상 컬럼 (8개). 원본 컬럼은 유지하고 _missing만 추가 —
# imputation 금지 (HGB는 NaN을 네이티브로 처리).
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
    """pitcher_id/batter_id → TargetEncoder, pitcher_enc/batter_enc 신규 컬럼 추가.

    원본 ID 유지(C3 add). sklearn 1.8.0: cv=int만, shuffle=False+random_state 검증 없음
    → shuffle=False만 지정. set_output(pandas)로 DataFrame 반환(R1). 훈련 fit_transform은
    cross-fitting(cv=5), transform의 unseen ID는 global_mean_(y_train.mean())으로 대체.
    """

    def __init__(self):
        self.encoder_ = None
        self.global_mean_ = None

    def _make_encoder(self):
        return TargetEncoder(
            cv=5, smooth="auto", target_type="binary", shuffle=False
        ).set_output(transform="pandas")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("IDTargetEncoder.fit: y(타깃) 필수")
        y = np.asarray(y, dtype=float)
        self.encoder_ = self._make_encoder()
        self.encoder_.fit(X[["pitcher_id", "batter_id"]], y)
        self.global_mean_ = float(y.mean())
        return self

    def fit_transform(self, X, y=None):
        """cross-fitting 인코딩 (TargetEncoder.fit_transform이 내부적으로 수행)."""
        if y is None:
            raise ValueError("IDTargetEncoder.fit_transform: y(타깃) 필수")
        y = np.asarray(y, dtype=float)
        self.encoder_ = self._make_encoder()
        enc = self.encoder_.fit_transform(X[["pitcher_id", "batter_id"]], y)
        self.global_mean_ = float(y.mean())
        return self._attach(X, enc)

    def transform(self, X):
        if self.encoder_ is None:
            raise RuntimeError("IDTargetEncoder.transform: fit/fit_transform 먼저 호출")
        enc = self.encoder_.transform(X[["pitcher_id", "batter_id"]])
        return self._attach(X, enc)

    def _attach(self, X, enc):
        # set_output 미적용 환경 대비 ndarray면 DataFrame으로 복원
        if not isinstance(enc, pd.DataFrame):
            enc = pd.DataFrame(
                np.asarray(enc), columns=["pitcher_id", "batter_id"], index=X.index
            )
        out = X.copy()
        out["pitcher_enc"] = enc["pitcher_id"].fillna(self.global_mean_)
        out["batter_enc"] = enc["batter_id"].fillna(self.global_mean_)
        return out


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """MISSING_COLS 8개 컬럼의 결측 여부 boolean 컬럼({col}_missing)을 추가.

    원본 컬럼은 유지하고 imputation하지 않는다 (HGB NaN 네이티브).
    """

    def fit(self, X, y=None):
        missing = [c for c in MISSING_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"MissingIndicatorAdder.fit: 컬럼 부재: {missing}")
        return self

    def transform(self, X):
        out = X.copy()
        for c in MISSING_COLS:
            out[f"{c}_missing"] = out[c].isna()
        return out


class InteractionAdder(BaseEstimator, TransformerMixin):
    """상호작용 피처 3종 추가 (원본 컬럼 유지).

    - base_state_li: base_state별 대표 li(중앙값)의 qcut(5구간, duplicates="drop")
      구간 → ordinal int (0~4). fit에서 qcut edges·base_state 매핑 저장, transform에서 재적용.
      fit에서 보지 못한 base_state → -1 (R5).
    - count_cat: balls_before*3 + strikes_before → 0~11 ordinal (C2).
    - runner_risk: num_runners_on*3 + li 3구간(pd.cut bins=[-1,1,2,1e9],
      include_lowest=True — li==0.0 포함) → ordinal int. bin은 fit에서 저장 (R3).
    """

    RUNNER_RISK_BINS = np.array([-1.0, 1.0, 2.0, 1e9])

    def __init__(self):
        self.li_edges_ = None
        self.base_state_map_ = {}

    def fit(self, X, y=None):
        need = {"base_state", "li", "balls_before", "strikes_before", "num_runners_on"}
        missing = need - set(X.columns)
        if missing:
            raise ValueError(f"InteractionAdder.fit: 컬럼 부재: {sorted(missing)}")
        li = X["li"].astype(float)
        # qcut 5구간 edges 저장 (duplicates="drop" — li==0.0 다수 시 구간 수 감소 가능)
        _, self.li_edges_ = pd.qcut(li, 5, duplicates="drop", retbins=True)
        # base_state별 대표 li(중앙값)가 속한 구간 → 매핑 저장
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
        # base_state_li: unseen base_state(또는 NaN) → -1
        out["base_state_li"] = (
            out["base_state"].map(self.base_state_map_).fillna(-1).astype(int)
        )
        # count_cat: balls(0~3)*3 + strikes(0~2) → 0~11
        out["count_cat"] = (
            out["balls_before"].astype(int) * 3 + out["strikes_before"].astype(int)
        ).astype(int)
        # runner_risk: li 3구간 (li==0.0은 [-1,1] 첫 구간, NaN li는 0으로 처리) × num_runners_on
        li_codes = pd.cut(
            out["li"].astype(float), bins=self.RUNNER_RISK_BINS, include_lowest=True
        ).cat.codes
        li_codes = li_codes.where(li_codes >= 0, 0)
        out["runner_risk"] = (
            out["num_runners_on"].astype(int).fillna(0) * 3 + li_codes
        ).astype(int)
        return out
