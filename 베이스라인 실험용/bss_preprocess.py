"""bss_preprocess.py — HGB 학습 공용 전처리 모듈.

모든 정의는 모듈 레벨에 둔다 (N1):
- joblib 1.5.3은 함수를 by-reference(`bss_preprocess._to_category`)로 직렬화하므로
  __main__ 블록이나 람다 안에서 정의하면 새 프로세스/평가 서버 로드 시
  AttributeError가 발생한다.
- 반드시 모듈 레벨 함수로만 정의하고, train_hgb.py 등에서 import 하여 사용한다.
- 9개 후보 중 pitcher_id/batter_id는 sklearn HGB 범주형 카디널리티 255 제한(실측 711/742 > 255)으로 수치형(int) 처리 — 사용자 승인 옵션 A.
"""

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


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InteractionAdder(BaseEstimator, TransformerMixin):
    """상호작용 피처 3종 추가 (원본 컬럼 유지).

    - base_state_li: base_state별 대표 li(중앙값)의 qcut(5구간, duplicates="drop")
      구간 → ordinal int (0~4). fit에서 qcut edges·base_state 매핑 저장, transform에서 재적용.
      fit에서 보지 못한 base_state → -1 (R5).
    - count_cat: balls_before*3 + strikes_before → 0~11 ordinal (C2).
    - runner_risk: num_runners_on*3 + li 3구간(pd.cut bins=[-1,1,2,1e9],
      include_lowest=True — li==0.0 포함) → ordinal int. bin은 fit에서 저장 (R3).

    게이트 검증 (2026-08-08, diag_stage_ablation.py): base(47)+interact(3) = 2024 BSS 494.36
    (Wave A 439.00 대비 +55.36). 모듈 레벨 정의 (N1 — joblib by-reference 직렬화).
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
