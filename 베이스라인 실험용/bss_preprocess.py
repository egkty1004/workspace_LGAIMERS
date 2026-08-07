"""bss_preprocess.py — HGB 학습 공용 전처리 모듈.

모든 정의는 모듈 레벨에 둔다 (N1):
- joblib 1.5.3은 함수를 by-reference(`bss_preprocess._to_category`)로 직렬화하므로
  __main__ 블록이나 람다 안에서 정의하면 새 프로세스/평가 서버 로드 시
  AttributeError가 발생한다.
- 반드시 모듈 레벨 함수로만 정의하고, train_hgb.py 등에서 import 하여 사용한다.
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
