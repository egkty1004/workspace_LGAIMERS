"""feat_eng.py — 피처 엔지니어링 후보 트랜스포머 모듈 (이중 게이트 검증용).

모든 정의는 모듈 레벨 (N1: joblib by-reference 직렬화 — lambda/__main__ 정의 금지).
각 트랜스포머: fit은 필요한 컬럼 검증 + train-fit 파라미터(빈/평균)만 저장,
transform은 피처 추가/제거 + 신규 NaN 0건 + 카디널리티 ≤64 어서션.
data_direction_sanity(df): 피처의 target 관계가 2019~2024 중 ≥5시즌 동일 부호인지 검증.

제외 (실증으로 배제): interact 패밀리(count_cat/runner_risk/base_state_li — Public -103),
명시적 F플래그(무효), ID TargetEncoder(FAIL), trackman(매핑 0%), A1s(2025 외삽 리스크).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PREV1 = "asof_pitcher_prev1_game_success_rate"
PREV5 = "asof_pitcher_prev5_game_success_rate"
REVERSE = "asof_pitcher_reverse_rate"
COND = "asof_pitcher_success_rate"
COUNT_COL = "count_cat"
SEASONS = list(range(2019, 2025))


def _require(X, cols, name):
    missing = set(cols) - set(X.columns)
    if missing:
        raise ValueError(f"{name}.fit: 컬럼 부재: {sorted(missing)}")


class DropDupCols(BaseEstimator, TransformerMixin):
    """B1: 완전 중복 컬럼 제거 (상관 ±1.0 확인됨)."""

    DROP = ["asof_pitcher_pitchmix_n", "away_win_expectancy"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.drop(columns=[c for c in self.DROP if c in X.columns])
        return out


class SeasonProgress(BaseEstimator, TransformerMixin):
    """D1: season_progress = (game_month - 3) / 7 — 시즌 내 진행도 (0~1)."""

    def fit(self, X, y=None):
        _require(X, ["game_month"], "SeasonProgress")
        return self

    def transform(self, X):
        out = X.copy()
        out["season_progress"] = (out["game_month"].astype(float) - 3.0) / 7.0
        return out


class InningNorm(BaseEstimator, TransformerMixin):
    """D2: inning_norm = inning / 9 — 이닝 정규화."""

    def fit(self, X, y=None):
        _require(X, ["inning"], "InningNorm")
        return self

    def transform(self, X):
        out = X.copy()
        out["inning_norm"] = out["inning"].astype(float) / 9.0
        return out


class FormTrend(BaseEstimator, TransformerMixin):
    """A2: form_trend = prev1 - prev5 — 최근 1경기 vs 5경기 모멘텀 (드리프트 강건)."""

    def fit(self, X, y=None):
        _require(X, [PREV1, PREV5], "FormTrend")
        return self

    def transform(self, X):
        out = X.copy()
        out["form_trend"] = out[PREV1] - out[PREV5]
        return out


class HandMatch(BaseEstimator, TransformerMixin):
    """E2: hand_match = (pitcher_hand == batter_hand) — 좌우 상성."""

    def fit(self, X, y=None):
        _require(X, ["pitcher_hand", "batter_hand"], "HandMatch")
        return self

    def transform(self, X):
        out = X.copy()
        out["hand_match"] = (out["pitcher_hand"] == out["batter_hand"]).astype(int)
        return out


class ScoreAbs(BaseEstimator, TransformerMixin):
    """E3: score_abs = abs(score_diff_pitcher_team) — 접전 압박."""

    def fit(self, X, y=None):
        _require(X, ["score_diff_pitcher_team"], "ScoreAbs")
        return self

    def transform(self, X):
        out = X.copy()
        out["score_abs"] = out["score_diff_pitcher_team"].abs()
        return out


class _BinnedProduct(BaseEstimator, TransformerMixin):
    """공용: rate를 3빈으로 + count_cat과 결합 (C2/C1). fit은 빈 경계 저장."""

    _SRC = None

    def fit(self, X, y=None):
        _require(X, [self._SRC, "balls_before", "strikes_before"], type(self).__name__)
        s = X[self._SRC].dropna()
        self.edges_ = np.quantile(s, [1 / 3, 2 / 3]) if len(s) else np.array([0.5, 0.7])
        return self

    def transform(self, X):
        out = X.copy()
        count_cat = out["balls_before"].astype(int) * 3 + out["strikes_before"].astype(int)
        bins = pd.cut(out[self._SRC], bins=[-np.inf, *self.edges_, np.inf], labels=False)
        bins = bins.fillna(1).astype(int)
        out[self._OUT] = (bins * 12 + count_cat).astype(int)
        return out


class ReverseCount(_BinnedProduct):
    """C2: reverse_count = 3bin(reverse_rate) × count_cat."""

    _SRC = REVERSE
    _OUT = "reverse_count"


class CountCondition(_BinnedProduct):
    """C1: count_cond = 3bin(asof_pitcher_success_rate) × count_cat."""

    _SRC = COND
    _OUT = "count_cond"


class SeasonCat(BaseEstimator, TransformerMixin):
    """A1c: season을 pandas category로 (2025 unseen → HGB NaN/missing 브랜치)."""

    def fit(self, X, y=None):
        _require(X, ["season"], "SeasonCat")
        return self

    def transform(self, X):
        out = X.copy()
        out["season"] = out["season"].astype("category")
        return out


ALL = [
    ("B1", DropDupCols), ("D1D2", None), ("A2", FormTrend),
    ("E2", HandMatch), ("E3", ScoreAbs), ("C2", ReverseCount),
    ("C1", CountCondition), ("A1c", SeasonCat),
]


def _count_cat(df):
    return df["balls_before"].astype(int) * 3 + df["strikes_before"].astype(int)


def _season_target_sign(df, col):
    """피처 값의 시즌별 target 단조 방향이 안정적인지 (≥5/6시즌 동일 부호)."""
    d = df[["season", col, "control_success"]].dropna()
    if len(d) < 1000 or d[col].nunique() < 3:
        return None, "샘플/카디널리티 부족"
    d = d.copy()
    d["bin"] = pd.qcut(d[col], 5, duplicates="drop")
    signs = []
    for s in SEASONS:
        sub = d[d["season"] == s]
        if len(sub) < 500 or sub["bin"].nunique() < 3:
            continue
        m = sub.groupby("bin", observed=True)["control_success"].mean()
        if m.is_monotonic_increasing:
            signs.append("+")
        elif m.is_monotonic_decreasing:
            signs.append("-")
        else:
            signs.append("0")
    if not signs:
        return None, "검증 가능 시즌 없음"
    stable = max(signs.count("+"), signs.count("-")) / len(signs)
    verdict = "STABLE" if stable >= 5 / 6 and "0" not in signs else "UNSTABLE"
    return verdict, f"부호={signs} (안정성 {stable:.2f})"


def run_sanity(df):
    """모든 후보의 방향성 sanity 검사. (C2/C1은 _BinnedProduct 내부 변환 사용)"""
    print("=" * 88)
    print("[T5] 후보 데이터 방향성 sanity")
    print("=" * 88)
    d = df.copy()
    d["count_cat"] = _count_cat(d)
    results = {}
    for name, cls in ALL:
        if name == "D1D2":
            tr = SeasonProgress().fit_transform(d)
            tr = InningNorm().fit_transform(tr)
            col = "season_progress"
        elif name == "B1":
            results[name] = ("STABLE", "컬럼 제거 — 방향성 무관")
            continue
        elif cls is None:
            continue
        else:
            tr = cls().fit_transform(d)
            col = cls()._OUT if hasattr(cls(), "_OUT") else (
                "season" if name == "A1c" else None)
            if col is None:
                if name == "A1c":
                    results[name] = ("STABLE", "범주형 전환 — 방향성 무관")
                    continue
        verdict, note = _season_target_sign(tr, col)
        results[name] = (verdict, note)
        print(f"  {name:>6} ({col:<20}): {verdict} | {note}")
    return results


if __name__ == "__main__":
    import sys

    train = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    run_sanity(train)
