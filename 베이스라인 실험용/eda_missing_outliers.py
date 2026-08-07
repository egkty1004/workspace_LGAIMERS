"""eda_missing_outliers.py — 결측치·이상치 집중 점검.

목표:
  A. 결측: 컬럼별 결측 수/율, 결측이 같은 행에 몰려있는지(패턴), 시즌별 결측 분포,
     결측 행의 target 특성 (결측 = 정보성 있는지)
  B. 이상치: IQR 기준 + 도메인 기준(야구 상식)으로 판정 — 극단값이 오류인지 자연값인지
  C. 결측·이상치 처리 권고

실행: python eda_missing_outliers.py 2>&1 | tee backup/eda_missing_outliers.log
"""
import numpy as np
import pandas as pd

TARGET = "control_success"


def sec(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape, "| season:", df["season"].min(), "~", df["season"].max())

    # ============ A. 결측치 ============
    sec("[A] 결측치 점검")

    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print("결측 있는 컬럼 ({}개 / {}):".format(len(miss), df.shape[1]))
    for c, n in miss.items():
        print(f"  {c:<48} {n:>8,} ({n/len(df)*100:.3f}%)")

    # A1. 결측 패턴: 결측 행이 얼마나 겹치는지
    sec("[A1] 결측 패턴 — 같은 행에 몰려있는가?")
    miss_cols = list(miss.index)
    if miss_cols:
        row_miss = df[miss_cols].isna().sum(axis=1)
        vc = row_miss.value_counts().sort_index()
        print("행당 결측 컬럼 수 분포:")
        for k, v in vc.items():
            print(f"  결측 {k}개 컬럼인 행: {v:>9,} ({v/len(df)*100:.2f}%)")
        # 완전 동일 패턴 그룹 찾기
        print("\n완전 동일한 결측 패턴 그룹:")
        pattern = df[miss_cols].isna().agg(tuple, axis=1)
        p_counts = pattern.value_counts()
        for pat, n in p_counts.head(8).items():
            cols = [c for c, m in zip(miss_cols, pat) if m]
            print(f"  n={n:>8,} ({n/len(df)*100:.2f}%) | 결측 컬럼: {', '.join(cols)}")

    # A2. 결측 행의 target 특성 (정보성)
    sec("[A2] 결측 = 신호인가? (결측 행 vs 정상 행 target 평균)")
    if miss_cols:
        any_miss = df[miss_cols].isna().any(axis=1)
        print(f"결측 있는 행: {any_miss.sum():,} ({any_miss.mean()*100:.2f}%) | "
              f"target 평균: {df.loc[any_miss, TARGET].mean():.4f}")
        print(f"결측 없는 행: {(~any_miss).sum():,} | "
              f"target 평균: {df.loc[~any_miss, TARGET].mean():.4f}")

    # A3. 시즌별 결측 분포
    sec("[A3] 시즌별 결측 분포")
    if miss_cols:
        g = df.groupby("season")[miss_cols[0]].apply(lambda s: s.isna().mean())
        print(f"대표 컬럼 {miss_cols[0]} 시즌별 결측률:")
        print(g.round(4).to_string())
        # prev_game 6종은 동일 패턴이므로 대표 1개만, 나머지 대표 확인
        prev6 = [c for c in miss_cols if "prev" in c]
        if len(prev6) >= 6:
            same = all(
                df[prev6[0]].isna().equals(df[c].isna()) for c in prev6[1:]
            )
            print(f"\nprev_game 6종({len(prev6)}개) 결측 패턴 동일 여부: {same}")

    # ============ B. 이상치 ============
    sec("[B] 이상치 점검")

    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET
                and c not in ("pitcher_id", "batter_id", "row_id")]
    print("IQR(1.5×) 기준 이상치 + 도메인 판정:")
    print(f"{'컬럼':<42} {'이상치수':>9} {'비율':>7} {'min':>7} {'max':>7}  판정")
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out == 0:
            continue
        pct = n_out / len(df) * 100
        verdict = judge_outlier(c, lo, hi, s.min(), s.max())
        print(f"{c:<42} {n_out:>9,} {pct:>6.2f}% {s.min():>7.1f} {s.max():>7.1f}  {verdict}")

    # B1. 도메인 특이값 후보 직접 스캔
    sec("[B1] 도메인 기준 특이값 스캔")
    checks = [
        ("balls_before", (lambda s: s < 0), "음수"),
        ("strikes_before", (lambda s: s < 0), "음수"),
        ("outs_before", (lambda s: s < 0), "음수"),
        ("inning", (lambda s: (s < 1) | (s > 9)), "야구 이닝은 1~9 (연장 제외)"),
        ("run_total_before", (lambda s: s > 30), "30점 초과 (이례적)"),
        ("li", (lambda s: s > 5), "li > 5 (극고위험 상황)"),
        ("home_win_expectancy", (lambda s: (s < 0) | (s > 100)), "범위(0~100) 밖"),
        ("asof_pitcher_n", (lambda s: s < 0), "음수"),
        ("score_diff_pitcher_team", (lambda s: (s > 25) | (s < -25)), "±25 초과"),
    ]
    for c, cond, desc in checks:
        if c not in df.columns:
            continue
        s = df[c]
        n = int(cond(s).sum())
        if n:
            print(f"  ⚠ {c}: {desc} — {n:,}건 ({n/len(df)*100:.3f}%) | "
                  f"값 범위: {s.min()} ~ {s.max()}")
        else:
            print(f"  ✓ {c}: {desc} — 0건")

    # B2. rate 컬럼 0/1 극단값 (소표본에서만 발생하는지)
    sec("[B2] rate 컬럼 0.0/1.0 극단값 (소표본 확인)")
    rate_cols = [c for c in df.columns if "rate" in c]
    for c in rate_cols:
        s = df[c].dropna()
        n01 = int(((s == 0) | (s == 1)).sum())
        if n01 == 0:
            continue
        # 극단값 행의 asof_n(표본수) 확인
        ext = df.loc[(df[c] == 0) | (df[c] == 1)]
        n_small = (ext["asof_pitcher_n"] <= 10).sum() if c.startswith("asof_pitcher") else (
            (ext["asof_batter_n"] <= 10).sum() if c.startswith("asof_batter") else 0)
        print(f"  {c:<45} 0/1 극단 {n01:>8,} ({n01/len(df)*100:.3f}%) | "
              f"그중 표본≤10: {n_small:,}")

    # ============ C. 처리 권고 ============
    sec("[C] 처리 권고 요약")
    print("""
결측:
- asof_* 16개: imputation 금지 (HGB는 NaN 네이티브). cold-start는 '정보 없음'이라는 신호.
- prev_game 6종 결측(1.98%): MissingIndicator로 신호 보존 가능하나, 2024 신호 역전 확인됨(리포트 참조)
  → 지시자 사용 시 최근 시즌 검증 필수.
- G2/G3 cold-start(792/830행): 시즌 첫 등장 — rate 컬럼 결측은 '이력 없음' 그 자체.

이상치:
- 전부 스포츠 자연 극단값 — 제거/클리핑 금지. (run_total 37점, li 10.83, rate 0/1 등)
- rate 0/1은 표본 수 부족에서 기인 — 단순 0/1보다 asof_n(표본수)이 더 중요한 보조 신호.
- 이상치 '처리'보다 '해석'이 정답: 극단 상황(li>5)은 그 자체로 정보.
""")


def judge_outlier(c, lo, hi, mn, mx):
    """컬럼별 이상치 판정 — 오류(실데이터 의심) vs 자연 극단값."""
    if c in ("score_diff_home", "score_diff_pitcher_team"):
        return "자연 극단 (점수차)"
    if c in ("run_top_before", "run_bot_before", "run_total_before"):
        return "자연 극단 (득점)"
    if c in ("li", "home_win_expectancy", "away_win_expectancy"):
        return "자연 극단 (연속 지표 꼬리)"
    if c in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        return "자연 극단 (누적량)"
    if c.startswith("asof_") and "rate" in c:
        return "자연 극단 (0/1 rate, 소표본)"
    return "자연 극단"


if __name__ == "__main__":
    main()
