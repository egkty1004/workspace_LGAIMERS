"""eda_descriptive_ts.py — 기술통계(왜도/첨도/최빈값/분산) + 시계열 추세 종합 분석.

목표:
  A. 수치형 컬럼 기술통계: min/max/mean/median/var/std/skew/kurt/mode + 분포 유형 판정
  B. 시계열: 컬럼별 시즌 평균 변화 (표준화 delta), 추세 강도 (monotonicity)
  C. target과 주요 피처의 시즌별 공변화 (드리프트 동조 여부)

실행: python eda_descriptive_ts.py 2>&1 | tee backup/eda_descriptive_ts.log
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 100)

TARGET = "control_success"


def sec(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape)

    # ============ A. 기술통계 ============
    sec("[A] 수치형 컬럼 기술통계 + 분포 유형 판정")
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and c not in (TARGET, "pitcher_id", "batter_id", "row_id")]
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        mode_series = s.mode()
        mode_val = mode_series.iloc[0] if len(mode_series) else np.nan
        mode_share = (s == mode_val).mean() if pd.notna(mode_val) else np.nan
        rows.append(dict(
            col=c, n=len(s), min=s.min(), max=s.max(), mean=s.mean(),
            median=s.median(), var=s.var(), std=s.std(),
            skew=s.skew(), kurt=s.kurt(), mode=mode_val, mode_share=mode_share,
        ))
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    tab.to_csv("backup/desc_stats.csv", index=False)

    # 분포 유형 판정 (skew/kurt 기준 + 값 특성)
    sec("[A2] 분포 유형 판정 (자동 라벨)")
    for _, r in tab.iterrows():
        label = classify_dist(r)
        print(f"  {r['col']:<42} skew={r['skew']:>8.2f} kurt={r['kurt']:>9.2f} "
              f"mode={r['mode']:>10.3f} ({(r['mode_share']*100):>5.1f}%) -> {label}")

    # ============ B. 시계열 추세 ============
    sec("[B] 시계열 — 시즌별 평균 변화 (전체 대비 표준화 delta)")
    g = df.groupby("season")
    tgt_by_season = g[TARGET].mean()
    seasons = sorted(df["season"].unique())
    rows = []
    for c in num_cols + [TARGET]:
        by_s = g[c].mean()
        overall = df[c].mean()
        delta_last = by_s[seasons[-1]] - by_s[seasons[0]]
        rel_last = (by_s[seasons[-1]] - overall) / (overall + 1e-12)
        # 단조 추세 (2019->2024 부호 일관성)
        diffs = np.sign(np.diff(by_s.values))
        mono_frac = (diffs == diffs[0]).mean() if len(diffs) else 1.0
        rows.append(dict(col=c, s2019=by_s[seasons[0]], s2024=by_s[seasons[-1]],
                         delta=delta_last, rel_delta=rel_last, mono_frac=mono_frac))
    ts = pd.DataFrame(rows).sort_values("rel_delta", key=abs, ascending=False)
    print("전체 대비 2024 편차(rel_delta) 큰 순 (|값|):")
    print(ts.round(4).to_string(index=False))
    ts.to_csv("backup/ts_season_trend.csv", index=False)

    # ============ C. target 드리프트와의 동조 ============
    sec("[C] target과 피처의 시즌별 동조 (corr of season-means)")
    feat_means = g[num_cols].mean()
    tgt_means = g[TARGET].mean()
    corrs = feat_means.corrwith(tgt_means)
    corrs = corrs.sort_values(key=abs, ascending=False)
    print("시즌별 평균이 target 평균과 함께 움직이는 피처 (상관 절대값 순):")
    for c, v in corrs.items():
        if pd.notna(v):
            print(f"  {c:<42} {v:+.4f}")
    corrs.to_csv("backup/ts_target_corr.csv")

    # ============ D. 월별 주기성 (게임월) ============
    sec("[D] 월별 패턴 (게임 진행 시즌 내 주기성)")
    m = df.groupby("game_month")[TARGET].agg(["mean", "count"])
    m["count_share"] = m["count"] / m["count"].sum()
    print(m.round(4).to_string())
    # 시즌별 월분포 (2024만)
    print("\n2024 월별 target:")
    print(df[df["season"] == 2024].groupby("game_month")[TARGET].mean().round(4).to_string())


def classify_dist(r):
    """분포 유형 판정."""
    if r["max"] - r["min"] < 1e-9:
        return "상수"
    if r["col"].startswith(("runner_", "num_")) or r["col"] in ("balls_before", "strikes_before", "outs_before"):
        return "저차원 이산 (카운트)"
    if r["col"] == "inning":
        return "이산 (1~13, 연장 포함)"
    if r["skew"] > 1.5:
        return f"강한 우측 꼬리 (log 정규/지수적)"
    if r["skew"] > 0.5:
        return "우측 꼬리"
    if r["skew"] < -0.5:
        return "좌측 꼬리"
    if abs(r["kurt"]) < 1:
        return "대략 정규/평평"
    return "종 모양 (정규 근사)"


if __name__ == "__main__":
    main()
