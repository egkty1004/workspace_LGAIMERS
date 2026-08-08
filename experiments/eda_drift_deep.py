"""eda_drift_deep.py — 시즌 드리프트 심층 분석.

목표:
  A. 2025(target=test) base rate 외삽: 선형 추세 / 최근 2시즌 평균 — 2025 예측 기준값
  B. game_type=F(+0.0795)의 진위: 10월(월) 효과와의 교란 분리
  C. middle_rate +35% 증가의 원인: 손별(좌/우)·시즌별 분해
  D. 월×시즌 교차: 드리프트가 시즌 내 월별로 어떻게 진행되는가
  E. prev1_game_missing(데뷔) 인구의 시즌별 성적 추이

실행: python eda_drift_deep.py 2>&1 | tee backup/eda_drift_deep.log
"""
import numpy as np
import pandas as pd

TARGET = "control_success"


def sec(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape)

    # ============ A. 2025 base rate 외삽 ============
    sec("[A] 2025(test) base rate 외삽")
    by_s = df.groupby("season")[TARGET].mean()
    years = np.arange(2019, 2025)
    vals = by_s.values
    # 선형 회귀
    slope, intercept = np.polyfit(years, vals, 1)
    pred_lin = slope * 2025 + intercept
    # 최근 2시즌 평균
    pred_last2 = vals[-2:].mean()
    # 시즌별 가중 (최근 가중) — 가중 평균
    w = np.arange(1, 7) / 21.0
    pred_wavg = (vals * w).sum()
    print("시즌별 target 평균:", dict(zip(years, vals.round(4))))
    print(f"선형 추세: y = {slope:.5f}x + {intercept:.3f} -> 2025 예상 {pred_lin:.4f}")
    print(f"최근 2시즌 평균(2023~24): {pred_last2:.4f}")
    print(f"가중 평균(최근 가중): {pred_wavg:.4f}")
    print(f"참고: 2024 = {vals[-1]:.4f}, 전체 평균 = {df[TARGET].mean():.4f}")

    # ============ B. game_type=F 진위 — 월 효과와 분리 ============
    sec("[B] game_type=F 신호의 진위 (10월/포스트시즌 효과 분리)")
    # F가 존재하는 월 확인
    f_by_month = df[df["game_type"] == "F"].groupby("game_month")[TARGET].agg(["mean", "count"])
    r_by_month = df[df["game_type"] == "R"].groupby("game_month")[TARGET].agg(["mean", "count"])
    print("F(포스트시즌) 월별 분포:")
    print(f_by_month.round(4).to_string())
    print("\nR(정규) 월별 분포:")
    print(r_by_month.round(4).to_string())
    # 같은 월 내 F vs R 비교 (10월만 있으면 월 교란 확인)
    f_oct = df[(df["game_type"] == "F") & (df["game_month"] == 10)][TARGET]
    r_oct = df[(df["game_type"] == "R") & (df["game_month"] == 10)][TARGET]
    print(f"\n10월 내 F vs R: F={f_oct.mean():.4f}(n={len(f_oct)}) R={r_oct.mean():.4f}(n={len(r_oct)})")
    # 시즌별 F 비율
    f_share = df.groupby("season")["game_type"].apply(lambda s: (s == "F").mean())
    print("\n시즌별 F 비율:")
    print(f_share.round(4).to_string())

    # ============ C. middle_rate 증가 원인 — 손별 분해 ============
    sec("[C] middle_rate +35% 증가의 원인 (손별 분해)")
    mid = "asof_pitcher_middle_rate"
    d = df.dropna(subset=[mid]).copy()
    d["hand"] = d["pitcher_hand"].map({1: "우투", 2: "좌투"})
    piv = d.pivot_table(index="season", columns="hand", values=mid, aggfunc="mean")
    print(piv.round(4).to_string())
    # reverse_rate도 함께
    rev = "asof_pitcher_reverse_rate"
    d2 = df.dropna(subset=[rev]).copy()
    d2["hand"] = d2["pitcher_hand"].map({1: "우투", 2: "좌투"})
    piv2 = d2.pivot_table(index="season", columns="hand", values=rev, aggfunc="mean")
    print("\nreverse_rate 손별:")
    print(piv2.round(4).to_string())
    # fastball 비율 변화 (구종 구성 변화?)
    fb = "asof_pitcher_fastball_rate"
    d3 = df.dropna(subset=[fb]).copy()
    d3["hand"] = d3["pitcher_hand"].map({1: "우투", 2: "좌투"})
    piv3 = d3.pivot_table(index="season", columns="hand", values=fb, aggfunc="mean")
    print("\nfastball_rate 손별:")
    print(piv3.round(4).to_string())

    # ============ D. 월×시즌 교차 ============
    sec("[D] 월×시즌 target 교차 (드리프트의 시즌 내 진행)")
    cross = df.pivot_table(index="season", columns="game_month", values=TARGET, aggfunc="mean")
    print(cross.round(4).to_string())
    # 시즌 내 월 슬로프 (3월→9월 변화) 시즌별
    slopes = {}
    for s in range(2019, 2025):
        sub = df[df["season"] == s]
        m = sub.groupby("game_month")[TARGET].mean()
        if 3 in m.index and 9 in m.index:
            slopes[s] = m[9] - m[3]
    print("\n시즌 내 3월→9월 target 변화:")
    for s, v in slopes.items():
        print(f"  {s}: {v:+.4f}")

    # ============ E. 데뷔(missing) 인구 시즌별 성적 ============
    sec("[E] prev1_game 결측(데뷔) 인구의 시즌별 성적")
    prev1 = "asof_pitcher_prev1_game_success_rate"
    d4 = df.copy()
    d4["is_debut"] = d4[prev1].isna()
    g = d4.groupby(["season", "is_debut"])[TARGET].mean().unstack()
    g.columns = ["기존(prev 있음)", "데뷔(prev 결측)"]
    print(g.round(4).to_string())
    g["delta"] = g["데뷔(prev 결측)"] - g["기존(prev 있음)"]
    print("\ndelta(데뷔-기존):")
    print(g["delta"].round(4).to_string())
    # 데뷔 인구 규모
    n_debut = d4.groupby("season")["is_debut"].mean()
    print("\n데뷔 비율:")
    print(n_debut.round(4).to_string())


if __name__ == "__main__":
    main()
