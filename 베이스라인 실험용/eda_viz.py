"""eda_viz.py — 데이터 시각화 배치 (그래프 PNG 생성).

저장: docs/plots/*.png (git 추적 — 시각화 산출물)
그래프:
  01 시즌 드리프트 (target + 주요 피처 추세)
  02 주요 컬럼 분포 히스토그램
  03 월별 패턴 (시즌 내 주기성)
  04 count_cat × target (단조 신호)
  05 base_state × target
  06 asof_pitcher_success_rate × target (최근 컨디션)
  07 game_type F/R × target
  08 상관 히트맵 (수치형)
  09 ID 시즌 간 안정성
  10 결측 패턴

실행: python eda_viz.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

TARGET = "control_success"
OUT = "docs/plots"
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "base_state", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

NUM_COLS = [c for c in FEATURES if pd.api.types.is_numeric_dtype(pd.Series(dtype=float)) or True]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {path}")


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape)

    # ---- 01 시즌 드리프트 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    by_s = df.groupby("season")[TARGET].mean()
    axes[0].plot(by_s.index, by_s.values, "o-", color="crimson", lw=2)
    axes[0].axhline(df[TARGET].mean(), ls="--", color="gray", label=f"전체 평균 {df[TARGET].mean():.4f}")
    axes[0].set_title("시즌별 제구 성공률 (드리프트)")
    axes[0].set_xlabel("season"); axes[0].set_ylabel("control_success 평균")
    axes[0].legend()
    for c, lbl in [("asof_pitcher_middle_rate", "middle_rate(우축)"),
                   ("asof_pitcher_success_rate", "pitcher_success(우축)")]:
        m = df.groupby("season")[c].mean()
        ax2 = axes[1].twinx()
        ax2.plot(m.index, m.values, "s--", alpha=0.6, label=lbl)
    axes[1].set_title("시즌별 주요 피처 추세")
    axes[1].set_xlabel("season")
    save(fig, "01_season_drift.png")

    # ---- 02 주요 컬럼 분포 ----
    hist_cols = ["li", "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
                 "home_win_expectancy", "run_total_before", "asof_pitcher_n"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), hist_cols):
        s = df[c].dropna()
        ax.hist(s, bins=60, color="steelblue", alpha=0.8)
        ax.set_title(f"{c}\n(n={len(s):,})")
        ax.set_yscale("log")
    fig.suptitle("주요 컬럼 분포 (로그 스케일)")
    save(fig, "02_distributions.png")

    # ---- 03 월별 패턴 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m = df.groupby("game_month")[TARGET].mean()
    axes[0].plot(m.index, m.values, "o-", color="darkorange", lw=2)
    axes[0].set_title("월별 target 평균 (전체)")
    axes[0].set_xlabel("game_month"); axes[0].set_ylabel("평균")
    for s in (2019, 2022, 2024):
        mm = df[df["season"] == s].groupby("game_month")[TARGET].mean()
        axes[1].plot(mm.index, mm.values, "o--", label=str(s))
    axes[1].set_title("월별 target (시즌별)")
    axes[1].set_xlabel("game_month"); axes[1].legend()
    save(fig, "03_month_pattern.png")

    # ---- 04 count_cat × target ----
    df["count_cat"] = df["balls_before"] * 3 + df["strikes_before"]
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df.groupby("count_cat")[TARGET].mean()
    sizes = df.groupby("count_cat").size()
    ax.bar(counts.index, counts.values, color="teal")
    ax.axhline(df[TARGET].mean(), ls="--", color="gray")
    for i, n in sizes.items():
        ax.text(i, counts[i] + 0.002, f"{n/1000:.0f}k", ha="center", fontsize=8)
    ax.set_title("count(balls*3+strikes)별 제구 성공률 — 단조 신호")
    ax.set_xlabel("count_cat (0-0 ~ 3-2)")
    ax.set_ylim(0.49, 0.54)
    save(fig, "04_count_cat.png")

    # ---- 05 base_state × target ----
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
    m = df.groupby("base_state")[TARGET].mean().reindex(order)
    n = df.groupby("base_state").size().reindex(order)
    ax.bar(m.index, m.values, color="purple", alpha=0.8)
    ax.axhline(df[TARGET].mean(), ls="--", color="gray")
    for i, (bs, c) in enumerate(zip(m.index, n)):
        ax.text(i, m[bs] + 0.001, f"{c/1000:.0f}k", ha="center", fontsize=8)
    ax.set_title("base_state별 제구 성공률")
    ax.set_ylim(0.515, 0.54)
    save(fig, "05_base_state.png")

    # ---- 06 asof 최근 컨디션 × target ----
    fig, ax = plt.subplots(figsize=(10, 5))
    d = df.dropna(subset=["asof_pitcher_success_rate"])
    d["cond_q"] = pd.qcut(d["asof_pitcher_success_rate"], 10, duplicates="drop")
    m = d.groupby("cond_q", observed=True)[TARGET].mean()
    ax.plot(range(len(m)), m.values, "o-", color="green", lw=2)
    ax.set_title("투수 최근 컨디션(asof_success_rate) 10분위별 target — 단조 관계")
    ax.set_xlabel("컨디션 분위(낮→높)")
    ax.set_ylabel("target 평균")
    save(fig, "06_condition.png")

    # ---- 07 game_type F/R ----
    fig, ax = plt.subplots(figsize=(10, 5))
    gt = df.groupby("game_type")[TARGET].agg(["mean", "count"])
    ax.bar(gt.index, gt["mean"], color=["steelblue", "crimson"])
    for i, (t, r) in enumerate(gt.iterrows()):
        ax.text(i, r["mean"] + 0.003, f"n={r['count']:,}", ha="center")
    ax.axhline(df[TARGET].mean(), ls="--", color="gray", label="전체")
    ax.set_title("game_type별 제구 성공률 (F=포스트시즌 +0.08)")
    ax.set_ylim(0.50, 0.62)
    ax.legend()
    save(fig, "07_game_type.png")

    # ---- 08 상관 히트맵 ----
    corr_cols = ["season", "game_month", "inning", "balls_before", "strikes_before",
                 "outs_before", "run_total_before", "score_diff_pitcher_team",
                 "num_runners_on", "li", "home_win_expectancy",
                 "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                 "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                 "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
                 "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
                 "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                 "asof_pitcher_offspeed_rate", TARGET]
    corr = df[corr_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if abs(corr.iloc[i, j]) > 0.25:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("수치형 컬럼 상관 (spearman, |r|>0.25만 표기)")
    save(fig, "08_corr_heatmap.png")

    # ---- 09 ID 시즌 간 안정성 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, id_col, title in [
        (axes[0], "pitcher_id", "투수"),
        (axes[1], "batter_id", "타자"),
    ]:
        means = df.groupby([id_col, "season"])[TARGET].mean().unstack()
        counts = df.groupby([id_col, "season"]).size().unstack()
        valid = counts >= 50
        corrs = []
        for s in range(2019, 2024):
            m1, m2 = means[s], means[s + 1]
            ok = valid[s] & valid[s + 1] & m1.notna() & m2.notna()
            if ok.sum() >= 10:
                corrs.append((s, m1[ok].corr(m2[ok])))
        if corrs:
            x = [c[0] for c in corrs]; y = [c[1] for c in corrs]
            ax.plot(x, y, "o-", lw=2)
            ax.axhline(0, ls="--", color="gray")
            ax.set_title(f"{title} 시즌 간 성적 재현성 (corr)")
            ax.set_ylim(-0.3, 0.8)
    save(fig, "09_id_stability.png")

    # ---- 10 결측 패턴 ----
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    fig, ax = plt.subplots(figsize=(12, 4))
    miss = df[miss_cols].isna().mean().sort_values()
    ax.barh(miss.index, miss.values * 100, color="darkred")
    ax.set_xlabel("결측 비율 (%)")
    ax.set_title("컬럼별 결측 비율")
    save(fig, "10_missing.png")

    print("\n모든 그래프 저장 완료:", OUT)


if __name__ == "__main__":
    main()
