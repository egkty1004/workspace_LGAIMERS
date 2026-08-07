"""eda_fail_diag.py — Wave B 게이트 FAIL 원인 진단용 EDA (모델 fit 없음, 데이터 레벨).

핵심 질문: 왜 60컬럼 전처리(특히 enc)가 2024에서 상수 예측보다 나쁜 Brier(0.274)를 만들었나?
가설: 과거 시즌 기반 ID 인코딩(2019~2023 평균)이 2024 예측력이 없거나 역전 → 모델이
강하게 신뢰한 enc 신호가 2024에서 정반대로 작동.

검증 항목 (모두 pandas groupby, fit 없음):
  A. 시즌별 target 평균 (드리프트 재확인)
  B. pitcher_enc 드리프트: 각 시즌 s에서 "이전 시즌(<s) 평균 성공률" vs "시즌 s 성공률"의
     상관/절대편차 — 2024에서 예측력이 무너지는지
  C. batter_enc 동일
  D. 2023→2024 ID별 성공률 재현성 (연속 등장 ID만): corr
  E. count_cat별 target 평균의 시즌별 안정성 (단조 신호가 2024에도 유지?)
  F. prev_game missing 지시자의 시즌별 target 차이 (+0.0276 신호가 2024에도?)
  G. asof_pitcher_success_rate (최근 컨디션) quantile별 target — 시즌별
  H. li 분포 시즌별 (base_state_li/runner_risk bin 입력)

실행 (CWD = 베이스라인 실험용/):
    python eda_fail_diag.py 2>&1 | tee backup/eda_fail_diag.log
"""
import pandas as pd
import numpy as np

TARGET = "control_success"
PREV_GAME_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
]


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape, "| season:", df["season"].min(), "~", df["season"].max())

    # ---- A. 시즌별 target 평균 ----
    section("A. 시즌별 target 평균 (드리프트)")
    a = df.groupby("season")[TARGET].agg(["count", "mean"])
    a.columns = ["n", "target_mean"]
    print(a.round(4).to_string())

    # ---- B/C. pitcher/batter enc 드리프트: 이전 시즌 누적 평균 vs 당해 시즌 ----
    section("B/C. ID 인코딩 예측력: 이전 시즌(<s) 성공률 vs 시즌 s 성공률")
    for id_col, label in [("pitcher_id", "pitcher"), ("batter_id", "batter")]:
        print(f"\n--- {label} ---")
        rows = []
        for s in range(2020, 2025):
            mask_s = df["season"] == s
            prior = df[df["season"] < s]
            # 이전 시즌 ID별 평균 (enc와 유사 — smoothing 없음, 근사)
            prior_mean = prior.groupby(id_col)[TARGET].mean().rename("prior_mean")
            cur = df[mask_s].merge(prior_mean, left_on=id_col, right_index=True, how="left")
            cur = cur.dropna(subset=["prior_mean"])
            corr = cur["prior_mean"].corr(cur[TARGET])
            # prior_mean 분위별 당해 시즌 실제 성공률 (단조성 유지?)
            cur["q"] = pd.qcut(cur["prior_mean"], 4, duplicates="drop")
            qt = cur.groupby("q", observed=True)[TARGET].agg(["mean", "count"])
            rows.append(dict(season=s, n=len(cur), corr=corr))
            print(f"  season={s}: n={len(cur):>7} corr(prior_mean, target)={corr:+.4f}")
            for q, r in qt.iterrows():
                print(f"      {q}: target_mean={r['mean']:.4f} n={int(r['count']):>6}")
        pd.DataFrame(rows).to_csv(f"backup/eda_{label}_enc_drift.csv", index=False)

    # ---- D. 2023->2024 연속 등장 ID 재현성 ----
    section("D. 2023->2024 연속 등장 ID의 성공률 재현성")
    for id_col, label in [("pitcher_id", "pitcher"), ("batter_id", "batter")]:
        m23 = df.loc[df["season"] == 2023].groupby(id_col)[TARGET].mean()
        m24 = df.loc[df["season"] == 2024].groupby(id_col)[TARGET].mean()
        both = pd.concat([m23.rename("rate23"), m24.rename("rate24")], axis=1).dropna()
        print(f"  {label}: 연속 등장 ID {len(both)}개 | "
              f"corr(rate23, rate24)={both['rate23'].corr(both['rate24']):+.4f} | "
              f"rate23 mean={both['rate23'].mean():.4f} rate24 mean={both['rate24'].mean():.4f}")

    # ---- E. count_cat 시즌별 안정성 ----
    section("E. count(balls*3+strikes)별 target — 시즌별")
    df["count_cat"] = df["balls_before"].astype(int) * 3 + df["strikes_before"].astype(int)
    piv = df.pivot_table(index="count_cat", columns="season", values=TARGET, aggfunc="mean")
    print(piv.round(4).to_string())

    # ---- F. prev_game missing 지시자 시즌별 신호 ----
    section("F. prev_game missing 지시자 시즌별 target 차이")
    frows = []
    for c in PREV_GAME_COLS:
        print(f"\n--- {c} ---")
        for s in range(2019, 2025):
            sub = df[df["season"] == s]
            miss = sub[c].isna()
            if miss.sum() < 10:
                continue
            m_miss = sub.loc[miss, TARGET].mean()
            m_pres = sub.loc[~miss, TARGET].mean()
            delta = m_miss - m_pres
            frows.append(dict(col=c, season=s, n_miss=int(miss.sum()),
                              miss_rate=miss.mean(), delta=delta))
            print(f"  {s}: missing n={int(miss.sum()):>6} ({miss.mean():.4f}) | "
                  f"target miss={m_miss:.4f} pres={m_pres:.4f} delta={delta:+.4f}")
    pd.DataFrame(frows).to_csv("backup/eda_missing_signal.csv", index=False)

    # ---- G. asof_pitcher_success_rate (최근 컨디션) quantile별 target — 시즌별 ----
    section("G. asof_pitcher_success_rate quantile별 target — 시즌별 (최근 컨디션 신호)")
    g = df.dropna(subset=["asof_pitcher_success_rate"]).copy()
    g["cond_q"] = pd.qcut(g["asof_pitcher_success_rate"], 5, duplicates="drop")
    piv2 = g.pivot_table(index="cond_q", columns="season", values=TARGET, aggfunc="mean")
    print(piv2.round(4).to_string())

    # ---- H. li 분포 시즌별 ----
    section("H. li 분포 시즌별 (base_state_li/runner_risk bin 입력)")
    h = df.groupby("season")["li"].describe(percentiles=[0.5, 0.9, 0.99])[
        ["50%", "90%", "99%", "max"]]
    print(h.round(3).to_string())

    print("\n완료: backup/eda_* 파일 저장")


if __name__ == "__main__":
    main()
