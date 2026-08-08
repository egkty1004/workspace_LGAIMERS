"""eda_explore_more.py — 6개 영역 추가 데이터 분석 (그래프+수치).

1. 매치업 심층: pitcher_hand × batter_hand 4조합 + 손×카운트 + 손×시즌
2. 투수 유형별: asof_pitcher_n 구간(신규/불펜/선발) × target, 피로도
3. 상황 조합 교차: count × 컨디션, F 내부 컨디션, 주자 × count
4. 팀/홈 원정: pitcher_team_id별, top_bottom(홈/원정)
5. (모델) Wave A 확률 분포 + calibration — 별도 실행
6. 데이터 품질: 완전 중복 행, ID 회전율(신규/이탈)

실행: python eda_explore_more.py 2>&1 | tee backup/eda_explore_more.log
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

TARGET = "control_success"


def sec(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def main():
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("train:", df.shape)

    # ============ 1. 매치업 심층 ============
    sec("[1] 매치업 심층 — pitcher_hand × batter_hand 4조합")
    d = df.copy()
    d["ph"] = d["pitcher_hand"].map({1: "우투", 2: "좌투"})
    d["bh"] = d["batter_hand"].map({1: "우타", 2: "좌타"})
    d["matchup"] = d["ph"] + "×" + d["bh"]
    print(d.groupby("matchup")[TARGET].agg(["mean", "count"]).round(4).to_string())

    sec("[1b] 손×카운트 교차 (풀카운트/스트라이크 우위)")
    d["count_cat"] = d["balls_before"] * 3 + d["strikes_before"]
    d["count_grp"] = np.select(
        [d["count_cat"] == 11, d["count_cat"] == 1,
         d["count_cat"].isin([9, 10]), d["count_cat"] == 0],
        ["풀카운트(3-2)", "스트라이크우위(0-1)", "볼우위(3-0/3-1)", "시작(0-0)"],
        default="기타")
    piv = d.pivot_table(index="matchup", columns="count_grp", values=TARGET,
                        aggfunc="mean")
    print(piv.round(4).to_string())

    sec("[1c] 손×시즌 — 상성 효과가 시즌 드리프트에도 유지되는가")
    piv2 = d.pivot_table(index="season", columns="matchup", values=TARGET, aggfunc="mean")
    print(piv2.round(4).to_string())
    # 대표 상성: 우투×우타 vs 우투×좌타 (같은 시즌 내 비교로 드리프트 제거)
    d["platoon"] = np.select(
        [(d["ph"] == "우투") & (d["bh"] == "좌타"), (d["ph"] == "우투") & (d["bh"] == "우타"),
         (d["ph"] == "좌투") & (d["bh"] == "우타"), (d["ph"] == "좌투") & (d["bh"] == "좌타")],
        ["우투vs좌타", "우투vs우타", "좌투vs우타", "좌투vs좌타"], default=None)
    t = d.pivot_table(index="season", columns="platoon", values=TARGET, aggfunc="mean")
    t["이점"] = t["우투vs좌타"] - t["우투vs우타"]  # 좌타 우위(스위치 개념)
    print("\n우투수 상대로 좌타 vs 우타 (플래툰 이점):")
    print(t.round(4).to_string())

    # ============ 2. 투수 유형별 ============
    sec("[2] 투수 유형별 — asof_pitcher_n 구간 × target")
    d2 = df.copy()
    d2["n_grp"] = pd.cut(d2["asof_pitcher_n"],
                         bins=[-1, 0, 50, 200, 500, 1500, 5000, 1e9],
                         labels=["0(신규)", "1~50", "51~200", "201~500",
                                 "501~1500", "1501~5000", "5000+"])
    print(d2.groupby("n_grp", observed=True)[TARGET].agg(["mean", "count"]).round(4).to_string())

    sec("[2b] 피로도 — 같은 경기 내에서 볼/스트라이크 이력의 영향 (outs_before, inning)")
    print(d2.groupby(["inning", "outs_before"])[TARGET].agg(["mean", "count"]).round(4).to_string())

    # ============ 3. 상황 조합 교차 ============
    sec("[3] count × 컨디션 (단조 신호 2개 결합)")
    d3 = df.dropna(subset=["asof_pitcher_success_rate"]).copy()
    d3["count_cat"] = d3["balls_before"] * 3 + d3["strikes_before"]
    d3["cond"] = pd.qcut(d3["asof_pitcher_success_rate"], 3, labels=["저컨디션", "중", "고컨디션"])
    d3["count_grp"] = np.select(
        [d3["count_cat"] == 11, d3["count_cat"] == 1, d3["count_cat"] == 0],
        ["3-2", "0-1", "0-0"], default="기타")
    piv3 = d3.pivot_table(index="cond", columns="count_grp", values=TARGET, aggfunc="mean")
    print(piv3.round(4).to_string())

    sec("[3b] 포스트시즌(F) 내부의 컨디션 효과")
    for gt, lbl in [("R", "정규"), ("F", "포스트시즌")]:
        sub = d3[d3["game_type"] == gt]
        m = sub.groupby("cond", observed=True)[TARGET].mean()
        print(f"  {lbl}: " + " | ".join(f"{c}={v:.4f}" for c, v in m.items()))

    # ============ 4. 팀/홈 원정 ============
    sec("[4] 팀/홈 원정 효과")
    d4 = df.copy()
    d4["home"] = (d4["top_bottom"] == "B").astype(int)  # 말 = 홈 공격
    print("홈(말) vs 원정(초):")
    print(d4.groupby("home")[TARGET].agg(["mean", "count"]).round(4).to_string())
    print("\n투수 팀별 target:")
    print(d4.groupby("pitcher_team_id")[TARGET].agg(["mean", "count"]).round(4).to_string())

    sec("[4b] 홈/원정 × game_type")
    piv4 = d4.pivot_table(index="game_type", columns="home", values=TARGET, aggfunc="mean")
    piv4.columns = ["원정(초)", "홈(말)"]
    print(piv4.round(4).to_string())

    # ============ 6. 데이터 품질 ============
    sec("[6] 데이터 품질 — 완전 중복 행 + ID 회전율")
    dups = df.duplicated(keep=False).sum()
    print(f"완전 중복 행(모든 컬럼 동일): {dups:,} ({dups/len(df)*100:.3f}%)")
    # row_id 제외 중복
    cols = [c for c in df.columns if c != "row_id"]
    dups2 = df[cols].duplicated(keep=False).sum()
    print(f"row_id 제외 중복 행: {dups2:,} ({dups2/len(df)*100:.3f}%)")

    sec("[6b] ID 회전율 — 시즌별 신규/이탈 선수")
    for id_col, label in [("pitcher_id", "투수"), ("batter_id", "타자")]:
        seen = set()
        rows = []
        for s in sorted(df["season"].unique()):
            cur = set(df[df["season"] == s][id_col].unique())
            new = cur - seen
            rows.append((s, len(cur), len(new), len(cur & seen)))
            seen |= cur
        print(f"\n{label}: 시즌 | 고유수 | 신규 | 잔존")
        for s, n, new, stay in rows:
            print(f"  {s} | {n:>4} | {new:>4} ({new/n*100:.0f}%) | {stay:>4} ({stay/n*100:.0f}%)")

    print("\n완료")


if __name__ == "__main__":
    main()
