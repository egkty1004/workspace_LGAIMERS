"""eda_full_profile.py — 전체 데이터 종합 프로파일 EDA.

대상: train.csv / test.csv / trackman_history.csv / sample_submission.csv
항목:
  [1] train.csv 컬럼별 프로파일: dtype·카디널리티·결측·기초통계·IQR 이상치
  [2] train.csv target(control_success) 상관 분석: 수치형 spearman / 범주형 평균
  [3] test.csv 구조·샘플 행 (형식 확인)
  [4] trackman_history.csv: 스키마·결측·수치형 분포·범주형 분포·시즌 분포
  [5] trackman ↔ train ID 매핑 실증 (직접/간접) + 활용 가능성
  [6] sample_submission.csv 형식
실행: python eda_full_profile.py 2>&1 | tee backup/eda_full_profile.log
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def sec(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def profile_frame(df, name, target=None):
    """컬럼별 프로파일 테이블 생성."""
    rows = []
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        miss = int(s.isna().sum())
        miss_r = miss / len(df)
        if pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            q = d.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = q[0.75] - q[0.25]
            lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
            n_out = int(((d < lo) | (d > hi)).sum())
            rows.append(dict(col=c, dtype=str(s.dtype), nun=nun, miss=miss, miss_r=miss_r,
                             min=d.min(), p1=q[0.01], med=q[0.5], p99=q[0.99], max=d.max(),
                             mean=d.mean(), std=d.std(), iqr_out=n_out,
                             out_hi=int((d > hi).sum()), out_lo=int((d < lo).sum())))
        else:
            top = s.value_counts(dropna=True).head(3)
            top_s = ";".join(f"{k}({v})" for k, v in top.items())
            rows.append(dict(col=c, dtype=str(s.dtype), nun=nun, miss=miss, miss_r=miss_r,
                             min=np.nan, p1=np.nan, med=np.nan, p99=np.nan, max=np.nan,
                             mean=np.nan, std=np.nan, iqr_out=np.nan,
                             out_hi=np.nan, out_lo=np.nan, top=top_s))
    out = pd.DataFrame(rows)
    if target is not None:
        # 수치형: target과의 spearman (범주형은 생략, 뒤에서 따로)
        corrs = {}
        for c in df.columns:
            if c == target or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            cc = df[[c, target]].dropna()
            if cc[c].nunique() <= 1:
                continue
            corrs[c] = cc[c].corr(cc[target], method="spearman")
        out["spear_tgt"] = out["col"].map(corrs)
    return out


def main():
    # ============ [1][2] train.csv ============
    sec("[1] train.csv 컬럼별 프로파일")
    tr = pd.read_csv("./data/train.csv", encoding="utf-8-sig")
    print("shape:", tr.shape)
    prof = profile_frame(tr, "train", target="control_success")
    prof["miss_r"] = (prof["miss_r"] * 100).round(3)
    prof["iqr_out"] = prof["iqr_out"].fillna("").astype(str)
    print(prof.to_string(index=False, formatters={
        "min": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "p1": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "med": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "p99": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "max": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "mean": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "std": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "spear_tgt": lambda v: f"{v:+.4f}" if pd.notna(v) else ""}))
    prof.to_csv("backup/profile_train.csv", index=False)

    # target 요약
    sec("[1b] target(control_success) 요약")
    print("positive rate:", tr["control_success"].mean().round(4))
    print(tr.groupby("season")["control_success"].agg(["count", "mean"]).round(4).to_string())

    # 범주형 × target 평균 (상위 카테고리)
    cat_cols = [c for c in tr.columns if tr[c].dtype == object or tr[c].dtype.name == "category"]
    sec("[2] 범주형 컬럼별 target 평균 편차 (상위 8개 카테고리)")
    g = tr["control_success"].mean()
    for c in cat_cols:
        m = tr.groupby(c, dropna=False)["control_success"].agg(["mean", "count"])
        m["dev"] = m["mean"] - g
        m = m.sort_values("count", ascending=False).head(8)
        print(f"\n--- {c} (전체 mean={g:.4f}) ---")
        for idx, r in m.iterrows():
            print(f"    {str(idx):>12} | mean={r['mean']:.4f} dev={r['dev']:+.4f} n={int(r['count']):>7}")

    # ============ [3] test.csv ============
    sec("[3] test.csv (형식 샘플 5행)")
    te = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    print("shape:", te.shape)
    print("컬럼:", list(te.columns))
    print(te.to_string(index=False))

    # ============ [4] trackman_history.csv ============
    sec("[4] trackman_history.csv 프로파일")
    tm = pd.read_csv("./data/trackman_history.csv", encoding="utf-8-sig",
                     low_memory=False)
    print("shape:", tm.shape)
    prof_tm = profile_frame(tm, "trackman")
    prof_tm["miss_r"] = (prof_tm["miss_r"] * 100).round(3)
    prof_tm["iqr_out"] = prof_tm["iqr_out"].fillna("").astype(str)
    print(prof_tm.to_string(index=False, formatters={
        "min": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "p1": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "med": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "p99": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "max": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "mean": lambda v: f"{v:.4g}" if pd.notna(v) else "",
        "std": lambda v: f"{v:.4g}" if pd.notna(v) else ""}))
    prof_tm.to_csv("backup/profile_trackman.csv", index=False)

    # 구종/구속 분포
    sec("[4b] trackman 구종·구속 분포")
    if "pitch_type_group" in tm.columns:
        print("pitch_type_group:\n", tm["pitch_type_group"].value_counts(dropna=False).to_string())
    if "tagged_pitch_type" in tm.columns:
        print("\ntagged_pitch_type top15:\n",
              tm["tagged_pitch_type"].value_counts(dropna=False).head(15).to_string())
    if "rel_speed" in tm.columns:
        print("\nrel_speed(구속) 분포:", tm["rel_speed"].describe(percentiles=[.01, .5, .99]).round(2).to_dict())
    if "spin_rate" in tm.columns:
        print("spin_rate 분포:", tm["spin_rate"].describe(percentiles=[.01, .5, .99]).round(1).to_dict())

    # 시즌 분포
    sec("[4c] trackman 시즌·월 분포")
    print(tm.groupby(["season"]).size().to_string())

    # ============ [5] trackman ↔ train ID 매핑 ============
    sec("[5] trackman ↔ train 매핑 실증")
    tm_ids = tm[["pitcher_trackman_id", "batter_trackman_id"]].copy()
    tr_ids = tr[["pitcher_id", "batter_id"]].copy()
    for tc, mc in [("pitcher_trackman_id", "pitcher_id"), ("batter_trackman_id", "batter_id")]:
        tn = tm_ids[tc].nunique()
        mn = tr_ids[mc].nunique()
        inter = len(set(tm_ids[tc].dropna()) & set(tr_ids[mc].dropna()))
        print(f"{tc}({tn}) ∩ {mc}({mn}) = {inter}")

    # 간접 매핑 후보: (season, hand, team) 조합으로 교차 확인
    print("\n--- 간접 매핑 후보: pitcher (season, hand, team) 조합 ---")
    tm_p = tm.dropna(subset=["pitcher_trackman_id"])[["season", "pitcher_hand", "pitcher_team"]].drop_duplicates()
    tr_p = tr[["season", "pitcher_hand", "pitcher_team_id"]].drop_duplicates()
    tm_p["key"] = tm_p["season"].astype(str) + "|" + tm_p["pitcher_hand"].astype(str) + "|" + tm_p["pitcher_team"].astype(str)
    tr_p["key"] = tr_p["season"].astype(str) + "|" + tr_p["pitcher_hand"].astype(str) + "|" + tr_p["pitcher_team_id"].astype(str)
    inter_k = len(set(tm_p["key"]) & set(tr_p["key"]))
    print(f"trackman (season,hand,team) 고유조합 {len(tm_p)} ∩ train {len(tr_p)} = {inter_k}")
    print("trackman pitcher_team 값:", sorted(tm["pitcher_team"].dropna().unique())[:20])
    print("train pitcher_team_id 값:", sorted(tr["pitcher_team_id"].dropna().unique())[:20])

    # game_date 존재 여부 (train에는 날짜 없음)
    print("\ntrain 날짜성 컬럼:", [c for c in tr.columns if "date" in c.lower() or "month" in c.lower()])
    print("trackman 날짜성 컬럼:", [c for c in tm.columns if "date" in c.lower() or "month" in c.lower()])

    # 행 수 비교 (같은 투구의 중복 여부 가설)
    print(f"\ntrackman 행 수 {len(tm):,} vs train 행 수 {len(tr):,} "
          f"(비율 {len(tm)/len(tr):.3f})")

    # ============ [6] sample_submission ============
    sec("[6] sample_submission.csv")
    ss = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    print("shape:", ss.shape, "| 컬럼:", list(ss.columns))
    print(ss.to_string(index=False))

    print("\n완료")


if __name__ == "__main__":
    main()
