"""eda_id_stability.py — ID(투수/타자) 시즌 간 성적 변동성 분석.

질문: IDTargetEncoder(enc)가 FAIL한 근본 원인은?
  - 선수 성적이 시즌마다 크게 변한다면(불안정) ID 인코딩은 근본적으로 부적합
  - 안정하다면 설계(시즌 가중 등)로 구제 가능

검증:
  A. 시즌 간 상관: rate(시즌 s) vs rate(시즌 s+1) — 연속 시즌 재현성 (투수/타자)
  B. 분산 분해 (ANOVA): total var = between-ID + within-ID(시즌 간) + residual
     -> between-ID 비중(ICC)이 작으면 ID는 '누구인가'보다 '어느 시즌인가'가 결정
  C. asof vs ID 추가 예측력: 시즌별로 ID 고정효과가 asof(현재 성적) 이상 정보를 주는지
  D. 사용량별 안정성: 고사용(주전) vs 저사용(불펜/플래툰) 투수의 시즌 간 안정성 차이

실행: python eda_id_stability.py 2>&1 | tee backup/eda_id_stability.log
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

    for id_col, label in [("pitcher_id", "투수"), ("batter_id", "타자")]:
        # ============ A. 시즌 간 상관 ============
        sec(f"[A] {label} 시즌 간 성적 재현성 (rate s vs rate s+1)")
        means = df.groupby([id_col, "season"])[TARGET].mean().unstack()
        # 최소 표본 요건: 시즌당 n>=50인 선수만 (노이즈 제거)
        counts = df.groupby([id_col, "season"]).size().unstack()
        valid = counts >= 50
        for s in range(2019, 2024):
            m1, m2 = means[s], means[s + 1]
            ok = valid[s] & valid[s + 1] & m1.notna() & m2.notna()
            if ok.sum() >= 10:
                corr = m1[ok].corr(m2[ok])
                n = int(ok.sum())
                print(f"  {s}->{s+1}: corr={corr:+.4f} (선수 {n}명)")

        # ============ B. 분산 분해 (ICC) ============
        sec(f"[B] {label} 분산 분해 — 성적 분산의 몇 %가 '선수 정체성'에서 오는가?")
        # ID별 평균과 시즌별 평균
        id_mean = df.groupby(id_col)[TARGET].mean()
        id_season_mean = df.groupby([id_col, "season"])[TARGET].mean()
        # between-ID 분산: ID별 평균의 분산
        var_between_id = id_mean.var(ddof=1)
        # within-ID (시즌 간) 분산: ID 내 시즌별 평균의 변동
        var_within = id_season_mean.groupby(id_col).var(ddof=1).mean()
        # 행 수준 residual 분산 근사 (이진 -> ID+시즌 평균 대비)
        merged = df.merge(id_season_mean.rename("idseason_mean"), left_on=[id_col, "season"], right_index=True)
        merged["id_mean"] = merged[id_col].map(id_mean)
        var_total = df[TARGET].var(ddof=1)
        var_resid = ((merged[TARGET] - merged["idseason_mean"]) ** 2).mean()
        icc_id = var_between_id / var_total
        icc_idseason = var_between_id / (var_between_id + var_within)
        print(f"  total var           = {var_total:.5f}")
        print(f"  between-ID var      = {var_between_id:.5f}  -> ID가 설명: {icc_id*100:.1f}%")
        print(f"  within-ID(시즌 간)  = {var_within:.5f}  -> ID 내 시즌 변동")
        print(f"  ID+시즌이 설명(ICC) = {icc_idseason*100:.1f}%  (ID 고정 시 남는 변동)")

        # ============ C. asof가 이미 ID를 대체? ============
        sec(f"[C] {label} — asof(현재 성적)가 ID 정보를 이미 포함?")
        asof_col = f"asof_{'pitcher' if label == '투수' else 'batter'}_success_rate"
        d = df.dropna(subset=[asof_col]).copy()
        # 1) ID별 평균과 asof와의 상관 (ID=asof의 함수인가?)
        id_asof_mean = d.groupby(id_col)[asof_col].mean()
        corr_id_asof = id_mean.reindex(id_asof_mean.index).corr(id_asof_mean)
        print(f"  ID별 평균 target vs ID별 asof 평균: corr={corr_id_asof:+.4f}")
        # 2) 시즌 s에서 asof로 ID 고정효과를 재현 가능한가 (2024 검증)
        for s_test in (2023, 2024):
            tr = d[d["season"] < s_test]
            te = d[d["season"] == s_test]
            # asof 기반 예측
            from numpy import corrcoef
            c_asof = te[asof_col].corr(te[TARGET])
            # ID 평균(asof와 무관하게 ID 정체성) 기반
            id_prev = tr.groupby(id_col)[TARGET].mean()
            te2 = te.merge(id_prev.rename("id_prev_mean"), left_on=id_col, right_index=True, how="left")
            c_id = te2["id_prev_mean"].corr(te2[TARGET]) if te2["id_prev_mean"].notna().sum() > 10 else np.nan
            print(f"  {s_test} 검증: corr(asof, target)={c_asof:+.4f} | "
                  f"corr(이전시즌ID평균, target)={c_id:+.4f}")

        # ============ D. 사용량별 안정성 ============
        sec(f"[D] {label} 사용량별 시즌 간 안정성")
        usage = df.groupby(id_col).size()
        q_lo, q_hi = usage.quantile(0.4), usage.quantile(0.6)
        for s in (2022, 2023):
            m1, m2 = means[s], means[s + 1]
            cnt_s = counts[s]
            rows = []
            for grp_name, mask in [
                ("저사용(<p40)", usage < q_lo), ("중사용(p40~p60)", (usage >= q_lo) & (usage <= q_hi)),
                ("고사용(>p60)", usage > q_hi),
            ]:
                ok = mask & (cnt_s >= 30) & m1.notna() & m2.notna()
                if ok.sum() >= 10:
                    rows.append((grp_name, m1[ok].corr(m2[ok]), int(ok.sum())))
            print(f"  {s}->{s+1}: " + " | ".join(f"{g}: corr={c:+.3f}(n={n})" for g, c, n in rows))

    print("\n완료")


if __name__ == "__main__":
    main()
