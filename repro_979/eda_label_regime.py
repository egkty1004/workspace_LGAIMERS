#!/usr/bin/env python3
"""eda_label_regime.py — F/R 레이블 레짐 시계열 + ABS 카운트별 delta + test 구조 추론 (2026-08-09)

REPORT_kbo_insights.md §5의 구조적 드리프트를 train 전체에서 재현하고,
2025 test(형식 샘플 5행)의 game_type 구조를 점검한다.

1. F/R 성공률 연도별 시계열 (game_type 'F'/'R' × season 2019~2024, success rate + n)
2. F 레짐 전환 확인: 2019-22 F 0.59~0.71 → 2023 0.473 → 2024 0.459 (§5-2 재현)
   - 2019-22: F가 R보다 높음(관대한 운영 존/판정) → 2023~: F가 R보다 낮음(표준 존)
3. ABS 카운트별 delta matrix: R 2023→2024 count_state별 success Δ
   (1-1 −2.1pp, 2-0 −2.0pp, 2-1 −2.0pp, 3-1 −1.9pp, 0-0/1-0 −1.6pp, 3-2 −0.5pp … §5-1 재현)
   count_state = balls_before*3 + strikes_before → 12범주 ("0-0"~"3-2")
   + experiments/kbo_insights_out/ABS_R_count_delta.csv 원본과 비교
4. F post-2023(2023-24) 카운트 구조: 0-0~2-2 평평 0.46~0.48, 3볼 셀 n<3000 표본부족
   + S_2024_F_vs_R_count.csv 원본(2024 F/R count)과 비교
5. R 3-2 장기 하락: 2019-22 평균 0.501 → 2023-24 평균 0.462 (−3.9pp 추가 하락, §5-1)
6. test 구조 예측: open/data/test.csv (5행, season 2025, R-only) 로드
   → 결론 "test F 미확정": 실제 평가 test는 245,789행이며 2025 F 존재 여부는
     로컬에서 확인 불가 명시.

출력: experiments/eda_regime.json + stdout 요약 표
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# count_state = balls_before*3 + strikes_before (12범주, common.add_count_state_feature와 동일)
CS_LABELS = {0: "0-0", 1: "0-1", 2: "0-2", 3: "1-0", 4: "1-1", 5: "1-2",
             6: "2-0", 7: "2-1", 8: "2-2", 9: "3-0", 10: "3-1", 11: "3-2"}
CS_ORDER = [CS_LABELS[i] for i in range(12)]

REPORT_DELTA_CSV = "../experiments/kbo_insights_out/ABS_R_count_delta.csv"
REPORT_FR_COUNT_CSV = "../experiments/kbo_insights_out/S_2024_F_vs_R_count.csv"

EVAL_TEST_ROWS = 245_789  # 평가 환경 추론 10분/245,789행 (AGENTS.md 대회 제약)


def rate_table(df, group_cols):
    """groupby 성공률/표본수 테이블."""
    g = df.groupby(group_cols, observed=True)[common.TARGET].agg(["count", "mean"])
    g.columns = ["n", "rate"]
    return g


def row_rate(sr, n):
    """(n, rate) → 테이블 행 dict."""
    return dict(n=int(n), rate=float(sr) if n else None)


def main():
    t0 = time.time()
    print("[eda_label_regime] F/R 레이블 레짐 + ABS 카운트 delta + test 구조 추론", flush=True)

    train, _ = common.load_train()
    train["cs"] = (train["balls_before"] * 3 + train["strikes_before"]).astype(int)
    train["cs_lbl"] = train["cs"].map(CS_LABELS).astype("category")
    print(f"train: {train.shape} | season {train['season'].min()}~{train['season'].max()} | "
          f"memory: {train.memory_usage(deep=True).sum()/1e6:.0f}MB", flush=True)

    isF = train["game_type"] == "F"
    isR = train["game_type"] == "R"
    seasons = sorted(train["season"].unique())

    # ── 1) F/R × season 성공률 시계열 ──
    fr_by_season = {}
    for gt in ["F", "R"]:
        sub = train[train["game_type"] == gt]
        fr_by_season[gt] = {
            str(s): row_rate(v["mean"], v["count"])
            for s, v in sub.groupby("season", observed=True)[common.TARGET].agg(
                ["count", "mean"]).iterrows()
        }

    # ── 2) F 레짐 전환 확인 ──
    def pooled_rate(mask):
        v = train.loc[mask, common.TARGET]
        return float(v.mean()), int(v.count())

    f_pre, f_pre_n = pooled_rate(isF & (train["season"] <= 2022))
    f_23, f_23_n = pooled_rate(isF & (train["season"] == 2023))
    f_24, f_24_n = pooled_rate(isF & (train["season"] == 2024))
    r_pre, r_pre_n = pooled_rate(isR & (train["season"] <= 2022))
    r_23, r_23_n = pooled_rate(isR & (train["season"] == 2023))
    r_24, r_24_n = pooled_rate(isR & (train["season"] == 2024))
    f_regime = dict(
        f_pre2023=dict(rate=f_pre, n=f_pre_n, label="2019-22"),
        f_2023=dict(rate=f_23, n=f_23_n),
        f_2024=dict(rate=f_24, n=f_24_n),
        r_pre2023=dict(rate=r_pre, n=r_pre_n, label="2019-22"),
        r_2023=dict(rate=r_23, n=r_23_n),
        r_2024=dict(rate=r_24, n=r_24_n),
        f_minus_r_pre2023_pp=float(100 * (f_pre - r_pre)),
        f_minus_r_2023_pp=float(100 * (f_23 - r_23)),
        f_minus_r_2024_pp=float(100 * (f_24 - r_24)),
        regime_transition=("관대한 존(2019-22) → 표준 존(2023~)" if f_pre > r_pre and f_23 < r_23
                           else "전환 미관측"),
    )

    # ── 3) ABS 카운트별 delta matrix (R 2023→2024) ──
    r23 = rate_table(train[isR & (train["season"] == 2023)], ["cs_lbl"])
    r24 = rate_table(train[isR & (train["season"] == 2024)], ["cs_lbl"])
    abs_delta = {}
    for lbl in CS_ORDER:
        a = r23.loc[lbl, "rate"] if lbl in r23.index else np.nan
        b = r24.loc[lbl, "rate"] if lbl in r24.index else np.nan
        na = int(r23.loc[lbl, "n"]) if lbl in r23.index else 0
        nb = int(r24.loc[lbl, "n"]) if lbl in r24.index else 0
        abs_delta[lbl] = dict(
            r2023=float(a) if pd.notna(a) else None,
            r2024=float(b) if pd.notna(b) else None,
            n2023=na, n2024=nb,
            delta_pp=float(100 * (b - a)) if pd.notna(a) and pd.notna(b) else None,
        )
    delta_all = pd.concat([r23[["rate"]].rename(columns={"rate": "r2023"}),
                           r24[["rate"]].rename(columns={"rate": "r2024"})], axis=1)
    delta_all = delta_all.reindex(CS_ORDER)
    abs_delta["overall"] = dict(
        r2023=float(train.loc[isR & (train["season"] == 2023), common.TARGET].mean()),
        r2024=float(train.loc[isR & (train["season"] == 2024), common.TARGET].mean()),
        n2023=int((isR & (train["season"] == 2023)).sum()),
        n2024=int((isR & (train["season"] == 2024)).sum()),
        delta_pp=float(100 * (train.loc[isR & (train["season"] == 2024), common.TARGET].mean()
                              - train.loc[isR & (train["season"] == 2023), common.TARGET].mean())),
    )

    # REPORT 원본 CSV와 비교 (없으면 skip)
    report_delta = {}
    if os.path.exists(REPORT_DELTA_CSV):
        rc = pd.read_csv(REPORT_DELTA_CSV)
        for _, rrow in rc.iterrows():
            lbl = rrow["count_state"]
            report_delta[lbl] = dict(report_delta_pp=float(100 * rrow["delta_2024_minus_2023"]),
                                     recompute_delta_pp=abs_delta[lbl]["delta_pp"])
        abs_delta["report_comparison"] = report_delta

    # ── 4) F post-2023(2023-24) 카운트 구조 ──
    f_post = train[isF & (train["season"] >= 2023)]
    f_post_count = {}
    ft = rate_table(f_post, ["cs_lbl"]).reindex(CS_ORDER)
    for lbl in CS_ORDER:
        n = int(ft.loc[lbl, "n"]) if lbl in ft.index else 0
        f_post_count[lbl] = dict(
            n=n,
            rate=float(ft.loc[lbl, "rate"]) if lbl in ft.index else None,
            sample_insufficient=bool(n < 3000),
        )
    f_post_count["flat_range"] = [lbl for lbl in CS_ORDER
                                  if lbl in f_post_count and f_post_count[lbl]["rate"] is not None
                                  and 0.46 <= f_post_count[lbl]["rate"] <= 0.48]

    # 2024 F/R count 원본 CSV와 비교
    report_fr_count = {}
    if os.path.exists(REPORT_FR_COUNT_CSV):
        rc = pd.read_csv(REPORT_FR_COUNT_CSV)
        for _, rrow in rc.iterrows():
            key = (rrow["game_type"], rrow["count_state"])
            report_fr_count[f"{key[0]}_{key[1]}"] = dict(
                report_rate=float(rrow["succ"]), report_n=int(rrow["n"]))
        f_post_count["report_2024_comparison"] = report_fr_count

    # ── 5) R 3-2 장기 하락 ──
    r32 = train[isR & (train["balls_before"] == 3) & (train["strikes_before"] == 2)]
    r32_pre = r32[r32["season"] <= 2022][common.TARGET]
    r32_post = r32[r32["season"] >= 2023][common.TARGET]
    r32_longterm = dict(
        r32_pre2023_rate=float(r32_pre.mean()), r32_pre2023_n=int(r32_pre.count()),
        r32_post2023_rate=float(r32_post.mean()), r32_post2023_n=int(r32_post.count()),
        drop_pp=float(100 * (r32_post.mean() - r32_pre.mean())),
    )

    # ── 6) test 구조 예측 ──
    test_path = os.path.join(common.DATA_DIR, "test.csv")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    test["cs"] = (test["balls_before"] * 3 + test["strikes_before"]).astype(int)
    test["cs_lbl"] = test["cs"].map(CS_LABELS)
    gt_counts = {str(k): int(v) for k, v in test["game_type"].value_counts().items()}
    season_counts = {str(k): int(v) for k, v in test["season"].value_counts().items()}
    cs_counts = {lbl: int((test["cs_lbl"] == lbl).sum()) for lbl in CS_ORDER
                 if (test["cs_lbl"] == lbl).sum()}
    test_structure = dict(
        n=int(len(test)),
        game_type_counts=gt_counts,
        season_counts=season_counts,
        has_F=bool((test["game_type"] == "F").any()),
        has_R=bool((test["game_type"] == "R").any()),
        count_state_counts=cs_counts,
    )
    test_inference = dict(
        conclusion="test F 미확정",
        local_test_is_format_sample=True,
        local_test_n=int(len(test)),
        eval_test_rows=EVAL_TEST_ROWS,
        note=("로컬 test.csv는 5행 형식 확인용 샘플(season 2025, 전부 R). 실제 평가 test는 "
              f"{EVAL_TEST_ROWS:,}행이며 2025 F 존재 여부는 로컬에서 확인 불가. "
              "만약 2025 test에 F가 포함되면 post-2023 레짐(≈0.46)으로 매핑해야 하고, "
              "game_type을 상수 피처로 쓰면 안 된다(§5-2 모델 함의)."),
        recommended_handling=dict(
            if_F_present="F를 post-2023 레짐(≈0.46)으로 매핑, F 카운트 계수는 R과 분리 학습 권장",
            if_F_absent="R 단일 레짐으로 학습 가능, F 게이트 불필요",
        ),
    )

    results = dict(
        meta=dict(
            script="eda_label_regime.py",
            date="2026-08-09",
            source="experiments/REPORT_kbo_insights.md §5-1/§5-2 (F/R 레짐 + ABS 카운트별 Δ 재현)",
            count_state_def="balls_before*3 + strikes_before (12범주, common.add_count_state_feature 동일)",
            delta_pp="percentage point (R 2024 − R 2023)",
            eval_test_rows=f"평가 환경 추론 제한 10분/{EVAL_TEST_ROWS:,}행 (AGENTS.md)",
            columns=dict(rate="success rate(control_success 평균)", n="행 수"),
        ),
        series_fr_by_season=fr_by_season,
        f_regime=f_regime,
        abs_r_count_delta=abs_delta,
        f_post2023_count=f_post_count,
        r32_longterm=r32_longterm,
        test_structure=test_structure,
        test_inference=test_inference,
        conclusion=(
            "F 레이블 레짐은 2019-22(관대한 존, R보다 높음) → 2023~(표준 존, R보다 낮음)로 "
            "전환되었으며, ABS(2024) 도입은 R success를 전체 −1.3pp, 카운트별 비균등 "
            "(1-1 −2.1pp 최대, 3-2 −0.5pp 최소)으로 하락시켰다. 2025 test의 F 존재 여부는 "
            "로컬 5행 샘플로는 미확정."
        ),
    )

    # ── stdout 요약 ──
    print("\n" + "=" * 96, flush=True)
    print("[1] F/R × season 성공률 시계열 (rate / n)", flush=True)
    print("=" * 96, flush=True)
    hdr = f"  {'season':>6s} {'F_rate':>7s} {'F_n':>9s} {'R_rate':>7s} {'R_n':>9s} {'F-R_pp':>8s}"
    print(hdr, flush=True)
    for s in seasons:
        fv = fr_by_season["F"].get(str(s), {})
        rv = fr_by_season["R"].get(str(s), {})
        fr = None
        if fv.get("rate") is not None and rv.get("rate") is not None:
            fr = 100 * (fv["rate"] - rv["rate"])
        print("  %6d %7.4f %9s %7.4f %9s %+8.2f" % (
            s, fv.get("rate", float("nan")), f"{fv.get('n', 0):,}",
            rv.get("rate", float("nan")), f"{rv.get('n', 0):,}",
            fr if fr is not None else float("nan")), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("[2] F 레짐 전환", flush=True)
    print("=" * 96, flush=True)
    print("  F 2019-22=%.4f (n=%s) | 2023=%.4f (n=%s) | 2024=%.4f (n=%s)" % (
        f_pre, f"{f_pre_n:,}", f_23, f"{f_23_n:,}", f_24, f"{f_24_n:,}"), flush=True)
    print("  R 2019-22=%.4f (n=%s) | 2023=%.4f (n=%s) | 2024=%.4f (n=%s)" % (
        r_pre, f"{r_pre_n:,}", r_23, f"{r_23_n:,}", r_24, f"{r_24_n:,}"), flush=True)
    print("  F−R: 2019-22 %+.2fpp → 2023 %+.2fpp → 2024 %+.2fpp → %s" % (
        100 * (f_pre - r_pre), 100 * (f_23 - r_23), 100 * (f_24 - r_24),
        f_regime["regime_transition"]), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("[3] ABS 카운트별 delta matrix (R, 2024−2023) — §5-1 재현", flush=True)
    print("=" * 96, flush=True)
    hdr = f"  {'count':>5s} {'r2023':>7s} {'r2024':>7s} {'Δpp':>7s} {'n23':>8s} {'n24':>8s}"
    print(hdr, flush=True)
    for lbl in CS_ORDER:
        d = abs_delta[lbl]
        dpp = d["delta_pp"]
        print("  %5s %7.4f %7.4f %+7.2f %8s %8s" % (
            lbl, d["r2023"] or float("nan"), d["r2024"] or float("nan"),
            dpp if dpp is not None else float("nan"),
            f"{d['n2023']:,}", f"{d['n2024']:,}"), flush=True)
    ov = abs_delta["overall"]
    print("  ----- R 전체: %7.4f → %7.4f (%+.2fpp, n %s→%s) -----" % (
        ov["r2023"], ov["r2024"], ov["delta_pp"], f"{ov['n2023']:,}", f"{ov['n2024']:,}"),
        flush=True)
    if abs_delta.get("report_comparison"):
        print("  REPORT ABS_R_count_delta.csv 대조 (Δpp):", flush=True)
        for lbl, rc in abs_delta["report_comparison"].items():
            print("    %5s report=%+.2f recompute=%+.2f" % (
                lbl, rc["report_delta_pp"], rc["recompute_delta_pp"]), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("[4] F post-2023(2023-24) 카운트 구조 — §5-2 (3볼 셀 n<3000 표본부족)", flush=True)
    print("=" * 96, flush=True)
    hdr = f"  {'count':>5s} {'rate':>7s} {'n':>8s} {'n<3000?':>8s}"
    print(hdr, flush=True)
    for lbl in CS_ORDER:
        fc = f_post_count[lbl]
        if fc["rate"] is None:
            print("  %5s     -   %8s    -" % (lbl, f"{fc['n']:,}"), flush=True)
        else:
            print("  %5s %7.4f %8s %8s" % (
                lbl, fc["rate"], f"{fc['n']:,}",
                "O" if fc["sample_insufficient"] else ""), flush=True)
    print("  flat_range(0.46~0.48): %s" % ", ".join(f_post_count["flat_range"]), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("[5] R 3-2 장기 하락", flush=True)
    print("=" * 96, flush=True)
    print("  R 3-2 2019-22 평균=%.4f (n=%s) → 2023-24 평균=%.4f (n=%s) (%+.2fpp)" % (
        r32_longterm["r32_pre2023_rate"], f"{r32_longterm['r32_pre2023_n']:,}",
        r32_longterm["r32_post2023_rate"], f"{r32_longterm['r32_post2023_n']:,}",
        r32_longterm["drop_pp"]), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("[6] test 구조 예측", flush=True)
    print("=" * 96, flush=True)
    print("  로컬 test.csv: n=%d | game_type=%s | season=%s | F 존재=%s" % (
        test_structure["n"], gt_counts, season_counts, test_structure["has_F"]), flush=True)
    print("  결론: %s — 평가 test는 %d행, 2025 F 존재 여부 로컬 확인 불가" % (
        test_inference["conclusion"], EVAL_TEST_ROWS), flush=True)

    with open("experiments/eda_regime.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/eda_regime.json (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
