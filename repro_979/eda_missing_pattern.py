#!/usr/bin/env python3
"""eda_missing_pattern.py — G1(복귀/공백)·G2(투수 데뷔)·G3(타자 데뷔) 결측 정보성 분석 (2026-08-09)

REPORT_data_quality.md §1.3~1.4의 결측 그룹 3종을 4폴드(primary/r2022/r2023/r2024)에서
재검증하고, fold별 target Δ 시계열과 BSS 잠재력을 수치화한다.

그룹 정의 (행 단위, 누수 없음 — 결측 패턴은 예측 시점에 관측 가능):
  G1 (29,185행): asof_pitcher_prev{1,3,5}_game_{success,middle}_rate 6종 전부 결측
                 (asof_pitcher_n은 존재 → "복귀/공백", 진짜 데뷔 아님)
  G2 (792행)   : asof_pitcher_{success,reverse,middle,ball,strike,fastball,breaking,offspeed}_rate
                 8종 결측 & asof_pitcher_n==0 → 투수 데뷔
  G3 (830행)   : asof_batter_{success,middle}_rate 2종 결측 & asof_batter_n==0 → 타자 데뷔
  ※ 실측: G2 ⊂ G1 (n==0이면 prev1~5도 결측), G1∩G3 = 161, G1 max asof_pitcher_n = 150.

분석 항목:
  1. G1 × asof_pitcher_n 곡선: [0,10)/[10,50)/[50,100)/[100,1000)/[1000+) 구간별 n + target rate
  2. fold별(primary/r2022/r2023/r2024) G1 target Δ 시계열
     — REPORT §1.3: 2019~2022 +0.021~+0.027 → 2023 −0.010 → 2024 −0.024 역전 재현
     (season 단위 시계열 + fold val 연도 기준 양쪽을 기록)
  3. debut_flag 제안: G2=asof_pitcher_n==0, G3=asof_batter_n==0, return_gap_flag=prev1결측 & asof_n<100
  4. BSS 잠재력: G1/G2/G3 + 제안 플래그의 fold별 target Δ + 예상 기여
     (train 버킷 평균 → val 적용 + base-rate(logit shift) 보정 = eda_recent_gap.py 방식)

규칙: 결측 0개 컬럼은 분석에서 건너뛰고 경고 로그. common.py 수정 금지.
출력: experiments/eda_missing.json + stdout 요약 표
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

G1_PREV_COLS = [f"asof_pitcher_prev{n}_game_{r}_rate"
                for n in (1, 3, 5) for r in ("success", "middle")]
G2_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
G3_COLS = ["asof_batter_success_rate", "asof_batter_middle_rate"]

ASOF_N_BINS = [0, 10, 50, 100, 1000, np.inf]
ASOF_N_LABELS = ["[0,10)", "[10,50)", "[50,100)", "[100,1000)", "[1000+)"]
FOLD_ORDER = ["primary", "r2022", "r2023", "r2024"]
FLAG_ORDER = ["g1_flag", "return_gap_flag", "g1_hist_flag", "g2_flag", "g3_flag"]


def build_flags(train):
    """결측 그룹별 boolean 플래그. 모두 int8 (메모리 최소)."""
    m1 = train[G1_PREV_COLS].isna().all(axis=1)
    m2 = train[G2_COLS].isna().all(axis=1)
    m3 = train[G3_COLS].isna().all(axis=1)
    n = train["asof_pitcher_n"]
    flags = {
        "g1_flag": m1.to_numpy(dtype=bool),
        "g2_flag": m2.to_numpy(dtype=bool),            # == asof_pitcher_n==0 (실측 확인)
        "g3_flag": m3.to_numpy(dtype=bool),            # == asof_batter_n==0
        "return_gap_flag": ((m1.values) & (n < 100).to_numpy(dtype=bool)),  # 복귀/공백(소이력)
        "g1_hist_flag": ((m1.values) & (n >= 100).to_numpy(dtype=bool)),    # 복귀/공백(충분이력)
    }
    return flags


def flag_bss(f_tr, y_tr, f_va, yv, r_val):
    """이진 플래그 버킷 모델: train 그룹 평균 → val 예측 + base-rate 보정 BSS.
    bss_bucket_raw = 비캘리브레이션, bss_bucket_calib = base-rate(logit shift) 보정 = 예상 기여,
    bss_oracle = val 그룹 평균 적용(기술 상한)."""
    rate_tr = y_tr.groupby(f_tr).mean().to_dict()
    p = pd.Series(f_va, index=yv.index).map(rate_tr).astype("float32").fillna(r_val).values
    z = common.logit(p) + (common.logit(np.full(1, r_val))[0] - common.logit(p.mean()))
    p_calib = common.sigmoid(z)
    rate_va = yv.groupby(f_va).mean().to_dict()
    p_or = pd.Series(f_va, index=yv.index).map(rate_va).astype("float32").fillna(r_val).values
    return dict(
        bss_bucket_raw=float(common.score(p, yv.values)),
        bss_bucket_calib=float(common.score(p_calib, yv.values)),
        expected_gain_estimate=float(common.score(p_calib, yv.values)),
        bss_oracle=float(common.score(p_or, yv.values)),
    )


def group_stats(y, mask, overall_rate):
    n = int(mask.sum())
    rate = float(y[mask].mean()) if n else None
    return dict(n=n, n_pct=float(n / len(y)), target_rate=rate,
                delta_pp=float(100 * (rate - overall_rate)) if n else None)


def main():
    t0 = time.time()
    print("[eda_missing_pattern] G1/G2/G3 결측 정보성 분석 시작", flush=True)

    train, _ = common.load_train()
    y = train[common.TARGET]
    overall_rate = float(y.mean())
    n_all = len(train)
    flags = build_flags(train)

    # ── 결측 컬럼 스캔: 0개 컬럼 skip 경고, 미분류 컬럼 경고 ──
    miss = train.isna().sum()
    miss_cols = miss[miss > 0]
    zero_cols = sorted(miss[miss == 0].index.tolist())
    known = set(G1_PREV_COLS) | set(G2_COLS) | set(G3_COLS)
    unknown = [c for c in miss_cols.index if c not in known]
    print(f"[WARN] 결측 0개 컬럼 {len(zero_cols)}개 — 분석에서 제외 (skip): "
          f"{', '.join(zero_cols[:8])}{' …' if len(zero_cols) > 8 else ''}", flush=True)
    if unknown:
        print(f"[WARN] 미분류 결측 컬럼 {len(unknown)}개: {unknown}", flush=True)

    results = dict(
        meta=dict(
            script="eda_missing_pattern.py", date="2026-08-09",
            source="REPORT_data_quality.md §1.3~1.4 (G1 복귀/공백·G2 투수데뷔·G3 타자데뷔)",
            groups=dict(
                G1=dict(def_="asof_pitcher_prev{1,3,5}_game_{success,middle}_rate 6종 전부 결측 "
                             "(asof_pitcher_n 존재 = 복귀/공백)", cols=G1_PREV_COLS),
                G2=dict(def_="asof_pitcher_{success,reverse,middle,ball,strike,fastball,breaking,offspeed}"
                             "_rate 8종 결측 & asof_pitcher_n==0 (투수 데뷔)", cols=G2_COLS),
                G3=dict(def_="asof_batter_{success,middle}_rate 2종 결측 & asof_batter_n==0 (타자 데뷔)",
                        cols=G3_COLS),
            ),
            fold_defs=dict(
                primary="train: season<=2023 / val: season==2024",
                r2022="train: (season<=2021)&R / val: (season==2022)&R",
                r2023="train: (season<=2022)&R / val: (season==2023)&R",
                r2024="train: (season<=2023)&R / val: (season==2024)&R",
            ),
            bss_interpretation="bss_bucket_raw=비캘리브레이션 train→val 버킷 BSS, "
                               "bss_bucket_calib=base-rate(logit shift) 보정 후 BSS = 예상 기여, "
                               "bss_oracle=val 그룹 평균 적용 상한, delta_pp=percentage point",
            note="G2 ⊂ G1 (asof_pitcher_n==0이면 prev1~5도 결측), G1∩G3=161. "
                 "G1 flag는 return_gap_flag(<100) + g1_hist_flag(>=100)로 분해.",
        ),
        overall=dict(n=n_all, target_rate=overall_rate,
                     groups={k: group_stats(y, flags[k], overall_rate) for k in
                             ("g1_flag", "g2_flag", "g3_flag")},
                     overlaps=dict(g1_and_g2=int((flags["g1_flag"] & flags["g2_flag"]).sum()),
                                   g1_and_g3=int((flags["g1_flag"] & flags["g3_flag"]).sum()),
                                   g1_excl_g2=int((flags["g1_flag"] & ~flags["g2_flag"]).sum())),
                     g1_by_game_type=dict(),
                     skipped_zero_missing_cols=zero_cols),
        g1_asof_n_curve=dict(),
        g1_by_season=dict(),
        debut_flags_proposed=dict(
            pitcher_debut_flag=dict(
                formula="asof_pitcher_n == 0", identical_to="g2_flag",
                n=int(flags["g2_flag"].sum()),
                rationale="투수 데뷔 (이력 0구). fold별 Δ는 2024에서 음수 전환"),
            batter_debut_flag=dict(
                formula="asof_batter_n == 0", identical_to="g3_flag",
                n=int(flags["g3_flag"].sum()),
                rationale="타자 데뷔. 전체 +7.9pp로 가장 큰 Δ지만 소표본(830) 및 시즌 편차 큼"),
            return_gap_flag=dict(
                formula="prev1_game 결측 & asof_pitcher_n < 100",
                n=int(flags["return_gap_flag"].sum()),
                rationale="G1(복귀/공백) 중 소이력 구간만 분리. n>=100은 기준치로 회귀(Δ~0)하므로 제외"),
        ),
        folds={},
    )

    # ── G1 × asof_n 곡선 (반열림 [a,b) — task 표기, REPORT §1.4 재현) ──
    g1 = flags["g1_flag"]
    c = pd.cut(train["asof_pitcher_n"], bins=ASOF_N_BINS, labels=ASOF_N_LABELS,
               right=False)
    for lbl in ASOF_N_LABELS:
        m = g1 & (c == lbl)
        results["g1_asof_n_curve"][lbl] = group_stats(y, m, overall_rate)

    # ── G1 season 시계열 (REPORT §1.3 재현) ──
    # game_type 분해 병기: G1 Δ의 상당분이 F(비정규)게임 집중 구성 효과 —
    # F baseline(0.603) > R(0.514)이고 G1 발생률 F 5.6% vs R 1.5%.
    # 층내(within-stratum) Δ는 R·F 모두 |1pp| 내외로 "G1 자체 정보성"은 약함.
    isR = train["game_type"] == "R"
    df = pd.DataFrame({"season": train["season"], "y": y,
                       "g1": g1.astype(int), "isR": isR})
    for season, sub in df.groupby("season"):
        g1m = sub["g1"].astype(bool).values
        rec = dict(
            n=int(len(sub)),
            g1_n=int(g1m.sum()),
            g1_rate=float(sub.loc[g1m, "y"].mean()) if g1m.sum() else None,
            overall_rate=float(sub["y"].mean()),
            delta_pp=float(100 * (sub.loc[g1m, "y"].mean() - sub["y"].mean()))
            if g1m.sum() else None,
        )
        for lbl, m_gt in (("R", sub["isR"].values), ("F", ~sub["isR"].values)):
            g1gt = g1m & m_gt
            if g1gt.sum():
                rec[f"delta_pp_{lbl}"] = float(
                    100 * (sub.loc[g1gt, "y"].mean() - sub.loc[m_gt, "y"].mean()))
                rec[f"g1_n_{lbl}"] = int(g1gt.sum())
        results["g1_by_season"][str(season)] = rec

    # ── game_type 구성 요약 (G1 Δ 해석용) ──
    for lbl, m_gt in (("R", isR.values), ("F", ~isR.values)):
        g1gt = g1 & m_gt
        results["overall"]["g1_by_game_type"][lbl] = dict(
            n=int(m_gt.sum()), g1_n=int(g1gt.sum()),
            g1_prevalence_pct=float(100 * g1gt.sum() / m_gt.sum()),
            target_rate=float(y[m_gt].mean()),
            g1_target_rate=float(y[g1gt].mean()),
            delta_pp=float(100 * (y[g1gt].mean() - y[m_gt].mean())),
        )

    # ── fold별 플래그 분석 ──
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }
    for fn in FOLD_ORDER:
        tr_m, va_m = folds[fn]
        yv = y.loc[va_m]
        r_val = float(yv.mean())
        res = dict(n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
                   val_year=int(train.loc[va_m, "season"].mode().iloc[0]),
                   target_rate_val=r_val, flags={})
        for fk in FLAG_ORDER:
            f_tr, f_va = flags[fk][tr_m.to_numpy(dtype=bool)], flags[fk][va_m.to_numpy(dtype=bool)]
            fva_s = pd.Series(f_va, index=yv.index)
            ftr_s = pd.Series(f_tr, index=y.loc[tr_m].index)
            bss = flag_bss(ftr_s, y.loc[tr_m], fva_s, yv, r_val)
            res["flags"][fk] = dict(
                n_train=int(f_tr.sum()), n_val=int(f_va.sum()),
                prevalence_val=float(f_va.mean()),
                target_rate_val_flag=float(yv[f_va].mean()) if f_va.sum() else None,
                delta_pp=float(100 * (yv[f_va].mean() - r_val)) if f_va.sum() else None,
                bss=bss,
            )
        results["folds"][fn] = res

    # ── stdout 요약 ──
    print("\n" + "=" * 96, flush=True)
    print("G1/G2/G3 결측 정보성 요약 (전체 target=%.4f, n=%s)" % (overall_rate, f"{n_all:,}"), flush=True)
    print("=" * 96, flush=True)
    print("  %-8s %-18s %10s %8s %9s %8s %9s" %
          ("그룹", "정의", "행", "비율", "target", "Δpp", "겹침"), flush=True)
    for gk, gl in (("g1_flag", "G1 복귀/공백"), ("g2_flag", "G2 투수 데뷔"), ("g3_flag", "G3 타자 데뷔")):
        s = results["overall"]["groups"][gk]
        ov = results["overall"]["overlaps"]
        extra = {"g1_flag": f"∩G2={ov['g1_and_g2']}",
                 "g2_flag": f"⊂G1 ({ov['g1_and_g2']})",
                 "g3_flag": f"∩G1={ov['g1_and_g3']}"}[gk]
        print("  %-8s %-18s %10s %7.2f%% %9.4f %+8.2f %9s" %
              (gk, gl, f"{s['n']:,}", 100 * s["n_pct"], s["target_rate"],
               s["delta_pp"], extra), flush=True)
    print("-" * 96, flush=True)
    print("G1 × asof_pitcher_n 곡선:", flush=True)
    for lbl in ASOF_N_LABELS:
        s = results["g1_asof_n_curve"][lbl]
        print("  %-10s n=%8s  target=%.4f  Δ= %+6.2fpp" %
              (lbl, f"{s['n']:,}", s["target_rate"] or float("nan"),
               s["delta_pp"] or float("nan")), flush=True)
    print("-" * 96, flush=True)
    print("G1 season 시계열 (REPORT §1.3: 2019~22 +0.021~+0.027 → 2023 -0.010 → 2024 -0.024), "
          "game_type 층내 Δ 병기:", flush=True)
    for s, v in results["g1_by_season"].items():
        dR = v.get("delta_pp_R")
        dF = v.get("delta_pp_F")
        print("  season %s: G1 n=%7s  G1 rate=%.4f  전체=%.4f  Δ=%+7.2fpp  "
              "(R Δ=%+6.2f / F Δ=%+6.2f)" %
              (s, f"{v['g1_n']:,}", v["g1_rate"], v["overall_rate"], v["delta_pp"],
               dR if dR is not None else float("nan"),
               dF if dF is not None else float("nan")), flush=True)
    print("  game_type 구성 (전체):", flush=True)
    for gt, v in results["overall"]["g1_by_game_type"].items():
        print("    %s: baseline=%.4f  G1 n=%7s (%.2f%%)  G1 rate=%.4f  Δ=%+6.2fpp" %
              (gt, v["target_rate"], f"{v['g1_n']:,}", v["g1_prevalence_pct"],
               v["g1_target_rate"], v["delta_pp"]), flush=True)
    print("  → 해석: R·F 층내 Δ는 ±1pp 내외로 미약. REPORT의 2019~22 +0.02~0.03은 "
          "G1이 F게임(기준 target 높음)에 집중(5.6% vs 1.5%)된 구성 효과.", flush=True)
    print("-" * 96, flush=True)
    hdr = "  %-11s %8s %8s %10s %10s %9s %9s %9s" % \
          ("fold(val)", "n_val", "target", "G1 Δpp", "G1 BSS", "G2 Δpp", "G3 Δpp", "RG Δpp")
    print(hdr, flush=True)
    for fn in FOLD_ORDER:
        r = results["folds"][fn]
        def dp(k):  # noqa: E306
            v = r["flags"][k]["delta_pp"]
            return float("nan") if v is None else v
        print("  %-11s %8s %8.4f %+10.2f %9.1f %+9.2f %+9.2f %+9.2f" % (
            f"{fn}({r['val_year']})", f"{r['n_val']:,}", r["target_rate_val"],
            dp("g1_flag"), r["flags"]["g1_flag"]["bss"]["bss_bucket_calib"],
            dp("g2_flag"), dp("g3_flag"), dp("return_gap_flag")), flush=True)
    print("-" * 96, flush=True)
    print("예상 기여(bss_bucket_calib) — fold별:", flush=True)
    for fk in FLAG_ORDER:
        line = "  %-17s" % fk
        for fn in FOLD_ORDER:
            line += "  %s=%6.1f" % (fn.split("r")[-1], results["folds"][fn]["flags"][fk]["bss"]["bss_bucket_calib"])
        print(line, flush=True)

    out = "experiments/eda_missing.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: {out} (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
