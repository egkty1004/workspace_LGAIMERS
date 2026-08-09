#!/usr/bin/env python3
"""eda_recent_gap.py — recent gap(prev1−asof) 신호 EDA + BSS 잠재력 측정 (2026-08-09)

GIHO 979.31 파이프라인의 미검증 유도 신호 중 REPORT_data_quality.md §5-1
【검증 1순위】 recent gap = prev1_success − asof_success (+3.6pp 스프레드 실측)을
4폴드(primary/r2022/r2023/r2024)에서 재검증하고 BSS 잠재력을 수치화한다.

gap 정의 (행 단위, 누수 없음):
  recent_gap_success = asof_pitcher_prev1_game_success_rate − asof_pitcher_success_rate
  recent_gap_middle  = asof_pitcher_prev1_game_middle_rate  − asof_pitcher_middle_rate

NaN 처리: prev1 결측(G1 29,185행) → gap=0.0 대체하되 prev1_missing 플래그 분포를
별도로 기록 (asof 결측 792행은 데뷔 n==0 → prev1도 결측, 동일 대체).

BSS 잠재력 (누수 없는 버킷 모델):
  fold 학습 행의 gap 구간별 target 평균(rate) → 검증 행에 적용 → common.score 로 BSS.
  시즌 드리프트로 인한 레벨 편차를 base-rate(logit shift)로 보정한 bss_bucket_calib를
  예상 기여로 보고 (evaluate_fold의 extrap-shift 기법 동일). oracle(검증 구간 평균 적용)은
  기술적 상한으로 병기. F3 기준선(cache/*.npy 로짓) BSS도 참고 수치로 병기.

구간: <-0.1 / [-0.1,0) / [0,0.05] / >0.05  (pd.cut right=True → (−∞,−0.1], (−0.1,0], (0,0.05], (0.05,∞))

출력: experiments/eda_recent_gap.json + stdout 요약 표
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GAP_SUCCESS = "recent_gap_success"
GAP_MIDDLE = "recent_gap_middle"
BINS = [-np.inf, -0.1, 0.0, 0.05, np.inf]
BIN_LABELS = ["<-0.1", "-0.1~0", "0~0.05", ">0.05"]
FOLD_ORDER = ["primary", "r2022", "r2023", "r2024"]
NPLY = {  # F3 기준선 로짓 (gen_lgb_f3_preds.py 산출, 폴드 val 행 순서와 일치)
    "primary": "cache/preds_primary_lgb_f3.npy",
    "r2022": "cache/h2b_r2022_lgb_f3.npy",
    "r2023": "cache/h2b_r2023_lgb_f3.npy",
    "r2024": "cache/h2b_r2024_lgb_f3.npy",
}
PREV1_COL = "asof_pitcher_prev1_game_success_rate"
ASOF_COL = "asof_pitcher_success_rate"


def add_gap_features(t):
    """gap 컬럼 2개 + 결측 플래그 추가. prev1 결측 → gap=0.0 대체."""
    for gap, prev1, asof in [
        (GAP_SUCCESS, PREV1_COL, ASOF_COL),
        (GAP_MIDDLE, "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_middle_rate"),
    ]:
        t[gap] = (t[prev1] - t[asof]).astype("float32")
    t["prev1_missing"] = t[PREV1_COL].isna().astype("int8")
    t["asof_missing"] = t[ASOF_COL].isna().astype("int8")
    t[GAP_SUCCESS] = t[GAP_SUCCESS].fillna(0.0).astype("float32")
    t[GAP_MIDDLE] = t[GAP_MIDDLE].fillna(0.0).astype("float32")
    return t


def quantiles(s):
    """gap 분위수 1/10/50/90/99."""
    q = s.quantile([0.01, 0.10, 0.50, 0.90, 0.99])
    return {f"q{int(p * 100)}": float(v) for p, v in q.items()}


def bin_target_table(y, gap_s, bins=BINS, labels=BIN_LABELS):
    """구간별 target 통계 (기술 통계용)."""
    c = pd.cut(gap_s, bins=bins, labels=labels, include_lowest=True)
    g = pd.DataFrame({"bin": c, "y": y}).groupby("bin", observed=True)["y"]
    return g.agg(count="count", target_rate="mean")


def spread_2525(y, gap_s):
    """q25/q75 양끝 target 스프레드 (REPORT §5-1 +3.6pp 재현). 단위: percentage point."""
    q25, q75 = gap_s.quantile([0.25, 0.75])
    lo = y[gap_s.values <= q25].mean()
    hi = y[gap_s.values >= q75].mean()
    return dict(q25=float(q25), q75=float(q75),
                lo_target=float(lo), hi_target=float(hi),
                spread_pp=float(100 * (hi - lo)))


def bucket_bss(gap_tr, y_tr, gap_va, y_va):
    """누수 없는 버킷 모델: train 구간 평균 → val 예측값 반환.
    상수 baseline BSS=0 이므로 (원시 예측의) BSS는 신호의 순수 예측력 하한치."""
    rate_tr = y_tr.groupby(
        pd.cut(gap_tr, bins=BINS, labels=BIN_LABELS, include_lowest=True),
        observed=True).mean()
    c_va = pd.cut(gap_va, bins=BINS, labels=BIN_LABELS, include_lowest=True)
    p_val = pd.Series(c_va.astype(object), index=gap_va.index) \
        .map(rate_tr.to_dict()).astype("float32").fillna(float(y_va.mean())).values
    return p_val


def fold_analysis(t, tr_mask, va_mask, name):
    """한 폴드: 분포·상관·구간 Δ·BSS 잠재력."""
    tr, va = t.loc[tr_mask], t.loc[va_mask]
    yv = va[common.TARGET].values
    r_val = float(yv.mean())

    res = dict(
        name=name,
        n_train=int(len(tr)), n_val=int(len(va)),
        target_rate_val=r_val,
        quantiles_success=quantiles(va[GAP_SUCCESS]),
        quantiles_middle=quantiles(va[GAP_MIDDLE]),
        corr_success=float(np.corrcoef(va[GAP_SUCCESS], yv)[0, 1]),
        corr_middle=float(np.corrcoef(va[GAP_MIDDLE], yv)[0, 1]),
        missing=dict(
            prev1_n=int(va["prev1_missing"].sum()),
            prev1_rate=float(va["prev1_missing"].mean()),
            prev1_target_rate=float(yv[va["prev1_missing"].values == 1].mean())
            if va["prev1_missing"].sum() else None,
            asof_n=int(va["asof_missing"].sum()),
        ),
    )

    # ── 구간별 target Δ (기술 통계, 검증 행 직접) ──
    for gap, key in ((GAP_SUCCESS, "bins_success"), (GAP_MIDDLE, "bins_middle")):
        br = bin_target_table(yv, va[gap])
        res[key] = {lbl: dict(n=int(br.loc[lbl, "count"]),
                              pct=float(br.loc[lbl, "count"] / len(va)),
                              target_rate=float(br.loc[lbl, "target_rate"]),
                              delta_pp=float(100 * (br.loc[lbl, "target_rate"] - r_val)))
                    for lbl in BIN_LABELS if lbl in br.index}

    # ── q25/q75 스프레드 재현 ──
    res["spread_2525_success"] = spread_2525(yv, va[GAP_SUCCESS])
    res["spread_2525_middle"] = spread_2525(yv, va[GAP_MIDDLE])

    # ── BSS 잠재력: 버킷 모델 (train→val) + base-rate 캘리브레이션 + oracle 상한 ──
    for gap, key in ((GAP_SUCCESS, "bss_bucket_success"), (GAP_MIDDLE, "bss_bucket_middle")):
        p = bucket_bss(tr[gap], tr[common.TARGET], va[gap], yv)
        # base-rate 캘리브레이션 (logit shift, common.evaluate_fold의 extrap-shift 기법):
        # 시즌 드리프트로 인한 레벨 편차를 제거해 신호의 순수 구분력을 측정.
        z = common.logit(p) + (common.logit(np.full(1, r_val))[0] - common.logit(p.mean()))
        p_calib = common.sigmoid(z)
        br = bin_target_table(yv, va[gap])  # oracle: val 구간 평균 적용 (기술 상한)
        c_va = pd.cut(va[gap], bins=BINS, labels=BIN_LABELS, include_lowest=True)
        p_or = pd.Series(c_va.astype(object), index=va.index) \
            .map(br["target_rate"].to_dict()).astype("float32").fillna(r_val).values
        res[key] = dict(
            bss_bucket_raw=float(common.score(p, yv)),          # 비캘리브레이션 (레벨 편차 포함)
            bss_bucket_calib=float(common.score(p_calib, yv)),  # base-rate 보정 후 (예상 기여)
            expected_gain_estimate=float(common.score(p_calib, yv)),
            bss_oracle=float(common.score(p_or, yv)),
        )
    return res


def main():
    t0 = time.time()
    print("[eda_recent_gap] recent gap(prev1−asof) EDA + BSS 잠재력 측정", flush=True)

    train, _ = common.load_train()
    add_gap_features(train)
    prev1_missing = int(train["prev1_missing"].sum())
    asof_missing = int(train["asof_missing"].sum())
    print(f"train: {train.shape} | prev1 결측: {prev1_missing:,} (G1) "
          f"| asof 결측: {asof_missing:,} | memory: "
          f"{train.memory_usage(deep=True).sum()/1e6:.0f}MB", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    # ── 전체(전체 train 행) 요약 ──
    overall = dict(
        n=int(len(train)),
        quantiles_success=quantiles(train[GAP_SUCCESS]),
        quantiles_middle=quantiles(train[GAP_MIDDLE]),
        corr_success=float(np.corrcoef(train[GAP_SUCCESS], train[common.TARGET])[0, 1]),
        corr_middle=float(np.corrcoef(train[GAP_MIDDLE], train[common.TARGET])[0, 1]),
        spread_2525_success=spread_2525(train[common.TARGET].values, train[GAP_SUCCESS]),
        spread_2525_middle=spread_2525(train[common.TARGET].values, train[GAP_MIDDLE]),
        missing=dict(prev1_n=prev1_missing, prev1_rate=prev1_missing / len(train),
                     asof_n=asof_missing,
                     prev1_target_rate=float(train.loc[train["prev1_missing"] == 1,
                                                        common.TARGET].mean())),
    )
    by_season = {}
    for season, sub in train.groupby("season", observed=True):
        by_season[str(season)] = dict(
            n=int(len(sub)),
            gap_success_mean=float(sub[GAP_SUCCESS].mean()),
            gap_success_median=float(sub[GAP_SUCCESS].median()),
            gap_middle_mean=float(sub[GAP_MIDDLE].mean()),
            gap_middle_median=float(sub[GAP_MIDDLE].median()),
            target_rate=float(sub[common.TARGET].mean()),
            prev1_missing_n=int(sub["prev1_missing"].sum()),
            prev1_missing_rate=float(sub["prev1_missing"].mean()),
        )

    results = dict(
        meta=dict(script="eda_recent_gap.py", date="2026-08-09",
                  source="REPORT_data_quality.md §5-1 (recent_gap +3.6pp 검증 1순위)",
                  gap_success="asof_pitcher_prev1_game_success_rate - asof_pitcher_success_rate",
                  gap_middle="asof_pitcher_prev1_game_middle_rate - asof_pitcher_middle_rate",
                  nan_handling="prev1 결측 → gap=0.0, prev1_missing 플래그 병기",
                  bins=BIN_LABELS, bin_semantics="pd.cut right=True",
                  folds=FOLD_ORDER,
                  bss_interpretation="bss_bucket_raw=비캘리브레이션 train→val 버킷 BSS, "
                                     "bss_bucket_calib=base-rate(logit shift) 보정 후 BSS = 예상 기여, "
                                     "bss_oracle=검증행 구간평균 적용 상한, delta_pp/spread_pp=percentage point"),
        overall=overall,
        by_season=by_season,
        folds={},
    )

    # ── 폴드별 분석 ──
    for fn in FOLD_ORDER:
        tr_m, va_m = folds[fn]
        res = fold_analysis(train, tr_m, va_m, fn)
        # F3 기준선 BSS (참고용)
        z = np.load(NPLY[fn])
        p_f3 = common.sigmoid(z)
        res["f3_bss_ref"] = float(common.score(p_f3, train.loc[va_m, common.TARGET].values))
        results["folds"][fn] = res

    # ── stdout 요약 ──
    print("\n" + "=" * 90, flush=True)
    print("recent gap EDA 요약 (bins: <-0.1 / -0.1~0 / 0~0.05 / >0.05)", flush=True)
    print("=" * 90, flush=True)
    print("전체: n=%s | q1/q50/q99(gap_success)=%.4f/%.4f/%.4f | corr=%.3f | "
          "spread_25/25=%.1fpp" % (
              f"{overall['n']:,}", overall["quantiles_success"]["q1"],
              overall["quantiles_success"]["q50"], overall["quantiles_success"]["q99"],
              overall["corr_success"], overall["spread_2525_success"]["spread_pp"]), flush=True)
    print("prev1 결측: %s행 (%.2f%%) target=%.4f | asof 결측: %s행"
          % (f"{overall['missing']['prev1_n']:,}", 100 * overall["missing"]["prev1_rate"],
             overall["missing"]["prev1_target_rate"],
             f"{overall['missing']['asof_n']:,}"), flush=True)
    print("-" * 90, flush=True)
    hdr = f"  {'fold':<9s} {'n_val':>8s} {'target':>7s} {'Δlo(bins<-0.1)':>13s} " \
          f"{'Δhi(>0.05)':>10s} {'spread25/75':>11s} {'BSS_cal':>9s} {'BSS_or':>9s} " \
          f"{'F3_ref':>9s} {'prev1miss':>9s}"
    print(hdr, flush=True)
    for fn in FOLD_ORDER:
        r = results["folds"][fn]
        b = r["bins_success"]
        d_lo = b["<-0.1"]["delta_pp"] if "<-0.1" in b else float("nan")
        d_hi = b[">0.05"]["delta_pp"] if ">0.05" in b else float("nan")
        print("  %-9s %8s %7.4f %+13.1f %+10.1f %+11.1f %9.1f %9.1f %9.1f %9s" % (
            fn, f"{r['n_val']:,}", r["target_rate_val"], d_lo, d_hi,
            r["spread_2525_success"]["spread_pp"],
            r["bss_bucket_success"]["bss_bucket_calib"],
            r["bss_bucket_success"]["bss_oracle"],
            r["f3_bss_ref"],
            f"{r['missing']['prev1_n']:,}"), flush=True)
    print("-" * 90, flush=True)
    print("시즌별 gap_success mean/median:", flush=True)
    for s, v in by_season.items():
        print("  season %s: n=%7s  mean=%+.4f  median=%+.4f  target=%.4f  prev1miss=%d"
              % (s, f"{v['n']:,}", v["gap_success_mean"], v["gap_success_median"],
                 v["target_rate"], v["prev1_missing_n"]), flush=True)

    with open("experiments/eda_recent_gap.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: experiments/eda_recent_gap.json (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
