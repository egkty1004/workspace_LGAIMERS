#!/usr/bin/env python3
"""eda_cross_signals.py — count×platoon·score_diff×LI·outs×count·game_type×count 교차 EDA + BSS 잠재력 (2026-08-09)

REPORT_kbo_insights.md §3의 교차 신호 4종을 4폴드(primary/r2022/r2023/r2024)에서 재검증하고,
fold별 양성 셀 n·target Δ·base-rate 보정 버킷 BSS(예상 기여)를 수치화한다.

전처리: common.preprocess_for_submission(train) 호출 → platoon(4범주: pitcher_hand*2+batter_hand,
3=L-L/4=L-R/5=R-L/6=R-R)과 count_state(12범주: balls_before*3+strikes_before) 생성. common.py 수정 금지.

교차 신호 정의 (행 단위, 누수 없음 — pos/ref 모두 예측 시점 관측 가능):
  1. count_platoon_3b2_same: 3-2(balls==3&strikes==2) & 동손(pitcher_hand==batter_hand)
     기준(pos−ref 음의 방향): 0-1 & 이손. REPORT §3: 최근(23-24) 0.4362 vs 0.5198 → Δ8.4pp
     (실측 재현: 2023-24 R+F 0.4365 n=11,211 vs 0.5205 n=30,879 → Δ−8.4pp)
  2. score_diff_binary: |score_diff_pitcher_team|<=1 (close) vs >=6 (blowout)
     REPORT H8: close 0.5283 vs blowout 0.5121 → Δ1.8pp (전체)
  3. li_risp_flag: RISP(runner_on_2b==1 | runner_on_3b==1) & li>=1.0 vs RISP & li<0.5
     REPORT §3: LI 1.0-2.0 0.5324 (n=159,226) vs LI<0.5 0.5094 (n=86,250) → Δ2.3pp (전체)
  4. outs_count_3b2_2out: 3-2 & outs_before==2 vs 3-2 & outs_before==0
     REPORT §3: 0.5087 (n=24,070) vs 0.4918 (n=23,441) → Δ1.7pp (전체)

추가: game_type×count 교차표 (fold val별 R/F count_state 성공률, REPORT §3: R 3-2 0.462 vs
0-1 0.512 Δ5.0pp / F 평평 0.46~0.48). 교차표만 산출 — 인코딩 제안은 경계 해소 단계에서 판정.

폴드 정의:
  primary = train(season<=2023) / val(season==2024)
  r2022   = train((season<=2021)&isR) / val((season==2022)&isR)
  r2023   = train((season<=2022)&isR) / val((season==2023)&isR)
  r2024   = train((season<=2023)&isR) / val((season==2024)&isR)

BSS 잠재력 (eda_recent_gap.py / eda_missing_pattern.py 방식 재사용, 누수 없음):
  bss_bucket_calib: train 셀 평균 → val 적용 + base-rate(logit shift) 보정 = 예상 기여
  bss_bucket_raw:   비캘리브레이션 (시즌 레벨 편차 포함)
  bss_oracle:       val 셀 평균 적용 (기술 상한)
  flag 모델 = pos vs 나머지 이진, pair 모델 = pos/ref/기타 3셀 (교차 신호 쌍 구분력 병기)

규칙: 양성(n_pos) 또는 기준(n_ref) < 3,000 셀은 "표본 부족"으로 표기하고 세부 통계 생략.
출력: experiments/eda_cross.json + stdout 요약 표
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FOLD_ORDER = ["primary", "r2022", "r2023", "r2024"]
MIN_CELL_N = 3000
COUNT_STATES = [
    "0-0", "0-1", "0-2", "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2", "3-0", "3-1", "3-2",
]


def build_signal_masks(t):
    """교차 신호 4종의 pos/ref boolean 마스크 (numpy, 전체 train 행)."""
    balls = t["balls_before"].to_numpy()
    strikes = t["strikes_before"].to_numpy()
    outs = t["outs_before"].to_numpy()
    p_hand = t["pitcher_hand"].to_numpy()
    b_hand = t["batter_hand"].to_numpy()
    same = p_hand == b_hand
    c32 = (balls == 3) & (strikes == 2)
    c01 = (balls == 0) & (strikes == 1)
    risp = (t["runner_on_2b"].to_numpy() == 1) | (t["runner_on_3b"].to_numpy() == 1)
    sd = t["score_diff_pitcher_team"].to_numpy()

    return {
        "count_platoon_3b2_same": dict(
            desc="3-2 카운트 & 동손(pitcher_hand==batter_hand)",
            formula="(balls==3 & strikes==2) & (pitcher_hand==batter_hand)",
            ref_desc="0-1 카운트 & 이손",
            ref_formula="(balls==0 & strikes==1) & (pitcher_hand!=batter_hand)",
            pos=c32 & same,
            ref=c01 & ~same,
            report=dict(
                period="recent(2023-24)",
                report_delta_pp=-8.4, report_pos_n=7653,
                source="REPORT §3 카운트×플래툰 (3-2+동손 0.4362 vs 0-1+이손 0.5198)",
                note="pos가 ref보다 낮은 '붕괴 셀' → delta_pp는 음수",
            ),
        ),
        "score_diff_binary": dict(
            desc="|score_diff_pitcher_team|<=1 (close)",
            formula="abs(score_diff_pitcher_team)<=1",
            ref_desc="|score_diff_pitcher_team|>=6 (blowout)",
            ref_formula="abs(score_diff_pitcher_team)>=6",
            pos=np.abs(sd) <= 1,
            ref=np.abs(sd) >= 6,
            report=dict(
                period="full(2019-24)",
                report_delta_pp=1.8,
                source="REPORT H8 접전 vs 대량 점수차 (close 0.5283 vs blowout 0.5121)",
            ),
        ),
        "li_risp_flag": dict(
            desc="RISP & LI>=1.0",
            formula="(runner_on_2b==1 | runner_on_3b==1) & li>=1.0",
            ref_desc="RISP & LI<0.5",
            ref_formula="(runner_on_2b==1 | runner_on_3b==1) & li<0.5",
            pos=risp & (t["li"].to_numpy() >= 1.0),
            ref=risp & (t["li"].to_numpy() < 0.5),
            report=dict(
                period="full(2019-24)",
                report_delta_pp=2.3,
                source="REPORT §3 주자×LI (RISP+LI 1.0-2.0 0.5324 n=159,226 vs LI<0.5 0.5094 n=86,250)",
                note="REPORT는 LI 1.0-2.0 밴드 — 본 스크립트 pos는 li>=1.0 (2.0 초과 포함, 보수적)",
            ),
        ),
        "outs_count_3b2_2out": dict(
            desc="3-2 카운트 & outs_before==2",
            formula="(balls==3 & strikes==2) & outs_before==2",
            ref_desc="3-2 카운트 & outs_before==0",
            ref_formula="(balls==3 & strikes==2) & outs_before==0",
            pos=c32 & (outs == 2),
            ref=c32 & (outs == 0),
            report=dict(
                period="full(2019-24)",
                report_delta_pp=1.7,
                source="REPORT §3 카운트×아웃 (3-2 2아웃 0.5087 n=24,070 vs 0아웃 0.4918 n=23,441)",
            ),
        ),
    }


def cell_bss(cell_tr, y_tr, cell_va, yv, r_val):
    """셀(카테고리) 버킷 모델: train 셀 평균 → val 예측 + base-rate(logit shift) 보정.
    eda_recent_gap.bucket_bss / eda_missing_pattern.flag_bss 방식 재사용.
    bss_bucket_calib = 예상 기여, bss_oracle = val 셀 평균 적용 상한."""
    rate_tr = y_tr.groupby(cell_tr, observed=True).mean().to_dict()
    p = pd.Series(cell_va.astype(object), index=yv.index).map(rate_tr) \
        .astype("float32").fillna(r_val).values
    z = common.logit(p) + (common.logit(np.full(1, r_val))[0] - common.logit(p.mean()))
    p_calib = common.sigmoid(z)
    rate_va = yv.groupby(cell_va, observed=True).mean().to_dict()
    p_or = pd.Series(cell_va.astype(object), index=yv.index).map(rate_va) \
        .astype("float32").fillna(r_val).values
    return dict(
        bss_bucket_raw=float(common.score(p, yv.values)),
        bss_bucket_calib=float(common.score(p_calib, yv.values)),
        expected_gain_estimate=float(common.score(p_calib, yv.values)),
        bss_oracle=float(common.score(p_or, yv.values)),
    )


def main():
    t0 = time.time()
    print("[eda_cross_signals] 교차 신호 4종 EDA + BSS 잠재력 측정 시작", flush=True)

    train, _ = common.load_train()
    common.preprocess_for_submission(train)  # platoon + count_state 생성 (common.py 미수정)
    y = train[common.TARGET]
    n_all = len(train)
    overall_rate = float(y.mean())
    print(f"train: {train.shape} | target=%.4f | platoon={train['platoon'].dtype} "
          f"count_state={train['count_state'].dtype} | memory: "
          f"{train.memory_usage(deep=True).sum()/1e6:.0f}MB" % overall_rate, flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }
    signals = build_signal_masks(train)

    # ── REPORT 재현 체크: 전체(2019-24) 또는 최근(2023-24) 셀 통계 ──
    recent = train["season"].isin([2023, 2024]).to_numpy()
    report_check = {}
    for key, s in signals.items():
        m = recent if s["report"]["period"].startswith("recent") else np.ones(n_all, dtype=bool)
        m_pos = s["pos"] & m
        m_ref = s["ref"] & m
        n_pos, n_ref = int(m_pos.sum()), int(m_ref.sum())
        rate_pos = float(y[m_pos].mean()) if n_pos else None
        rate_ref = float(y[m_ref].mean()) if n_ref else None
        report_check[key] = dict(
            period=s["report"]["period"], source=s["report"]["source"],
            report_delta_pp=s["report"]["report_delta_pp"],
            n_pos=n_pos, rate_pos=rate_pos, n_ref=n_ref, rate_ref=rate_ref,
            delta_pp=float(100 * (rate_pos - rate_ref)) if n_pos and n_ref else None,
        )
    # game_type×count 최근(2023-24) 교차표 (REPORT §3 재현용)
    recent_r = recent & isR.to_numpy()
    recent_f = recent & ~isR.to_numpy()
    gt_recent = {}
    for lbl, m in (("R", recent_r), ("F", recent_f)):
        rows = []
        for cs in COUNT_STATES:
            bb, st = int(cs[0]), int(cs[2])
            cm = (train["balls_before"].to_numpy() == bb) & \
                 (train["strikes_before"].to_numpy() == st) & m
            n = int(cm.sum())
            rows.append(dict(count_state=cs, n=n,
                             target_rate=float(y[cm].mean()) if n else None,
                             status="표본 부족" if n < MIN_CELL_N else "ok"))
        gt_recent[lbl] = rows
    report_check["game_type_x_count_recent_2023_24"] = gt_recent

    # ── game_type×count fold별 교차표 (val) ──
    gt_tables = {}
    for fn in FOLD_ORDER:
        va_m = folds[fn][1].to_numpy()
        tbl = {}
        for lbl, m_gt in (("R", isR.to_numpy()), ("F", (~isR).to_numpy())):
            rows = []
            for cs in COUNT_STATES:
                bb, st = int(cs[0]), int(cs[2])
                cm = (train["balls_before"].to_numpy() == bb) & \
                     (train["strikes_before"].to_numpy() == st) & va_m & m_gt
                n = int(cm.sum())
                rows.append(dict(count_state=cs, n=n,
                                 target_rate=float(y[cm].mean()) if n else None,
                                 status="표본 부족" if n < MIN_CELL_N else "ok"))
            tbl[lbl] = rows
        # R-F Δ (양쪽 n>=3000일 때만)
        deltas = []
        for i, cs in enumerate(COUNT_STATES):
            r_, f_ = tbl["R"][i], tbl["F"][i]
            if r_["status"] == "ok" and f_["status"] == "ok":
                deltas.append(dict(count_state=cs, delta_pp=float(
                    100 * (r_["target_rate"] - f_["target_rate"]))))
        gt_tables[fn] = dict(tables=tbl, r_minus_f_delta_pp=deltas,
                             val_year=int(train.loc[folds[fn][1], "season"].mode().iloc[0]))

    # ── fold별 신호 분석 ──
    results = dict(
        meta=dict(
            script="eda_cross_signals.py", date="2026-08-09",
            source="REPORT_kbo_insights.md §3 (카운트×플래툰 / 주자×LI / 카운트×아웃 / game_type×카운트)",
            preprocess="common.preprocess_for_submission(train) → platoon(pitcher_hand*2+batter_hand, "
                       "3=L-L/4=L-R/5=R-L/6=R-R) + count_state(balls*3+strikes)",
            fold_defs=dict(
                primary="train: season<=2023 / val: season==2024",
                r2022="train: (season<=2021)&R / val: (season==2022)&R",
                r2023="train: (season<=2022)&R / val: (season==2023)&R",
                r2024="train: (season<=2023)&R / val: (season==2024)&R",
            ),
            signals={k: dict(desc=s["desc"], formula=s["formula"],
                             ref_desc=s["ref_desc"], ref_formula=s["ref_formula"])
                     for k, s in signals.items()},
            min_cell_n=MIN_CELL_N,
            cell_rule=f"양성/기준 셀 n<{MIN_CELL_N:,}이면 '표본 부족'으로 표기하고 세부 통계 생략",
            bss_interpretation="bss_bucket_calib=base-rate(logit shift) 보정 버킷 BSS = 예상 기여, "
                               "bss_bucket_raw=비캘리브레이션, bss_oracle=val 셀 평균 적용 상한. "
                               "flag=pos vs 나머지 이진, pair=pos/ref/기타 3셀. "
                               "delta_pp=percentage point (pos−ref 기준, s1은 pos<ref로 음수).",
        ),
        overall=dict(n=n_all, target_rate=overall_rate),
        report_check=report_check,
        game_type_count_tables=gt_tables,
        folds={},
    )

    for fn in FOLD_ORDER:
        tr_m, va_m = folds[fn]
        tr_b, va_b = tr_m.to_numpy(dtype=bool), va_m.to_numpy(dtype=bool)
        y_tr, yv = y.loc[tr_m], y.loc[va_m]
        r_val = float(yv.mean())
        res = dict(
            n_train=int(tr_b.sum()), n_val=int(va_b.sum()),
            val_year=int(train.loc[va_m, "season"].mode().iloc[0]),
            target_rate_val=r_val, signals={},
        )
        for key, s in signals.items():
            pos_tr, pos_va = s["pos"][tr_b], s["pos"][va_b]
            ref_tr, ref_va = s["ref"][tr_b], s["ref"][va_b]
            n_pos, n_ref = int(pos_va.sum()), int(ref_va.sum())
            rec = dict(n_pos=n_pos, n_pos_pct=float(pos_va.mean()),
                       n_ref=n_ref, n_ref_pct=float(ref_va.mean()),
                       status="표본 부족" if (n_pos < MIN_CELL_N or n_ref < MIN_CELL_N) else "ok")
            if rec["status"] == "ok":
                yv_pos = yv[pos_va]
                rec.update(
                    rate_pos=float(yv_pos.mean()),
                    rate_ref=float(yv[ref_va].mean()),
                    rate_all=r_val,
                    delta_vs_all_pp=float(100 * (yv_pos.mean() - r_val)),
                    delta_vs_ref_pp=float(100 * (yv_pos.mean() - yv[ref_va].mean())),
                )
                # flag 모델 (pos vs 나머지)
                cflag_tr = pd.Series(pos_tr.astype("int8"), index=y_tr.index)
                cflag_va = pd.Series(pos_va.astype("int8"), index=yv.index)
                rec["bss_flag"] = cell_bss(cflag_tr, y_tr, cflag_va, yv, r_val)
                # pair 모델 (pos=2 / ref=1 / 기타=0) — full-length 마스크 필요
                pos_sel, ref_sel = s["pos"] & tr_b, s["ref"] & tr_b
                pos_sel_v, ref_sel_v = s["pos"] & va_b, s["ref"] & va_b
                cpair_tr = np.zeros(n_all, dtype="int8")
                cpair_tr[ref_sel] = 1
                cpair_tr[pos_sel] = 2
                cpair_va = np.zeros(n_all, dtype="int8")
                cpair_va[ref_sel_v] = 1
                cpair_va[pos_sel_v] = 2
                rec["bss_pair"] = cell_bss(
                    pd.Series(cpair_tr[tr_b], index=y_tr.index),
                    y_tr,
                    pd.Series(cpair_va[va_b], index=yv.index),
                    yv, r_val)
            res["signals"][key] = rec
        results["folds"][fn] = res

    # ── stdout 요약 ──
    print("\n" + "=" * 120, flush=True)
    print("교차 신호 EDA 요약 — fold별 (전체 target=%.4f, n=%s)" % (overall_rate, f"{n_all:,}"), flush=True)
    print("=" * 120, flush=True)
    for key, s in signals.items():
        print("\n■ %s — %s  [REFORT: %s]" % (key, s["desc"], s["report"]["source"]), flush=True)
        hdr = "  %-9s %8s %7s %8s %8s %10s %10s %8s %9s %9s %9s" % (
            "fold", "n_val", "target", "n_pos", "rate_pos", "rate_ref",
            "Δvs_ref", "Δvs_all", "BSS_flag", "BSS_pair", "BSS_or")
        print(hdr, flush=True)
        for fn in FOLD_ORDER:
            r = results["folds"][fn]
            sg = r["signals"][key]
            if sg["status"] == "ok":
                bf, bp = sg["bss_flag"]["bss_bucket_calib"], sg["bss_pair"]["bss_bucket_calib"]
                bo = max(sg["bss_pair"]["bss_oracle"], sg["bss_flag"]["bss_oracle"])
                print("  %-9s %8s %7.4f %8s %8.4f %10.4f %+10.2f %+10.2f %9.1f %9.1f %9.1f" % (
                    f"{fn}({r['val_year']})", f"{r['n_val']:,}", r["target_rate_val"],
                    f"{sg['n_pos']:,}", sg["rate_pos"], sg["rate_ref"],
                    sg["delta_vs_ref_pp"], sg["delta_vs_all_pp"], bf, bp, bo), flush=True)
            else:
                print("  %-9s %8s %7.4f %8s %s" % (
                    f"{fn}({r['val_year']})", f"{r['n_val']:,}", r["target_rate_val"],
                    f"{sg['n_pos']:,}", "표본 부족(n<3000) — 세부 통계 생략"), flush=True)
        # report 재현 요약
        rc = report_check[key]
        print("  └ report_check(%s): n_pos=%s rate=%.4f / n_ref=%s rate=%.4f → Δ=%+.2fpp "
              "(보고 Δ=%+.1fpp)" % (
                  rc["period"], f"{rc['n_pos']:,}", rc["rate_pos"], f"{rc['n_ref']:,}",
                  rc["rate_ref"], rc["delta_pp"], rc["report_delta_pp"]), flush=True)

    print("\n" + "-" * 120, flush=True)
    print("game_type×count 교차표 (fold val, '표본 부족'=n<3,000) — R−F Δ(pp):", flush=True)
    for fn in FOLD_ORDER:
        t = gt_tables[fn]
        r_, f_ = t["tables"]["R"], t["tables"]["F"]
        dmap = {d["count_state"]: d["delta_pp"] for d in t["r_minus_f_delta_pp"]}
        print("  [%s (val %s)]  " % (fn, t["val_year"]), flush=True)
        for i, cs in enumerate(COUNT_STATES):
            rv, fv = r_[i], f_[i]
            rr = "%.4f" % rv["target_rate"] if rv["status"] == "ok" else "표본부족"
            fr = "%.4f" % fv["target_rate"] if fv["status"] == "ok" else "표본부족"
            d = ("%+.2f" % dmap[cs]) if cs in dmap else "  -  "
            print("    %-4s R %8s %s | F %8s %s | R−F %s" % (
                cs, f"{rv['n']:,}", rr, f"{fv['n']:,}", fr, d), flush=True)

    print("\n" + "-" * 120, flush=True)
    print("BSS 예상 기여(bss_bucket_calib, flag/pair) — fold별:", flush=True)
    for key in signals:
        line = "  %-26s" % key
        for fn in FOLD_ORDER:
            sg = results["folds"][fn]["signals"][key]
            if sg["status"] == "ok":
                line += "  %s flag=%6.1f/pair=%6.1f" % (fn.split("r")[-1].rjust(4),
                                                        sg["bss_flag"]["bss_bucket_calib"],
                                                        sg["bss_pair"]["bss_bucket_calib"])
            else:
                line += "  %s flag=   -  /pair=   -  " % fn.split("r")[-1].rjust(4)
        print(line, flush=True)
    print("  (primary=2024전체 / r*=R-only. F3 기준선 BSS: primary 729.7 / r2022 578.4 / r2023 551.0 / r2024 711.4)", flush=True)

    out = "experiments/eda_cross.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n저장: {out} (총 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
