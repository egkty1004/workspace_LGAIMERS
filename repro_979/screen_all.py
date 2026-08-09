#!/usr/bin/env python3
"""screen_all.py — Wave C 경계 통과 유도 피처 전면 스크리닝 (계획 Todo 8, 2026-08-09)

F3_EXTRA(10종: platoon/count_state + 신규 8종) 전부를 포함한 **baseline_fe**를 기준으로,
신규 비범주형 피처 8종 각각을 제거(ablation)했을 때의 5시드 × 4폴드 로짓 평균 BSS를
측정한다. delta = baseline_fe − reduced (제거 시 성능 하락 = 해당 피처 기여 있음).
count_platoon_3b2_same은 F3_EXTRA 미포함 → "baseline_fe + 추가" 실험으로 처리.
delta = added − baseline_fe.

⚠️ cats 분리 (핵심):
    신규 피처(recent_gap_success float32 / return_gap·debut·asof_n_bucket·교차셀 int8)는
    **범주형이 아니다**. cats = CAT_COLS + ["platoon", "count_state"] 로 한정하고
    나머지는 numeric으로 취급한다 (cats = CAT_COLS + F3_EXTRA 금지).

게이트: primary delta ≥ +20 & R-only 2/3 → "채택"
        primary +10~19 & R-only 2/3 → "경계" (--combo 로 통합 확인)
        미달 → "기각"

--combo 모드: 통과/경계 후보 전부 동시 포함한 모델 BSS 측정 (통합 확인용).
    - ablation 후보는 baseline_fe에 이미 포함되어 있으므로 추가 후보(count_platoon_3b2_same)가
      통과/경계면 baseline_fe + 추가로 결합 모델을 학습한다.
    - 대조 기준 baseline_old(F3 신규 8종 제외: base + platoon + count_state)도 학습해
      그룹 이득(baseline_fe − baseline_old)을 보고한다 — 경계 판정의 통합 확인 근거.

출력: experiments/screen_all.json + 요약 테이블 (stdout)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEEDS = list(range(42, 47))  # 5시드
FOLDS = ["primary", "r2022", "r2023", "r2024"]
R_FOLDS = ["r2022", "r2023", "r2024"]

# 신규 비범주형 F3_EXTRA 멤버 — ablation 후보 (baseline_fe minus 1개)
ABLATION_CANDIDATES = [
    "recent_gap_success",
    "return_gap",
    "pitcher_debut",
    "batter_debut",
    "asof_n_bucket",
    "score_diff_binary",
    "li_risp_flag",
    "outs_count_3b2_2out",
]
# F3_EXTRA에 없는 추가 후보 — baseline_fe + 추가 실험
ADD_CANDIDATES = ["count_platoon_3b2_same"]

# ⚠️ 범주형으로 취급할 컬럼: CAT_COLS + platoon + count_state 만.
#    신규 int8/float32 피처는 numeric (cats = CAT_COLS + F3_EXTRA 금지).
CATS = common.CAT_COLS + ["platoon", "count_state"]

GATE = ("primary delta ≥ +20 & R-only 2/3 → 채택 | "
        "+10~19 & R-only 2/3 → 경계(combo) | 미달 → 기각")


def build_folds(train):
    isR = train["game_type"] == "R"
    return {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }


def run_config(train, feats, folds):
    """한 구성에 대해 4폴드 BSS. 5시드 로짓 평균 (common.PARAMS deterministic=True)."""
    out = {}
    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        zs = []
        for seed in SEEDS:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CATS)
            dva = lgb.Dataset(X_va, y_va, categorical_feature=CATS, reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            zs.append(common.logit(m.predict(X_va, num_iteration=m.best_iteration)))
        z = np.mean(zs, axis=0)
        p = common.sigmoid(z)
        out[fn] = dict(bss=float(common.score(p, yv)), pred_mean=float(p.mean()))
    return out


def r_only_count(delta):
    """R-only 폴드 중 개선(delta>0) 카운트."""
    return sum(1 for fn in R_FOLDS if delta[fn] > 1e-9)


def gate_verdict(delta):
    """delta 기준 게이트 판정: 채택 / 경계 / 기각."""
    primary = delta["primary"]
    r_imp = r_only_count(delta)
    if primary >= 20.0 and r_imp >= 2:
        return "채택"
    if primary >= 10.0 and r_imp >= 2:
        return "경계"
    return "기각"


def main():
    ap = argparse.ArgumentParser(description="Wave C 유도 피처 전면 스크리닝")
    ap.add_argument("--combo", action="store_true",
                    help="통과/경계 후보 전부 동시 포함 모델 BSS 측정 (통합 확인용)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[screen_all] Wave C 유도 피처 전면 스크리닝 (5시드 × 4폴드, cats 분리, "
          f"deterministic=True)", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    base_feats = common.get_feature_cols(
        pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    folds = build_folds(train)
    print(f"train: {train.shape} | feats: {len(base_feats)} | cats: {CATS} | "
          f"seeds: {SEEDS}", flush=True)

    # ── configs: baseline_fe + ablation 8종 + addition 1종 ──
    configs = [("baseline_fe", list(base_feats))]
    for c in ABLATION_CANDIDATES:
        configs.append((f"reduced_{c}", [f for f in base_feats if f != c]))
    for c in ADD_CANDIDATES:
        configs.append((f"added_{c}", list(base_feats) + [c]))

    results = {}
    for cfg_name, feats in configs:
        t1 = time.time()
        results[cfg_name] = run_config(train, feats, folds)
        row = " ".join(f"{results[cfg_name][fn]['bss']:>8.1f}" for fn in FOLDS)
        print(f"  [{cfg_name:<28s}] {row}  ({time.time()-t1:.0f}s)", flush=True)

    base = results["baseline_fe"]

    # ── delta 계산: ablation = baseline_fe − reduced, addition = added − baseline_fe ──
    deltas = {}
    for c in ABLATION_CANDIDATES:
        deltas[c] = {fn: base[fn]["bss"] - results[f"reduced_{c}"][fn]["bss"]
                     for fn in FOLDS}
    for c in ADD_CANDIDATES:
        deltas[c] = {fn: results[f"added_{c}"][fn]["bss"] - base[fn]["bss"]
                     for fn in FOLDS}

    # ── 요약 표 ──
    print("\n" + "=" * 100, flush=True)
    print("screen_all 결과 (BSS, delta vs baseline_fe — ablation: 제거 시 변화, "
          "addition: 추가 시 변화)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'config':<28s} " + " ".join(f"{fn:>9s}" for fn in FOLDS), flush=True)
    for cfg_name, _ in configs:
        row = " ".join(f"{results[cfg_name][fn]['bss']:>9.1f}" for fn in FOLDS)
        print(f"  {cfg_name:<28s} {row}", flush=True)
    print("-" * 100, flush=True)
    for c in ABLATION_CANDIDATES + ADD_CANDIDATES:
        drow = " ".join(f"{deltas[c][fn]:>+9.1f}" for fn in FOLDS)
        print(f"  {'Δ'+c:<28s} {drow}", flush=True)

    # ── 게이트 판정 ──
    print("-" * 100, flush=True)
    print("  게이트: " + GATE, flush=True)
    verdicts = {c: gate_verdict(deltas[c]) for c in ABLATION_CANDIDATES + ADD_CANDIDATES}
    for c in ABLATION_CANDIDATES + ADD_CANDIDATES:
        d = deltas[c]
        print(f"  {c:<28s} primary {d['primary']:+.1f} / R-only {r_only_count(d)}/3 "
              f"→ {verdicts[c]}", flush=True)

    # ── combo: 통과/경계 후보 전부 동시 포함 ──
    combo_info = {"status": "not_run"}
    if args.combo:
        print("\n" + "=" * 100, flush=True)
        print("[--combo] 통과/경계 후보 통합 확인", flush=True)
        passed = [c for c in ABLATION_CANDIDATES + ADD_CANDIDATES
                  if verdicts[c] in ("채택", "경계")]
        join_add = [c for c in ADD_CANDIDATES if verdicts[c] in ("채택", "경계")]
        combo_info = {
            "status": "run",
            "passed_or_boundary": passed,
            "ablation_kept": [c for c in ABLATION_CANDIDATES
                              if verdicts[c] in ("채택", "경계")],
            "addition_joined": join_add,
        }
        print(f"  통과/경계 후보: {passed}", flush=True)
        print(f"  ablation 후보는 baseline_fe에 이미 포함 | 추가 합류: {join_add}",
              flush=True)
        if join_add:
            combo_feats = list(base_feats) + join_add
            t1 = time.time()
            cr = run_config(train, combo_feats, folds)
            combo_info["bss"] = cr
            combo_info["delta_vs_baseline_fe"] = {
                fn: cr[fn]["bss"] - base[fn]["bss"] for fn in FOLDS}
            combo_info["r_only_improved"] = r_only_count(combo_info["delta_vs_baseline_fe"])
            row = " ".join(f"{cr[fn]['bss']:>9.1f}" for fn in FOLDS)
            print(f"  [combo{'(+'+'+'.join(join_add)+')':<24s}] {row} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            drow = " ".join(f"{combo_info['delta_vs_baseline_fe'][fn]:>+9.1f}"
                            for fn in FOLDS)
            print(f"  {'Δcombo':<28s} {drow}", flush=True)
        else:
            combo_info["status"] = "skipped"
            combo_info["reason"] = ("추가 후보(count_platoon_3b2_same) 채택/경계 없음 — "
                                    "ablation 후보는 baseline_fe에 이미 동시 포함")
            print(f"  → 추가 합류 후보 없음: baseline_fe가 이미 전부 포함 (스킵)", flush=True)

        # 그룹 이득 대조: baseline_old (F3 신규 8종 제외) vs baseline_fe
        old_feats = [f for f in base_feats if f not in ABLATION_CANDIDATES]
        t1 = time.time()
        old_res = run_config(train, old_feats, folds)
        combo_info["baseline_old"] = old_res  # base + platoon + count_state
        combo_info["group_delta_vs_old"] = {
            fn: base[fn]["bss"] - old_res[fn]["bss"] for fn in FOLDS}
        combo_info["group_r_only_improved"] = r_only_count(combo_info["group_delta_vs_old"])
        row = " ".join(f"{old_res[fn]['bss']:>9.1f}" for fn in FOLDS)
        print(f"  [baseline_old(F3-신규8종 제외)] {row}  ({time.time()-t1:.0f}s)",
              flush=True)
        drow = " ".join(f"{combo_info['group_delta_vs_old'][fn]:>+9.1f}" for fn in FOLDS)
        print(f"  {'Δ그룹(baseline_fe−old)':<28s} {drow}", flush=True)

    # ── JSON 저장 ──
    json_results = {"baseline_fe": results["baseline_fe"]}
    for c in ABLATION_CANDIDATES:
        json_results[c] = {
            "mode": "ablation",
            "bss": results[f"reduced_{c}"],
            "delta_vs_baseline_fe": deltas[c],
            "r_only_improved": r_only_count(deltas[c]),
            "verdict": verdicts[c],
        }
    for c in ADD_CANDIDATES:
        json_results[c] = {
            "mode": "addition",
            "bss": results[f"added_{c}"],
            "delta_vs_baseline_fe": deltas[c],
            "r_only_improved": r_only_count(deltas[c]),
            "verdict": verdicts[c],
        }
    json_results["_meta"] = dict(
        seeds=SEEDS, folds=FOLDS, gate=GATE, cats=CATS,
        baseline_fe="F3_EXTRA 전체(10종: base + platoon + count_state + 신규 8종)",
        ablation_candidates=ABLATION_CANDIDATES,
        add_candidates=ADD_CANDIDATES,
        reference="이번 실행 baseline_fe (deterministic=True, 신규 피처 포함 — "
                  "기존 726.3/729.7 캐시 기준선과 직접 비교 금지)",
        combo=combo_info,
        total_time=time.time() - t0)
    out_path = "experiments/screen_all.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, default=float)
    print(f"\n저장: {out_path} (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
