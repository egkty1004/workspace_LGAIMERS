#!/usr/bin/env python3
"""screen_all_10seed.py — 10시드 필수 + 강화 게이트 피처 전면 스크리닝 (하드닝 v2)

배경 (승자 저주 대응): 5시드 스크리닝(screen_all.py)에서 asof_n_bucket / score_diff_binary가
+10.4/+13.0으로 통과했으나 10시드 검증에서 primary +3.4, R-only 1/3으로 붕괴.
diag_mean_sensitivity.py 감사 결과 모든 후보가 mean-safe(Δmean≤0.005) → 진짜 필터는
시드 수 + R-only 일관성. 이 하네스는 10시드(42..51)를 필수로 하고 게이트를 강화한다.

실험 설계 (screen_all.py와 동일):
    baseline_fe(F3_EXTRA 10종 전체) 기준,
    ablation 후보 8종 각각 제거 → delta = baseline_fe − reduced,
    추가 후보 count_platoon_3b2_same → delta = added − baseline_fe.
    cats = CAT_COLS + ["platoon", "count_state"] (신규 int8/float32 피처는 numeric).
    4폴드(primary/r2022/r2023/r2024) × 10시드 로짓 평균 BSS (common.PARAMS deterministic=True).

강화 게이트:
    채택: primary delta ≥ +15 AND R-only 3/3 AND max|Δmean| ≤ 0.005 (4폴드 전체)
    기각: 그 외 전부
    Δmean = 후보 폴드별 pred_mean − baseline_fe 폴드별 pred_mean (mean-shift 가드).

--smoke: SEEDS=[42,43] + baseline_fe + ablation 후보 1종만 실행해 코드 경로
    (폴드/delta/게이트 판정/JSON)를 검증하고 PASS 출력. 기본 전체 동작에 영향 없음.
    --combo: 단순 보고 모드 — 채택 후보 목록만 출력 (추가 학습 없음).

출력: experiments/screen_all_10seed.json (screen_all.json 스키마 + max_abs_dmean/dmean_per_fold)
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

SEEDS = list(range(42, 52))  # 10시드 (42..51)
FOLDS = ["primary", "r2022", "r2023", "r2024"]
R_FOLDS = ["r2022", "r2023", "r2024"]

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
ADD_CANDIDATES = ["count_platoon_3b2_same"]

CATS = common.CAT_COLS + ["platoon", "count_state"]

# 강화 게이트: primary ≥ +15 & R-only 3/3 & max|Δmean| ≤ 0.005
PRIMARY_THRESHOLD = 15.0
R_ONLY_REQUIRED = 3
MEAN_POISON_THRESHOLD = 0.005

GATE = (
    f"primary delta ≥ +{PRIMARY_THRESHOLD:.0f} & R-only {R_ONLY_REQUIRED}/3 "
    f"& max|Δmean| ≤ {MEAN_POISON_THRESHOLD} → 채택 | 그 외 → 기각"
)


def build_folds(train):
    isR = train["game_type"] == "R"
    return {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }


def run_config(train, feats, folds, seeds):
    """한 구성에 대해 4폴드 BSS. seeds 시드 로짓 평균 (common.PARAMS deterministic=True)."""
    out = {}
    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        yv = y_va.values
        zs = []
        for seed in seeds:
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


def dmean_per_fold(cand_res, base_res):
    """후보 폴드별 pred_mean − baseline_fe 폴드별 pred_mean (mean-shift 가드)."""
    return {fn: cand_res[fn]["pred_mean"] - base_res[fn]["pred_mean"]
            for fn in FOLDS}


def gate_verdict(delta, dmean):
    """강화 게이트 판정. (verdict, 실패 조건 목록) 반환 — 빈 목록 = 채택."""
    primary = delta["primary"]
    r_imp = r_only_count(delta)
    max_abs_dmean = max(abs(v) for v in dmean.values())
    failures = []
    if primary < PRIMARY_THRESHOLD:
        failures.append(f"primary {primary:+.1f}<+{PRIMARY_THRESHOLD:.0f}")
    if r_imp < R_ONLY_REQUIRED:
        failures.append(f"R-only {r_imp}/{R_ONLY_REQUIRED}")
    if max_abs_dmean > MEAN_POISON_THRESHOLD:
        failures.append(f"maxΔmean {max_abs_dmean:.4f}>{MEAN_POISON_THRESHOLD:.3f}")
    return ("채택" if not failures else "기각"), failures


def main():
    ap = argparse.ArgumentParser(
        description="10시드 필수 + 강화 게이트 피처 스크리닝 (하드닝 v2)")
    ap.add_argument("--combo", action="store_true",
                    help="채택 후보 목록 보고 (단순 보고 — 추가 학습 없음)")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크 테스트: SEEDS=[42,43] + baseline_fe + ablation 후보 1종")
    args = ap.parse_args()

    t0 = time.time()
    seeds = list(SEEDS)
    abl_cands = list(ABLATION_CANDIDATES)
    add_cands = list(ADD_CANDIDATES)
    smoke = False
    if args.smoke:
        smoke = True
        seeds = [42, 43]
        abl_cands = ["recent_gap_success"]
        add_cands = []
        print(f"[--smoke] 축소 실행: SEEDS={seeds} ablation 1종만 "
              "(기본 동작과 무관한 테스트 모드)", flush=True)

    print(f"[screen_all_10seed] 강화 게이트 피처 스크리닝 "
          f"({len(seeds)}시드 × 4폴드, cats 분리, deterministic=True)", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    base_feats = common.get_feature_cols(
        pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    folds = build_folds(train)
    print(f"train: {train.shape} | feats: {len(base_feats)} | cats: {CATS} | "
          f"seeds: {seeds}", flush=True)

    # ── configs: baseline_fe + ablation + addition ──
    configs = [("baseline_fe", list(base_feats))]
    for c in abl_cands:
        configs.append((f"reduced_{c}", [f for f in base_feats if f != c]))
    for c in add_cands:
        configs.append((f"added_{c}", list(base_feats) + [c]))

    results = {}
    for cfg_name, feats in configs:
        t1 = time.time()
        results[cfg_name] = run_config(train, feats, folds, seeds)
        row = " ".join(f"{results[cfg_name][fn]['bss']:>8.1f}" for fn in FOLDS)
        print(f"  [{cfg_name:<28s}] {row}  ({time.time()-t1:.0f}s)", flush=True)

    base = results["baseline_fe"]

    # ── delta 계산: ablation = baseline_fe − reduced, addition = added − baseline_fe ──
    deltas = {}
    for c in abl_cands:
        deltas[c] = {fn: base[fn]["bss"] - results[f"reduced_{c}"][fn]["bss"]
                     for fn in FOLDS}
    for c in add_cands:
        deltas[c] = {fn: results[f"added_{c}"][fn]["bss"] - base[fn]["bss"]
                     for fn in FOLDS}

    cands = abl_cands + add_cands
    dmeans = {c: dmean_per_fold(
        results[f"reduced_{c}" if c in abl_cands else f"added_{c}"], base)
        for c in cands}
    max_dmeans = {c: max(abs(v) for v in dmeans[c].values()) for c in cands}

    # ── 요약 표 ──
    print("\n" + "=" * 100, flush=True)
    print("screen_all_10seed 결과 (BSS, delta vs baseline_fe — ablation: 제거 시 변화, "
          "addition: 추가 시 변화)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'config':<28s} " + " ".join(f"{fn:>9s}" for fn in FOLDS), flush=True)
    for cfg_name, _ in configs:
        row = " ".join(f"{results[cfg_name][fn]['bss']:>9.1f}" for fn in FOLDS)
        print(f"  {cfg_name:<28s} {row}", flush=True)
    print("-" * 100, flush=True)
    for c in cands:
        drow = " ".join(f"{deltas[c][fn]:>+9.1f}" for fn in FOLDS)
        print(f"  {'Δ'+c:<28s} {drow}", flush=True)

    # ── 강화 게이트 판정 ──
    print("-" * 100, flush=True)
    print("  게이트: " + GATE, flush=True)
    verdicts = {c: gate_verdict(deltas[c], dmeans[c]) for c in cands}
    for c in cands:
        verdict, failures = verdicts[c]
        fail_txt = f"  ✗ {' '.join(failures)}" if failures else ""
        print(f"  {c:<28s} primary {deltas[c]['primary']:+.1f} / R-only "
              f"{r_only_count(deltas[c])}/3 / maxΔmean {max_dmeans[c]:.4f} "
              f"→ {verdict}{fail_txt}", flush=True)

    # ── combo: 채택 후보 보고 (단순 모드 — 추가 학습 없음) ──
    combo_info = {"status": "not_run"}
    if args.combo:
        passed = [c for c in cands if verdicts[c][0] == "채택"]
        combo_info = {
            "status": "run",
            "adopted": passed,
            "note": "단순 보고 모드 — 추가 학습 없음 (10시드 강화 게이트 기준)",
        }
        print(f"\n  [--combo] 채택 후보: {passed}", flush=True)

    # ── JSON 저장 ──
    json_results = {"baseline_fe": results["baseline_fe"]}
    for c in abl_cands:
        json_results[c] = {
            "mode": "ablation",
            "bss": results[f"reduced_{c}"],
            "delta_vs_baseline_fe": deltas[c],
            "r_only_improved": r_only_count(deltas[c]),
            "verdict": verdicts[c][0],
            "max_abs_dmean": max_dmeans[c],
            "dmean_per_fold": dmeans[c],
        }
    for c in add_cands:
        json_results[c] = {
            "mode": "addition",
            "bss": results[f"added_{c}"],
            "delta_vs_baseline_fe": deltas[c],
            "r_only_improved": r_only_count(deltas[c]),
            "verdict": verdicts[c][0],
            "max_abs_dmean": max_dmeans[c],
            "dmean_per_fold": dmeans[c],
        }
    json_results["_meta"] = dict(
        seeds=seeds, folds=FOLDS, gate=GATE, cats=CATS,
        baseline_fe="F3_EXTRA 전체(10종: base + platoon + count_state + 신규 8종)",
        ablation_candidates=abl_cands,
        add_candidates=add_cands,
        smoke=smoke,
        note=("10시드 필수 + 강화 게이트: primary delta ≥ +15 & R-only 3/3 & "
              "max|Δmean| ≤ 0.005 → 채택. 5시드 승자 저주(5시드 +10.4/+13.0 → 10시드 "
              "+3.4/R-only 1/3) 대응용."),
        reference="이번 실행 baseline_fe (deterministic=True, 신규 피처 포함 — "
                  "기존 726.3/729.7 캐시 기준선과 직접 비교 금지)",
        combo=combo_info,
        total_time=time.time() - t0)
    out_path = "experiments/screen_all_10seed.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, default=float)
    print(f"\n저장: {out_path} (총 {time.time()-t0:.0f}s)", flush=True)

    if smoke:
        ok = ("baseline_fe" in json_results
              and "recent_gap_success" in json_results
              and len(seeds) == 2
              and all(fn in json_results["baseline_fe"] for fn in FOLDS)
              and "max_abs_dmean" in json_results["recent_gap_success"]
              and "dmean_per_fold" in json_results["recent_gap_success"])
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 폴드/delta/게이트/JSON 경로 "
              f"검증 완료 (seeds={len(seeds)})", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
