#!/usr/bin/env python3
"""
diag_mean_sensitivity.py — 후보 피처 pred_mean 감사 (Oracle 게이트 재설계 근거).

목적: 로컬 게이트(primary 2024)의 후보 채택/기각 판정이
      (a) 예측 평균 이동(mean shift)에서 왔는지, (b) 진짜 판별력 변화에서 왔는지 분해.
      Oracle 전략: mean 이동은 2025에서 40pts/0.01 페널티 → Δmean>0.005 후보는
      게이트 통과와 무관하게 폐기. 단, 감사 결과 피처 단위 후보는 전부 mean-safe라
      진짜 필터는 10시드 필수 + R-only 3/3 임을 확인.

입력: experiments/screen_all.json (5시드, 폴드별 pred_mean 포함)
      experiments/e6c_blend_adopted.json (10시드 게이트 결과)
출력: stdout 요약 + experiments/mean_sensitivity_report.json
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

SCREEN_ALL = "experiments/screen_all.json"
E6C_ADOPTED = "experiments/e6c_blend_adopted.json"
OUT_PATH = "experiments/mean_sensitivity_report.json"
FOLDS = ["primary", "r2022", "r2023", "r2024"]
MEAN_POISON_THRESHOLD = 0.005

CANDIDATES = [
    "recent_gap_success", "return_gap", "pitcher_debut", "batter_debut",
    "asof_n_bucket", "score_diff_binary", "li_risp_flag",
    "outs_count_3b2_2out", "count_platoon_3b2_same",
]


def load_screen(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def audit(screen: dict) -> dict:
    base = screen["baseline_fe"]
    rows = {}
    for name in CANDIDATES:
        c = screen[name]
        bs = c["bss"]
        dm = c.get("delta_vs_baseline_fe", {})
        dmeans = {fn: bs[fn]["pred_mean"] - base[fn]["pred_mean"] for fn in FOLDS}
        max_abs = max(abs(v) for v in dmeans.values())
        rows[name] = {
            "mode": c.get("mode"),
            "verdict": c.get("verdict"),
            "primary_delta_bss": dm.get("primary"),
            "r_only_improved": c.get("r_only_improved"),
            "dmean_per_fold": {fn: round(v, 6) for fn, v in dmeans.items()},
            "max_abs_dmean": round(max_abs, 6),
            "mean_poisoned": max_abs > MEAN_POISON_THRESHOLD,
        }
    return rows


def load_e6c(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    screen = load_screen(SCREEN_ALL)
    rows = audit(screen)
    e6c = load_e6c(E6C_ADOPTED)

    print("=== mean-sensitivity audit: Δmean (후보 − baseline) ===")
    print(f"{'후보':<22} {'prim Δbss':>9} {'prim Δmean':>10} {'max|Δmean|':>10}  mean-poisoned")
    print("-" * 80)
    for name, r in rows.items():
        flag = "⚠️" if r["mean_poisoned"] else "-"
        print(f"{name:<22} {r['primary_delta_bss']:>+9.1f} "
              f"{r['dmean_per_fold']['primary']:>+10.5f} {r['max_abs_dmean']:>10.5f}  {flag}")

    report = {
        "threshold_me": MEAN_POISON_THRESHOLD,
        "note": ("피처 단위 후보 전부 mean-safe → mean 게이트만으로는 여과 불가. "
                 "진짜 필터는 10시드 필수 + R-only 3/3."),
        "candidates": rows,
        "e6c_blend_adopted_10seed": None,
    }
    if e6c:
        report["e6c_blend_adopted_10seed"] = {
            "verdict": e6c.get("verdict"),
            "gate": e6c.get("gate"),
            "gain_primary_10seed": e6c.get("gate", {}).get("gain_primary"),
            "r_only_improved_10seed": e6c.get("gate", {}).get("r_only_improved"),
            "primary_w": e6c.get("primary_w"),
            "primary_blend": e6c.get("primary_blend"),
        }
        print()
        print("=== 10시드 재검증 (e6c_blend_adopted) ===")
        print(f"  verdict: {e6c.get('verdict')}")
        print(f"  10시드 primary gain: {e6c['gate']['gain_primary']:+.1f} | "
              f"R-only {e6c['gate']['r_only_improved']}/3")
        print(f"  주: 5시드 +10.4/+13.0 통과 후보가 10시드에서 +3.4/R-only 1/3 붕괴 = 승자 저주")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n저장: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
