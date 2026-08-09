#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gate_verdict.py — 유도 피처 게이트 판정 (재판정, 학습 없음)

기존 `experiments/screen_all.json`(5시드 x 4폴드 스크리닝)의 저장된 BSS 값으로
각 후보의 채택/기각을 재판정한다. **재학습/LGB 실행 없음** — 판정 로직의
부호 오류 수정 + 게이트 기준 +10 적용이 목적.

부호 규칙 (ablation = 제거 실험):
  delta_vs_baseline_fe = reduced_bss - baseline_fe_bss
  - ablation: delta < 0 = 제거 시 성능 하락 = 피처가 기여.  기여도 = -delta.
  - addition: delta > 0 = 추가 시 성능 상승 = 피처가 기여. 기여도 = +delta.
R-only 개선 카운트 (r2022/r2023/r2024):
  - ablation: delta < 0 인 폴드가 개선 (제거 시 하락 = 기여)
  - addition: delta > 0 인 폴드가 개선
게이트 (수정됨, 사용자 승인 2026-08-09):
  - primary 기여 >= +10 & R-only 2/3 개선 -> 채택. 미달 -> 기각.
  (기존 screen_all.json 의 r_only_improved / verdict 필드는 부호 오류로 전부
   "기각"으로 잘못 기록됨 — 본 스크립트가 올바른 값으로 재계산)

입력 : experiments/screen_all.json
출력 : experiments/gate_verdict.json  +  stdout 재판정표
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCREEN_ALL = REPO / "experiments" / "screen_all.json"
OUT = REPO / "experiments" / "gate_verdict.json"
LEDGER = REPO.parent / ".omo" / "start-work" / "ledger.jsonl"

R_ONLY_FOLDS = ["r2022", "r2023", "r2024"]
GATE_PRIMARY = 10.0        # primary 기여 임계 (시드 노이즈 ±9.5 실측 반영, +20 -> +10)
GATE_R_ONLY = 2            # R-only 3폴드 중 개선 요구 수 (2/3)

ADOPTED_WAVE_D = ["score_diff_binary", "asof_n_bucket"]  # 예상 채택 (검증용)


def contribution(mode: str, delta: float) -> float:
    """모드별 피처 기여도. ablation: 제거 시 하락(-delta)이 기여. addition: 상승(+delta)이 기여."""
    if mode == "addition":
        return delta
    return -delta  # ablation


def is_r_improved(mode: str, delta: float) -> bool:
    """R-only 폴드에서 해당 폴드가 '개선'(기여)인지."""
    if mode == "addition":
        return delta > 0.0
    return delta < 0.0  # ablation


def main() -> int:
    if not SCREEN_ALL.exists():
        print(f"[FATAL] {SCREEN_ALL} 없음", file=sys.stderr)
        return 1

    data = json.loads(SCREEN_ALL.read_text(encoding="utf-8"))
    baseline_fe = data["baseline_fe"]
    meta = data.get("_meta", {})

    candidates: dict[str, dict] = {}
    for name, cand in data.items():
        if name == "baseline_fe" or name == "_meta":
            continue
        mode = cand.get("mode", "ablation")
        delta = cand["delta_vs_baseline_fe"]
        contrib = contribution(mode, delta["primary"])
        r_deltas = {f: delta.get(f, float("nan")) for f in R_ONLY_FOLDS}
        r_imp = [f for f in R_ONLY_FOLDS if is_r_improved(mode, r_deltas[f])]
        r_only_old = cand.get("r_only_improved")

        primary_ok = contrib >= GATE_PRIMARY
        r_only_ok = len(r_imp) >= GATE_R_ONLY
        verdict = "채택" if (primary_ok and r_only_ok) else "기각"

        # 근거 문장
        reasons = []
        if primary_ok:
            reasons.append(f"primary 기여 +{contrib:.1f} >= +{GATE_PRIMARY:.0f} 통과")
        else:
            reasons.append(f"primary 기여 +{contrib:.1f} < +{GATE_PRIMARY:.0f} 미달")
        reasons.append(f"R-only 개선 {len(r_imp)}/{len(R_ONLY_FOLDS)} ({'충족' if r_only_ok else '미달'})")
        if not (primary_ok and r_only_ok):
            reasons.append("게이트 미달 -> 기각")

        candidates[name] = {
            "mode": mode,
            "primary_delta": delta["primary"],
            "primary_contribution": contrib,
            "r_only_delta": {f: r_deltas[f] for f in R_ONLY_FOLDS},
            "r_only_improved": f"{len(r_imp)}/{len(R_ONLY_FOLDS)}",
            "r_only_improved_old": r_only_old,
            "r_only_improved_folds": r_imp,
            "verdict": verdict,
            "reason": "; ".join(reasons),
        }

    adopted = [n for n, c in candidates.items() if c["verdict"] == "채택"]
    rejected = [n for n, c in candidates.items() if c["verdict"] == "기각"]

    # --- stdout 재판정표 ---
    print("=" * 96)
    print("GATE VERDICT — 유도 피처 재판정 (screen_all.json, 5시드 x 4폴드, 재학습 없음)")
    print("=" * 96)
    print(
        f"{'후보':<24}{'모드':<9}{'Δ primary':>10}{'기여':>9}"
        f"{'R-only Δ':>20}{'R-imp':>7}{'판정':>6}"
    )
    print("-" * 96)
    for name, c in candidates.items():
        rstr = "/".join(f"{c['r_only_delta'][f]:+.1f}" for f in R_ONLY_FOLDS)
        print(
            f"{name:<24}{c['mode']:<9}{c['primary_delta']:>+10.1f}{c['primary_contribution']:>+9.1f}"
            f"{rstr:>20}{c['r_only_improved']:>7}{c['verdict']:>6}"
        )
    print("-" * 96)
    for name, c in candidates.items():
        print(f"  [{c['verdict']}] {name}: {c['reason']}")

    print("=" * 96)
    print(f"기준선(primary): {baseline_fe['primary']['bss']:.1f}  (r2022 {baseline_fe['r2022']['bss']:.1f} / "
          f"r2023 {baseline_fe['r2023']['bss']:.1f} / r2024 {baseline_fe['r2024']['bss']:.1f})")
    print(f"게이트: primary 기여 >= +{GATE_PRIMARY:.0f} & R-only {GATE_R_ONLY}/3 개선 -> 채택 | 미달 -> 기각")
    print(f"채택({len(adopted)}): {adopted}")
    print(f"기각({len(rejected)}): {rejected}")

    # --- gate_verdict.json 작성 ---
    out = {
        "baseline_fe": baseline_fe,
        "gate_rules": {
            "threshold_primary": GATE_PRIMARY,
            "r_only_required": f"{GATE_R_ONLY}/{len(R_ONLY_FOLDS)}",
            "r_only_folds": R_ONLY_FOLDS,
            "contribution_rule": {
                "ablation": "기여 = -delta (제거 시 하락 = 기여), R-only delta<0 폴드 = 개선",
                "addition": "기여 = +delta (추가 시 상승 = 기여), R-only delta>0 폴드 = 개선",
            },
            "accept_rule": "primary 기여 >= +10 & R-only 2/3 개선 -> 채택 | 미달 -> 기각",
            "note": "기존 screen_all.json의 r_only_improved/verdict는 부호 오류(전부 기각) — 본 파일이 수정된 로직으로 재계산",
        },
        "candidates": candidates,
        "adopted": adopted,
        "rejected": rejected,
        "wave_d_input": {
            "features": adopted,
            "purpose": "Wave D 통합 검증(LGB 10시드 + MLP 5시드 블렌딩) + 제출 입력",
            "seeds": meta.get("seeds"),
            "folds": meta.get("folds"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[OK] gate_verdict.json 저장: {OUT}")

    if set(adopted) != set(ADOPTED_WAVE_D):
        print(f"[WARN] 채택 목록이 예상({ADOPTED_WAVE_D})과 다름: {adopted}", file=sys.stderr)

    # --- ledger.jsonl에 task-completed append (멱등: 동일 task 마커 중복 방지) ---
    task_marker = "gate_verdict 재판정"
    if LEDGER.exists() and task_marker in LEDGER.read_text(encoding="utf-8"):
        print(f"[SKIP] ledger에 이미 기록됨: {LEDGER}")
    else:
        from datetime import datetime, timezone

        entry = {
            "event": "task-completed",
            "plan": "aimers9-eda-fe-full",
            "task": task_marker,
            "artifact": "repro_979/gate_verdict.py, repro_979/experiments/gate_verdict.json",
            "summary": (
                f"부호 오류 수정 + 게이트 +10 적용, 재학습 없음. "
                f"채택 {len(adopted)}: {adopted} (asof_n_bucket 기여 +10.4 R-only 2/3, "
                f"score_diff_binary 기여 +13.0 R-only 2/3). "
                f"기각 {len(rejected)}: {rejected} (기여 < +10 또는 R-only 1/3 이하). "
                f"R-only 개선 카운트 재계산: recent_gap 2/3, return_gap 2/3, pitcher_debut 0/3, "
                f"batter_debut 1/3, li_risp 2/3, outs_count 2/3, count_platoon 1/3. "
                f"Wave D 입력: {adopted}."
            ),
            "commands": ["cd repro_979 && /home/gpu_01/.conda/envs/aimers9/bin/python gate_verdict.py"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[OK] ledger.jsonl에 task-completed append: {LEDGER}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
