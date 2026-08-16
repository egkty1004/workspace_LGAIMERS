#!/usr/bin/env python3
"""test_mlp_preprocess_runner.py — pytest-free QA for Todo 6 mlp_preprocess_runner.

Covers (plan Task 6 acceptance + QA scenarios):
  (a) --fixture combined-variants → exit 2 (proves A+B never combine)
  (b) --fixture validation-fit → exit 2 (proves no validation-row fitting)
  (c) --fixture blocked-relation → exit 2 (proves a registry-listed blocked
      hypothesis, return_gap, is not revived)
  (d) config digest sensitivity: any frozen spec/gate mutation changes the hash
  (e) gate verdict boundaries: ΔBSS=15.0/mean-shift=0.005 → PASS (inclusive);
      ΔBSS=14.99 or mean-shift=0.0051 → REJECT
  (f) blocked-relation records: real variant specs pass all blocked entries;
      an injected return_gap spec is flagged not-ok
  (g) asof_*/prev_game_* scoping: 19/6 columns within the frozen 49 features
  (h) policy check passes on the frozen policy (primary-only selection)

Run: python3 repro_979/test_mlp_preprocess_runner.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _run_runner(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "mlp_preprocess_runner.py"), *argv],
                          capture_output=True, text=True, timeout=1800, cwd=str(REPO))


def test_fixtures() -> None:
    for name in ("combined-variants", "validation-fit", "blocked-relation"):
        proc = _run_runner("--fixture", name)
        check(f"fixture {name} exits 2", proc.returncode == 2,
              f"rc={proc.returncode} (expected 2)")
        if proc.returncode != 2:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])


def test_config_digest_sensitivity() -> None:
    import mlp_preprocess_runner as r
    h_base = r._config_hash("clip_z5")
    h_base_b = r._config_hash("missing_flags")
    check("variant config hashes distinct", h_base != h_base_b,
          f"clip_z5={h_base[:16]}… missing_flags={h_base_b[:16]}…")

    import types
    orig = r.GATE_DELTA_BSS_MIN
    r.GATE_DELTA_BSS_MIN = 16.0
    h_mut = r._config_hash("clip_z5")
    r.GATE_DELTA_BSS_MIN = orig
    check("gate threshold mutation changes hash", h_mut != h_base,
          f"{h_base[:16]}… vs {h_mut[:16]}…")

    orig_clip = r.CLIP_Z
    r.CLIP_Z = 4.0
    h_mut2 = r._config_hash("clip_z5")
    r.CLIP_Z = orig_clip
    check("clip bound mutation changes hash", h_mut2 != h_base,
          f"{h_base[:16]}… vs {h_mut2[:16]}…")


def test_gate_verdict_boundaries() -> None:
    import mlp_preprocess_runner as r
    ok, gates = r._gate_verdict(15.0, 0.005)
    check("ΔBSS=15.0 & shift=0.005 → PASS (inclusive)", ok, str(gates))
    ok2, _ = r._gate_verdict(14.99, 0.005)
    check("ΔBSS=14.99 → REJECT", not ok2)
    ok3, _ = r._gate_verdict(15.0, 0.0051)
    check("shift=0.0051 → REJECT", not ok3)
    ok4, _ = r._gate_verdict(15.0, 0.005)
    ok5, _ = r._gate_verdict(16.0, 0.004)
    check("ΔBSS=16.0 & shift=0.004 → PASS", ok4 and ok5)


def test_registry_relations() -> None:
    import mlp_preprocess_runner as r
    reg = r._registry()
    specs_real = r._variant_feature_specs("missing_flags")
    recs_real = r._blocked_relation_records(specs_real, reg)
    check("real missing_flags specs pass all blocked entries",
          all(x["ok"] for x in recs_real),
          f"{len(recs_real)} blocked entries checked")
    specs_injected = specs_real + [{
        "name": "return_gap", "mechanism": "isna_conjunction",
        "source_columns": ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_n"],
        "role": "binary_flag"}]
    recs_bad = r._blocked_relation_records(specs_injected, reg)
    bad = [x for x in recs_bad if not x["ok"]]
    check("injected return_gap (registry-listed blocked id) flagged",
          any(x["blocked_id"] == "return_gap" for x in bad),
          f"blocked ids flagged: {[x['blocked_id'] for x in bad]}")
    specs_clip = r._variant_feature_specs("clip_z5")
    recs_clip = r._blocked_relation_records(specs_clip, reg)
    check("clip_z5 (pure transform, no features) has no relations",
          all(x["ok"] for x in recs_clip))


def test_column_scoping() -> None:
    import mlp_preprocess_runner as r
    from repro_979.qualification_runner import CHAMPION_FEATURES
    check("ASOF_COLS = 19 within frozen 49", len(r.ASOF_COLS) == 19,
          f"n={len(r.ASOF_COLS)}")
    check("PREV_COLS = 6 within frozen 49", len(r.PREV_COLS) == 6,
          f"n={len(r.PREV_COLS)}")
    check("all scoped cols are frozen features",
          set(r.ASOF_COLS) | set(r.PREV_COLS) <= set(CHAMPION_FEATURES))
    check("asof_n_bucket excluded (blocked, non-model)", "asof_n_bucket" not in r.ASOF_COLS)


def test_policy() -> None:
    import mlp_preprocess_runner as r
    try:
        policy, violations, phash = r._check_policy()
        check("frozen policy check PASS", not violations, f"policy hash {phash[:16]}…")
    except r.PolicyViolation as exc:
        check("frozen policy check PASS", False, str(exc))


def main() -> int:
    t0 = time.time()
    print("== test_mlp_preprocess_runner ==")
    test_config_digest_sensitivity()
    test_gate_verdict_boundaries()
    test_registry_relations()
    test_column_scoping()
    test_policy()
    test_fixtures()  # real-data fixture proofs (exit 2) — slowest, last
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    print(f"\n{len(RESULTS)} checks, {n_fail} failures, {time.time() - t0:.0f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
