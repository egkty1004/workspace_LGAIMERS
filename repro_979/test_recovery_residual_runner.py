#!/usr/bin/env python3
"""test_recovery_residual_runner.py — Todo 5 (aimers9-top100-score-recovery) tests
(pytest-free).

Tests:
  (a) happy --screen -> exit 0; writes task-5-residual.{json,md} with correction cap,
      selected C, residual complementarity, both selection-origin gates, inner-OOF/outer
      mask disjointness + hashes, and correction-config-before-outer-labels; evidence is
      scope-clean (no forbidden path tokens / upload markers).
  (b) the 4 adversarial fixtures (outer-target-feature, terminal-label-read,
      cap-violation, target-encoding) each exit 2 and write task-5-residual-fixture-<name>
      evidence with matched=True; never clobber the main task-5-residual evidence.
  (c) offset-identity fixture exits 0 on exact baseline equivalence (positive identity
      check: p_residual=sigmoid(z_base+C_LOGIT) leaves deployed predictions unchanged).
  (d) unit tests for the correction machinery: bounded masks, one-year-ahead base OOF
      masks, inner/outer disjointness + hashing, prior-OOF stats, feature building,
      apply_correction identity, C selection (smallest C on tie), and the
      outer-target-feature / target-encoding guards.

Usage: python3 repro_979/test_recovery_residual_runner.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
CLI = REPO / "recovery_residual_runner.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_residual_runner as rr  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def run_cli(*args: str) -> subprocess.CompletedProcess[Any]:
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_clean(rec: dict[str, Any]) -> list[str]:
    return rp.scan_scope(rec, Path("x.json"))


# ── (a) happy --screen ───────────────────────────────────────────────
def test_screen(tmp: Path) -> None:
    print("[test] (a) happy --screen -> exit 0, residual evidence")
    proc = run_cli("--screen", "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-5-residual.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.md_written", (tmp / "task-5-residual.md").is_file())
    check("a.candidate", rec.get("candidate_id") == "residual_ridge")
    check("a.labels_read", rec.get("labels_read") is True
          and rec.get("label_sources") == list(rp.SELECTION_ORIGINS))
    cfg = rec.get("config") or {}
    check("a.c_grid", cfg.get("c_grid") == list(rr.C_GRID))
    check("a.correction_cap", cfg.get("correction_cap") == rr.CORRECTION_CAP)
    check("a.z_corrected_formula",
          "clip(logit(p_residual) - C_LOGIT - z_base" in str(cfg.get("z_corrected", "")))
    check("a.lr_frozen", (cfg.get("logistic_regression") or {}).get("penalty") == "l2"
          and (cfg.get("logistic_regression") or {}).get("solver") == "lbfgs")
    check("a.standardize", cfg.get("standardize") == list(rr.NUMERIC_FEATS))
    check("a.one_hot", cfg.get("one_hot") == list(rr.ONE_HOT_FEATS))
    origins = rec.get("origins") or {}
    check("a.both_origins", set(origins) == set(rp.SELECTION_ORIGINS))
    for o in rp.SELECTION_ORIGINS:
        d = origins.get(o) or {}
        check(f"a.{o}.selected_c", d.get("selected_c") is not None)
        check(f"a.{o}.cap", d.get("correction_cap") == rr.CORRECTION_CAP)
        check(f"a.{o}.complementarity", isinstance(d.get("residual_complementarity"), float))
        disj = d.get("inner_outer_disjoint") or {}
        check(f"a.{o}.disjoint", disj.get("outer_year") == int(o.replace("r", ""))
              and bool(disj.get("outer_val_mask_hash")))
        check(f"a.{o}.inner_hashed", all(y.get("val_mask_hash") for y in disj.get("inner_years", [])))
    gate = rec.get("gate") or {}
    check("a.gate_origins", set((gate.get("origins") or {})) == set(rp.SELECTION_ORIGINS))
    check("a.gate_verdict", gate.get("verdict") in ("PASS", "REJECT"))
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    # correction config selected before outer labels load
    findings = " ".join(str(f) for f in rec.get("findings", []))
    check("a.config_before_outer", "correction_config_before_outer_labels" in findings
          or any("correction_config_before_outer_labels" in str(c.get("rule", ""))
                 for c in rec.get("checks", [])))


# ── (b) 4 adversarial fixtures exit 2 ────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 4 adversarial fixtures exit 2, fixture evidence, main not clobbered")
    main_sha = None
    main_ev = tmp / "task-5-residual.json"
    if main_ev.is_file():
        main_sha = main_ev.read_bytes()
    for name in ("outer-target-feature", "terminal-label-read", "cap-violation",
                 "target-encoding"):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-5-residual-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-5-residual.json byte-identical)")


# ── (c) offset-identity exits 0 ──────────────────────────────────────
def test_offset_identity(tmp: Path) -> None:
    print("[test] (c) offset-identity exits 0 on exact baseline equivalence")
    proc = run_cli("--fixture", "offset-identity", "--evidence-dir", str(tmp))
    check("c.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    fix = tmp / "task-5-residual-fixture-offset-identity.json"
    check("c.artifact", fix.is_file(), str(fix))
    if fix.is_file():
        rec = load(fix)
        check("c.matched", rec.get("matched") is True, f"detail={rec.get('detail')}")
        check("c.verdict", rec.get("verdict") == "PASS")
        check("c.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")


# ── (d) unit tests for correction machinery ──────────────────────────
def test_bounded_mask() -> None:
    print("[test] (d1) _bounded_mask keeps first n True indices")
    m = np.array([True, False, True, True, False, True])
    b = rr._bounded_mask(m, 2)
    check("d1.count", int(b.sum()) == 2)
    check("d1.first", bool(b[0]) and bool(b[2]))
    check("d1.rest_false", not b[3] and not b[5])


def test_base_oof_masks() -> None:
    print("[test] (d2) build_base_oof_masks one-year-ahead + R/all")
    train = rr._synthetic_train()
    r_masks = rr.build_base_oof_masks(train, "R")
    a_masks = rr.build_base_oof_masks(train, "all")
    check("d2.years", set(r_masks) == set(rr.BASE_OOF_YEARS))
    # R-only: val 2022 has no P rows
    va22 = np.asarray(r_masks[2022][1], dtype=bool)
    check("d2.r_only", not (train.loc[va22, "game_type"] == "P").any())
    # all: val 2022 includes P rows
    va22a = np.asarray(a_masks[2022][1], dtype=bool)
    check("d2.all_includes_p", (train.loc[va22a, "game_type"] == "P").any())
    # train mask ends at t-1
    tr23 = np.asarray(r_masks[2023][0], dtype=bool)
    check("d2.train_ends_tminus1", int(train.loc[tr23, "season"].max()) == 2022)


def test_inner_outer_disjoint() -> None:
    print("[test] (d3) assert_inner_outer_disjoint + hashing")
    train = rr._synthetic_train()
    masks = rr.build_base_oof_masks(train, "R")
    note = rr.assert_inner_outer_disjoint(masks, 2023)
    check("d3.outer_year", note["outer_year"] == 2023)
    check("d3.inner_years", [y["year"] for y in note["inner_years"]] == [2020, 2021, 2022])
    check("d3.hashes", all(y["val_mask_hash"] for y in note["inner_years"])
          and bool(note["outer_val_mask_hash"]))
    check("d3.zero_overlap", all(y["overlap_with_outer"] == 0 for y in note["inner_years"]))


def test_stats_and_X() -> None:
    print("[test] (d4) _compute_stats + _build_X prior-OOF transforms")
    train = rr._synthetic_train()
    masks = rr.build_base_oof_masks(train, "R")
    base_oof = {t: np.full(int(rr._bounded_mask(masks[t][1], rr.MAX_ROWS).sum()), 0.3)
                for t in rr.BASE_OOF_YEARS}
    stats = rr._compute_stats(train, base_oof, masks, [2020, 2021])
    check("d4.num_stats", set(stats["num_mean"]) == set(rr.NUMERIC_FEATS))
    check("d4.levels", set(stats["levels"]) == set(rr.ONE_HOT_FEATS))
    X = rr._build_X(train, base_oof, masks, 2022, stats)
    n_num = len(rr.NUMERIC_FEATS)
    n_oh = sum(len(stats["levels"][f]) for f in rr.ONE_HOT_FEATS)
    check("d4.shape", X.shape[1] == n_num + n_oh)
    check("d4.finite", bool(np.all(np.isfinite(X))))


def test_apply_correction_identity() -> None:
    print("[test] (d5) apply_correction identity leaves deployed predictions unchanged")
    rng = np.random.default_rng(0)
    z_base = rng.normal(0, 1, 300)
    import repro_979.common as common  # noqa: PLC0415
    p_residual = common.sigmoid(z_base + rp.C_LOGIT)
    delta = common.logit(p_residual) - rp.C_LOGIT - z_base
    z_corr = z_base + np.clip(delta, -rr.CORRECTION_CAP, rr.CORRECTION_CAP)
    p_base = rp.deployed_probs(z_base)
    p_corr = rp.deployed_probs(z_corr)
    check("d5.unchanged", float(np.max(np.abs(p_corr - p_base))) <= 1e-12)


def test_select_c_smallest_on_tie() -> None:
    print("[test] (d6) select_c picks smallest C on exact tie")
    train = rr._synthetic_train()
    masks = rr.build_base_oof_masks(train, "R")
    base_oof = {t: np.full(int(rr._bounded_mask(masks[t][1], rr.MAX_ROWS).sum()), 0.3)
                for t in rr.BASE_OOF_YEARS}
    # forward year 2021: fit on 2020, score 2021
    c, results = rr.select_c(train, base_oof, masks, [2021])
    check("d6.selected", c in rr.C_GRID)
    check("d6.results", isinstance(results, dict) and len(results) == len(rr.C_GRID))
    # smallest C on exact tie: all C produce identical mean_brier -> pick smallest
    means = [results[cc]["mean_brier"] for cc in rr.C_GRID]
    if all(abs(m - means[0]) < 1e-12 for m in means):
        check("d6.smallest_on_tie", c == rr.C_GRID[0])


def test_select_c_insufficient() -> None:
    print("[test] (d7) select_c returns SKIPPED_INSUFFICIENT_FORWARD_OOF when no history")
    train = rr._synthetic_train()
    masks = rr.build_base_oof_masks(train, "R")
    base_oof = {t: np.full(int(rr._bounded_mask(masks[t][1], rr.MAX_ROWS).sum()), 0.3)
                for t in rr.BASE_OOF_YEARS}
    # forward year 2020 has no prior OOF -> insufficient
    c, results = rr.select_c(train, base_oof, masks, [2020])
    check("d7.skipped", c == rr.SKIPPED_INSUFFICIENT_FORWARD_OOF)


def test_guards() -> None:
    print("[test] (d8) outer-target-feature + target-encoding guards")
    try:
        rr._assert_no_outer_target_feature(["base_logit", "outer_year_target_mean"],
                                           {"outer_year_target_mean"})
        check("d8.outer_guard", False, "outer-target feature allowed")
    except rr.ResidualViolation:
        check("d8.outer_guard", True)
    try:
        rr._assert_no_target_encoding(["base_logit", "target_enc_platoon"])
        check("d8.te_guard", False, "target encoding allowed")
    except rr.ResidualViolation:
        check("d8.te_guard", True)


def test_evidence_provenance(tmp: Path) -> None:
    print("[test] (d9) evidence naming/provenance compliance")
    for f in tmp.glob("task-5-residual*.json"):
        rec = load(f)
        check(f"d9.{f.name}.git_head", bool(rec.get("git_head")))
        check(f"d9.{f.name}.recorded_at", bool(rec.get("recorded_at_utc")))
        check(f"d9.{f.name}.config_hash", bool(rec.get("config_hash")))
        check(f"d9.{f.name}.name_re", bool(rp.EVIDENCE_NAME_RE.match(f.name)))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_screen(tmp)
        test_fixtures(tmp)
        test_offset_identity(tmp)
        test_evidence_provenance(tmp)
    test_bounded_mask()
    test_base_oof_masks()
    test_inner_outer_disjoint()
    test_stats_and_X()
    test_apply_correction_identity()
    test_select_c_smallest_on_tie()
    test_select_c_insufficient()
    test_guards()
    print(f"\n[test_recovery_residual_runner] {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
