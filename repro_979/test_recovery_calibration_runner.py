#!/usr/bin/env python3
"""test_recovery_calibration_runner.py — Todo 6 (aimers9-top100-score-recovery) tests
(pytest-free).

Tests:
  (a) happy --screen -> exit 0; writes task-6-calibration.{json,md} with
      label_sources=[r2022,r2023], labels_read=True, time-causal fit diagnostics
      (r2022 cold-start identity, r2023 fits on 2022 OOF), frozen deployed formula
      preserved, and scope-clean evidence.
  (b) each of the 4 fixtures (free-intercept, outer-fit, terminal-label-read,
      nonmonotonic-isotonic-output) exits 2, writes
      task-6-calibration-fixture-<name>.{json,md}, never clobbers the main evidence.
  (c) transform correctness: identity is identity; beta/isotonic fit on prior OOF and
      apply once; z_candidate = logit(p_transform) - C_LOGIT preserves the deployed
      formula (clip(sigmoid(z_candidate + C_LOGIT), .30, .70) == clip(p_transform)).
  (d) structural guards: free-intercept / outer-fit / terminal-label-read /
      nonmonotonic-isotonic-output each raise the expected exception.
  (e) causal panel: r2022 (outer year 2022) has no prior OOF year -> cold-start;
      r2023 (outer year 2023) fits on 2022 OOF only.
  (f) evidence naming / provenance compliance.

Usage: python3 repro_979/test_recovery_calibration_runner.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
CLI = REPO / "recovery_calibration_runner.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_calibration_runner as rc  # noqa: E402

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
    print("[test] (a) happy --screen -> exit 0, time-causal evidence")
    proc = run_cli("--screen", "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-6-calibration.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.md_written", (tmp / "task-6-calibration.md").is_file())
    check("a.verdict_in", rec.get("verdict") in ("PASS", "REJECT"),
          str(rec.get("verdict")))
    check("a.labels", rec.get("label_sources") == list(rp.SELECTION_ORIGINS)
          and rec.get("labels_read") is True, f"{rec.get('label_sources')}")
    check("a.candidates", set(rec.get("candidates") or [])
          == {"calibration_beta", "calibration_isotonic", "calibration_identity"})
    check("a.selectable", set(rec.get("selectable_candidates") or [])
          == {"calibration_beta", "calibration_isotonic"})
    check("a.control", set(rec.get("control_candidates") or [])
          == {"calibration_identity"})
    # time-causal fit diagnostics
    per = rec.get("per_candidate") or {}
    beta = per.get("calibration_beta") or {}
    fit = beta.get("fit") or {}
    check("a.r2022_cold_start", fit.get("r2022", {}).get("cold_start") is True,
          f"{fit.get('r2022')}")
    check("a.r2023_fit_2022", fit.get("r2023", {}).get("fit_years") == [2022],
          f"{fit.get('r2023')}")
    # frozen formula preserved
    frozen = rec.get("frozen") or {}
    check("a.c_logit", frozen.get("c_logit") == rp.C_LOGIT)
    check("a.clip", frozen.get("clip") == [rp.CLIP_LO, rp.CLIP_HI])
    check("a.z_candidate", "logit(p_transform) - C_LOGIT" in str(frozen.get("z_candidate")))
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")


# ── (b) 4 fixtures ───────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 4 fixtures exit 2, fixture evidence, main not clobbered")
    main_ev = tmp / "task-6-calibration.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in sorted(rc.FIXTURES):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-6-calibration-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-6-calibration.json byte-identical)")


# ── (c) transform correctness ────────────────────────────────────────
def test_transforms(tmp: Path) -> None:
    print("[test] (c) identity/beta/isotonic transforms + z_candidate formula")
    rng = __import__("numpy").random.default_rng(0)
    import numpy as np  # noqa: PLC0415
    z_fit = rng.normal(0, 1, 2000)
    y_fit = (rng.random(2000) < 0.45).astype(np.float64)
    z_apply = rng.normal(0, 1, 500)

    # identity: p_transform == p0
    p0 = rc._p0(z_apply)
    p_id = rc.apply_transform("calibration_identity", {}, [], z_apply)
    check("c.identity", np.allclose(p_id, p0), "identity == p0")

    # beta fit/apply
    model_b = rc.fit_beta(z_fit, y_fit)
    p_beta = rc.apply_beta(model_b, z_apply)
    check("c.beta_finite", bool(np.isfinite(p_beta).all())
          and bool((p_beta > 0).all()) and bool((p_beta < 1).all()))
    check("c.beta_has_intercept", model_b.intercept_ is not None)

    # isotonic fit/apply (monotonic)
    model_i = rc.fit_isotonic(z_fit, y_fit)
    p_iso = rc.apply_isotonic(model_i, z_apply)
    order = np.argsort(rc._p0(z_apply), kind="stable")
    check("c.isotonic_monotonic",
          bool(np.all(np.diff(p_iso[order]) >= -1e-12)),
          "isotonic output non-decreasing in sorted p0")

    # z_candidate preserves deployed formula: clip(sigmoid(z_cand+C_LOGIT)) == clip(p_transform)
    for p in (p_id, p_beta, p_iso):
        z_cand = rc.z_candidate_from_p(p)
        p_deployed = rp.deployed_probs(z_cand)
        check("c.formula_preserved",
              bool(np.allclose(p_deployed, np.clip(p, rp.CLIP_LO, rp.CLIP_HI))),
              "clip(sigmoid(z_candidate+C_LOGIT)) == clip(p_transform)")


# ── (d) structural guards ────────────────────────────────────────────
def test_guards(tmp: Path) -> None:
    print("[test] (d) structural guards raise expected exceptions")
    import numpy as np  # noqa: PLC0415
    try:
        rc._assert_no_standalone_offset([2022, 2023], outer_year=2023)
        check("d.free_intercept", False, "did not raise")
    except rc.PolicyViolation:
        check("d.free_intercept", True, "free-intercept rejected")
    try:
        rc._assert_no_outer_fit([2022, 2023], outer_year=2023)
        check("d.outer_fit", False, "did not raise")
    except rc.LeakageError:
        check("d.outer_fit", True, "outer-fit rejected")
    try:
        rc._assert_no_terminal_read({"r2022": np.zeros(1), "primary": np.zeros(1)})
        check("d.terminal_read", False, "did not raise")
    except rc.LeakageError:
        check("d.terminal_read", True, "terminal-label-read rejected")
    try:
        rc._assert_monotonic_isotonic(np.array([0.3, 0.2, 0.4]),
                                      np.array([0.1, 0.2, 0.3]))
        check("d.nonmonotonic", False, "did not raise")
    except rc.PolicyViolation:
        check("d.nonmonotonic", True, "nonmonotonic-isotonic-output rejected")
    # valid monotonic passes
    rc._assert_monotonic_isotonic(np.array([0.2, 0.3, 0.4]),
                                  np.array([0.1, 0.2, 0.3]))
    check("d.monotonic_ok", True, "monotonic output accepted")


# ── (e) causal panel ─────────────────────────────────────────────────
def test_causal_panel(tmp: Path) -> None:
    print("[test] (e) causal panel: r2022 cold-start, r2023 fits on 2022 only")
    import numpy as np  # noqa: PLC0415
    # synthetic train + masks
    train = __import__("repro_979.recovery_evaluator", fromlist=["_synthetic_train"])._synthetic_train()
    masks = rp.build_origin_masks(train)
    # synthetic baseline logits (bounded length)
    baseline = {}
    for o in rp.SELECTION_ORIGINS:
        n = int(masks[o][1].sum())
        baseline[o] = np.full(min(n, rc.BOUNDED_ROWS), 0.1, dtype=np.float64)
    panel = rc.build_causal_panel(train, masks, baseline)
    check("e.panel_years", set(panel) == {2022, 2023}, f"{sorted(panel)}")
    # r2022 outer year 2022 -> no prior year
    fit_r2022 = [y for y in rc.PANEL_YEARS if y < 2022]
    check("e.r2022_no_prior", fit_r2022 == [], f"{fit_r2022}")
    # r2023 outer year 2023 -> fit on 2022 only
    fit_r2023 = [y for y in rc.PANEL_YEARS if y < 2023]
    check("e.r2023_prior_2022", fit_r2023 == [2022], f"{fit_r2023}")


# ── (f) evidence compliance ──────────────────────────────────────────
def test_evidence_compliance(tmp: Path) -> None:
    print("[test] (f) evidence naming / provenance compliance")
    for path in sorted(tmp.glob("*.json")):
        check(f"f.name_{path.name}", bool(rp.EVIDENCE_NAME_RE.match(path.name)), path.name)
        rec = load(path)
        if rec.get("fixture") is None:
            check(f"f.recorded_at_{path.name}",
                  bool(rec.get("recorded_at_utc")) and bool(rec.get("git_head")))
            check(f"f.config_hash_{path.name}", bool(rec.get("config_hash")))
            check(f"f.labels_{path.name}",
                  isinstance(rec.get("label_sources"), list)
                  and rec.get("labels_read") is True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="t6cal_test_") as td:
        tmp = Path(td)
        test_screen(tmp / "a")
        test_fixtures(tmp / "a")
        test_transforms(tmp / "c")
        test_guards(tmp / "d")
        test_causal_panel(tmp / "e")
        test_evidence_compliance(tmp / "a")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
