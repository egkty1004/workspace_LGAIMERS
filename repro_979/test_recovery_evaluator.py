#!/usr/bin/env python3
"""test_recovery_evaluator.py — Todo 2 (aimers9-top100-score-recovery) tests (pytest-free).

Tests:
  (a) happy --smoke -> exit 0; writes task-2-evaluator.{json,md} with an immutable
      manifest (candidate id, code/config/data hashes, origins, seeds, formula,
      selection keys, bootstrap definition, terminal-firewall state); evidence is
      scope-clean (no forbidden path tokens / upload markers).
  (b) each of the 3 fixtures (primary-read-before-freeze, r2024-sort-key,
      manifest-mutation) exits 2, writes task-2-evaluator-fixture-<name>.{json,md}
      with fixture+matched fields, never clobbers the main task-2-evaluator evidence.
  (c) --validate-manifest --candidate baseline -> exit 0; manifest validates.
  (d) --screen reads ONLY r2022/r2023 labels (structural firewall); primary read
      before freeze raises TerminalFirewallError.
  (e) --freeze flips the firewall to FROZEN only after the screen gate passes;
      --terminal-check requires FROZEN (exit 2 when UNFROZEN).
  (f) --audit-compliance / --audit-quality / --audit-scope exit 0 on a clean
      evidence dir; --audit-scope rejects a forbidden-path-token evidence file.
  (g) naming/provenance compliance of all evidence the CLI writes.

Usage: python3 repro_979/test_recovery_evaluator.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
CLI = REPO / "recovery_evaluator.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_evaluator as re  # noqa: E402

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


# ── (a) happy --smoke ─────────────────────────────────────────────────
def test_smoke(tmp: Path) -> None:
    print("[test] (a) happy --smoke -> exit 0, immutable manifest evidence")
    proc = run_cli("--smoke", "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-2-evaluator.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.verdict", rec.get("verdict") == "PASS", str(rec.get("verdict")))
    check("a.md_written", (tmp / "task-2-evaluator.md").is_file())
    man = rec.get("manifest") or {}
    check("a.manifest_candidate", bool(man.get("candidate_id")))
    check("a.manifest_code_hashes", isinstance(man.get("code_hashes"), dict)
          and len(man["code_hashes"]) >= 3)
    check("a.manifest_config_hash", bool(man.get("config_hash")))
    check("a.manifest_data_hash", bool(man.get("data_hash")))
    check("a.manifest_origins",
          man.get("origins", {}).get("selection") == list(rp.SELECTION_ORIGINS)
          and man.get("origins", {}).get("terminal") == rp.TERMINAL_ORIGIN)
    check("a.manifest_formula", "common.score" in str(man.get("formula", {}).get("scoring", "")))
    check("a.manifest_selection_keys",
          man.get("selection_keys") == list(rp.SELECTION_KEYS))
    check("a.manifest_bootstrap", isinstance(man.get("bootstrap"), dict)
          and man["bootstrap"].get("n_resamples") == rp.BOOTSTRAP_N_RESAMPLES)
    check("a.manifest_firewall",
          man.get("terminal_firewall", {}).get("state") == rp.FIREWALL_UNFROZEN)
    check("a.manifest_hash_valid", not rp.validate_manifest(man))
    check("a.labels_unread", rec.get("labels_read") is False and rec.get("label_sources") == [])
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")


# ── (b) 3 fixtures ────────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 3 fixtures exit 2, fixture evidence, main not clobbered")
    main_sha = None
    main_ev = tmp / "task-2-evaluator.json"
    if main_ev.is_file():
        main_sha = main_ev.read_bytes()
    for name in sorted(re.FIXTURES):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-2-evaluator-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-2-evaluator.json byte-identical)")


# ── (c) --validate-manifest ───────────────────────────────────────────
def test_validate_manifest(tmp: Path) -> None:
    print("[test] (c) --validate-manifest baseline -> exit 0")
    proc = run_cli("--validate-manifest", "--candidate", "baseline",
                   "--evidence-dir", str(tmp))
    check("c.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-2-evaluator-validate-manifest.json"
    check("c.evidence", ev.is_file(), str(ev))
    if ev.is_file():
        rec = load(ev)
        check("c.manifest_valid", rec.get("verdict") == "PASS"
              and not rec.get("violations"), f"{rec.get('violations')}")


# ── (d) structural firewall: screen reads only selection labels ───────
def test_firewall(tmp: Path) -> None:
    print("[test] (d) screen reads only r2022/r2023; primary read before freeze blocked")
    train = re._synthetic_train()
    masks = rp.build_origin_masks(train)
    labels = rp.read_selection_labels(train, masks)
    check("d.selection_only", set(labels) == set(rp.SELECTION_ORIGINS),
          f"{sorted(labels)}")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)
    try:
        rp.read_primary_labels(train, masks)
        check("d.primary_blocked", False, "primary read before freeze did NOT raise")
    except rp.TerminalFirewallError:
        check("d.primary_blocked", True, "primary read before freeze raised TerminalFirewallError")
    # --screen CLI path never reads primary
    proc = run_cli("--screen", "--candidate", "t", "--evidence-dir", str(tmp))
    check("d.screen_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-2-evaluator-screen.json"
    if ev.is_file():
        rec = load(ev)
        check("d.screen_labels", rec.get("label_sources") == list(rp.SELECTION_ORIGINS)
              and rec.get("labels_read") is True, f"{rec.get('label_sources')}")


# ── (e) freeze flips firewall; terminal-check requires FROZEN ─────────
def test_freeze_terminal(tmp: Path) -> None:
    print("[test] (e) freeze flips firewall; terminal-check requires FROZEN")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)
    # terminal-check before freeze -> exit 2
    proc = run_cli("--terminal-check", "--candidate", "t", "--evidence-dir", str(tmp))
    check("e.terminal_before_freeze_exit2", proc.returncode == 2, f"(rc={proc.returncode})")
    # freeze (synthetic gate is REJECT -> NO_PROMOTION, firewall stays UNFROZEN)
    proc = run_cli("--freeze", "--candidate", "t", "--evidence-dir", str(tmp))
    check("e.freeze_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    # firewall mechanics: FROZEN allows primary read; --terminal-check exits 0 when FROZEN
    train = re._synthetic_train()
    masks = rp.build_origin_masks(train)
    rp.set_firewall(rp.FIREWALL_FROZEN)
    check("e.firewall_frozen", rp.firewall_state() == rp.FIREWALL_FROZEN)
    y_primary = rp.read_primary_labels(train, masks)
    check("e.primary_read_when_frozen", y_primary is not None and len(y_primary) > 0)
    # in-process terminal-check (firewall state is module-level, not cross-process)
    args = argparse.Namespace(evidence_dir=str(tmp), candidate="t")
    check("e.terminal_after_freeze_exit0",
          re.cmd_terminal_check(args) == 0, "in-process terminal-check exit 0 when FROZEN")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)  # reset for other tests


# ── (f) audits ────────────────────────────────────────────────────────
def test_audits(root: Path) -> None:
    print("[test] (f) audits exit 0 on clean dir; scope rejects forbidden token")
    d = root / "audit_ev"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    # seed a clean smoke evidence
    run_cli("--smoke", "--evidence-dir", str(d))
    check("f.compliance.happy",
          run_cli("--audit-compliance", "--evidence-dir", str(d)).returncode == 0)
    check("f.quality.happy",
          run_cli("--audit-quality", "--evidence-dir", str(d)).returncode == 0)
    check("f.scope.happy",
          run_cli("--audit-scope", "--evidence-dir", str(d)).returncode == 0)

    # forbidden token evidence -> scope audit exit 2
    d2 = root / "audit_bad"
    shutil.rmtree(d2, ignore_errors=True)
    d2.mkdir(parents=True, exist_ok=True)
    (d2 / "task-2-evaluator.json").write_text(
        json.dumps({"데이터/secret": True, "verdict": "PASS", "exit_code": 0}),
        encoding="utf-8")
    check("f.scope.forbidden_reject",
          run_cli("--audit-scope", "--evidence-dir", str(d2)).returncode == 2)

    check("f.missing_dir",
          run_cli("--audit-compliance", "--evidence-dir",
                  str(root / "no_such_dir")).returncode == 1)


# ── (g) evidence compliance ───────────────────────────────────────────
def test_evidence_compliance(tmp: Path) -> None:
    print("[test] (g) evidence naming / provenance compliance")
    for path in sorted(tmp.glob("*.json")):
        check(f"g.name_{path.name}", bool(rp.EVIDENCE_NAME_RE.match(path.name)), path.name)
        rec = load(path)
        if rec.get("fixture") is None:
            check(f"g.recorded_at_{path.name}",
                  bool(rec.get("recorded_at_utc")) and bool(rec.get("git_head")))
            check(f"g.config_hash_{path.name}", bool(rec.get("config_hash")))
            check(f"g.labels_{path.name}",
                  isinstance(rec.get("label_sources"), list)
                  and rec.get("labels_read") is False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="t2ev_test_") as td:
        tmp = Path(td)
        test_smoke(tmp / "a")
        test_fixtures(tmp / "a")
        test_validate_manifest(tmp / "c")
        test_firewall(tmp / "d")
        test_freeze_terminal(tmp / "e")
        test_audits(tmp)
        test_evidence_compliance(tmp / "a")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
