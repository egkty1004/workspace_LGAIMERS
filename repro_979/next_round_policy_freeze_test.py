#!/usr/bin/env python3
"""next_round_policy_freeze_test.py — Task 9 freeze 결정 테스트 (pytest-free).

Tests:
  (a) happy --freeze on the real evidence dir → NO_PROMOTION, exit 0, evidence written,
      and the evidence passes the F1 provenance/forbidden-key scans.
  (b) R-only / leaderboard insensitivity: injecting+perturbing R-only fold values and
      leaderboard/public-score values (in-memory copies) must NOT change the frozen
      result — asserted both via decide_freeze() directly and via the CLI --fixture
      guard runs (which exit 2 per plan QA, with the result unchanged).
  (c) synthetic fixture: task-3 PRIMARY_PASS → that exact candidate is frozen;
      task-8 PASS derivative supersedes the base (priority rule proof).

Usage: python3 repro_979/next_round_policy_freeze_test.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
EVIDENCE = ROOT / ".omo" / "evidence" / "aimers9-next-round"
CLI = REPO / "next_round_policy.py"

sys.path.insert(0, str(REPO))
import next_round_policy as nrp  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_happy_no_promotion() -> None:
    print("[test] (a) happy --freeze on real evidence → NO_PROMOTION exit 0")
    proc = run_cli("--freeze", "--evidence-dir", str(EVIDENCE))
    check("a.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
    ev_path = EVIDENCE / "task-9-freeze.json"
    check("a.evidence_written", ev_path.is_file())
    ev = load(ev_path)
    check("a.verdict", ev.get("verdict") == "NO_PROMOTION", f"verdict={ev.get('verdict')}")
    check("a.no_candidate", ev["decision"].get("frozen_candidate_id") is None)
    check("a.rollback_retained",
          ev.get("retained_rollback_baseline_candidate_id") == nrp.ROLLBACK_BASELINE_CANDIDATE_ID)
    check("a.provenance", bool(ev.get("git_head")) and bool(ev.get("recorded_at_utc"))
          and bool(ev.get("policy_config_hash")) and isinstance(ev.get("label_sources"), list))
    check("a.rule_trace", len(ev.get("priority_tie_rule_trace") or []) >= 3,
          f"trace_len={len(ev.get('priority_tie_rule_trace') or [])}")
    policy = load(REPO / "next_round_policy.json")
    f1 = nrp.scan_forbidden_usage(ev, set(policy["selection"]["forbidden_sort_keys"]))
    check("a.f1_forbidden_keys_clean", not f1, f"problems={f1}")
    f1p = nrp.scan_provenance_fields(ev, ev_path)
    check("a.f1_provenance_clean", not f1p, f"problems={f1p}")


def test_insensitivity() -> None:
    print("[test] (b) R-only / leaderboard insensitivity (frozen result unchanged)")
    policy = load(REPO / "next_round_policy.json")
    t3 = load(EVIDENCE / "task-3-cat-boundary-gate.json")
    t7 = load(EVIDENCE / "task-7-promotion.json")
    t8 = load(EVIDENCE / "task-8-calibration.json")
    clean = nrp.decide_freeze(policy, t3, t7, t8)
    clean_key = (clean["decision"]["terminal_verdict"], clean["decision"]["frozen_candidate_id"])

    r_only_norms = {nrp._norm_key(k) for k in policy["selection"]["forbidden_sort_keys"]}
    lb_norms = {nrp._norm_key(k) for k in nrp.FREEZE_LB_KEYS}
    for label, norms in (("altered-r-only-values", r_only_norms),
                         ("altered-leaderboard-values", lb_norms)):
        tampered = nrp._tampered_copies([r for r in (t3, t7, t8) if r is not None], norms)
        res = nrp.decide_freeze(policy, *tampered)
        tampered_key = (res["decision"]["terminal_verdict"], res["decision"]["frozen_candidate_id"])
        check(f"b.{label}.direct", tampered_key == clean_key,
              f"clean={clean_key} tampered={tampered_key}")
        proc = run_cli("--freeze", "--evidence-dir", str(EVIDENCE), "--fixture", label)
        guard_held = proc.returncode == 2 and "frozen result unchanged" in proc.stdout
        check(f"b.{label}.cli_fixture_guard", guard_held,
              f"exit={proc.returncode} (plan QA: exit 2, result unchanged)")
        check(f"b.{label}.no_fatal", "FATAL" not in proc.stdout + proc.stderr)


def test_synthetic_primary_pass() -> None:
    print("[test] (c) synthetic fixtures: PRIMARY_PASS frozen / Task-8 derivative supersedes")
    tmp = Path(tempfile.mkdtemp(prefix="nrp_freeze_fixture_"))
    try:
        base_id = "abcdef1234567890"
        t3_pass = {
            "schema_version": 1, "title": "synthetic task-3 PASS",
            "task": "aimers9-next-round/task-3-cat-boundary-gate",
            "mode": "primary-only", "verdict": "PRIMARY_PASS", "exit_code": 0,
            "recorded_at_utc": "2026-08-16T00:00:00+00:00", "git_head": "synthetic",
            "config_hash": "synthetic", "label_sources": ["primary"], "labels_read": True,
            "gate": {"frozen": {"candidate_id": base_id, "primary_bss": 800.1234}},
        }
        t7_skip = {"verdict": "SKIPPED", "git_head": "synthetic",
                   "recorded_at_utc": "2026-08-16T00:00:00+00:00",
                   "config_hash": "synthetic", "label_sources": [], "labels_read": False}
        t8_no_base = {"verdict": "SKIPPED_NO_BASE", "git_head": "synthetic",
                      "recorded_at_utc": "2026-08-16T00:00:00+00:00",
                      "config_hash": "synthetic", "label_sources": [], "labels_read": False}
        (tmp / "task-3-cat-boundary-gate.json").write_text(json.dumps(t3_pass), encoding="utf-8")
        (tmp / "task-7-promotion.json").write_text(json.dumps(t7_skip), encoding="utf-8")
        (tmp / "task-8-calibration.json").write_text(json.dumps(t8_no_base), encoding="utf-8")

        proc = run_cli("--freeze", "--evidence-dir", str(tmp))
        ev = load(tmp / "task-9-freeze.json")
        check("c.t3_pass_exit", proc.returncode == 0, f"exit={proc.returncode}")
        check("c.t3_pass_frozen",
              ev["decision"].get("frozen_candidate_id") == base_id
              and ev["decision"].get("source") == "task-3-cat-boundary-gate",
              f"frozen={ev['decision'].get('frozen_candidate_id')}")

        # Task-8 PASS calibrated derivative supersedes the Task-3 base.
        deriv_id = "deadbeefcafe0001"
        t8_pass = dict(t8_no_base)
        t8_pass["verdict"] = "PASS"
        t8_pass["calibrated_candidate_id"] = deriv_id
        t8_pass["primary_bss"] = 802.5
        (tmp / "task-8-calibration.json").write_text(json.dumps(t8_pass), encoding="utf-8")
        proc = run_cli("--freeze", "--evidence-dir", str(tmp))
        ev = load(tmp / "task-9-freeze.json")
        check("c.t8_derivative_supersedes",
              ev["decision"].get("frozen_candidate_id") == deriv_id
              and ev["decision"].get("source") == "task-8-calibration",
              f"frozen={ev['decision'].get('frozen_candidate_id')} "
              f"source={ev['decision'].get('source')}")

        # Task-8 PASS without any base (t3/t7 not passed) is a contradiction → exit 2.
        t3_reject = dict(t3_pass)
        t3_reject["verdict"] = "PRIMARY_REJECT"
        (tmp / "task-3-cat-boundary-gate.json").write_text(json.dumps(t3_reject), encoding="utf-8")
        proc = run_cli("--freeze", "--evidence-dir", str(tmp))
        check("c.derivative_without_base_reject", proc.returncode == 2,
              f"exit={proc.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_happy_no_promotion()
    test_insensitivity()
    test_synthetic_primary_pass()
    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)}/{total} PASS, {len(FAILED)} FAIL")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
