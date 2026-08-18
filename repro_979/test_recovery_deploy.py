#!/usr/bin/env python3
"""test_recovery_deploy.py — Todo 9 (aimers9-top100-score-recovery) tests (pytest-free).

Tests:
  (a) happy --replay-frozen with the real Task 8 terminal receipt -> exit 0; writes
      task-9-deploy.{json,md} with verdict SKIPPED_NO_PROMOTION; no package / manifest /
      model artifact created; leaderboard_state.json untouched; evidence scope-clean.
  (b) each of the 3 fixtures (manifest-hash-mismatch, 2025-train-row,
      no-promotion-package) exits 2, writes task-9-deploy-fixture-<name>.{json,md} with
      fixture+matched fields, never clobbers the main task-9-deploy evidence.
  (c) absent terminal receipt -> exit 2.
  (d) invalid terminal receipt (unknown terminal_verdict) -> exit 2.
  (e) STATISTICAL_PASS receipt without a frozen candidate -> exit 2 (REJECT).
  (f) naming/provenance compliance of all evidence the CLI writes.

Usage: python3 repro_979/test_recovery_deploy.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
CLI = REPO / "recovery_deploy.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_deploy as rd  # noqa: E402

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


def copy_terminal_receipt(tmp: Path) -> None:
    """실제 Task 8 터미널 영수증을 임시 디렉토리로 복사 (--replay-frozen 입력용)."""
    tmp.mkdir(parents=True, exist_ok=True)
    src = rp.DEFAULT_EVIDENCE_DIR / rd.TERMINAL_RECEIPT_NAME
    shutil.copy(src, tmp / rd.TERMINAL_RECEIPT_NAME)


def _mk_receipt(terminal_verdict: str, *, frozen_id: Any = None,
                freeze_verdict: str = "NO_PROMOTION") -> dict[str, Any]:
    """합성 터미널 영수증 — 배포 분기 단위 테스트용."""
    return {
        "schema_version": 1,
        "title": "Todo 8 — spend the 2024 terminal check exactly once",
        "task": "aimers9-top100-recovery/task-8-terminal",
        "mode": "terminal-check",
        "verdict": terminal_verdict,
        "terminal_verdict": terminal_verdict,
        "freeze_verdict": freeze_verdict,
        "exit_code": 0,
        "recorded_at_utc": "2026-08-18T03:53:24+00:00",
        "git_head": "48bad325d9423b26b3f6a87756328b6134f9f0cd",
        "config_hash": "158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420",
        "pre_read_freeze_hash": "7fc1b6969a2788526932ed600d0ad4ee1a6ff14eae7ea405dc95a4436cb0e12a",
        "label_sources": [],
        "labels_read": False,
        "freeze_decision": {
            "terminal_verdict": freeze_verdict,
            "frozen_candidate_id": frozen_id,
            "mean_delta_bss": None,
            "selection_key": "mean_selection_delta_bss",
            "tie_rule": "smallest canonical candidate ID on exact tie",
        },
    }


# ── (a) happy --replay-frozen ─────────────────────────────────────────
def test_replay_frozen(tmp: Path) -> None:
    print("[test] (a) happy --replay-frozen -> exit 0, SKIPPED_NO_PROMOTION, no package")
    copy_terminal_receipt(tmp)
    proc = run_cli("--replay-frozen", "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-9-deploy.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.verdict", rec.get("verdict") == "SKIPPED_NO_PROMOTION", str(rec.get("verdict")))
    check("a.md_written", (tmp / "task-9-deploy.md").is_file())
    check("a.terminal_verdict", rec.get("terminal_verdict") == "SKIPPED")
    check("a.freeze_verdict", rec.get("freeze_verdict") == "NO_PROMOTION")
    check("a.frozen_candidate_id", rec.get("frozen_candidate_id") is None)
    check("a.pre_read_terminal_hash", bool(rec.get("pre_read_terminal_hash")))
    check("a.no_package", rec.get("package", {}).get("created") is False)
    check("a.no_state_mutation", rec.get("leaderboard_state", {}).get("mutated") is False)
    check("a.labels_unread", rec.get("labels_read") is False and rec.get("label_sources") == [])
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    # no package artifact anywhere under tmp (a package would contain script.py)
    scripts = list(tmp.rglob("script.py"))
    check("a.no_package_script", not scripts, f"{scripts}")


# ── (b) 3 fixtures ────────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 3 fixtures exit 2, fixture evidence, main not clobbered")
    copy_terminal_receipt(tmp)
    main_ev = tmp / "task-9-deploy.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in sorted(rd.FIXTURES):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-9-deploy-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-9-deploy.json byte-identical)")


# ── (c) absent terminal receipt ───────────────────────────────────────
def test_absent_receipt(tmp: Path) -> None:
    print("[test] (c) absent terminal receipt -> exit 2")
    proc = run_cli("--replay-frozen", "--evidence-dir", str(tmp))
    check("c.absent_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


# ── (d) invalid terminal receipt ──────────────────────────────────────
def test_invalid_receipt(tmp: Path) -> None:
    print("[test] (d) invalid terminal receipt -> exit 2")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / rd.TERMINAL_RECEIPT_NAME).write_text(
        json.dumps({"schema_version": 1, "terminal_verdict": "BOGUS",
                    "labels_read": False, "label_sources": []}),
        encoding="utf-8")
    proc = run_cli("--replay-frozen", "--evidence-dir", str(tmp))
    check("d.invalid_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


# ── (e) STATISTICAL_PASS without frozen candidate -> exit 2 ───────────
def test_statistical_pass_guard(tmp: Path) -> None:
    print("[test] (e) STATISTICAL_PASS receipt without frozen candidate -> exit 2")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / rd.TERMINAL_RECEIPT_NAME).write_text(
        json.dumps(_mk_receipt("STATISTICAL_PASS", frozen_id=None,
                               freeze_verdict="FROZEN"), ensure_ascii=False),
        encoding="utf-8")
    proc = run_cli("--replay-frozen", "--evidence-dir", str(tmp))
    check("e.statistical_pass_exit2", proc.returncode == 2, f"(rc={proc.returncode})")
    ev = tmp / "task-9-deploy.json"
    if ev.is_file():
        rec = load(ev)
        check("e.reject_verdict", rec.get("verdict") == "REJECT", str(rec.get("verdict")))
        check("e.reject_violation", bool(rec.get("violations")), f"{rec.get('violations')}")


# ── (f) evidence compliance ───────────────────────────────────────────
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
                  and rec.get("labels_read") is False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="t9dep_test_") as td:
        tmp = Path(td)
        test_replay_frozen(tmp / "a")
        test_fixtures(tmp / "a")
        test_absent_receipt(tmp / "c")
        test_invalid_receipt(tmp / "d")
        test_statistical_pass_guard(tmp / "e")
        test_evidence_compliance(tmp / "a")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
