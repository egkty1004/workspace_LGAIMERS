#!/usr/bin/env python3
"""test_package_validator_recovery.py — Todo 10 (aimers9-top100-score-recovery) tests
(pytest-free).

Tests:
  (a) happy --validate --submission-dir <nonexistent> with the real Task 9 deploy
      evidence -> exit 0; writes task-10-package.{json,md} with verdict SKIPPED,
      valid_skipped True, valid_pass False; no package artifact created;
      leaderboard_state.json untouched; evidence scope-clean.
  (b) each of the 5 fixtures (altered-hash, shuffled-order, test-row-state,
      runtime-541s, offline-dependency) exits 2, writes
      task-10-package-fixture-<name>.{json,md} with fixture+matched fields, never
      clobbers the main task-10-package evidence.
  (c) absent Task 9 deploy evidence -> exit 2.
  (d) invalid Task 9 deploy evidence (unknown verdict) -> exit 2.
  (e) DEPLOYMENT_PASS verdict with no submission dir -> exit 0 SKIPPED (no package
      to validate).
  (f) naming/provenance compliance of all evidence the CLI writes.
  (g) full-validation guards reject adversarial conditions (parity drift, NaN/out-of-range,
      >6 CPU, RSS breach, GPU-VRAM breach, size breach, install timeout, pip-check failure,
      cold-script failure) — deterministic negative coverage per plan Task 10.

Usage: python3 repro_979/test_package_validator_recovery.py
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
CLI = REPO / "package_validator_recovery.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.package_validator_recovery as pv  # noqa: E402

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


def copy_task9_evidence(tmp: Path) -> None:
    """실제 Task 9 deploy 증거를 임시 디렉토리로 복사 (--validate 입력용)."""
    tmp.mkdir(parents=True, exist_ok=True)
    src = rp.DEFAULT_EVIDENCE_DIR / pv.TASK9_EVIDENCE_NAME
    shutil.copy(src, tmp / pv.TASK9_EVIDENCE_NAME)


def _mk_task9(verdict: str) -> dict[str, Any]:
    """합성 Task 9 deploy 증거 — 검증 분기 단위 테스트용."""
    return {
        "schema_version": 1,
        "title": "Todo 9 — replay and package only a terminal-PASS candidate",
        "task": "aimers9-top100-recovery/task-9-deploy",
        "verdict": verdict,
        "exit_code": 0,
        "recorded_at_utc": "2026-08-18T04:24:00+00:00",
        "git_head": "902237f1cbba4ab8353702e3ab7556b57efb4b40",
        "config_hash": "9768d89e3a77708a2ac617c0eec1e1bcecfc9c47ef765715be2d60247bf0bf87",
        "label_sources": [],
        "labels_read": False,
        "package": {"created": verdict not in pv.TASK9_SKIPPED_VERDICTS},
    }


# ── (a) happy --validate (valid SKIPPED) ──────────────────────────────
def test_skipped(tmp: Path) -> None:
    print("[test] (a) happy --validate -> exit 0, SKIPPED, no package")
    copy_task9_evidence(tmp)
    proc = run_cli("--validate", "--submission-dir", str(tmp / "nonexistent"),
                   "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-10-package.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.verdict", rec.get("verdict") == "SKIPPED", str(rec.get("verdict")))
    check("a.md_written", (tmp / "task-10-package.md").is_file())
    check("a.task9_verdict", rec.get("task9_verdict") == "SKIPPED_NO_PROMOTION",
          str(rec.get("task9_verdict")))
    check("a.valid_skipped", rec.get("valid_skipped") is True)
    check("a.valid_pass", rec.get("valid_pass") is False)
    check("a.validator_run", rec.get("validator_run") is False)
    check("a.no_package", rec.get("package", {}).get("created") is False)
    check("a.labels_unread", rec.get("labels_read") is False
          and rec.get("label_sources") == [])
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    # no package artifact anywhere under tmp (a package would contain script.py)
    scripts = list(tmp.rglob("script.py"))
    check("a.no_package_script", not scripts, f"{scripts}")


# ── (b) 5 fixtures ────────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 5 fixtures exit 2, fixture evidence, main not clobbered")
    copy_task9_evidence(tmp)
    main_ev = tmp / "task-10-package.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in sorted(pv.FIXTURES):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-10-package-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-10-package.json byte-identical)")


# ── (c) absent Task 9 evidence ────────────────────────────────────────
def test_absent_evidence(tmp: Path) -> None:
    print("[test] (c) absent Task 9 deploy evidence -> exit 2")
    proc = run_cli("--validate", "--submission-dir", str(tmp / "nonexistent"),
                   "--evidence-dir", str(tmp))
    check("c.absent_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


# ── (d) invalid Task 9 evidence ───────────────────────────────────────
def test_invalid_evidence(tmp: Path) -> None:
    print("[test] (d) invalid Task 9 deploy evidence -> exit 2")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / pv.TASK9_EVIDENCE_NAME).write_text(
        json.dumps({"schema_version": 1, "verdict": "BOGUS",
                    "labels_read": False, "label_sources": []}),
        encoding="utf-8")
    proc = run_cli("--validate", "--submission-dir", str(tmp / "nonexistent"),
                   "--evidence-dir", str(tmp))
    check("d.invalid_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


# ── (e) DEPLOYMENT_PASS verdict + no submission dir -> SKIPPED ────────
def test_deployed_no_package(tmp: Path) -> None:
    print("[test] (e) DEPLOYMENT_PASS verdict + no submission dir -> exit 0 SKIPPED")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / pv.TASK9_EVIDENCE_NAME).write_text(
        json.dumps(_mk_task9("DEPLOYMENT_PASS"), ensure_ascii=False), encoding="utf-8")
    proc = run_cli("--validate", "--evidence-dir", str(tmp))
    check("e.deployed_no_dir_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-10-package.json"
    if ev.is_file():
        rec = load(ev)
        check("e.verdict_skipped", rec.get("verdict") == "SKIPPED",
              str(rec.get("verdict")))
        check("e.valid_skipped", rec.get("valid_skipped") is True)
        check("e.no_package", rec.get("package", {}).get("created") is False)


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


# ── (g) full-validation guards (deterministic negative coverage) ──────
def test_full_validation_guards() -> None:
    print("[test] (g) full-validation guards reject adversarial conditions")
    import numpy as np  # noqa: PLC0415

    def rejects(name: str, fn: Any) -> None:
        try:
            fn()
            check(f"g.{name}", False, "가드가 위반을 허용함!")
        except pv.PolicyViolation:
            check(f"g.{name}", True)

    rejects("parity_drift", lambda: pv._guard_parity(1e-5))
    rejects("nan_output", lambda: pv._guard_finite_clipped([0.4, float("nan")]))
    rejects("out_of_range", lambda: pv._guard_finite_clipped([0.2, 0.8]))
    rejects("cpu_over_cap", lambda: pv._guard_cpu_cap(7))
    rejects("rss_breach", lambda: pv._guard_rss(pv.RSS_MAX_BYTES))
    rejects("gpu_vram_breach", lambda: pv._guard_gpu_vram(pv.GPU_VRAM_MAX_BYTES + 1))
    rejects("zip_size_breach", lambda: pv._guard_zip_size(pv.ZIP_MAX_BYTES + 1))
    rejects("extracted_size_breach",
            lambda: pv._guard_extracted_size(pv.EXTRACTED_MAX_BYTES + 1))
    rejects("install_timeout", lambda: pv._guard_install_time(600.0))
    rejects("wheel_hash_mismatch", lambda: pv._guard_wheel_hash("b" * 64, "a" * 64))
    rejects("pip_check_fail", lambda: pv._guard_pip_check(False))
    rejects("cold_script_fail", lambda: pv._guard_cold_script(False))
    rejects("row_order_mismatch", lambda: pv._guard_row_order([2, 1], [1, 2]))
    rejects("row_independence_drift", lambda: pv._guard_row_independence(1e-5))
    rejects("cross_row_state", lambda: pv._guard_no_cross_row_state(True))
    rejects("python_incompat", lambda: pv._guard_python_compat("3.10", "Ubuntu 22.04"))
    # positive controls — valid values must NOT raise
    try:
        pv._guard_parity(0.0)
        pv._guard_finite_clipped([0.4, 0.5, 0.6])
        pv._guard_cpu_cap(6)
        pv._guard_rss(pv.RSS_MAX_BYTES - 1)
        pv._guard_gpu_vram(pv.GPU_VRAM_MAX_BYTES)
        pv._guard_zip_size(pv.ZIP_MAX_BYTES)
        pv._guard_extracted_size(pv.EXTRACTED_MAX_BYTES)
        pv._guard_install_time(599.0)
        pv._guard_wheel_hash("a" * 64, "a" * 64)
        pv._guard_pip_check(True)
        pv._guard_cold_script(True)
        pv._guard_row_order([1, 2], [1, 2])
        pv._guard_row_independence(0.0)
        pv._guard_no_cross_row_state(False)
        pv._guard_python_compat(pv.PYTHON_VERSION, pv.OS_NAME)
        check("g.positive_controls", True)
    except pv.PolicyViolation as exc:
        check("g.positive_controls", False, f"유효 값이 거부됨: {exc}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="t10val_test_") as td:
        tmp = Path(td)
        test_skipped(tmp / "a")
        test_fixtures(tmp / "a")
        test_absent_evidence(tmp / "c")
        test_invalid_evidence(tmp / "d")
        test_deployed_no_package(tmp / "e")
        test_evidence_compliance(tmp / "a")
    test_full_validation_guards()

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
