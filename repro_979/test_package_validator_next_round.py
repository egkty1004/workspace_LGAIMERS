#!/usr/bin/env python3
"""test_package_validator_next_round.py — Task 11 package validator tests (pytest-free).

Tests:
  (a) happy SKIPPED path on the real Task 10 deploy evidence (verdict SKIPPED) → exit 0,
      verdict SKIPPED, label_sources=[], labels_read=false, config pre-registered BEFORE
      any label/package access, retained rollback baseline, NO package dir created,
      evidence provenance clean (provenance fields / EVIDENCE_NAME_RE / forbidden usage /
      forbidden path tokens).
  (b) each of the 3 fixtures (altered-hash, swapped-row-order, missing-clip-check) exits 2,
      writes its own fixture-tagged evidence/config/log, and NEVER clobbers the main
      task-11-package evidence.
  (c) synthetic DEPLOYED task-10 evidence + synthetic package dir → the candidate branch
      reaches the validation guards → REJECT exit 2; a manifest with a mismatched
      candidate_id fails the candidate binding guard; DEPLOYED evidence without a
      manifest/provenance in the submission dir → SKIPPED exit 0.
  (d) missing task-10 evidence → fatal exit 1.
  (e) naming/provenance compliance of all NEW evidence (EVIDENCE_NAME_RE, git_head /
      recorded_at_utc / config hash / label_sources, no forbidden path tokens, no
      forbidden R-only/leaderboard metric keys as numeric values).

Usage: python3 repro_979/test_package_validator_next_round.py
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
TASK10_EV = EVIDENCE / "task-10-deploy.json"
CLI = REPO / "package_validator_next_round.py"

sys.path.insert(0, str(REPO))
import next_round_policy as nrp  # noqa: E402
import package_validator_next_round as pvn  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def run_cli(cli: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(cli), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_temp_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="nrp_pkg_test_"))
    shutil.copy2(TASK10_EV, tmp / "task-10-deploy.json")
    return tmp


def scan_clean(rec: dict, path: Path, forbidden: set[str]) -> list[str]:
    """provenance + forbidden-key + path-token compliance problems for one evidence record."""
    problems: list[str] = []
    problems += nrp.scan_provenance_fields(rec, path)
    problems += nrp.scan_forbidden_usage(rec, forbidden)
    for s in nrp._iter_strs(rec):
        for tok in nrp.FORBIDDEN_PATH_TOKENS:
            if tok in s:
                problems.append(f"forbidden path token {tok!r} in {path.name}")
                break
    return problems


# ── (a) SKIPPED path ─────────────────────────────────────────────────
def test_skipped_path() -> None:
    print("[test] (a) happy SKIPPED path on real task-10 evidence (verdict SKIPPED)")
    tmp = make_temp_dir()
    sub_dir = tmp / "submit_next_round"
    try:
        proc = run_cli(CLI, "--submission-dir", str(sub_dir),
                       "--evidence-dir", str(tmp),
                       "--task10-evidence", str(tmp / "task-10-deploy.json"))
        check("a.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = tmp / "task-11-package.json"
        check("a.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("a.verdict", ev.get("verdict") == "SKIPPED", f"verdict={ev.get('verdict')}")
        check("a.exit_recorded", ev.get("exit_code") == 0)
        check("a.no_labels", ev.get("label_sources") == [] and ev.get("labels_read") is False)
        check("a.reason_task10_skipped", "SKIPPED" in (ev.get("reason") or ""))
        check("a.reason_no_promotion", "NO_PROMOTION" in (ev.get("reason") or ""))
        check("a.retained_baseline",
              ev.get("retained_rollback_baseline_candidate_id")
              == nrp.ROLLBACK_BASELINE_CANDIDATE_ID)
        cpr = ev.get("config_pre_registered") or {}
        check("a.config_pre_registered", cpr.get("written_before_labels_read") is True
              and len(str(cpr.get("sha256") or "")) == 64)
        check("a.config_file_written", (tmp / "task-11-package-config.json").is_file())
        check("a.log_written", (tmp / "task-11-package.log").is_file())
        check("a.md_written", (tmp / "task-11-package.md").is_file())
        check("a.no_package_dir_created", not sub_dir.exists(),
              "SKIPPED path must not create any package dir")
        check("a.no_zip_created", not any(p.suffix == ".zip" for p in tmp.rglob("*")))
        check("a.no_manifest_hash", "manifest_hash" not in ev
              and "manifest" not in ev, "F2-friendly: no manifest/manifest_hash keys")
        policy = load(REPO / "next_round_policy.json")
        problems = scan_clean(ev, ev_path, set(policy["selection"]["forbidden_sort_keys"]))
        check("a.evidence_clean", not problems, f"problems={problems}")
        check("a.f4_name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
        check("a.config_name_re",
              bool(nrp.EVIDENCE_NAME_RE.match((tmp / "task-11-package-config.json").name)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (b) 3 fixtures ───────────────────────────────────────────────────
def test_fixtures() -> None:
    print("[test] (b) each of the 3 fixtures exits 2, fixture-tagged evidence, main not clobbered")
    for name in sorted(pvn.PACKAGE_FIXTURES):
        tmp = make_temp_dir()
        try:
            proc = run_cli(CLI, "--submission-dir", str(tmp / "submit_next_round"),
                           "--evidence-dir", str(tmp),
                           "--task10-evidence", str(tmp / "task-10-deploy.json"),
                           "--fixture", name)
            fix_path = tmp / f"task-11-package-fixture-{name}.json"
            check(f"b.{name}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
            check(f"b.{name}.evidence", fix_path.is_file()
                  and load(fix_path).get("exit_code") == 2)
            check(f"b.{name}.fixture_field", fix_path.is_file()
                  and load(fix_path).get("fixture") == name)
            check(f"b.{name}.config", (tmp / f"task-11-package-config-fixture-{name}.json").is_file())
            check(f"b.{name}.log", (tmp / f"task-11-package-fixture-{name}.log").is_file())
            check(f"b.{name}.name_re", fix_path.is_file()
                  and bool(nrp.EVIDENCE_NAME_RE.match(fix_path.name)))
            check(f"b.{name}.main_not_clobbered", not (tmp / "task-11-package.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── (c) future-proof candidate branch ────────────────────────────────
def test_candidate_branch() -> None:
    print("[test] (c) synthetic DEPLOYED evidence → candidate branch reaches validation guards")
    cid = "d15788255a7596bb"

    # (c1) DEPLOYED + manifest present → binding guards run → REJECT exit 2
    tmp = make_temp_dir()
    try:
        ev = load(tmp / "task-10-deploy.json")
        ev["verdict"] = "DEPLOYED"
        ev["deployed_candidate_id"] = cid
        (tmp / "task-10-deploy.json").write_text(json.dumps(ev), encoding="utf-8")
        sub_dir = tmp / "submit_next_round"
        sub_dir.mkdir()
        manifest = {
            "schema_version": 1,
            "candidate_id": cid,
            "package_path": str(sub_dir),
            "formula": {"c_logit": -0.0404, "clip": [0.30, 0.70]},
            "model_file_sha256": {},
        }
        (sub_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        proc = run_cli(CLI, "--submission-dir", str(sub_dir),
                       "--evidence-dir", str(tmp),
                       "--task10-evidence", str(tmp / "task-10-deploy.json"))
        rec = load(tmp / "task-11-package.json")
        route = rec.get("candidate_route") or {}
        check("c1.exit2", proc.returncode == 2, f"exit={proc.returncode}")
        check("c1.reject", rec.get("verdict") == "REJECT" and rec.get("exit_code") == 2)
        check("c1.route", route.get("candidate_id") == cid
              and route.get("manifest") == "manifest.json", f"route={route}")
        checks = {c.get("rule"): c for c in rec.get("checks") or []}
        check("c1.binding_guards", checks.get("candidate_id_binding", {}).get("ok") is True
              and checks.get("formula_clip_binding", {}).get("ok") is True)
        check("c1.guard_reason", "전체 검증 미실행" in (rec.get("reason") or ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # (c2) manifest candidate_id mismatch → candidate binding guard fires
    tmp = make_temp_dir()
    try:
        ev = load(tmp / "task-10-deploy.json")
        ev["verdict"] = "DEPLOYED"
        ev["deployed_candidate_id"] = cid
        (tmp / "task-10-deploy.json").write_text(json.dumps(ev), encoding="utf-8")
        sub_dir = tmp / "submit_next_round"
        sub_dir.mkdir()
        manifest = {"schema_version": 1, "candidate_id": "beefbeefbeefbeef",
                    "package_path": str(sub_dir), "formula": {"c_logit": -0.0404,
                                                              "clip": [0.30, 0.70]},
                    "model_file_sha256": {}}
        (sub_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        proc = run_cli(CLI, "--submission-dir", str(sub_dir),
                       "--evidence-dir", str(tmp),
                       "--task10-evidence", str(tmp / "task-10-deploy.json"))
        rec = load(tmp / "task-11-package.json")
        check("c2.exit2", proc.returncode == 2, f"exit={proc.returncode}")
        check("c2.binding_reason", "candidate ID binding" in (rec.get("reason") or ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # (c3) DEPLOYED + submission dir lacks manifest/provenance → SKIPPED exit 0
    tmp = make_temp_dir()
    try:
        ev = load(tmp / "task-10-deploy.json")
        ev["verdict"] = "DEPLOYED"
        ev["deployed_candidate_id"] = cid
        (tmp / "task-10-deploy.json").write_text(json.dumps(ev), encoding="utf-8")
        sub_dir = tmp / "submit_next_round"  # exists but empty → no manifest/provenance
        sub_dir.mkdir()
        proc = run_cli(CLI, "--submission-dir", str(sub_dir),
                       "--evidence-dir", str(tmp),
                       "--task10-evidence", str(tmp / "task-10-deploy.json"))
        rec = load(tmp / "task-11-package.json")
        check("c3.exit0", proc.returncode == 0, f"exit={proc.returncode}")
        check("c3.skipped", rec.get("verdict") == "SKIPPED" and rec.get("exit_code") == 0)
        check("c3.no_manifest_reason", "manifest/provenance" in (rec.get("reason") or ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (d) missing task-10 evidence ─────────────────────────────────────
def test_missing_task10() -> None:
    print("[test] (d) missing task-10 evidence → fatal exit 1")
    tmp = Path(tempfile.mkdtemp(prefix="nrp_pkg_test_"))
    try:
        proc = run_cli(CLI, "--submission-dir", str(tmp / "submit_next_round"),
                       "--evidence-dir", str(tmp),
                       "--task10-evidence", str(tmp / "task-10-deploy.json"))
        check("d.exit1", proc.returncode == 1, f"exit={proc.returncode}")
        check("d.no_evidence", not (tmp / "task-11-package.json").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (e) naming/provenance compliance of NEW evidence ─────────────────
def test_evidence_compliance() -> None:
    print("[test] (e) naming/provenance compliance of all new task-11 evidence")
    policy = load(REPO / "next_round_policy.json")
    forbidden = set(policy["selection"]["forbidden_sort_keys"])
    scanned: list[tuple[Path, dict]] = []

    tmp = make_temp_dir()
    try:
        run_cli(CLI, "--submission-dir", str(tmp / "submit_next_round"),
                "--evidence-dir", str(tmp),
                "--task10-evidence", str(tmp / "task-10-deploy.json"))
        scanned.append((tmp / "task-11-package.json", load(tmp / "task-11-package.json")))
        scanned.append((tmp / "task-11-package-config.json", load(tmp / "task-11-package-config.json")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for name in sorted(pvn.PACKAGE_FIXTURES):
        tmp = make_temp_dir()
        try:
            run_cli(CLI, "--submission-dir", str(tmp / "submit_next_round"),
                    "--evidence-dir", str(tmp),
                    "--task10-evidence", str(tmp / "task-10-deploy.json"),
                    "--fixture", name)
            for fname in (f"task-11-package-fixture-{name}.json",
                          f"task-11-package-config-fixture-{name}.json"):
                scanned.append((tmp / fname, load(tmp / fname)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    check("e.new_evidence_found", len(scanned) >= 5, f"n={len(scanned)}")
    for path, rec in scanned:
        problems = scan_clean(rec, path, forbidden)
        check(f"e.clean.{path.name}", not problems, f"problems={problems}")
        check(f"e.name.{path.name}", bool(nrp.EVIDENCE_NAME_RE.match(path.name)))
        check(f"e.no_upload.{path.name}", not nrp.scan_upload_markers(rec))


def main() -> int:
    test_skipped_path()
    test_fixtures()
    test_candidate_branch()
    test_missing_task10()
    test_evidence_compliance()
    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)}/{total} PASS, {len(FAILED)} FAIL")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
