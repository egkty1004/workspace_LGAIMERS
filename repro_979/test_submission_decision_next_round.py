#!/usr/bin/env python3
"""test_submission_decision_next_round.py — Task 12 submission decision tests (pytest-free).

Tests:
  (a) SKIPPED register path on the real Task 11 evidence → exit 0, verdict
      SKIPPED_NO_PACKAGE, label_sources=[], labels_read=false, config pre-registered
      BEFORE any state/label access, retained rollback baseline, REAL
      leaderboard_state.json sha256 byte-identical before/after, evidence provenance clean.
  (b) next-round `--check` (no round-marked qualified package) → exit 0, verdict
      SKIPPED_NO_QUALIFIED_CANDIDATE (never ALLOW); state file missing → BLOCK exit 1.
  (c) each of the 4 fixtures (stale-cutoff, consumed-slot, altered-package,
      unauthorized-state-mutation) exits 2, writes fixture-tagged evidence/config, never
      clobbers the main task-12-decision evidence, real state untouched (in-memory copies).
  (d) synthetic future round: DEPLOYED task-11 evidence + temp state + temp package
      (manifest/provenance/model) → register binds candidate → qualified_package with
      round marker written to TEMP state only (real state untouched) → exit 0; then
      `--check` against temp state evaluates the gates (fresh cutoff) → ALLOW exit 0.
  (e) missing task-11 evidence → fatal exit 1 (no config written).
  (f) top100 compat: `--mode register` / `--mode check` with temp dirs still work
      (evidence to temp base; real top100 evidence paths untouched).
  (g) naming/provenance compliance of ALL new evidence (EVIDENCE_NAME_RE, git_head /
      recorded_at_utc / config hash / label_sources, no forbidden path tokens, no
      forbidden R-only/leaderboard metric keys as numeric values, no upload markers).

Usage: python3 repro_979/test_submission_decision_next_round.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
EVIDENCE = ROOT / ".omo" / "evidence" / "aimers9-next-round"
TASK11_EV = EVIDENCE / "task-11-package.json"
TASK11_EV_REL = ".omo/evidence/aimers9-next-round/task-11-package.json"
STATE = REPO / "leaderboard_state.json"
TASK8_EV = ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-8-package.json"
TOP100_BASE_JSON = ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-9-submission.json"
TOP100_BASE_MD = ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-9-submission.md"
REAL_PKG = REPO / "submit_champ_cat_20260814-0929"
CLI = REPO / "submission_decision.py"

sys.path.insert(0, str(REPO))
import next_round_policy as nrp  # noqa: E402
import recovery_live_state as rls  # noqa: E402
import submission_decision as sd  # noqa: E402

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha(path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True,
                          cwd=str(ROOT))
    assert proc.returncode == 0, f"git show HEAD:{rel} 실패"
    return hashlib.sha256(proc.stdout).hexdigest()


def scan_clean(rec: dict, path: Path, forbidden: set[str]) -> list[str]:
    problems: list[str] = []
    problems += nrp.scan_provenance_fields(rec, path)
    problems += nrp.scan_forbidden_usage(rec, forbidden)
    for s in nrp._iter_strs(rec):
        for tok in nrp.FORBIDDEN_PATH_TOKENS:
            if tok in s:
                problems.append(f"forbidden path token {tok!r} in {path.name}")
                break
    problems += nrp.scan_upload_markers(rec)
    return problems


# ── (a) SKIPPED register path (real task-11 evidence) ─────────────────
def test_skipped_register() -> None:
    print("[test] (a) SKIPPED register path on real task-11 evidence (verdict SKIPPED_NO_PACKAGE)")
    state_before = sha256_file(STATE)
    try:
        proc = run_cli(CLI, "--register-qualified", "--evidence-path", TASK11_EV_REL,
                       "--state", str(STATE))
        check("a.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = EVIDENCE / "task-12-decision.json"
        check("a.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("a.verdict", ev.get("verdict") == "SKIPPED_NO_PACKAGE",
              f"verdict={ev.get('verdict')}")
        check("a.exit_recorded", ev.get("exit_code") == 0)
        check("a.no_labels", ev.get("label_sources") == [] and ev.get("labels_read") is False)
        check("a.reason_task11_skipped", "SKIPPED" in (ev.get("reason") or ""))
        check("a.reason_no_register", "등록 미실행" in (ev.get("reason") or ""))
        cpr = ev.get("config_pre_registered") or {}
        check("a.config_pre_registered", cpr.get("written_before_labels_read") is True
              and len(str(cpr.get("sha256") or "")) == 64)
        check("a.config_file_written", (EVIDENCE / "task-12-decision-config.json").is_file())
        check("a.config_own_hash", len(str(ev.get("config_hash") or "")) == 64
              and len(str(ev.get("policy_config_hash") or "")) == 64)
        check("a.task11_evidence_ref", str(ev.get("task11_evidence") or "") == str(TASK11_EV))
        sm = ev.get("state_mutation") or {}
        check("a.no_state_mutation", sm.get("mutated") is False)
        check("a.state_sha_recorded", sm.get("sha256_before") == state_before
              and sm.get("sha256_after") == state_before)
        checks = {c.get("rule"): c for c in ev.get("checks") or []}
        check("a.checks_present", checks.get("policy_check", {}).get("ok") is True
              and checks.get("task11_evidence_present", {}).get("ok") is True
              and checks.get("no_qualified_package", {}).get("ok") is True
              and checks.get("no_state_mutation", {}).get("ok") is True)
        check("a.violations_empty", ev.get("violations") == [])
        check("a.md_written", (EVIDENCE / "task-12-decision.md").is_file())
        policy = load(REPO / "next_round_policy.json")
        problems = scan_clean(ev, ev_path, set(policy["selection"]["forbidden_sort_keys"]))
        check("a.evidence_clean", not problems, f"problems={problems}")
        check("a.name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
        check("a.config_name_re",
              bool(nrp.EVIDENCE_NAME_RE.match((EVIDENCE / "task-12-decision-config.json").name)))
        config = load(EVIDENCE / "task-12-decision-config.json")
        check("a.config_clean", not scan_clean(
            config, EVIDENCE / "task-12-decision-config.json",
            set(policy["selection"]["forbidden_sort_keys"])), "config evidence clean")
    finally:
        check("a.real_state_untouched", sha256_file(STATE) == state_before,
              "sha256 identical before/after")


# ── (b0) plain `--check` (pure defaults) routes to next-round ─────────
def test_plain_check_defaults() -> None:
    print("[test] (b0) plain `--check` (all defaults) → next-round "
          "SKIPPED_NO_QUALIFIED_CANDIDATE exit 0, top100 evidence untouched")
    state_before = sha256_file(STATE)
    t9_head_json = head_sha(TOP100_BASE_JSON)
    t9_head_md = head_sha(TOP100_BASE_MD)
    try:
        proc = run_cli(CLI, "--check")
        check("b0.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = EVIDENCE / "task-12-decision-check.json"
        check("b0.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("b0.verdict", ev.get("verdict") == "SKIPPED_NO_QUALIFIED_CANDIDATE",
              f"verdict={ev.get('verdict')}")
        check("b0.exit_recorded", ev.get("exit_code") == 0)
        check("b0.never_allow", "ALLOW" not in str(ev.get("verdict")))
    finally:
        check("b0.real_state_untouched", sha256_file(STATE) == state_before)
        check("b0.top100_evidence_untouched",
              sha256_file(TOP100_BASE_JSON) == t9_head_json
              and sha256_file(TOP100_BASE_MD) == t9_head_md)


# ── (b) next-round --check → SKIPPED_NO_QUALIFIED_CANDIDATE ──────────
def test_check_skipped() -> None:
    print("[test] (b) next-round --check → SKIPPED_NO_QUALIFIED_CANDIDATE exit 0 (never ALLOW)")
    state_before = sha256_file(STATE)
    try:
        proc = run_cli(CLI, "--check", "--evidence-path", TASK11_EV_REL, "--state", str(STATE))
        check("b.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = EVIDENCE / "task-12-decision-check.json"
        check("b.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("b.verdict", ev.get("verdict") == "SKIPPED_NO_QUALIFIED_CANDIDATE",
              f"verdict={ev.get('verdict')}")
        check("b.exit_recorded", ev.get("exit_code") == 0)
        check("b.never_allow", "ALLOW" not in str(ev.get("verdict")))
        check("b.no_labels", ev.get("label_sources") == [] and ev.get("labels_read") is False)
        policy = load(REPO / "next_round_policy.json")
        problems = scan_clean(ev, ev_path, set(policy["selection"]["forbidden_sort_keys"]))
        check("b.evidence_clean", not problems, f"problems={problems}")
        check("b.name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
    finally:
        check("b.real_state_untouched", sha256_file(STATE) == state_before)

    print("  [test] (b2) next-round --check with MISSING state file → BLOCK exit 1")
    tmp = Path(tempfile.mkdtemp(prefix="nr_t12_missing_state_"))
    try:
        proc = run_cli(CLI, "--check", "--evidence-path", str(tmp / "x.json"),
                       "--state", str(tmp / "no-such-state.json"))
        check("b2.exit1", proc.returncode == 1, f"exit={proc.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (c) 4 fixtures ───────────────────────────────────────────────────
def test_fixtures() -> None:
    print("[test] (c) each of the 4 fixtures exits 2, fixture evidence, main not clobbered")
    main_ev_before = (sha256_file(EVIDENCE / "task-12-decision.json")
                      if (EVIDENCE / "task-12-decision.json").is_file() else None)
    state_before = sha256_file(STATE)
    for name in sorted(sd.NEXT_ROUND_FIXTURES):
        tmp = Path(tempfile.mkdtemp(prefix=f"nr_t12_fix_{name}_"))
        try:
            proc = run_cli(CLI, "--fixture", name, "--evidence-path", str(tmp / "x.json"),
                           "--state", str(STATE))
            fix_path = tmp / f"task-12-decision-fixture-{name}.json"
            check(f"c.{name}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
            check(f"c.{name}.evidence", fix_path.is_file()
                  and load(fix_path).get("exit_code") == 2)
            check(f"c.{name}.verdict", fix_path.is_file()
                  and load(fix_path).get("verdict") == "REJECT")
            check(f"c.{name}.fixture_field", fix_path.is_file()
                  and load(fix_path).get("fixture") == name)
            check(f"c.{name}.reason", fix_path.is_file() and load(fix_path).get("reason"))
            check(f"c.{name}.config",
                  (tmp / f"task-12-decision-config-fixture-{name}.json").is_file())
            check(f"c.{name}.name_re", fix_path.is_file()
                  and bool(nrp.EVIDENCE_NAME_RE.match(fix_path.name)))
            check(f"c.{name}.main_not_clobbered",
                  not (tmp / "task-12-decision.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if main_ev_before is not None:
        check("c.main_evidence_unchanged",
              sha256_file(EVIDENCE / "task-12-decision.json") == main_ev_before)
    check("c.real_state_untouched", sha256_file(STATE) == state_before)


# ── (d) synthetic future round: bound register + gate-evaluating check ─
def test_synthetic_future_round() -> None:
    print("[test] (d) synthetic DEPLOYED task-11 → bound register (round marker) → gate check")
    cid = "deadbeefcafe0001"
    state_before = sha256_file(STATE)
    tmp = Path(tempfile.mkdtemp(prefix="nr_t12_future_"))
    try:
        pkg = tmp / "pkg"
        (pkg / "model").mkdir(parents=True)
        (pkg / "model" / "x.bin").write_bytes(b"model-bytes-t12")
        model_sha = sha256_file(pkg / "model" / "x.bin")
        provenance = {
            "schema_version": 1,
            "task": "aimers9-next-round",
            "candidate_id": cid,
            "candidate": "synthetic_cat",
            "blend": {"weights": {"champion": 0.85, "catboost": 0.15},
                      "c_logit": -0.0404, "clip": [0.30, 0.70], "seeds": [42]},
            "model_file_sha256": {"x.bin": model_sha},
        }
        (pkg / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "candidate_id": cid,
            "candidate": "synthetic_cat",
            "package_path": str(pkg),
            "model_file_sha256": {"model/x.bin": model_sha},
        }
        manifest["manifest_hash"] = nrp._canonical_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_hash"})
        (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        state = load(STATE)
        state.pop("qualified_package", None)
        state["date_captured"] = sd.now_utc()
        today, _tz = sd.local_today()
        state["submissions_by_date"] = {today: 0}
        state_path = tmp / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        ev = load(TASK11_EV)
        ev["verdict"] = "DEPLOYED"
        ev["deployed_candidate_id"] = cid
        t11 = tmp / "task-11-package.json"
        t11.write_text(json.dumps(ev), encoding="utf-8")

        proc = run_cli(CLI, "--register-qualified", "--candidate-id", cid,
                       "--package-path", str(pkg), "--manifest-path", str(pkg / "manifest.json"),
                       "--evidence-path", str(t11), "--state", str(state_path),
                       "--task8-evidence", str(TASK8_EV))
        check("d.register.exit0", proc.returncode == 0, f"exit={proc.returncode}")
        rec = load(tmp / "task-12-decision.json")
        check("d.register.verdict", rec.get("verdict") == "REGISTERED",
              f"verdict={rec.get('verdict')}")
        state2 = load(state_path)
        qp = state2.get("qualified_package") or {}
        check("d.register.round_marker", qp.get("round") == sd.NEXT_ROUND_MARKER
              and qp.get("candidate_id") == cid)
        check("d.register.digests", len(str(qp.get("qualification_digest") or "")) == 64
              and len(str(qp.get("binding_digest") or "")) == 64)
        check("d.register.state_write_recorded",
              (rec.get("state_mutation") or {}).get("mutated") is True)

        proc2 = run_cli(CLI, "--check", "--candidate-id", cid,
                        "--package-path", str(pkg), "--manifest-path", str(pkg / "manifest.json"),
                        "--evidence-path", str(t11), "--state", str(state_path),
                        "--task8-evidence", str(TASK8_EV))
        check("d.check.exit0", proc2.returncode == 0, f"exit={proc2.returncode}")
        check_rec = load(tmp / "task-12-decision-check.json")
        check("d.check.verdict", check_rec.get("verdict") == "ALLOW",
              f"verdict={check_rec.get('verdict')}")
        check("d.check.gates_evaluated", len(check_rec.get("checks") or []) >= 5
              and all(c.get("ok") for c in check_rec.get("checks") or []))
        check("d.check.registered_marker_used",
              (check_rec.get("qualification_digest_registered") or "")
              == qp.get("qualification_digest"))
        check("d.check.leaderboard_source",
              check_rec.get("leaderboard_source") == "user-reported DACON leaderboard")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        check("d.real_state_untouched", sha256_file(STATE) == state_before)


# ── (e) missing task-11 evidence ─────────────────────────────────────
def test_missing_task11() -> None:
    print("[test] (e) missing task-11 evidence → fatal exit 1")
    tmp = Path(tempfile.mkdtemp(prefix="nr_t12_missing_ev_"))
    try:
        proc = run_cli(CLI, "--register-qualified", "--candidate-id", "x",
                       "--package-path", str(tmp / "pkg"), "--manifest-path",
                       str(tmp / "manifest.json"), "--evidence-path", str(tmp / "missing.json"),
                       "--state", str(tmp / "state.json"))
        check("e.exit1", proc.returncode == 1, f"exit={proc.returncode}")
        check("e.no_config_written", not (tmp / "task-12-decision-config.json").exists())
        check("e.no_evidence_written", not (tmp / "task-12-decision.json").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (f) top100 compat (temp dirs only) ───────────────────────────────
def test_top100_compat() -> None:
    print("[test] (f) top100 --mode register / --mode check still work (temp dirs)")
    t9_json_before = sha256_file(TOP100_BASE_JSON)
    t9_md_before = sha256_file(TOP100_BASE_MD)
    state_before = sha256_file(STATE)
    tmp = Path(tempfile.mkdtemp(prefix="nr_t12_top100_"))
    try:
        state_path = tmp / "state.json"
        state = load(STATE)
        # Task 1 이 date_captured 를 보고일(2026-08-17)로 갱신해 보고일 당일엔
        # 컷오프가 fresh 해진다 — 레거시 BLOCK 경로를 결정적으로 테스트하려면
        # temp 복사본의 캡처 시각을 과거로 고정한다 (실제 상태는 미변경).
        state["date_captured"] = "2020-01-01T00:00:00+00:00"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        base = tmp / "task-9-submission"
        proc = run_cli(CLI, "--mode", "register", "--package-dir", str(REAL_PKG),
                       "--task8-evidence", str(TASK8_EV), "--state", str(state_path),
                       "--evidence-base", str(base))
        check("f.register.exit0", proc.returncode == 0, f"exit={proc.returncode}")
        check("f.register.evidence", (base.with_suffix(".json")).is_file()
              and load(base.with_suffix(".json")).get("decision") == "REGISTERED")
        check("f.register.state_updated",
              (load(state_path).get("qualified_package") or {}).get("package_validated") is True)

        proc2 = run_cli(CLI, "--mode", "check", "--package-dir", str(REAL_PKG),
                        "--task8-evidence", str(TASK8_EV), "--state", str(state_path),
                        "--evidence-base", str(base))
        check("f.check.gates_run", proc2.returncode == 1,  # stale cutoff → BLOCK (existing logic)
              f"exit={proc2.returncode}")
        rec = load(base.with_suffix(".json"))
        check("f.check.block_verdict", rec.get("decision") == "BLOCK"
              and "cutoff_stale" in (rec.get("block_reasons") or []))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        check("f.real_top100_untouched",
              sha256_file(TOP100_BASE_JSON) == t9_json_before
              and sha256_file(TOP100_BASE_MD) == t9_md_before)
        check("f.real_state_untouched", sha256_file(STATE) == state_before)


# ── (g) naming/provenance compliance of NEW evidence ─────────────────
def test_evidence_compliance() -> None:
    print("[test] (g) naming/provenance compliance of all new task-12 evidence")
    policy = load(REPO / "next_round_policy.json")
    forbidden = set(policy["selection"]["forbidden_sort_keys"])
    scanned: list[tuple[Path, dict]] = []

    for fname in ("task-12-decision.json", "task-12-decision-config.json",
                  "task-12-decision-check.json"):
        path = EVIDENCE / fname
        if path.is_file():
            scanned.append((path, load(path)))

    for name in sorted(sd.NEXT_ROUND_FIXTURES):
        tmp = Path(tempfile.mkdtemp(prefix=f"nr_t12_comp_{name}_"))
        try:
            run_cli(CLI, "--fixture", name, "--evidence-path", str(tmp / "x.json"),
                    "--state", str(STATE))
            for fname in (f"task-12-decision-fixture-{name}.json",
                          f"task-12-decision-config-fixture-{name}.json"):
                p = tmp / fname
                scanned.append((p, load(p)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    check("g.new_evidence_found", len(scanned) >= 11, f"n={len(scanned)}")
    for path, rec in scanned:
        problems = scan_clean(rec, path, forbidden)
        check(f"g.clean.{path.name}", not problems, f"problems={problems}")
        check(f"g.name.{path.name}", bool(nrp.EVIDENCE_NAME_RE.match(path.name)))
        check(f"g.no_upload.{path.name}", not nrp.scan_upload_markers(rec))


# ── (h) Task 11 recovery readiness (aimers9-top100-recovery) ─────────
def test_recovery_baseline_block() -> None:
    print("[test] (h1) --check-recovery BLOCK path → SKIPPED_BASELINE_BLOCK exit 0 "
          "(label-free); (h2) idempotent re-run does not clobber")
    state_before = sha256_file(STATE)
    tmp = Path(tempfile.mkdtemp(prefix="nr_t11_check_"))
    try:
        proc = run_cli(CLI, "--check-recovery", "--state", str(STATE),
                       "--evidence-dir", str(tmp))
        check("h1.exit0", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = tmp / "task-11-readiness.json"
        check("h1.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("h1.verdict", ev.get("verdict") == "SKIPPED_BASELINE_BLOCK",
              f"verdict={ev.get('verdict')}")
        check("h1.exit_recorded", ev.get("exit_code") == 0)
        check("h1.no_labels", ev.get("label_sources") == []
              and ev.get("labels_read") is False)
        check("h1.baseline_verdict", ev.get("baseline_verdict") == "BASELINE_PROVENANCE_BLOCK")
        tr = ev.get("task_routing") or {}
        check("h1.blocked_tasks", tr.get("blocked_tasks") == [str(i) for i in range(2, 11)],
              f"blocked={tr.get('blocked_tasks')}")
        check("h1.tasks_2_10_absent", tr.get("tasks_2_10_artifacts_present") is False)
        check("h1.task_routing_note", bool(tr.get("note")))
        check("h1.no_notion_row", (ev.get("notion") or {}).get("row_written") is False)
        sm = ev.get("state_mutation") or {}
        check("h1.no_state_mutation", sm.get("mutated") is False
              and sm.get("sha256_before") == state_before)
        check("h1.git_head", len(str(ev.get("git_head") or "")) >= 7)
        check("h1.recorded_at_utc", bool(ev.get("recorded_at_utc")))
        check("h1.name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
        sha1 = sha256_file(ev_path)
        proc2 = run_cli(CLI, "--check-recovery", "--state", str(STATE),
                        "--evidence-dir", str(tmp))
        check("h2.exit0", proc2.returncode == 0, f"exit={proc2.returncode}")
        check("h2.not_clobbered", sha256_file(ev_path) == sha1,
              "second run must not rewrite main evidence")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        check("h1.real_state_untouched", sha256_file(STATE) == state_before)


def test_recovery_fixtures() -> None:
    print("[test] (h3) 4 recovery fixtures exit 2, fixture evidence, main not clobbered")
    state_before = sha256_file(STATE)
    for name in sorted(sd.RECOVERY_FIXTURES):
        tmp = Path(tempfile.mkdtemp(prefix=f"nr_t11_fix_{name}_"))
        try:
            proc = run_cli(CLI, "--fixture", name, "--state", str(STATE),
                           "--evidence-dir", str(tmp))
            fix_path = tmp / f"task-11-readiness-fixture-{name}.json"
            check(f"h3.{name}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
            check(f"h3.{name}.evidence", fix_path.is_file()
                  and load(fix_path).get("exit_code") == 2)
            check(f"h3.{name}.verdict", fix_path.is_file()
                  and load(fix_path).get("verdict") == "REJECT")
            check(f"h3.{name}.fixture_field", fix_path.is_file()
                  and name in load(fix_path).get("fixture", ""))
            check(f"h3.{name}.reason", fix_path.is_file() and load(fix_path).get("reason"))
            check(f"h3.{name}.name_re", fix_path.is_file()
                  and bool(nrp.EVIDENCE_NAME_RE.match(fix_path.name)))
            check(f"h3.{name}.main_not_clobbered",
                  not (tmp / "task-11-readiness.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    check("h3.real_state_untouched", sha256_file(STATE) == state_before)


def test_recovery_register_blocked() -> None:
    print("[test] (h4) --register-recovery-qualified BLOCK path → refuse exit 1, no mutation")
    state_before = sha256_file(STATE)
    tmp = Path(tempfile.mkdtemp(prefix="nr_t11_reg_"))
    try:
        proc = run_cli(CLI, "--register-recovery-qualified", "--state", str(STATE),
                       "--evidence-dir", str(tmp))
        check("h4.exit1", proc.returncode == 1, f"exit={proc.returncode}")
        ev_path = tmp / "task-11-readiness-register-rejected.json"
        check("h4.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("h4.verdict", ev.get("verdict") == "SKIPPED_BASELINE_BLOCK",
              f"verdict={ev.get('verdict')}")
        check("h4.exit_recorded", ev.get("exit_code") == 1)
        sm = ev.get("state_mutation") or {}
        check("h4.no_state_mutation", sm.get("mutated") is False
              and sm.get("sha256_before") == state_before)
        check("h4.mode", ev.get("mode") == "register-recovery-qualified")
        check("h4.name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        check("h4.real_state_untouched", sha256_file(STATE) == state_before)


def test_recovery_evidence_compliance() -> None:
    print("[test] (h5) naming/provenance/scope compliance of recovery evidence "
          "(incl. value-level upload-norm scan like --audit-scope-baseline-block)")
    scanned: list[tuple[Path, dict]] = []
    for mode in ("check", "register"):
        tmp = Path(tempfile.mkdtemp(prefix=f"nr_t11_comp_{mode}_"))
        try:
            cli_args = (["--check-recovery"] if mode == "check"
                        else ["--register-recovery-qualified"])
            run_cli(CLI, *cli_args, "--state", str(STATE), "--evidence-dir", str(tmp))
            for fname in ("task-11-readiness.json",
                          "task-11-readiness-register-rejected.json"):
                p = tmp / fname
                if p.is_file():
                    scanned.append((p, load(p)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for name in sorted(sd.RECOVERY_FIXTURES):
        tmp = Path(tempfile.mkdtemp(prefix=f"nr_t11_comp_{name}_"))
        try:
            run_cli(CLI, "--fixture", name, "--state", str(STATE), "--evidence-dir", str(tmp))
            p = tmp / f"task-11-readiness-fixture-{name}.json"
            scanned.append((p, load(p)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    check("h5.new_evidence_found", len(scanned) == 6, f"n={len(scanned)}")
    for path, rec in scanned:
        check(f"h5.name.{path.name}", bool(nrp.EVIDENCE_NAME_RE.match(path.name)))
        problems = scan_clean(rec, path, set())
        check(f"h5.clean.{path.name}", not problems, f"problems={problems}")
        bad = [s for s in rls._iter_strs(rec) if rls._norm_key(s) in rls.UPLOAD_KEY_NORMS]
        check(f"h5.scope.{path.name}", not bad, f"norm-collisions={bad}")


def main() -> int:
    test_skipped_register()
    test_plain_check_defaults()
    test_check_skipped()
    test_fixtures()
    test_synthetic_future_round()
    test_missing_task11()
    test_top100_compat()
    test_evidence_compliance()
    test_recovery_baseline_block()
    test_recovery_fixtures()
    test_recovery_register_blocked()
    test_recovery_evidence_compliance()
    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)}/{total} PASS, {len(FAILED)} FAIL")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
