#!/usr/bin/env python3
"""test_next_round_deploy.py — Task 10 deploy dispatcher + F2 audit tests (pytest-free).

Tests:
  (a) happy SKIPPED path on the real Task 9 freeze evidence (NO_PROMOTION) → exit 0,
      verdict SKIPPED, label_sources=[], labels_read=false, config pre-registered
      BEFORE any label access, retained rollback baseline, F1/F4-clean evidence.
  (b) synthetic freeze evidence naming a candidate (verdict FROZEN) → the dispatcher
      reaches family routing/guards → REJECT exit 2 (no candidate manifest exists this
      round). Covers catboost-boundary (task-3 source) and deepfm (task-7 source) routes.
  (c) each of the 7 deploy fixtures exits 2, writes its own fixture-tagged evidence,
      and NEVER clobbers the main task-10-deploy evidence.
  (d) qualification_runner.py --audit-next-round (F2) PASSes on the real evidence dir
      (exit 0, f2-quality.json verdict PASS); each F2 fixture (future-row-read,
      changed-c-logit, source-package-drift) exits 2.
  (e) naming/provenance compliance of all NEW evidence (EVIDENCE_NAME_RE, git_head /
      recorded_at_utc / config hash / label_sources, no forbidden path tokens, no
      forbidden R-only/leaderboard metric keys as numeric values).

Usage: python3 repro_979/test_next_round_deploy.py
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
FREEZE_EV = EVIDENCE / "task-9-freeze.json"
POLICY_CLI = REPO / "next_round_policy.py"
QA_CLI = REPO / "qualification_runner.py"

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


def run_cli(cli: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(cli), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_temp_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="nrp_deploy_test_"))
    shutil.copy2(FREEZE_EV, tmp / "task-9-freeze.json")
    return tmp


# ── (a) SKIPPED path ─────────────────────────────────────────────────
def test_skipped_path() -> None:
    print("[test] (a) happy SKIPPED path on real task-9 freeze evidence (NO_PROMOTION)")
    tmp = make_temp_dir()
    try:
        proc = run_cli(POLICY_CLI, "--qualify-and-deploy-frozen",
                       "--evidence-dir", str(tmp),
                       "--freeze-evidence", str(tmp / "task-9-freeze.json"))
        check("a.exit_code", proc.returncode == 0, f"exit={proc.returncode}")
        ev_path = tmp / "task-10-deploy.json"
        check("a.evidence_written", ev_path.is_file())
        ev = load(ev_path)
        check("a.verdict", ev.get("verdict") == "SKIPPED", f"verdict={ev.get('verdict')}")
        check("a.exit_recorded", ev.get("exit_code") == 0)
        check("a.no_labels", ev.get("label_sources") == [] and ev.get("labels_read") is False)
        check("a.reason_no_promotion", "NO_PROMOTION" in (ev.get("reason") or ""))
        check("a.retained_baseline",
              ev.get("retained_rollback_baseline_candidate_id") == nrp.ROLLBACK_BASELINE_CANDIDATE_ID)
        cpr = ev.get("config_pre_registered") or {}
        check("a.config_pre_registered", cpr.get("written_before_labels_read") is True
              and len(str(cpr.get("sha256") or "")) == 64)
        check("a.config_file_written", (tmp / "task-10-deploy-config.json").is_file())
        check("a.provenance", bool(ev.get("git_head")) and bool(ev.get("recorded_at_utc"))
              and bool(ev.get("policy_config_hash")) and isinstance(ev.get("label_sources"), list))
        policy = load(REPO / "next_round_policy.json")
        check("a.f1_forbidden_keys_clean",
              not nrp.scan_forbidden_usage(ev, set(policy["selection"]["forbidden_sort_keys"])))
        check("a.f1_provenance_clean", not nrp.scan_provenance_fields(ev, ev_path))
        check("a.f4_name_re", bool(nrp.EVIDENCE_NAME_RE.match(ev_path.name)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (b) future-proof candidate routing ───────────────────────────────
def test_candidate_routing() -> None:
    print("[test] (b) synthetic FROZEN evidence → dispatcher reaches family routing/guards")
    base = load(FREEZE_EV)
    for source, family in (("task-3-cat-boundary-gate", "catboost-boundary"),
                           ("task-7-promotion", "deepfm")):
        tmp = Path(tempfile.mkdtemp(prefix="nrp_deploy_cand_"))
        try:
            ev = dict(base)
            ev["verdict"] = "FROZEN"
            ev["decision"] = {"terminal_verdict": "FROZEN",
                              "frozen_candidate_id": "d15788255a7596bb",
                              "primary_bss": 795.2586, "source": source}
            (tmp / "task-9-freeze.json").write_text(json.dumps(ev), encoding="utf-8")
            proc = run_cli(POLICY_CLI, "--qualify-and-deploy-frozen",
                           "--evidence-dir", str(tmp),
                           "--freeze-evidence", str(tmp / "task-9-freeze.json"))
            rec = load(tmp / "task-10-deploy.json")
            route = rec.get("candidate_route") or {}
            check(f"b.{family}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
            check(f"b.{family}.reject", rec.get("verdict") == "REJECT"
                  and rec.get("exit_code") == 2)
            check(f"b.{family}.route", route.get("family") == family
                  and route.get("route") in {"rehash-only", "replay"}
                  and route.get("candidate_id") == "d15788255a7596bb",
                  f"route={route}")
            check(f"b.{family}.manifest_guard",
                  "manifest" in (rec.get("reason") or "").lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── (c) 7 deploy fixtures ────────────────────────────────────────────
def test_deploy_fixtures() -> None:
    print("[test] (c) each of the 7 deploy fixtures exits 2, fixture-tagged evidence")
    for name in sorted(nrp.DEPLOY_FIXTURES):
        tmp = make_temp_dir()
        try:
            proc = run_cli(POLICY_CLI, "--qualify-and-deploy-frozen",
                           "--evidence-dir", str(tmp),
                           "--freeze-evidence", str(tmp / "task-9-freeze.json"),
                           "--fixture", name)
            fix_path = tmp / f"task-10-deploy-fixture-{name}.json"
            check(f"c.{name}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
            check(f"c.{name}.evidence", fix_path.is_file()
                  and load(fix_path).get("exit_code") == 2)
            check(f"c.{name}.name_re", fix_path.is_file()
                  and bool(nrp.EVIDENCE_NAME_RE.match(fix_path.name)))
            check(f"c.{name}.main_not_clobbered", not (tmp / "task-10-deploy.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── (d) F2 audit + fixtures ──────────────────────────────────────────
def test_f2_audit() -> None:
    print("[test] (d) qualification_runner --audit-next-round PASS + 3 F2 fixtures exit 2")
    proc = run_cli(QA_CLI, "--audit-next-round", "--evidence-dir", str(EVIDENCE))
    check("d.audit_exit0", proc.returncode == 0, f"exit={proc.returncode}")
    f2 = EVIDENCE / "f2-quality.json"
    check("d.audit_evidence", f2.is_file() and load(f2).get("verdict") == "PASS")
    check("d.audit_provenance", not nrp.scan_provenance_fields(load(f2), f2))
    for name in ("future-row-read", "changed-c-logit", "source-package-drift"):
        proc = run_cli(QA_CLI, "--audit-next-round", "--evidence-dir", str(EVIDENCE),
                       "--fixture", name)
        fix_path = EVIDENCE / f"f2-quality-fixture-{name}.json"
        check(f"d.{name}.exit2", proc.returncode == 2, f"exit={proc.returncode}")
        check(f"d.{name}.evidence", fix_path.is_file()
              and load(fix_path).get("exit_code") == 2)


# ── (e) naming/provenance compliance of NEW evidence ─────────────────
def test_evidence_compliance() -> None:
    print("[test] (e) naming/provenance compliance of all new task-10/f2 evidence")
    policy = load(REPO / "next_round_policy.json")
    forbidden = set(policy["selection"]["forbidden_sort_keys"])
    new_names = sorted(p.name for p in EVIDENCE.glob("*.json")
                       if p.name.startswith(("task-10", "f2-quality"))
                       and p.name not in {"f2-quality.json"})
    new_names += ["f2-quality.json"]
    check("e.new_evidence_found", len(new_names) >= 5, f"n={len(new_names)}")
    for name in new_names:
        path = EVIDENCE / name
        rec = load(path)
        check(f"e.name.{name}", bool(nrp.EVIDENCE_NAME_RE.match(name)))
        probs = nrp.scan_provenance_fields(rec, path)
        check(f"e.prov.{name}", not probs, f"problems={probs}")
        fk = nrp.scan_forbidden_usage(rec, forbidden)
        check(f"e.forbidden.{name}", not fk, f"problems={fk}")
        toks = [t for s in nrp._iter_strs(rec) for t in nrp.FORBIDDEN_PATH_TOKENS if t in s]
        check(f"e.path_tokens.{name}", not toks, f"tokens={toks}")


# ── (f) F1/F4 audit waive fixes (final-wave prep) ────────────────────
UNDERSCORE_EVIDENCE = [
    "task-6-mlp-preprocess-clip_z5.json",
    "task-6-mlp-preprocess-config-clip_z5-promote.json",
    "task-6-mlp-preprocess-config-clip_z5.json",
    "task-6-mlp-preprocess-config-missing_flags-promote.json",
    "task-6-mlp-preprocess-config-missing_flags.json",
    "task-6-mlp-preprocess-missing_flags.json",
    "task-7-promotion-mlp-clip_z5.json",
    "task-7-promotion-mlp-missing_flags.json",
]
BAD_NAMES = ["Task_6_foo.json", "task-6-Foo.json", "x-task-6.json"]


def test_audit_waive_fixes() -> None:
    print("[test] (f) F1/F4 audit waive fixes: underscore slugs + nested config hash")
    # (a) real task-6 aggregate evidence passes provenance (nested per-variant config_hash)
    t6 = load(EVIDENCE / "task-6-mlp-preprocess.json")
    check("f.a.task6_provenance_clean",
          not nrp.scan_provenance_fields(t6, EVIDENCE / "task-6-mlp-preprocess.json"))
    # (b) EVIDENCE_NAME_RE accepts all 8 underscore-named files, rejects genuinely bad names
    check("f.b.underscore_all_match",
          all(bool(nrp.EVIDENCE_NAME_RE.match(n)) for n in UNDERSCORE_EVIDENCE))
    check("f.b.bad_names_rejected",
          all(not nrp.EVIDENCE_NAME_RE.match(n) for n in BAD_NAMES))
    # (c) negative: no config hash anywhere still REJECTs (check not gutted)
    no_hash = {"git_head": "abc", "recorded_at_utc": "2026-08-16T00:00:00+00:00",
               "label_sources": []}
    probs = nrp.scan_provenance_fields(no_hash, Path("no-hash.json"))
    check("f.c.no_hash_rejected", any("config hash" in p for p in probs), f"problems={probs}")
    nested_ok = dict(no_hash)
    nested_ok["variants"] = {"clip_z5": {"config_hash": "a" * 64}}
    check("f.c.nested_hash_accepted",
          not nrp.scan_provenance_fields(nested_ok, Path("nested.json")))
    # (d) full audits PASS on the real evidence dir (zero violations)
    proc = run_cli(POLICY_CLI, "--audit-compliance", "--evidence-dir", str(EVIDENCE))
    check("f.d.audit_compliance_exit0", proc.returncode == 0, f"exit={proc.returncode}")
    check("f.d.f1_verdict", load(EVIDENCE / "f1-compliance.json").get("verdict") == "PASS")
    proc = run_cli(POLICY_CLI, "--audit-scope", "--evidence-dir", str(EVIDENCE))
    check("f.d.audit_scope_exit0", proc.returncode == 0, f"exit={proc.returncode}")
    check("f.d.f4_verdict", load(EVIDENCE / "f4-scope.json").get("verdict") == "PASS")
    # (e) negative fixtures still exit 2: injected R-sort key (F1) / bad name (F4)
    tmp = Path(tempfile.mkdtemp(prefix="nrp_audit_neg_"))
    try:
        shutil.copy2(EVIDENCE / "task-1-state-policy.json", tmp / "task-1-state-policy.json")
        injected = {"git_head": "abc", "recorded_at_utc": "2026-08-16T00:00:00+00:00",
                    "config_hash": "a" * 64, "label_sources": ["primary"],
                    "delta_r2022": 5.0}
        (tmp / "task-2-injected-r-sort.json").write_text(
            json.dumps(injected), encoding="utf-8")
        proc = run_cli(POLICY_CLI, "--audit-compliance", "--evidence-dir", str(tmp))
        check("f.e.f1_r_sort_exit2", proc.returncode == 2, f"exit={proc.returncode}")
        (tmp / "Task_6_foo.json").write_text("{}", encoding="utf-8")
        proc = run_cli(POLICY_CLI, "--audit-scope", "--evidence-dir", str(tmp))
        check("f.e.f4_bad_name_exit2", proc.returncode == 2, f"exit={proc.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_skipped_path()
    test_candidate_routing()
    test_deploy_fixtures()
    test_f2_audit()
    test_evidence_compliance()
    test_audit_waive_fixes()
    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)}/{total} PASS, {len(FAILED)} FAIL")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
