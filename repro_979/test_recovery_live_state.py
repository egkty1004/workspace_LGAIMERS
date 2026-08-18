#!/usr/bin/env python3
"""test_recovery_live_state.py — Todo 1 (aimers9-top100-score-recovery) tests (pytest-free).

Tests:
  (a) happy --check on the REAL reconciled leaderboard_state.json → exit 0;
      temp-dir evidence task-1-live-state.json written with verdict
      BASELINE_RECONCILED, three fact classes distinguished
      (user_observed / worker_reported_at / package_proven), all six
      provenance elements proven (v93 6-leg package), rollback
      5890a4c54f502c4e/992.8390640403 preserved, blocked_tasks [] (Tasks 2-10
      proceed), task_11=normal; REAL state file sha256 byte-identical
      before/after. The reconciled evidence legitimately references the v93
      package archive filename (submit_v93_r0476.zip) — the ONLY forbidden
      path token present must be .zip.
  (b) each of the 3 fixtures (missing-reported-at, score-package-mismatch,
      invented-event) exits 2, writes task-1-live-state-fixture-<name>.{json,md}
      with fixture+matched fields, never mutates the real state, never clobbers
      the main task-1-live-state evidence.
  (c) --check on an INVALID state (mutated temp copy) exits 1 and writes nothing.
  (d) --notion-receipt / --notion-unavailable record honestly in evidence;
      an invented observation_timestamp in a fixture can never be written as
      user observation (covered by invented-event fixture).
  (e) --audit-baseline-block / --audit-scope-baseline-block: exit 0 ONLY when
      task-1-live-state.json (BLOCK verdict) + task-11-readiness.json
      (SKIPPED_BASELINE_BLOCK) exist and NO task-{2..10}-* artifacts; exit 2
      on missing task-1 / task-11 / wrong verdict / task-2..10 present / scope
      violations (forbidden path tokens, bad evidence name).
  (f) naming/provenance compliance of ALL evidence the CLI writes
      (EVIDENCE_NAME_RE, no forbidden path tokens, no upload markers,
      label_sources=[], labels_read=false).

Usage: python3 repro_979/test_recovery_live_state.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
STATE = REPO / "leaderboard_state.json"
EVIDENCE_DIR = ROOT / ".omo" / "evidence" / "aimers9-top100-recovery"
CLI = REPO / "recovery_live_state.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_live_state as rls  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_clean(rec: dict[str, Any], allowed_tokens: set[str] | None = None) -> list[str]:
    """증거 범위 스캔. allowed_tokens 는 정당하게 존재할 수 있는 금지 토큰
    (예: reconciled 패키지 아카이브 파일명의 .zip) 을 제외한다."""
    allowed = allowed_tokens or set()
    problems: list[str] = []
    for s in rls._iter_strs(rec):
        for tok in rls.FORBIDDEN_PATH_TOKENS:
            if tok in s and tok not in allowed:
                problems.append(f"forbidden path token {tok!r}")
                break
    for key in rls._iter_strs(rec):
        if rls._norm_key(key) in rls.UPLOAD_KEY_NORMS:
            problems.append(f"upload marker key {key!r}")
            break
    return problems


def forbidden_tokens_present(rec: dict[str, Any]) -> set[str]:
    return {t for s in rls._iter_strs(rec)
            for t in rls.FORBIDDEN_PATH_TOKENS if t in s}


# ── (a) happy --check (real state, temp evidence) ─────────────────────
def test_happy_check(tmp: Path) -> None:
    print("[test] (a) happy --check on real reconciled state -> exit 0")
    before = sha256_file(STATE)
    proc = run_cli("--check", "--state", str(STATE), "--evidence-dir", str(tmp))
    after = sha256_file(STATE)
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    check("a.state_untouched", before == after, "(real state sha256 unchanged)")

    ev = tmp / "task-1-live-state.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.verdict", rec.get("verdict") == "BASELINE_RECONCILED",
          str(rec.get("verdict")))
    uo = rec["facts"]["user_observed"]
    check("a.user_observed", (uo["rank_100_cutoff"] == 1090.64249
                              and uo["current_best_public_score"] == 1001.74449
                              and uo["team_rank"] == 287),
          f"cutoff={uo['rank_100_cutoff']} best={uo['current_best_public_score']} "
          f"rank={uo['team_rank']}")
    check("a.obs_timestamp_unknown", uo["observation_timestamp"] == "UNKNOWN")
    wr = rec["facts"]["worker_reported_at"]
    check("a.reported_at_kst",
          bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$",
                        str(wr["reported_at"])))
          and wr["timezone"] == "Asia/Seoul",
          f"reported_at={wr['reported_at']}")
    pp = rec["facts"]["package_proven"]["elements"]
    check("a.package_proven_all_proven",
          all(pp[e]["proven"] is True for e in rls.PROVENANCE_ELEMENTS),
          f"{ {e: pp[e]['proven'] for e in rls.PROVENANCE_ELEMENTS} }")
    rp = rec["rollback_preserved"]
    check("a.rollback",
          (rp["candidate_id"] == "5890a4c54f502c4e"
           and abs(rp["public_score"] - 992.8390640403) < 1e-12
           and rp["submissions_by_date"] == {"2026-08-14": 2}))
    check("a.blocked_tasks", rec["task_routing"]["blocked_tasks"] == [],
          f"{rec['task_routing']['blocked_tasks']}")
    check("a.task11", rec["task_routing"]["task_11"] == "normal",
          f"{rec['task_routing']['task_11']}")
    check("a.notion_field", "notion" in rec, f"status={rec.get('notion', {}).get('status')}")
    check("a.checks_all_ok", all(c["ok"] for c in rec["checks"]),
          f"{[c['rule'] for c in rec['checks'] if not c['ok']]}")
    check("a.labels_unread", rec["labels_read"] is False and rec["label_sources"] == [])
    # reconciled 증거는 v93 패키지 아카이브 파일명(submit_v93_r0476.zip)을 정당하게
    # 참조하므로 .zip 토큰만 허용하고, 그 외 금지 토큰/업로드 마커는 없어야 한다.
    check("a.scan_clean", not scan_clean(rec, allowed_tokens={".zip"}),
          f"{scan_clean(rec)}")
    check("a.scan_only_reconciled_zip",
          forbidden_tokens_present(rec) <= {".zip"},
          f"tokens={forbidden_tokens_present(rec)}")
    check("a.md_written", (tmp / "task-1-live-state.md").is_file())


# ── (b) 3 fixtures ────────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 3 fixtures exit 2, fixture evidence, real state untouched")
    before = sha256_file(STATE)
    main_sha = sha256_file(tmp / "task-1-live-state.json")
    for name in sorted(rls.FIXTURES):
        proc = run_cli("--fixture", name, "--state", str(STATE), "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-1-live-state-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"expected={rec.get('expected_failure_rule')}")
            check(f"b.{name}.no_mutation",
                  rec.get("state_mutation", {}).get("mutated") is False)
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    after = sha256_file(STATE)
    check("b.state_untouched", before == after, "(real state sha256 unchanged)")
    check("b.main_not_clobbered",
          sha256_file(tmp / "task-1-live-state.json") == main_sha,
          "(main task-1-live-state.json byte-identical)")


# ── (c) invalid state -> exit 1, no evidence ──────────────────────────
def test_invalid_state(tmp: Path) -> None:
    print("[test] (c) invalid state -> exit 1, nothing written")
    bad = json.loads(STATE.read_text(encoding="utf-8"))
    bad["rank_100_cutoff"] = 1000.0  # 사용자 보고값과 불일치
    bad_path = tmp / "bad_state.json"
    bad_path.write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
    ev_dir = tmp / "bad_ev"
    proc = run_cli("--check", "--state", str(bad_path), "--evidence-dir", str(ev_dir))
    check("c.exit1", proc.returncode == 1, f"(rc={proc.returncode})")
    check("c.no_evidence", not ev_dir.exists() or not list(ev_dir.iterdir()),
          "(no evidence written on invalid state)")
    check("c.missing_state_exit1",
          run_cli("--check", "--state", str(tmp / "nope.json"),
                  "--evidence-dir", str(ev_dir)).returncode == 1,
          "(missing state file -> exit 1)")


# ── (d) notion receipt / unavailable ──────────────────────────────────
def test_notion(tmp: Path) -> None:
    print("[test] (d) --notion-receipt / --notion-unavailable honest recording")
    receipt = tmp / "receipt.json"
    receipt.write_text(json.dumps({"block": "abc123", "row": "1001.74449"}),
                       encoding="utf-8")
    ev = tmp / "notion_ok"
    proc = run_cli("--check", "--state", str(STATE), "--evidence-dir", str(ev),
                   "--notion-receipt", str(receipt))
    check("d.receipt.exit0", proc.returncode == 0)
    rec = load(ev / "task-1-live-state.json")
    check("d.receipt.available", rec["notion"]["status"] == "available"
          and rec["notion"]["receipt"]["block"] == "abc123")

    ev2 = tmp / "notion_unavail"
    proc2 = run_cli("--check", "--state", str(STATE), "--evidence-dir", str(ev2),
                    "--notion-unavailable", "api timeout")
    check("d.unavailable.exit0", proc2.returncode == 0)
    rec2 = load(ev2 / "task-1-live-state.json")
    check("d.unavailable.status", rec2["notion"]["status"] == "unavailable"
          and rec2["notion"]["reason"] == "api timeout",
          f"{rec2['notion']}")
    check("d.unavailable.no_fake_receipt", "receipt" not in rec2["notion"])
    check("d.conflict_exit2",
          run_cli("--check", "--state", str(STATE), "--evidence-dir", str(ev),
                  "--notion-receipt", str(receipt),
                  "--notion-unavailable", "x").returncode == 2,
          "(receipt + unavailable 동시 지정 -> exit 2)")


# ── (e) audits ────────────────────────────────────────────────────────
def _write_audit_dir(root: Path, *,
                     task11_verdict: str | None = "SKIPPED_BASELINE_BLOCK",
                     task2_present: bool = False,
                     forbidden_token: bool = False) -> Path:
    d = root / "audit_ev"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    # BLOCK-path 감사는 BLOCK verdict 의 task-1 증거를 요구한다. 실제 증거는 이제
    # BASELINE_RECONCILED 이므로, 결정적 BLOCK 경로를 위해 합성 BLOCK 레코드를 쓴다
    # (실제 증거/상태는 미변경).
    task1 = {
        "schema_version": 1,
        "title": "Todo 1 — reconcile live leaderboard state & 1001.74449 provenance",
        "task": "aimers9-top100-recovery/task-1-live-state",
        "verdict": "BASELINE_PROVENANCE_BLOCK",
        "baseline_verdict": "BASELINE_PROVENANCE_BLOCK",
        "exit_code": 0,
        "recorded_at_utc": "2026-08-17T04:00:00+00:00",
        "git_head": "c191a5864749aabedd9c3166475c52d59cddd21e",
        "label_sources": [],
        "labels_read": False,
    }
    (d / "task-1-live-state.json").write_text(
        json.dumps(task1, ensure_ascii=False, indent=2), encoding="utf-8")
    if task11_verdict is not None:
        t11 = {
            "schema_version": 1,
            "title": "Todo 11 — recovery readiness (baseline block)",
            "task": "aimers9-top100-recovery/task-11-readiness",
            "verdict": task11_verdict,
            "exit_code": 0,
            "recorded_at_utc": "2026-08-17T04:00:00+00:00",
            "git_head": "c191a5864749aabedd9c3166475c52d59cddd21e",
            "label_sources": [],
            "labels_read": False,
        }
        (d / "task-11-readiness.json").write_text(
            json.dumps(t11, ensure_ascii=False, indent=2), encoding="utf-8")
    if task2_present:
        (d / "task-2-evaluator.json").write_text(
            json.dumps({"task": "aimers9-top100-recovery/task-2-evaluator",
                        "verdict": "PASS", "exit_code": 0}), encoding="utf-8")
    if forbidden_token:
        (d / "task-1-live-state.json").write_text(
            json.dumps({"데이터/secret": True, "verdict": "BASELINE_PROVENANCE_BLOCK"}),
            encoding="utf-8")
    return d


def test_audits(root: Path) -> None:
    print("[test] (e) --audit-baseline-block / --audit-scope-baseline-block gates")

    d_happy = _write_audit_dir(root)
    p = run_cli("--audit-baseline-block", "--evidence-dir", str(d_happy))
    check("e.baseline.happy", p.returncode == 0, f"(rc={p.returncode})")
    p = run_cli("--audit-scope-baseline-block", "--evidence-dir", str(d_happy))
    check("e.scope.happy", p.returncode == 0, f"(rc={p.returncode})")

    d_no_t11 = _write_audit_dir(root, task11_verdict=None)
    check("e.no_task11",
          run_cli("--audit-baseline-block", "--evidence-dir", str(d_no_t11)).returncode == 2)

    d_no_t1 = root / "audit_no_t1"
    d_no_t1.mkdir(parents=True, exist_ok=True)
    (d_no_t1 / "task-11-readiness.json").write_text(
        json.dumps({"verdict": "SKIPPED_BASELINE_BLOCK"}), encoding="utf-8")
    check("e.no_task1",
          run_cli("--audit-baseline-block", "--evidence-dir", str(d_no_t1)).returncode == 2)

    d_wrong_t11 = _write_audit_dir(root, task11_verdict="SKIPPED_NO_PROMOTION")
    check("e.wrong_task11_verdict",
          run_cli("--audit-baseline-block", "--evidence-dir", str(d_wrong_t11)).returncode == 2)

    d_t2 = _write_audit_dir(root, task2_present=True)
    check("e.task2_present",
          run_cli("--audit-baseline-block", "--evidence-dir", str(d_t2)).returncode == 2)

    d_forbidden = _write_audit_dir(root, forbidden_token=True)
    # baseline-block 는 토큰을 스캔하지 않음 (task-1/11 + 2-10 부재만) → exit 0
    check("e.forbidden.baseline_ok",
          run_cli("--audit-baseline-block", "--evidence-dir",
                  str(d_forbidden)).returncode == 0)
    # scope audit 은 토큰을 스캔 → exit 2
    check("e.forbidden.scope_reject",
          run_cli("--audit-scope-baseline-block", "--evidence-dir",
                  str(d_forbidden)).returncode == 2)

    check("e.missing_dir",
          run_cli("--audit-baseline-block", "--evidence-dir",
                  str(root / "no_such_dir")).returncode == 2)


# ── (f) evidence compliance of everything the CLI wrote ──────────────
def test_evidence_compliance(tmp: Path) -> None:
    print("[test] (f) evidence naming / provenance compliance")
    for path in sorted(tmp.glob("*.json")):
        check(f"f.name_{path.name}", bool(rls.EVIDENCE_NAME_RE.match(path.name)), path.name)
        rec = load(path)
        if rec.get("fixture") is None:  # 비-fixture 메인/감사 대상만
            check(f"f.recorded_at_{path.name}",
                  bool(rec.get("recorded_at_utc")) and bool(rec.get("git_head")))
            check(f"f.labels_{path.name}",
                  rec.get("label_sources") == [] and rec.get("labels_read") is False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rls_test_") as td:
        tmp = Path(td)
        test_happy_check(tmp / "a")
        test_fixtures(tmp / "a")       # 같은 evidence dir 재사용 (메인 증거 보존 검증)
        test_invalid_state(tmp)
        test_notion(tmp)
        test_audits(tmp)
        test_evidence_compliance(tmp / "a")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
