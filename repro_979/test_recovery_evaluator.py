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

import numpy as np

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


def copy_screen_evidence(tmp: Path) -> None:
    """실제 Task 4-6 스크린 증거를 임시 디렉토리로 복사 (--freeze 입력용)."""
    tmp.mkdir(parents=True, exist_ok=True)
    src = rp.DEFAULT_EVIDENCE_DIR
    for fname in re.FREEZE_SCREEN_EVIDENCE:
        shutil.copy(src / fname, tmp / fname)


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
        if name in re.FREEZE_FIXTURES or name in re.TERMINAL_FIXTURES:
            continue  # Task 7/8 fixtures are covered by tests (m)/(r)
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
    ev = tmp / "task-3-baseline-registry.json"
    check("c.evidence", ev.is_file(), str(ev))
    if ev.is_file():
        rec = load(ev)
        check("c.manifest_valid", rec.get("verdict") == "PASS"
              and not rec.get("violations"), f"{rec.get('violations')}")
        check("c.registry_valid", rec.get("registry", {}).get("valid") is True)
        check("c.baseline_sha", rec.get("baseline", {}).get("sha256")
              == "8157e144090bcccbf1c44367c75a2d2427e040b8353c8c1e17334b41324ac5fb")


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


# ── (e) freeze reads Task 4-6 evidence; terminal-check requires FROZEN ─
def test_freeze_terminal(tmp: Path) -> None:
    print("[test] (e) freeze reads Task 4-6 evidence; terminal-check requires FROZEN")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)
    # terminal-check before freeze -> exit 2
    proc = run_cli("--terminal-check", "--candidate", "t", "--evidence-dir", str(tmp))
    check("e.terminal_before_freeze_exit2", proc.returncode == 2, f"(rc={proc.returncode})")
    # freeze with real Task 4-6 evidence -> NO_PROMOTION (all REJECT), firewall UNFROZEN
    copy_screen_evidence(tmp)
    proc = run_cli("--freeze", "--evidence-dir", str(tmp))
    check("e.freeze_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-7-freeze.json"
    if ev.is_file():
        rec = load(ev)
        check("e.freeze_verdict", rec.get("verdict") == "NO_PROMOTION",
              str(rec.get("verdict")))
        check("e.freeze_labels_unread", rec.get("labels_read") is False
              and rec.get("label_sources") == [])
        check("e.freeze_firewall_unfrozen",
              rec.get("terminal_firewall", {}).get("state") == rp.FIREWALL_UNFROZEN)
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
          re.cmd_terminal_check(args) == 0,
          "in-process terminal-check exit 0 (SKIPPED on NO_PROMOTION freeze)")
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


# ── (h) Todo 3: registry structure / config hashes / selectable+control ──
def test_registry(tmp: Path) -> None:
    print("[test] (h) registry structure, config hashes, selectable/control IDs")
    reg = re.load_registry()
    problems = re.validate_registry(reg)
    check("h.registry_valid", not problems, f"{problems}")
    check("h.selectable_ids",
          reg.get("selectable_ids") == [
              "catboost_c2_lossguide", "catboost_c3_ordered", "catboost_c4_rmse",
              "residual_ridge", "calibration_beta", "calibration_isotonic"])
    check("h.control_ids",
          reg.get("control_ids") == ["catboost_c1_control", "calibration_identity"])
    cands = reg.get("candidates") or {}
    for cid in reg.get("selectable_ids", []) + reg.get("control_ids", []):
        entry = cands.get(cid) or {}
        cfg = entry.get("config") or {}
        recomputed = rp._canonical_sha256(cfg)
        check(f"h.config_hash_{cid}", entry.get("config_hash") == recomputed)
        check(f"h.resource_cap_{cid}", isinstance(entry.get("resource_cap"), dict))
    # selectable vs control flags
    for cid in reg.get("selectable_ids", []):
        check(f"h.selectable_flag_{cid}", cands.get(cid, {}).get("selectable") is True)
    for cid in reg.get("control_ids", []):
        check(f"h.control_flag_{cid}", cands.get(cid, {}).get("selectable") is False)
    # blocked legacy present
    blocked = reg.get("blocked_legacy") or {}
    check("h.blocked_names", "asof_n_bucket" in (blocked.get("names") or [])
          and "score_diff_binary" in (blocked.get("names") or []))
    check("h.blocked_configs", "catboost9" in (blocked.get("configs") or [])
          and "deepfm_dcnv2" in (blocked.get("configs") or []))
    check("h.baseline_present", isinstance(reg.get("baseline"), dict)
          and reg.get("baseline", {}).get("candidate_id") == "submit_v93_6leg_r0477")


# ── (i) Todo 3: blocked legacy rejection before data loading ──────────
def test_blocked_rejection(tmp: Path) -> None:
    print("[test] (i) blocked legacy digest/name/config rejected before data loading")
    reg = re.load_registry()
    for blocked_id in ("deepfm_dcnv2", "catboost9", "asof_n_bucket",
                       "score_diff_binary", "unknown-public-baseline"):
        problems = re.reject_blocked(reg, blocked_id)
        check(f"i.reject_{blocked_id}", bool(problems), f"{problems}")
    # allowed candidates are NOT rejected
    for cid in reg.get("selectable_ids", []) + reg.get("control_ids", []):
        problems = re.reject_blocked(reg, cid)
        check(f"i.allow_{cid}", not problems, f"{problems}")
    # baseline is allowed
    check("i.allow_baseline", not re.reject_blocked(reg, "baseline"))


# ── (j) Todo 3: baseline package hashes ───────────────────────────────
def test_baseline_package(tmp: Path) -> None:
    print("[test] (j) baseline package hashes + sha256")
    hashes = re.baseline_package_hashes()
    src = hashes.get("source_files") or {}
    check("j.source_files", all(src.get(n) and src[n] != "missing"
                                for n in ("script.py", "common.py", "mlp_model.py",
                                          "requirements.txt")), f"{src}")
    model_files = hashes.get("model_files") or {}
    check("j.model_files_51", len(model_files) == 51, f"n={len(model_files)}")
    check("j.catboost_models", all(f"catboost_s{s}.cbm" in model_files
                                   for s in range(42, 52)))
    check("j.sha256", re.baseline_package_sha256()
          == "8157e144090bcccbf1c44367c75a2d2427e040b8353c8c1e17334b41324ac5fb")


# ── (k) Todo 3: 3 new fixtures exit 2 ─────────────────────────────────
def test_task3_fixtures(tmp: Path) -> None:
    print("[test] (k) 3 new fixtures exit 2, fixture evidence, main not clobbered")
    main_ev = tmp / "task-3-baseline-registry.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in ("legacy-deepfm", "catboost9-digest", "unknown-public-baseline"):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"k.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-2-evaluator-fixture-{name}.json"
        check(f"k.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"k.{name}.fixture_field", rec.get("fixture") == name)
            check(f"k.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"k.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("k.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-3-baseline-registry.json byte-identical)")


def _mk_task6_evidence(cands: list[dict[str, Any]]) -> dict[str, Any]:
    """합성 task-6 스타일 증거 — 후보 게이트 구조 (선택 로직 단위 테스트용)."""
    per: dict[str, Any] = {}
    for c in cands:
        per[c["candidate_id"]] = {
            "selectable": c["selectable"],
            "gate": {
                "origins": {
                    "r2022": {"delta_bss": c["mean_delta_bss"], "brier_candidate": 0.24,
                              "brier_baseline": 0.245, "bootstrap_lb5": 1.0,
                              "mean_shift": 0.001, "finite": True},
                    "r2023": {"delta_bss": c["mean_delta_bss"], "brier_candidate": 0.24,
                              "brier_baseline": 0.245, "bootstrap_lb5": 1.0,
                              "mean_shift": 0.001, "finite": True},
                },
                "passed": c["gate_passed"],
                "verdict": "PASS" if c["gate_passed"] else "REJECT",
            },
        }
    return {
        "schema_version": 1,
        "verdict": "PASS" if any(c["gate_passed"] for c in cands) else "REJECT",
        "config_hash": "a8d344c7a98fd9571c556c322c9f428dbb6ab940484d8951d08c8c19c96a84e5",
        "label_sources": ["r2022", "r2023"],
        "labels_read": True,
        "per_candidate": per,
    }


# ── (l) Todo 7: freeze happy path (NO_PROMOTION) ──────────────────────
def test_task7_freeze(tmp: Path) -> None:
    print("[test] (l) --freeze reads Task 4-6 evidence -> NO_PROMOTION, labels unread")
    copy_screen_evidence(tmp)
    proc = run_cli("--freeze", "--evidence-dir", str(tmp))
    check("l.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-7-freeze.json"
    check("l.evidence", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("l.verdict", rec.get("verdict") == "NO_PROMOTION", str(rec.get("verdict")))
    check("l.decision", rec.get("decision", {}).get("terminal_verdict") == "NO_PROMOTION"
          and rec.get("decision", {}).get("frozen_candidate_id") is None)
    check("l.selection_key", rec.get("decision", {}).get("selection_key")
          == "mean_selection_delta_bss")
    check("l.tie_rule", "smallest canonical" in str(rec.get("decision", {}).get("tie_rule")))
    check("l.survivors_empty", rec.get("survivors") == [], f"{rec.get('survivors')}")
    check("l.labels_unread", rec.get("labels_read") is False
          and rec.get("label_sources") == [])
    check("l.firewall_unfrozen", rec.get("terminal_firewall", {}).get("state")
          == rp.FIREWALL_UNFROZEN)
    check("l.screen_evidence", len(rec.get("screen_evidence") or {}) == 3)
    check("l.hashes", isinstance(rec.get("hashes", {}).get("code_hashes"), dict)
          and bool(rec.get("hashes", {}).get("config_hash")))
    check("l.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    cands = {c["candidate_id"]: c for c in rec.get("candidates") or []}
    for cid in ("catboost_c2_lossguide", "residual_ridge", "calibration_beta",
                "calibration_isotonic"):
        check(f"l.cand_{cid}", cid in cands and cands[cid]["gate_passed"] is False)
    check("l.md", (tmp / "task-7-freeze.md").is_file())


# ── (m) Todo 7: 3 new fixtures exit 2 ─────────────────────────────────
def test_task7_fixtures(tmp: Path) -> None:
    print("[test] (m) 3 Task 7 freeze fixtures exit 2, fixture evidence")
    copy_screen_evidence(tmp)
    main_ev = tmp / "task-7-freeze.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in re.FREEZE_FIXTURES:
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"m.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-7-freeze-fixture-{name}.json"
        check(f"m.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"m.{name}.fixture_field", rec.get("fixture") == name)
            check(f"m.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"m.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("m.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-7-freeze.json byte-identical)")


# ── (n) Todo 7: selection logic ───────────────────────────────────────
def test_task7_selection(tmp: Path) -> None:
    print("[test] (n) _select_survivor picks highest mean delta-BSS, tie -> smallest ID")

    def mk(cid: str, mean: float) -> dict[str, Any]:
        return {"candidate_id": cid, "selectable": True, "gate_passed": True,
                "mean_delta_bss": mean, "origins": {}}

    s = re._select_survivor([mk("a", 1.0)])
    check("n.single", s is not None and s["candidate_id"] == "a")
    s = re._select_survivor([mk("a", 1.0), mk("b", 5.0), mk("c", 3.0)])
    check("n.highest", s is not None and s["candidate_id"] == "b")
    s = re._select_survivor([mk("catboost_c2_lossguide", 2.0), mk("calibration_beta", 2.0)])
    check("n.tie_smallest", s is not None and s["candidate_id"] == "calibration_beta")
    check("n.empty", re._select_survivor([]) is None)
    # non-selectable control excluded from survivors even when gate passes
    cands = [{"candidate_id": "calibration_identity", "selectable": False,
              "gate_passed": True, "mean_delta_bss": 99.0, "origins": {}}]
    res = re._decide_freeze({"task-6-calibration.json": _mk_task6_evidence(cands)})
    check("n.control_excluded", res["survivors"] == [], f"{res['survivors']}")


# ── (o) Todo 7: evidence validation ───────────────────────────────────
def test_task7_evidence_validation(tmp: Path) -> None:
    print("[test] (o) absent/invalid screen evidence -> freeze exit 2")
    proc = run_cli("--freeze", "--evidence-dir", str(tmp))
    check("o.absent_exit2", proc.returncode == 2, f"(rc={proc.returncode})")
    copy_screen_evidence(tmp)
    (tmp / "task-4-catboost.json").write_text("{not json", encoding="utf-8")
    proc = run_cli("--freeze", "--evidence-dir", str(tmp))
    check("o.invalid_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


# ── (p) Todo 7: synthetic passing candidate -> FROZEN ─────────────────
def test_task7_frozen_path(tmp: Path) -> None:
    print("[test] (p) synthetic passing candidate -> FROZEN, firewall flips")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)
    cands = [{"candidate_id": "calibration_beta", "selectable": True,
              "gate_passed": True, "mean_delta_bss": 5.0, "origins": {}}]
    res = re._decide_freeze({"task-6-calibration.json": _mk_task6_evidence(cands)})
    check("p.decision_frozen", res["decision"]["terminal_verdict"] == "FROZEN"
          and res["decision"]["frozen_candidate_id"] == "calibration_beta")
    check("p.survivors", res["survivors"] == ["calibration_beta"])
    d = tmp / "frozen_ev"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    copy_screen_evidence(d)
    (d / "task-6-calibration.json").write_text(
        json.dumps(_mk_task6_evidence(cands), ensure_ascii=False), encoding="utf-8")
    args = argparse.Namespace(evidence_dir=str(d), candidate=None)
    check("p.cmd_freeze_frozen", re.cmd_freeze(args) == 0, "in-process freeze exit 0")
    check("p.firewall_frozen", rp.firewall_state() == rp.FIREWALL_FROZEN)
    rec = load(d / "task-7-freeze.json")
    check("p.verdict_frozen", rec.get("verdict") == "FROZEN")
    check("p.candidate_locked", rec.get("decision", {}).get("frozen_candidate_id")
          == "calibration_beta")
    check("p.candidate_config_hash", bool(rec.get("hashes", {}).get("candidate_config_hash")))
    rp.set_firewall(rp.FIREWALL_UNFROZEN)  # reset


# ── (q) Todo 8: SKIPPED path (NO_PROMOTION freeze) ────────────────────
def test_task8_skipped(tmp: Path) -> None:
    print("[test] (q) --terminal-check with NO_PROMOTION freeze -> SKIPPED, labels unread")
    copy_screen_evidence(tmp)
    proc = run_cli("--freeze", "--evidence-dir", str(tmp))
    check("q.freeze_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    freeze_sha = (tmp / "task-7-freeze.json").read_bytes()
    proc = run_cli("--terminal-check", "--evidence-dir", str(tmp))
    check("q.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-8-terminal.json"
    check("q.evidence", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("q.verdict", rec.get("verdict") == "SKIPPED", str(rec.get("verdict")))
    check("q.terminal_verdict", rec.get("terminal_verdict") == "SKIPPED")
    check("q.freeze_verdict", rec.get("freeze_verdict") == "NO_PROMOTION")
    check("q.pre_read_freeze_hash", bool(rec.get("pre_read_freeze_hash")))
    check("q.labels_unread", rec.get("labels_read") is False
          and rec.get("label_sources") == [])
    check("q.no_post_selection",
          rec.get("freeze_decision", {}).get("frozen_candidate_id") is None)
    check("q.firewall_unfrozen", rec.get("terminal_firewall", {}).get("state")
          == rp.FIREWALL_UNFROZEN)
    check("q.freeze_unchanged", (tmp / "task-7-freeze.json").read_bytes() == freeze_sha,
          "(freeze evidence byte-identical after terminal check)")
    check("q.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    check("q.md", (tmp / "task-8-terminal.md").is_file())


# ── (r) Todo 8: 4 terminal fixtures exit 2 ────────────────────────────
def test_task8_fixtures(tmp: Path) -> None:
    print("[test] (r) 4 Task 8 terminal fixtures exit 2, fixture evidence")
    main_ev = tmp / "task-8-terminal.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in re.TERMINAL_FIXTURES:
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"r.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-8-terminal-fixture-{name}.json"
        check(f"r.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"r.{name}.fixture_field", rec.get("fixture") == name)
            check(f"r.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"r.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("r.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-8-terminal.json byte-identical)")


# ── (s) Todo 8: terminal_gate pass/fail conditions ────────────────────
def test_task8_terminal_gate(tmp: Path) -> None:
    print("[test] (s) terminal_gate pass/fail conditions")
    rng = np.random.default_rng(20260818)
    n = 500
    y = rng.integers(0, 2, size=n).astype(np.float64)
    base_p = np.clip(0.5 + 0.1 * rng.standard_normal(n), rp.CLIP_LO, rp.CLIP_HI)
    # identical -> fail (delta_bss 0, LB5 0)
    gate = rp.terminal_gate(base_p.copy(), base_p, y)
    check("s.identical_reject", gate["passed"] is False, f"{gate['violations']}")
    # non-finite -> fail
    bad = base_p.copy()
    bad[0] = np.nan
    gate = rp.terminal_gate(bad, base_p, y)
    check("s.nonfinite_reject", gate["passed"] is False, f"{gate['violations']}")
    # mean shift -> fail
    shifted = np.clip(base_p + 0.05, rp.CLIP_LO, rp.CLIP_HI)
    gate = rp.terminal_gate(shifted, base_p, y)
    check("s.mean_shift_reject", gate["passed"] is False, f"{gate['violations']}")
    # better-calibrated candidate with same mean -> pass
    cand_p = np.clip(0.5 + 0.8 * (y - 0.5) + 0.01 * rng.standard_normal(n),
                     rp.CLIP_LO, rp.CLIP_HI)
    gate = rp.terminal_gate(cand_p, base_p, y)
    check("s.better_pass", gate["passed"] is True, f"{gate['violations']}")
    for key in ("delta_bss", "brier_candidate", "brier_baseline", "bootstrap_lb5",
                "mean_shift", "finite", "passed", "verdict", "violations"):
        check(f"s.key_{key}", key in gate, key)


# ── (t) Todo 8: FROZEN path (in-process) ──────────────────────────────
def test_task8_frozen_path(tmp: Path) -> None:
    print("[test] (t) FROZEN freeze evidence -> terminal check runs (in-process)")
    rp.set_firewall(rp.FIREWALL_FROZEN)
    d = tmp / "frozen_t8"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema_version": 1,
        "verdict": "FROZEN",
        "exit_code": 0,
        "config_hash": "x",
        "label_sources": [],
        "labels_read": False,
        "decision": {
            "terminal_verdict": "FROZEN",
            "frozen_candidate_id": "calibration_beta",
            "mean_delta_bss": 5.0,
            "selection_key": "mean_selection_delta_bss",
            "tie_rule": "smallest canonical candidate ID on exact tie",
        },
    }
    (d / "task-7-freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False), encoding="utf-8")
    args = argparse.Namespace(evidence_dir=str(d), candidate=None)
    rc = re.cmd_terminal_check(args)
    check("t.exit0", rc == 0, f"(rc={rc})")
    ev = d / "task-8-terminal.json"
    check("t.evidence", ev.is_file(), str(ev))
    if ev.is_file():
        rec = load(ev)
        check("t.terminal_verdict", rec.get("terminal_verdict") == "NO_PROMOTION",
              str(rec.get("terminal_verdict")))
        check("t.labels_read", rec.get("labels_read") is True
              and rec.get("label_sources") == [rp.TERMINAL_ORIGIN])
        check("t.candidate", rec.get("candidate_id") == "calibration_beta")
        check("t.gate", isinstance(rec.get("gate"), dict)
              and rec.get("gate", {}).get("passed") is False)
        check("t.pre_read_freeze_hash", bool(rec.get("pre_read_freeze_hash")))
    rp.set_firewall(rp.FIREWALL_UNFROZEN)  # reset


# ── (u) Todo 8: spent-exactly-once guard ──────────────────────────────
def test_task8_spent_once(tmp: Path) -> None:
    print("[test] (u) terminal check spent exactly once")
    copy_screen_evidence(tmp)
    run_cli("--freeze", "--evidence-dir", str(tmp))
    proc = run_cli("--terminal-check", "--evidence-dir", str(tmp))
    check("u.first_exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    proc = run_cli("--terminal-check", "--evidence-dir", str(tmp))
    check("u.second_exit2", proc.returncode == 2, f"(rc={proc.returncode})")
    # absent freeze evidence -> exit 2
    d2 = tmp / "no_freeze"
    d2.mkdir(parents=True, exist_ok=True)
    proc = run_cli("--terminal-check", "--evidence-dir", str(d2))
    check("u.absent_freeze_exit2", proc.returncode == 2, f"(rc={proc.returncode})")


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
        test_registry(tmp / "h")
        test_blocked_rejection(tmp / "i")
        test_baseline_package(tmp / "j")
        test_task3_fixtures(tmp / "k")
        test_task7_freeze(tmp / "l")
        test_task7_fixtures(tmp / "m")
        test_task7_selection(tmp / "n")
        test_task7_evidence_validation(tmp / "o")
        test_task7_frozen_path(tmp / "p")
        test_task8_skipped(tmp / "q")
        test_task8_fixtures(tmp / "r")
        test_task8_terminal_gate(tmp / "s")
        test_task8_frozen_path(tmp / "t")
        test_task8_spent_once(tmp / "u")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
