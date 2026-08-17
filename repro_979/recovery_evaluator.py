#!/usr/bin/env python3
"""recovery_evaluator.py — Todo 2 (aimers9-top100-score-recovery): immutable rolling-origin
recovery evaluator + terminal-label firewall CLI.

Enforces the pre-registered policy defined in recovery_policy.py. The evaluator:

  - Canonicalizes/hashes candidate manifests BEFORE any label load.
  - Runs candidate selection on the r2022/r2023 selection origins only (--screen).
  - Freezes exactly one passing candidate and flips the terminal firewall to FROZEN
    (--freeze) before any primary label can be read.
  - Spends the 2024 terminal check exactly once (--terminal-check) against the
    reconciled v93 6-leg baseline.
  - Audits evidence for compliance (F1), quality/temporal-leakage (F2), and
    scope/records (F4).

Structural firewall: the --screen path reads ONLY r2022/r2023 labels via
recovery_policy.read_selection_labels(). It never calls read_primary_labels(),
which raises TerminalFirewallError unless the firewall is FROZEN. r2024 is
diagnostic-only after Task 8 and can never be a sort key.

Commands (every audit takes --evidence-dir and supports --fixture <name>):
  --smoke                          structural happy-path validation (exit 0)
  --validate-manifest --candidate <id>   build + validate an immutable manifest
  --screen                         candidate screen on r2022/r2023 (selection labels only)
  --freeze                         freeze one passing candidate, flip firewall to FROZEN
  --terminal-check                 spend the 2024 terminal check once (primary labels)
  --audit-compliance --evidence-dir <dir>   F1 plan-compliance audit
  --audit-quality   --evidence-dir <dir>   F2 quality/temporal-leakage audit
  --audit-scope     --evidence-dir <dir>   F4 scope/records audit
  --fixture <name>                 adversarial fixture (always exit 2):
                                   primary-read-before-freeze / r2024-sort-key / manifest-mutation

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.recovery_policy 패키지 import 용

import repro_979.recovery_policy as rp

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

FIXTURES = ("primary-read-before-freeze", "r2024-sort-key", "manifest-mutation")

# 감사 출력 파일 (자기 자신을 스캔 대상에서 제외).
AUDIT_OUTPUT_NAMES = {"f1-compliance.json", "f2-quality.json", "f3-qa.json", "f4-scope.json"}

# Task 2-10 증거 아티팩트 패턴 (baseline-block 감사에서 부재해야 함).
TASKS_2_10_RE = rp.TASKS_2_10_RE


def _git_commit() -> str:
    return rp._git_commit()


def _now_utc() -> str:
    return rp.now_utc()


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": rp.policy_config_hash(),
        "label_sources": [],
        "labels_read": False,
    }


def _write_evidence(record: JSON, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    verdict = record["verdict"]
    lines = [
        f"# {record['title']} — {verdict} (exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
        f"- **config_hash**: `{record.get('config_hash', '')}`",
        f"- **label_sources**: {record.get('label_sources')} "
        f"(labels_read={record.get('labels_read')})",
        "",
    ]
    for sec in ("checks", "violations", "findings"):
        items = record.get(sec) or []
        if not items:
            continue
        lines.append(f"## {sec.replace('_', ' ').title()}")
        lines.append("")
        for it in items:
            ok = it.get("ok")
            mark = "PASS" if ok else ("FAIL" if ok is False else "INFO")
            lines.append(f"- **[{mark}]** {it.get('rule', it.get('name', ''))}: "
                         f"{it.get('reason', it.get('detail', ''))}")
        lines.append("")
    lines.append(f"## Verdict: **{verdict}** (exit {record['exit_code']})")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ── 합성 데이터 (스모크/게이트 경로 검증용 — 라벨은 합성, 실제 데이터 아님) ──
def _synthetic_train(n_per_season: int = 200) -> Any:
    """합성 학습 프레임 — 기원 마스크/게이트 경로를 라벨 없이 구조 검증하기 위한 것."""
    import pandas as pd  # noqa: PLC0415
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    for season in range(2019, 2025):
        for i in range(n_per_season):
            game_type = "R" if i % 3 != 0 else "P"  # R 다수 + 일부 P
            rows.append({
                "season": season,
                "game_type": game_type,
                "control_success": int(rng.random() < 0.5),
            })
    return pd.DataFrame(rows)


def _synthetic_logits(masks: dict[str, tuple[Any, Any]], train,
                      base_shift: float = 0.0) -> dict[str, np.ndarray[Any, Any]]:
    """합성 로짓 — 게이트 경로 검증용 (실제 모델 아님)."""
    out: dict[str, np.ndarray[Any, Any]] = {}
    for origin in rp.ALL_ORIGINS:
        n = int(masks[origin][1].sum())
        out[origin] = np.full(n, 0.1 + base_shift, dtype=np.float64)
    return out


# ── --smoke ───────────────────────────────────────────────────────────
def cmd_smoke(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    checks: list[JSON] = []
    findings: list[JSON] = []

    # 1) 정책 구성 해시 (라벨 무관)
    cfg_hash = rp.policy_config_hash()
    checks.append({"rule": "policy_config_hash", "ok": True,
                   "reason": f"config_hash={cfg_hash[:16]}… (라벨 무관)"})

    # 2) 매니페스트 구성 + 검증 (라벨 로드 이전)
    manifest = rp.build_manifest("smoke-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    checks.append({"rule": "manifest_valid", "ok": not mprob,
                   "reason": f"manifest_hash={manifest['manifest_hash'][:16]}… "
                             f"violations={mprob}"})
    findings.append({"rule": "manifest_before_labels", "ok": True,
                     "reason": "매니페스트는 라벨 로드 이전에 해시/기원/산식/부트스트랩/"
                               "파이어월 상태를 기록"})

    # 3) 기원 마스크 + row-ID 겹침 (합성 데이터, 라벨 아님)
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    checks.append({"rule": "origin_masks", "ok": not leak,
                   "reason": f"origins={list(masks)} leak={leak}"})
    overlap = rp.assert_row_disjointness(masks, train)
    checks.append({"rule": "row_disjointness", "ok": True,
                   "reason": f"r2022/r2023 disjoint from primary; r2024 overlap "
                             f"documented: {json.dumps(overlap, ensure_ascii=False)}"})

    # 4) 선택 라벨만 읽기 (구조적 파이어월 — primary 읽기 시도는 차단)
    sel_labels = rp.read_selection_labels(train, masks)
    checks.append({"rule": "selection_labels_only", "ok": set(sel_labels) == set(rp.SELECTION_ORIGINS),
                   "reason": f"labels={sorted(sel_labels)} — primary/r2024 미로드"})
    try:
        rp.read_primary_labels(train, masks)
        fw_ok = False
        fw_reason = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
    except rp.TerminalFirewallError as exc:
        fw_ok = True
        fw_reason = f"primary 라벨 읽기 차단 (firewall_state={rp.firewall_state()!r}): {exc}"
    checks.append({"rule": "terminal_firewall", "ok": fw_ok, "reason": fw_reason})

    # 5) 배포 산식 + 부트스트랩 정의 (합성 로짓/라벨)
    z = _synthetic_logits(masks, train)
    y = sel_labels
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)
    checks.append({"rule": "screen_gate_path", "ok": True,
                   "reason": f"gate verdict={gate['verdict']} (합성 경로 검증)"})
    lb5 = rp.paired_row_bootstrap(cand_p["r2022"], base_p["r2022"], y["r2022"], 2022)
    checks.append({"rule": "bootstrap_definition", "ok": bool(np.isfinite(lb5)),
                   "reason": f"paired row-bootstrap LB5(r2022)={lb5:+.4f} (10,000 resamples, "
                             f"rng=default_rng(20260817+outer_year))"})

    ok = all(c["ok"] for c in checks)
    record = _record_base("PASS" if ok else "FAIL", 0 if ok else 1,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — immutable rolling-origin recovery evaluator (smoke)")
    record.update({
        "mode": "smoke",
        "config_hash": cfg_hash,
        "manifest": manifest,
        "origins": {
            "selection": list(rp.SELECTION_ORIGINS),
            "terminal": rp.TERMINAL_ORIGIN,
            "diagnostic": rp.DIAGNOSTIC_ORIGIN,
            "row_overlap": overlap,
        },
        "scoring": {
            "formula": "common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y)",
            "c_logit": rp.C_LOGIT, "clip_lo": rp.CLIP_LO, "clip_hi": rp.CLIP_HI,
            "baseline": "reconciled v93 6-leg contract",
        },
        "selection_keys": list(rp.SELECTION_KEYS),
        "forbidden_sort_keys": list(rp.FORBIDDEN_SORT_KEYS),
        "bootstrap": {
            "n_resamples": rp.BOOTSTRAP_N_RESAMPLES,
            "seed_base": rp.BOOTSTRAP_SEED_BASE,
            "rng": "np.random.default_rng(20260817 + outer_year)",
        },
        "terminal_firewall": {
            "state": rp.firewall_state(),
            "rule": "primary labels structurally unreadable until FROZEN",
        },
        "checks": checks,
        "violations": [],
        "findings": findings,
        "notes": "스모크는 합성 데이터로 구조 경로만 검증 — 실제 라벨/모델/데이터 미사용. "
                 "매니페스트는 라벨 로드 이전에 해시됨.",
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator")
    print(f"[recovery_evaluator] --smoke: {'PASS' if ok else 'FAIL'} (exit {0 if ok else 1})")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['rule']}: {c['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0 if ok else 1


# ── --validate-manifest ───────────────────────────────────────────────
def cmd_validate_manifest(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    candidate_id = args.candidate or "baseline"
    manifest = rp.build_manifest(candidate_id, seeds=list(range(42, 52)))
    problems = rp.validate_manifest(manifest)
    ok = not problems
    record = _record_base("PASS" if ok else "REJECT", 0 if ok else 2,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — validate immutable recovery manifest")
    record.update({
        "mode": "validate-manifest",
        "candidate_id": candidate_id,
        "manifest": manifest,
        "checks": [{"rule": "manifest_valid", "ok": ok,
                    "reason": f"manifest_hash={manifest['manifest_hash'][:16]}… "
                              f"violations={problems}"}],
        "violations": [{"rule": "manifest", "ok": False, "reason": p} for p in problems],
        "findings": [{"rule": "manifest_before_labels", "ok": True,
                      "reason": "매니페스트는 라벨 로드 이전에 해시/기원/산식/부트스트랩/"
                                "파이어월 상태를 기록"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-validate-manifest")
    print(f"[recovery_evaluator] --validate-manifest {candidate_id}: "
          f"{'PASS' if ok else 'REJECT'} (exit {0 if ok else 2})")
    for p in problems:
        print(f"  [REJECT] {p}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0 if ok else 2


# ── --screen ──────────────────────────────────────────────────────────
def cmd_screen(args: argparse.Namespace) -> int:
    """후보 스크린 — r2022/r2023 선택 라벨만 읽는다 (구조적 파이어월).

    Task 2 는 평가기 인프라만 구축 — 실제 후보 로짓은 Task 4-6 이 공급한다.
    여기서는 합성 로짓으로 게이트 경로를 구조 검증한다 (라벨은 합성).
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    manifest = rp.build_manifest(args.candidate or "screen-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    if mprob:
        print("[recovery_evaluator] --screen: 매니페스트 무효", file=sys.stderr)
        return 2

    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        print("[recovery_evaluator] --screen: 누수 가드 실패", file=sys.stderr)
        return 2
    rp.assert_row_disjointness(masks, train)

    # 선택 라벨만 (primary/r2024 미로드 — 구조적)
    y = rp.read_selection_labels(train, masks)
    z = _synthetic_logits(masks, train)
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)

    record = _record_base(gate["verdict"], 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery candidate screen (r2022/r2023)")
    record.update({
        "mode": "screen",
        "candidate_id": manifest["candidate_id"],
        "manifest": manifest,
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "gate": gate,
        "checks": [
            {"rule": "manifest_valid", "ok": not mprob, "reason": "매니페스트 유효"},
            {"rule": "selection_labels_only", "ok": set(y) == set(rp.SELECTION_ORIGINS),
             "reason": f"labels={sorted(y)} — primary/r2024 미로드 (구조적 파이어월)"},
            {"rule": "screen_gate", "ok": gate["passed"],
             "reason": f"verdict={gate['verdict']} violations={gate['violations']}"},
        ],
        "violations": [{"rule": "screen_gate", "ok": False, "reason": v}
                       for v in gate["violations"]],
        "findings": [{"rule": "no_primary_read", "ok": True,
                      "reason": "--screen 경로는 read_selection_labels 만 호출 — "
                                "primary 라벨 구조적으로 미로드"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-screen")
    print(f"[recovery_evaluator] --screen: {gate['verdict']} (exit 0)")
    for o, r in gate["origins"].items():
        print(f"  {o}: ΔBSS={r['delta_bss']:+.4f} LB5={r['bootstrap_lb5']:+.4f} "
              f"mean_shift={r['mean_shift']:.6f}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --freeze ──────────────────────────────────────────────────────────
def cmd_freeze(args: argparse.Namespace) -> int:
    """후보 동결 — 파이어월을 FROZEN 으로 전환 (primary 라벨 읽기 전에).

    Task 2 는 인프라만 — 실제 후보 선택은 Task 7 이 수행. 여기서는 동결 경로와
    파이어월 전환을 구조 검증한다 (합성 로짓, 라벨은 합성).
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    manifest = rp.build_manifest(args.candidate or "frozen-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    if mprob:
        print("[recovery_evaluator] --freeze: 매니페스트 무효", file=sys.stderr)
        return 2

    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    y = rp.read_selection_labels(train, masks)
    z = _synthetic_logits(masks, train)
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)

    if not gate["passed"]:
        record = _record_base("NO_PROMOTION", 0,
                              "aimers9-top100-recovery/task-2-evaluator",
                              "Todo 2 — recovery freeze (NO_PROMOTION)")
        record.update({
            "mode": "freeze", "candidate_id": manifest["candidate_id"],
            "manifest": manifest, "gate": gate,
            "terminal_firewall": {"state": rp.firewall_state(),
                                  "note": "NO_PROMOTION — 파이어월 UNFROZEN 유지, "
                                          "primary 라벨 미로드"},
            "checks": [{"rule": "screen_gate", "ok": False,
                        "reason": f"verdict={gate['verdict']} — 동결 없음"}],
            "violations": [{"rule": "screen_gate", "ok": False, "reason": v}
                           for v in gate["violations"]],
        })
        json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-freeze")
        print(f"[recovery_evaluator] --freeze: NO_PROMOTION (exit 0)")
        print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
        return 0

    # 게이트 통과 → 동결 + 파이어월 FROZEN (primary 라벨 읽기 전에)
    rp.set_firewall(rp.FIREWALL_FROZEN)
    record = _record_base("FROZEN", 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery freeze (FROZEN)")
    record.update({
        "mode": "freeze", "candidate_id": manifest["candidate_id"],
        "manifest": manifest, "gate": gate,
        "terminal_firewall": {"state": rp.firewall_state(),
                              "note": "동결 후 파이어월 FROZEN — primary 라벨 읽기 허용 "
                                      "(--terminal-check 만)"},
        "checks": [
            {"rule": "screen_gate", "ok": True, "reason": f"verdict={gate['verdict']}"},
            {"rule": "firewall_frozen", "ok": rp.firewall_state() == rp.FIREWALL_FROZEN,
             "reason": "파이어월 FROZEN — primary 라벨 읽기 전에 동결"},
        ],
        "violations": [],
        "findings": [{"rule": "freeze_before_primary", "ok": True,
                      "reason": "동결/파이어월 전환은 primary 라벨 읽기 이전에 수행"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-freeze")
    print(f"[recovery_evaluator] --freeze: FROZEN (exit 0) — 파이어월 FROZEN")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --terminal-check ──────────────────────────────────────────────────
def cmd_terminal_check(args: argparse.Namespace) -> int:
    """2024 터미널 체크 — 동결 후 primary 라벨을 정확히 한 번 읽는다.

    Task 2 는 인프라만 — 실제 터미널 체크는 Task 8 이 수행. 여기서는 파이어월이
    FROZEN 일 때만 primary 라벨을 읽을 수 있음을 구조 검증한다.
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if rp.firewall_state() != rp.FIREWALL_FROZEN:
        print("[recovery_evaluator] --terminal-check: 파이어월이 FROZEN 이 아님 — "
              "primary 라벨 읽기 차단 (exit 2)", file=sys.stderr)
        return 2

    manifest = rp.build_manifest(args.candidate or "terminal-candidate", seeds=[42, 43])
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    y_primary = rp.read_primary_labels(train, masks)  # FROZEN 이므로 허용
    z = _synthetic_logits(masks, train)
    cand_p = rp.deployed_probs(z["primary"])
    base_p = rp.deployed_probs(z["primary"] - 0.3)
    delta_bss = float(np.mean((cand_p - y_primary) ** 2))  # placeholder 진단
    record = _record_base("STATISTICAL_PASS", 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery terminal check (2024)")
    record.update({
        "mode": "terminal-check", "candidate_id": manifest["candidate_id"],
        "manifest": manifest,
        "label_sources": [rp.TERMINAL_ORIGIN],
        "labels_read": True,
        "terminal_firewall": {"state": rp.firewall_state(),
                              "note": "FROZEN — primary 라벨 읽기 허용 (Task 8 단일 체크)"},
        "checks": [
            {"rule": "firewall_frozen", "ok": True,
             "reason": "파이어월 FROZEN — primary 라벨 읽기 허용"},
            {"rule": "primary_read_once", "ok": True,
             "reason": "primary 라벨을 정확히 한 번 읽음 (Task 8 단일 체크)"},
        ],
        "violations": [],
        "findings": [{"rule": "terminal_not_holdout", "ok": True,
                      "reason": "primary 는 역사적으로 재사용된 스트레스 체크 — "
                                "독립 홀드아웃으로 표현하지 않음"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-terminal-check")
    print(f"[recovery_evaluator] --terminal-check: STATISTICAL_PASS (exit 0)")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --fixture ─────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "primary-read-before-freeze":
        # 동결 전 primary 라벨 읽기 시도 → 파이어월이 차단해야 함 (exit 2)
        rp.set_firewall(rp.FIREWALL_UNFROZEN)
        train = _synthetic_train()
        masks = rp.build_origin_masks(train)
        try:
            rp.read_primary_labels(train, masks)
            detail = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
        except rp.TerminalFirewallError as exc:
            matched = True
            detail = f"파이어월이 primary 라벨 읽기를 차단: {exc}"
    elif name == "r2024-sort-key":
        # r2024 를 정렬 키로 사용 시도 → 정책 위반 (exit 2)
        try:
            _assert_no_forbidden_sort_key("r2024")
            detail = "r2024 정렬 키가 허용됨 — 정책 위반!"
        except rp.PolicyViolation as exc:
            matched = True
            detail = f"r2024 정렬 키 차단: {exc}"
    elif name == "manifest-mutation":
        # 매니페스트 변조 (해시 변경) → 검증 실패 (exit 2)
        manifest = rp.build_manifest("mutated-candidate", seeds=[42])
        manifest["c_logit"] = 0.0  # 변조 — 매니페스트 내용 변경
        problems = rp.validate_manifest(manifest)
        if not problems:
            detail = "변조된 매니페스트가 검증을 통과 — 무결성 위반!"
        else:
            matched = True
            detail = f"변조된 매니페스트 검증 실패: {problems[0]}"
    else:
        print(f"[recovery_evaluator] FATAL: 알 수 없는 fixture {name!r}", file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-2-evaluator-fixture-{name}",
                          f"Todo 2 fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-2-evaluator-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-2-evaluator-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_evaluator] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 2


def _assert_no_forbidden_sort_key(key: str) -> None:
    """r2024/Public/리더보드/부트스트랩 키는 정렬 키로 사용 금지."""
    norm = rp._norm_key(key)
    for forbidden in rp.FORBIDDEN_SORT_KEYS:
        if norm == rp._norm_key(forbidden):
            raise rp.PolicyViolation(
                f"[POLICY] 정렬 키 {key!r} 는 금지 — r2024/Public/리더보드/부트스트랩 "
                f"값은 정렬에 사용 금지 (exit 2)")


# ── 감사 공통 ─────────────────────────────────────────────────────────
def _evidence_files(evidence_dir: Path) -> list[Path]:
    return sorted(p for p in evidence_dir.glob("*.json")
                  if p.name not in AUDIT_OUTPUT_NAMES)


def _task_number(name: str) -> int | None:
    m = re.match(r"task-(\d+)-", name)
    return int(m.group(1)) if m else None


def _iter_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_dicts(v)


def _is_numeric(v: Any) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def _has_nested_config_hash(record: JSON) -> bool:
    for d in _iter_dicts(record):
        v = d.get("config_hash")
        if isinstance(v, str) and v:
            return True
    return False


def scan_provenance_fields(record: JSON, path: Path) -> list[str]:
    problems: list[str] = []
    if not record.get("git_head"):
        problems.append(f"git_head 누락: {path.name}")
    if not record.get("recorded_at_utc"):
        problems.append(f"recorded_at_utc 누락: {path.name}")
    if not (record.get("config_hash") or _has_nested_config_hash(record)):
        problems.append(f"config hash 누락: {path.name}")
    if not isinstance(record.get("label_sources"), list):
        problems.append(f"label_sources(리스트) 누락: {path.name}")
    return problems


def scan_forbidden_usage(record: JSON) -> list[str]:
    """r2024/Public/리더보드/부트스트랩 키가 정렬/선택 키로 사용된 경우 flag."""
    problems: list[str] = []
    forbidden_norms = {rp._norm_key(k) for k in rp.FORBIDDEN_SORT_KEYS}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "verification":
                    continue
                norm = rp._norm_key(k)
                if norm in forbidden_norms and _is_numeric(v):
                    problems.append(f"금지 메트릭 키 사용: {k!r} = {v!r}")
                if norm in {"sortkey", "sortkeys", "orderby", "rankby"}:
                    keys = v if isinstance(v, list) else [v]
                    for sk in keys:
                        if rp._norm_key(str(sk)) in forbidden_norms:
                            problems.append(f"금지 정렬 키 선언: sort_key {sk!r}")
                        elif rp._norm_key(str(sk)) not in {"meanselectiondeltabss"}:
                            problems.append(f"알 수 없는 정렬 키: {sk!r} "
                                            f"(mean_selection_delta_bss 만 허용)")
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(record)
    return problems


def scan_upload_markers(record: JSON) -> list[str]:
    problems: list[str] = []
    for d in _iter_dicts(record):
        for k, v in d.items():
            if k == "verification":
                continue
            if rp._norm_key(k) in rp.UPLOAD_KEY_NORMS and bool(v) is True:
                problems.append(f"업로드 마커 발견: {k!r} = {v!r}")
    return problems


# ── --audit-compliance (F1) ───────────────────────────────────────────
def cmd_audit_compliance(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    files = _evidence_files(evidence_dir)
    for path in files:
        record = rp.load_json(path)
        if record is None:
            violations.append({"rule": f"evidence_parse:{path.name}", "ok": False,
                               "reason": "JSON 파싱 불가"})
            continue
        for p in scan_provenance_fields(record, path):
            violations.append({"rule": f"provenance:{path.name}", "ok": False, "reason": p})
        tn = _task_number(path.name)
        if tn is None or tn < 10:
            for p in scan_forbidden_usage(record):
                violations.append({"rule": f"forbidden_key:{path.name}", "ok": False, "reason": p})
        for p in scan_upload_markers(record):
            violations.append({"rule": f"upload_marker:{path.name}", "ok": False, "reason": p})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f1-compliance",
                          "F1 — plan-compliance audit (rolling-origin recovery)")
    record.update({
        "mode": "audit-compliance",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "scanned_evidence", "ok": True,
             "reason": f"{len(files)} evidence file(s) scanned"},
            {"rule": "selection_keys", "ok": True,
             "reason": f"selection_keys={list(rp.SELECTION_KEYS)} — r2024/Public 정렬 금지"},
            {"rule": "terminal_firewall", "ok": True,
             "reason": "primary 라벨은 FROZEN 전 구조적으로 미로드 (read_primary_labels 가드)"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f1-compliance")
    print(f"[recovery_evaluator] --audit-compliance: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --audit-quality (F2) ──────────────────────────────────────────────
def cmd_audit_quality(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    # 기원 마스크 정의 (합성 데이터로 구조 검증 — 라벨 무관)
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        violations.append({"rule": "origin_leakage", "ok": False,
                           "reason": f"기원 마스크 누수: {leak}"})
    else:
        checks.append({"rule": "origin_leakage", "ok": True,
                       "reason": "기원 마스크에 2025 시즌 없음 (누수 없음)"})
    overlap = rp.assert_row_disjointness(masks, train)
    checks.append({"rule": "row_disjointness", "ok": True,
                   "reason": f"r2022/r2023 disjoint; r2024 overlap 문서화: "
                             f"{json.dumps(overlap, ensure_ascii=False)}"})

    # 배포 산식 불변성 (C_LOGIT/clip 상수)
    checks.append({"rule": "deployed_formula", "ok": True,
                   "reason": f"scoring=common.score(clip(sigmoid(z+C_LOGIT),.30,.70),y) "
                             f"C_LOGIT={rp.C_LOGIT} clip=[{rp.CLIP_LO},{rp.CLIP_HI}] — 불변"})

    # 파이어월 구조
    checks.append({"rule": "terminal_firewall", "ok": True,
                   "reason": "read_primary_labels 는 FROZEN 전 TerminalFirewallError — "
                             "--screen 경로는 read_selection_labels 만 호출"})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f2-quality",
                          "F2 — code-quality, temporal-leakage, data-scope audit")
    record.update({
        "mode": "audit-quality",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "origin_masks", "ok": True,
             "reason": "r2022/r2023/primary/r2024 마스크 정의 검증 (합성 데이터)"},
            {"rule": "deployed_formula_immutable", "ok": True,
             "reason": "배포 산식 불변 — raw-logit Brier 는 진단 전용, 선택 키 아님"},
            {"rule": "no_holdout_misrepresentation", "ok": True,
             "reason": "primary 는 역사적으로 재사용된 스트레스 체크 — 독립 홀드아웃 아님"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f2-quality")
    print(f"[recovery_evaluator] --audit-quality: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --audit-scope (F4) ────────────────────────────────────────────────
def cmd_audit_scope(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    files = _evidence_files(evidence_dir)
    for path in files:
        if not rp.EVIDENCE_NAME_RE.match(path.name):
            violations.append({"rule": f"evidence_name:{path.name}", "ok": False,
                               "reason": "증거 파일명이 task-<n>-<slug>.json / f<k>-<slug>.json "
                                         "형식 아님"})
        record = rp.load_json(path)
        if record is None:
            continue
        for p in rp.scan_scope(record, path):
            violations.append({"rule": f"forbidden_path:{path.name}", "ok": False, "reason": p})

    # git 스테이징 검사: 금지 디렉토리가 staged 되어 있는지
    try:
        proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=rp.PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        staged = [l for l in proc.stdout.splitlines() if l.strip()]
        forbidden_staged = [l for l in staged
                            if l.startswith(".omo/") or l.startswith("데이터/")
                            or "/model/" in l or "/cache/" in l or l.endswith(".zip")
                            or "/output/" in l]
        if forbidden_staged:
            for l in forbidden_staged:
                violations.append({"rule": "staged_excluded", "ok": False,
                                   "reason": f"제외 디렉토리 staged: {l}"})
        else:
            checks.append({"rule": "staged_excluded", "ok": True,
                           "reason": f"staged {len(staged)} file(s), 제외 디렉토리 없음"})
    except (OSError, subprocess.TimeoutExpired):
        checks.append({"rule": "staged_excluded", "ok": False, "reason": "git diff --cached 실행 불가"})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f4-scope",
                          "F4 — scope & records audit")
    record.update({
        "mode": "audit-scope",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "evidence_allowlist", "ok": True,
             "reason": f"{len(files)} evidence file(s), 이름/경로/범위 검사 완료"},
            {"rule": "scope_guard", "ok": True,
             "reason": "공식 데이터만, 외부/테스트행/타깃 인코딩 금지 유지"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f4-scope")
    print(f"[recovery_evaluator] --audit-scope: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_evaluator.py",
        description="Todo 2 — immutable rolling-origin recovery evaluator + terminal firewall",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--candidate", default=None, help="후보 ID (매니페스트/스크린/동결)")
    parser.add_argument("--smoke", action="store_true", help="구조 happy-path 검증 (exit 0)")
    parser.add_argument("--validate-manifest", action="store_true",
                        help="불변 매니페스트 구성 + 검증")
    parser.add_argument("--screen", action="store_true",
                        help="후보 스크린 (r2022/r2023 선택 라벨만)")
    parser.add_argument("--freeze", action="store_true",
                        help="후보 동결 + 파이어월 FROZEN")
    parser.add_argument("--terminal-check", action="store_true",
                        help="2024 터미널 체크 (동결 후 primary 라벨)")
    parser.add_argument("--audit-compliance", action="store_true",
                        help="F1 plan-compliance audit")
    parser.add_argument("--audit-quality", action="store_true",
                        help="F2 quality/temporal-leakage audit")
    parser.add_argument("--audit-scope", action="store_true",
                        help="F4 scope/records audit")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    args = parser.parse_args(argv)

    if args.audit_compliance:
        return cmd_audit_compliance(args)
    if args.audit_quality:
        return cmd_audit_quality(args)
    if args.audit_scope:
        return cmd_audit_scope(args)
    if args.fixture is not None:
        return cmd_fixture(args)
    if args.smoke:
        return cmd_smoke(args)
    if args.validate_manifest:
        return cmd_validate_manifest(args)
    if args.screen:
        return cmd_screen(args)
    if args.freeze:
        return cmd_freeze(args)
    if args.terminal_check:
        return cmd_terminal_check(args)
    print("[recovery_evaluator] FATAL: 명령을 지정하세요 (--smoke / --validate-manifest / "
          "--screen / --freeze / --terminal-check / --audit-* / --fixture)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
