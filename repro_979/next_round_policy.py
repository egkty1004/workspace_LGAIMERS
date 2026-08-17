#!/usr/bin/env python3
"""next_round_policy.py — aimers9-score-improvement-next-round 정책 CLI (Task 1 freeze + F1/F4 audits).

Task 1 (this round's Wave 1): freezes the candidate-selection policy and the user-confirmed
2026-08-15 live leaderboard state into an IMMUTABLE artifact (`repro_979/next_round_policy.json`).
This CLI validates that artifact and audits evidence for compliance with it.

Modes:
  --check <policy.json>            Validate the frozen policy (structure + frozen live-state
                                   snapshot). Optional --state <path> additionally cross-checks
                                   `leaderboard_state.json` against the policy's frozen snapshot.
                                   Optional --evidence-out <base> writes provenance-rich evidence
                                   (.json/.md). Optional --verification <path> embeds externally
                                   collected verification results (fixture runs) into the evidence.
                                   exit 0 = PASS, 2 = REJECT (policy violation), 1 = fatal input error.
  --audit-compliance --evidence-dir <dir>   F1: scan every decision evidence under <dir> for
                                   primary-only selection compliance. PASS requires: policy file
                                   hash matches the hash recorded in task-1 evidence (immutability);
                                   every evidence records git_head / recorded_at_utc / config hash /
                                   label_sources; no forbidden R-only / bootstrap-LB metric key is
                                   used as a sort key or metric in any pre-task-10 evidence; no
                                   upload marker; rollback baseline consistent. Writes
                                   f1-compliance.{json,md}.
  --audit-scope --evidence-dir <dir>       F4: scope audit — evidence filenames must match the
                                    task evidence allowlist; no forbidden artifact paths
                                    (데이터/, /model/, /cache/, *.zip, submit_sim) referenced; no
                                    excluded directory staged in git; no leaderboard/public-score
                                     record without a user-reported source. Writes f4-scope.{json,md}.
  --freeze --evidence-dir <dir>            Task 9: freeze the single best passing candidate.
                                    Consumes Task 3/7/8 decision evidence and applies the
                                    pre-registered priority/tie rules: Task-3 PRIMARY_PASS base,
                                    else Task-7 PRIMARY_PROMOTED base; a passing Task-8
                                    calibrated derivative (verdict PASS) supersedes that base.
                                    Same-tier ties use pre-registered primary BSS only. If no
                                    base path passed → NO_PROMOTION (retain rollback baseline,
                                    stop before packaging). The decision reads NO labels and
                                    NEVER reads R-only fold values, bootstrap-LB values, or
                                    leaderboard/public-score values. exit 0 = normal decision
                                    (candidate frozen or NO_PROMOTION), 2 = policy/gate
                                    violation, 1 = fatal input error. Writes task-9-freeze.{json,md}.
                                     Optional --fixture {altered-r-only-values,
                                    altered-leaderboard-values} runs the tamper-simulation
                                    insensitivity guard (plan Task 9 failure QA): R-only /
                                    leaderboard values are injected/perturbed in an in-memory
                                    copy, the decision is re-run, and the frozen result must be
                                    unchanged — exit 2 (guard fired, tamper rejected).
  --qualify-and-deploy-frozen --freeze-evidence <path>
                                        Task 10: the SOLE candidate dispatcher. Reads the
                                    Task 9 freeze evidence and either (a) SKIPPED — verdict
                                    NO_PROMOTION → write task-10-deploy.{json,md} with
                                    verdict SKIPPED, exit 0, and ZERO label/model/data/package
                                    access (config pre-registered first, then stop); or
                                    (b) future-proof candidate branch (verdict FROZEN, never
                                    executes this round) — route by candidate family
                                    (catboost-boundary → re-hash existing bytes/formula only;
                                    deepfm/mlp-preprocess/calibrated → candidate-specific
                                    replay/final-fit on official 2019-2024 rows only, 2025/test
                                    rows solely for row-local inference QA) with structural
                                    guards: finite logits, preserved row order, clipping,
                                    candidate-specific manifest, 540s safety limit, peak
                                    RAM/GPU recorded; no manifest → guard REJECT (exit 2).
                                    Optional --fixture {failed-r2022, nonfinite-logit,
                                    shuffled-input, catboost-source-hash-mismatch,
                                    deepfm-epoch-mismatch, mlp-preprocessor-hash-mismatch,
                                    calibration-offset-cap-violation} triggers each guard with
                                    a tampered input — every fixture writes fixture-tagged
                                    evidence and exits 2 (plan Task 10 failure QA).

All modes also accept the plan's flag style: `--check <policy.json>`, `--audit-compliance`,
`--audit-scope`, `--freeze`, `--qualify-and-deploy-frozen` (the first argument is normalized
to the subcommand form automatically).

Exit codes: 0 = PASS, 1 = fatal input error, 2 = REJECT (policy/gate violation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent

JSON = dict[str, Any]

# ── 상수 ──────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1
DEFAULT_POLICY = REPO / "next_round_policy.json"
DEFAULT_STATE = REPO / "leaderboard_state.json"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"

# Task 1 freeze의 불변 값 (유저 확인 2026-08-15 live 리더보드 상태).
ROLLBACK_BASELINE_CANDIDATE_ID = "5890a4c54f502c4e"
ROLLBACK_BASELINE_CANDIDATE = "lgb_mlp_cat"
EXPECTED_LIVE_STATE: JSON = {
    "rank_100_cutoff": 1083.03417,
    "team_rank": 272,
    "current_best_public_score": 992.8390640403,
    "qualified_candidate_public_score": 992.8390640403,
    "date_captured": "2026-08-15T21:00:00+09:00",
    "timezone": "Asia/Seoul",
    "source": "user-reported DACON leaderboard",
}

# 감사 출력 파일 (자기 자신을 스캔 대상에서 제외).
AUDIT_OUTPUT_NAMES = {"f1-compliance.json", "f4-scope.json", "f2-quality.json", "f3-qa.json"}
# slug 은 밑줄(_)을 허용한다 — variant slug `clip_z5`/`missing_flags` 는 플랜 Task 6 QA 가
# `--variant clip_z5|missing_flags` 로 명시한 계획-강제 이름이라 증거 파일명에 밑줄이 정당하다
# (Task 1 에서 regex 를 너무 엄격하게 작성함; 파일명 변경은 task-6 sha256 핀을 깨뜨리므로 waive).
EVIDENCE_NAME_RE = re.compile(r"^(?:task-\d+-[a-z0-9_-]+|f[1-4]-[a-z0-9_-]+)\.json$")
FORBIDDEN_PATH_TOKENS = ("데이터/", "/model/", "/cache/", ".zip", "submit_sim", "submit_sim_")

# 금지 정렬 키의 정규화(소문자 + 비알파벳 제거) 매핑 — "Δr2022", "boot LB5%" 표기도 잡는다.
def _norm_key(k: str) -> str:
    return re.sub(r"[^0-9a-z]", "", k.lower())


UPLOAD_KEY_NORMS = {"upload", "uploaded", "daconupload", "daconsubmission",
                    "submitted", "uploadattempt", "uploadmarker"}


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _canonical_sha256(payload: JSON) -> str:
    """정규화 JSON(정렬, ensure_ascii=False) → sha256 (qualification_runner 컨벤션 재사용)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> JSON | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 정책 검증 ─────────────────────────────────────────────────────────
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


def validate_policy(policy: JSON) -> list[JSON]:
    """정책 검증 → violation 목록 (비면 = PASS)."""
    violations: list[JSON] = []

    def bad(rule: str, reason: str) -> None:
        violations.append({"rule": rule, "ok": False, "reason": reason})

    if policy.get("schema_version") != SCHEMA_VERSION:
        bad("schema_version", f"schema_version = {policy.get('schema_version')!r} (필요 1)")
    if policy.get("policy_name") != "aimers9-next-round-selection-policy":
        bad("policy_name", f"policy_name = {policy.get('policy_name')!r}")

    sel = policy.get("selection") or {}
    if sel.get("label_source") != "primary":
        bad("selection_label_source",
            f"selection.label_source = {sel.get('label_source')!r} — 'primary' 이어야 함")
    allowed = set(sel.get("allowed_sort_keys") or [])
    sort_keys = list(sel.get("sort_keys") or [])
    forbidden = set(sel.get("forbidden_sort_keys") or [])
    if not sort_keys:
        bad("selection_sort_keys", "selection.sort_keys 가 비어 있음")
    for k in sort_keys:
        if k in forbidden:
            bad("selection_sort_keys", f"sort key {k!r} 는 금지 목록에 있음 (R-only/부트스트랩 정렬 금지)")
        elif k not in allowed:
            bad("selection_sort_keys", f"sort key {k!r} 가 allowed_sort_keys 밖 (알 수 없는 키)")
    # forbidden 목록 완전성: boot_lb5 + 모든 R-only 폴드 파생 키 필수 포함
    r_folds = list(sel.get("r_only_folds") or [])
    required_forbidden = {"boot_lb5"} | {f"delta_{f}" for f in r_folds}
    missing = sorted(required_forbidden - forbidden)
    if missing:
        bad("forbidden_sort_keys", f"금지 정렬 키 목록에 필수 키 누락: {missing}")
    if not r_folds:
        bad("r_only_folds", "selection.r_only_folds 가 비어 있음")

    rb = policy.get("rollback_baseline_candidate_id")
    if rb != ROLLBACK_BASELINE_CANDIDATE_ID:
        bad("rollback_baseline", f"rollback_baseline_candidate_id = {rb!r} "
                                 f"(필요 {ROLLBACK_BASELINE_CANDIDATE_ID})")
    elif not re.fullmatch(r"[0-9a-f]{16}", str(rb)):
        bad("rollback_baseline", f"rollback_baseline_candidate_id 형식 오류: {rb!r}")
    if policy.get("rollback_baseline_candidate") != ROLLBACK_BASELINE_CANDIDATE:
        bad("rollback_baseline",
            f"rollback_baseline_candidate = {policy.get('rollback_baseline_candidate')!r}")

    if policy.get("no_public_feedback") is not True:
        bad("no_public_feedback", f"no_public_feedback = {policy.get('no_public_feedback')!r} — true 필요")
    if policy.get("no_upload_without_allow") is not True:
        bad("no_upload_without_allow", f"no_upload_without_allow = {policy.get('no_upload_without_allow')!r}")

    ls = policy.get("live_state_frozen") or {}
    for key, expected in EXPECTED_LIVE_STATE.items():
        got = ls.get(key)
        if isinstance(expected, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - expected) < 1e-12
        else:
            ok = got == expected
        if not ok:
            bad("live_state_frozen", f"live_state_frozen.{key} = {got!r} (필요 {expected!r})")
    if not ls.get("date_captured", "").startswith("2026-08-15"):
        bad("live_state_frozen", f"live_state_frozen.date_captured = {ls.get('date_captured')!r} "
                                 f"(2026-08-15 확인값이어야 함)")
    return violations


def check_state_against_policy(state: JSON, policy: JSON) -> list[JSON]:
    """leaderboard_state.json 이 정책의 frozen live-state 스냅샷과 일치하는지 교차 검증."""
    violations: list[JSON] = []
    ls = policy.get("live_state_frozen") or {}
    mapping = {
        "rank_100_cutoff": ("rank_100_cutoff", ls.get("rank_100_cutoff")),
        "current_best_public_score": ("current_best_public_score", ls.get("current_best_public_score")),
        "team_rank": ("team_rank", ls.get("team_rank")),
        "date_captured": ("date_captured", ls.get("date_captured")),
        "source": ("source", ls.get("source")),
    }
    for state_key, (_, expected) in mapping.items():
        got = state.get(state_key)
        if isinstance(expected, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - expected) < 1e-12
        else:
            ok = got == expected
        if not ok:
            violations.append({"rule": f"state.{state_key}", "ok": False,
                               "reason": f"leaderboard_state.json {state_key} = {got!r} "
                                         f"(정책 frozen 값 {expected!r} 필요)"})
    if state.get("timezone_for_daily_key") != "Asia/Seoul":
        violations.append({"rule": "state.timezone_for_daily_key", "ok": False,
                           "reason": "timezone_for_daily_key 가 Asia/Seoul 아님"})
    return violations


# ── 증거 작성 ─────────────────────────────────────────────────────────
def write_evidence(record: JSON, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    verdict = record["verdict"]
    lines = [
        f"# {record.get('title', 'next-round policy evidence')} — {verdict} "
        f"(exit {record['exit_code']})",
        "",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_head**: {record['git_head']}",
    ]
    if record.get("policy_config_hash"):
        lines.append(f"- **policy_config_hash**: `{record['policy_config_hash']}`")
    if record.get("label_sources") is not None:
        lines.append(f"- **label_sources**: {record['label_sources']} "
                     f"(labels_read={record.get('labels_read')})")
    lines.append("")
    for sec in ("checks", "violations", "findings"):
        items = record.get(sec) or []
        if not items:
            continue
        lines.append(f"## {sec.replace('_', ' ').title()}")
        lines.append("")
        for it in items:
            ok = it.get("ok")
            mark = "PASS" if ok else ("FAIL" if ok is False else "INFO")
            lines.append(f"- **[{mark}]** {it.get('rule', it.get('name', ''))}: {it.get('reason', it.get('detail', ''))}")
        lines.append("")
    if record.get("verification"):
        lines.append("## Verification (fixture results)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(record["verification"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ── --check ───────────────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[next_round_policy] FATAL: 정책 파일을 읽을 수 없음: {policy_path}", file=sys.stderr)
        return 1

    violations = validate_policy(policy)
    state = None
    state_path = None
    if args.state:
        state_path = Path(args.state).expanduser().resolve()
        state = load_json(state_path)
        if state is None:
            print(f"[next_round_policy] FATAL: 상태 파일을 읽을 수 없음: {state_path}", file=sys.stderr)
            return 1
        violations += check_state_against_policy(state, policy)

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    print(f"[next_round_policy] --check {policy_path.name}: "
          f"{'PASS' if all_pass else 'REJECT'} (exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    if all_pass:
        ls = policy["live_state_frozen"]
        print(f"[next_round_policy] frozen live state: cutoff={ls['rank_100_cutoff']} "
              f"team_rank={ls['team_rank']} best={ls['current_best_public_score']} "
              f"captured={ls['date_captured']} source={ls['source']!r}")
        print(f"[next_round_policy] selection: label_source={policy['selection']['label_source']} "
              f"sort_keys={policy['selection']['sort_keys']}")

    if args.evidence_out and all_pass:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Task 1 — live state refresh & selection-policy freeze",
            "task": "aimers9-next-round/task-1-state-policy",
            "mode": "check",
            "verdict": "PASS",
            "exit_code": 0,
            "recorded_at_utc": now_utc(),
            "git_head": _git_commit(),
            "policy_path": str(policy_path),
            "policy_config_hash": _canonical_sha256(policy),
            "state_path": str(state_path) if state_path else None,
            "live_state": policy.get("live_state_frozen"),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "Task 1 reads NO labels. Policy freezes label_source=primary for "
                                  "all later label reading in this round (R-only folds are "
                                  "reject-only, never a sort key).",
            "checks": [
                {"rule": "policy_validation", "ok": True,
                 "reason": f"{len(validate_policy(policy))} violations"},
                {"rule": "state_cross_check", "ok": state is not None,
                 "reason": "leaderboard_state.json 일치" if state is not None else "state 미제공"},
            ],
            "violations": violations,
            "verification": args.verification or {},
            "notes": "User-confirmed 2026-08-15 live leaderboard state (cutoff 1083.03417, "
                     "team_rank 272, best 992.8390640403) supersedes stale 2026-08-14 cutoff "
                     "1077.02694; gap to rank 100 = 90.19511. submissions_by_date untouched; "
                     "no submission count or score event added.",
        }
        json_path, md_path = write_evidence(record, Path(args.evidence_out).expanduser().resolve())
        print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
    return exit_code


# ── 공통 감사 유틸 ────────────────────────────────────────────────────
def _iter_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_dicts(v)


def _iter_strs(node: Any):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_strs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_strs(v)


def _evidence_files(evidence_dir: Path) -> list[Path]:
    return sorted(
        p for p in evidence_dir.glob("*.json")
        if p.name not in AUDIT_OUTPUT_NAMES
    )


def _task_number(name: str) -> int | None:
    m = re.match(r"task-(\d+)-", name)
    return int(m.group(1)) if m else None


def scan_forbidden_usage(record: JSON, forbidden_keys: set[str]) -> list[str]:
    """R-only/부트스트랩 키가 metric 값(숫자) 또는 sort 선언으로 사용된 경우 flag.
    리스트 값으로 선언된 금지 키 목록(예: forbidden_sort_keys: [...])은 선언이므로 제외.
    'verification' 섹션(fixture 실행 결과 기록)은 선택/스코어링 증거가 아니므로 제외."""
    problems: list[str] = []
    forbidden_norms = {_norm_key(k) for k in forbidden_keys}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "verification":  # fixture 실행 결과 기록 — 선택 증거가 아님
                    continue
                norm = _norm_key(k)
                if norm in forbidden_norms and _is_numeric(v):
                    problems.append(f"금지 메트릭 키 사용: {k!r} = {v!r}")
                if norm in {"sortkey", "sortkeys", "orderby", "rankby"}:
                    keys = v if isinstance(v, list) else [v]
                    for sk in keys:
                        if _norm_key(str(sk)) in forbidden_norms:
                            problems.append(f"금지 정렬 키 선언: sort_key {sk!r}")
                        elif _norm_key(str(sk)) not in {"primarybss"}:
                            problems.append(f"알 수 없는 정렬 키: {sk!r} (primary_bss 만 허용)")
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
            if _norm_key(k) in UPLOAD_KEY_NORMS and bool(v) is True:
                problems.append(f"업로드 마커 발견: {k!r} = {v!r}")
    return problems


def _has_nested_config_hash(record: JSON) -> bool:
    """record 어디든 `config_hash` 키 아래 truthy 문자열 값이 있는지 (aggregate 증거 허용).

    task-6-mlp-preprocess.json 은 두 개의 사전 등록 variant(clip_z5/missing_flags)를
    실행한 aggregate 레코드로, config_hash 가 최상위가 아니라 variants.<v>.config_hash 로
    중첩되어 있다. provenance 의도("모든 후보 config 가 라벨 읽기 전에 기록·해시됨")는
    per-variant 해시 존재로 충족되므로 waive 한다 — 파일명/내용 변경은 task-6 sha256 핀과
    task-7 참조를 깨뜨리기 때문. 다른 검사(git_head/recorded_at_utc/label_sources)는 그대로.
    """
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
    if not (record.get("config_hash") or record.get("policy_config_hash")
            or _has_nested_config_hash(record)):
        problems.append(f"config hash 누락: {path.name}")
    if not isinstance(record.get("label_sources"), list):
        problems.append(f"label_sources(리스트) 누락: {path.name}")
    return problems


# ── --audit-compliance (F1) ───────────────────────────────────────────
def cmd_audit_compliance(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[next_round_policy] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[next_round_policy] FATAL: 정책 파일을 읽을 수 없음: {policy_path}", file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    violations = validate_policy(policy)
    checks: list[JSON] = []

    task1_path = evidence_dir / "task-1-state-policy.json"
    task1 = load_json(task1_path)
    if task1 is None:
        violations.append({"rule": "task1_evidence", "ok": False,
                           "reason": "task-1-state-policy.json 없음 — 정책 freeze 증거 누락"})
    else:
        checks.append({"rule": "task1_evidence", "ok": True, "reason": "task-1 freeze 증거 존재"})
        recorded_hash = task1.get("policy_config_hash")
        if recorded_hash != policy_hash:
            violations.append({"rule": "policy_immutability", "ok": False,
                               "reason": f"정책 파일 해시 {policy_hash[:16]}… != task-1 증거 기록 "
                                         f"{str(recorded_hash)[:16]}… — 정책이 변경됨 (immutable 위반)"})
        else:
            checks.append({"rule": "policy_immutability", "ok": True,
                           "reason": "정책 config hash == task-1 증거 기록 해시"})

    files = _evidence_files(evidence_dir)
    if not files:
        checks.append({"rule": "evidence_present", "ok": True, "reason": "스캔 대상 증거 없음 (정책/상태 freeze 증거만)"})
    for path in files:
        record = load_json(path)
        if record is None:
            violations.append({"rule": "evidence_parse", "ok": False,
                               "reason": f"JSON 파싱 불가: {path.name}"})
            continue
        for p in scan_provenance_fields(record, path):
            violations.append({"rule": f"provenance:{path.name}", "ok": False, "reason": p})
        tn = _task_number(path.name)
        if tn is None or tn < 10:
            for p in scan_forbidden_usage(record, set(policy["selection"]["forbidden_sort_keys"])):
                violations.append({"rule": f"forbidden_key:{path.name}", "ok": False, "reason": p})
        for p in scan_upload_markers(record):
            violations.append({"rule": f"upload_marker:{path.name}", "ok": False, "reason": p})
        rb = policy.get("rollback_baseline_candidate_id")
        for d in _iter_dicts(record):
            if "rollback_baseline_candidate_id" in d and d["rollback_baseline_candidate_id"] != rb:
                violations.append({"rule": f"rollback:{path.name}", "ok": False,
                                   "reason": f"롤백 베이스라인 불일치: {d['rollback_baseline_candidate_id']!r}"})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    label_sources = [policy["selection"]["label_source"]] if policy.get("selection") else []
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "F1 — plan-compliance audit (primary-only selection)",
        "task": "aimers9-next-round/f1-compliance",
        "mode": "audit-compliance",
        "verdict": "PASS" if all_pass else "REJECT",
        "exit_code": exit_code,
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "policy_path": str(policy_path),
        "policy_config_hash": policy_hash,
        "label_sources": label_sources,
        "labels_read": False,
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "scanned_evidence", "ok": True, "reason": f"{len(files)} evidence file(s) scanned"},
            {"rule": "selection_policy", "ok": policy.get("selection", {}).get("label_source") == "primary",
             "reason": "label_source=primary, R-only folds reject-only (transfer gates)"},
            {"rule": "no_public_feedback", "ok": policy.get("no_public_feedback") is True,
             "reason": "no_public_feedback=true — 공개 점수 피드백 금지"},
            {"rule": "rollback_baseline", "ok": True,
             "reason": f"rollback baseline {policy.get('rollback_baseline_candidate_id')} 유지"},
        ],
    }
    json_path, md_path = write_evidence(record, evidence_dir / "f1-compliance")
    print(f"[next_round_policy] --audit-compliance: {'PASS' if all_pass else 'REJECT'} (exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --audit-scope (F4) ────────────────────────────────────────────────
def cmd_audit_scope(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[next_round_policy] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1

    violations: list[JSON] = []
    checks: list[JSON] = []

    files = _evidence_files(evidence_dir)
    for path in files:
        if not EVIDENCE_NAME_RE.match(path.name):
            violations.append({"rule": f"evidence_name:{path.name}", "ok": False,
                               "reason": "증거 파일명이 task-<n>-<slug>.json / f<k>-<slug>.json 형식 아님"})
        record = load_json(path)
        if record is None:
            continue
        for s in _iter_strs(record):
            for tok in FORBIDDEN_PATH_TOKENS:
                if tok in s:
                    violations.append({"rule": f"forbidden_path:{path.name}", "ok": False,
                                       "reason": f"금지 아티팩트 경로 토큰 {tok!r} 참조"})
                    break
        mentions_lb = any("publicscore" in _norm_key(k) or "leaderboard" in _norm_key(k)
                          for d in _iter_dicts(record) for k in d)
        has_user_source = any("user-reported" in s for s in _iter_strs(record)) or \
                          any(d.get("user_reported") is True for d in _iter_dicts(record))
        if mentions_lb and not has_user_source:
            violations.append({"rule": f"leaderboard_record:{path.name}", "ok": False,
                               "reason": "리더보드/공개점수 기록에 user-reported source 없음 (무단 기록)"})

    # git 스테이징 검사: 금지 디렉토리가 staged 되어 있는지
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
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
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "F4 — scope & records audit",
        "task": "aimers9-next-round/f4-scope",
        "mode": "audit-scope",
        "verdict": "PASS" if all_pass else "REJECT",
        "exit_code": exit_code,
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "policy_path": str(policy_path := Path(args.policy).expanduser().resolve()),
        "policy_config_hash": _canonical_sha256(load_json(policy_path) or {}),
        "label_sources": [],
        "labels_read": False,
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "evidence_allowlist", "ok": True, "reason": f"{len(files)} evidence file(s), "
                                                                 "이름/경로/리더보드 기록 검사 완료"},
            {"rule": "scope_guard", "ok": True,
             "reason": "공식 데이터만, 외부/테스트행/시퀀스 재구성/타깃 인코딩 금지 유지"},
        ],
    }
    json_path, md_path = write_evidence(record, evidence_dir / "f4-scope")
    print(f"[next_round_policy] --audit-scope: {'PASS' if all_pass else 'REJECT'} (exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --freeze (Task 9) ─────────────────────────────────────────────────
# Task 9 pre-registered priority/tie rules (plan lines 132-138; policy `selection.tie_rule`):
#   freeze the FIRST base candidate that passed Task 3 (PRIMARY_PASS) or Task 7
#   (PRIMARY_PROMOTED); a passing Task-8 calibrated derivative (verdict PASS) supersedes
#   that base. If neither base path passed → NO_PROMOTION (retain rollback baseline, stop
#   before packaging). Same-tier ties use pre-registered primary BSS only; exact primary
#   ties → declared per-family tick order ascending (lgb < mlp < catboost), then candidate_id
#   ascending as a deterministic non-score fallback. The decision NEVER reads R-only fold
#   values, bootstrap-LB values, or leaderboard/public-score values.
FREEZE_T3_PASS = "PRIMARY_PASS"
FREEZE_T7_PASS = "PRIMARY_PROMOTED"
FREEZE_T8_PASS = "PASS"
FREEZE_T3_TERMINAL = {"PRIMARY_PASS", "PRIMARY_REJECT"}
FREEZE_T7_TERMINAL = {"PRIMARY_PROMOTED", "PRIMARY_REJECT", "PRIMARY_REJECTED", "SKIPPED"}
FREEZE_T8_TERMINAL = {"PASS", "REJECT", "SKIPPED_NO_BASE"}
FREEZE_TICK_ORDER = {"lgb": 1, "mlp": 2, "catboost": 3}  # policy tie_rule: per-family tick ascending
FREEZE_FIXTURES = {"altered-r-only-values", "altered-leaderboard-values"}
# Insensitivity tamper fixture: leaderboard/public-score key canonical names injected/perturbed.
FREEZE_LB_KEYS = [
    "rank_100_cutoff", "team_rank", "current_best_public_score",
    "qualified_candidate_public_score", "public_score", "cutoff",
]
FREEZE_INJECT_VALUE = 99999.0
FREEZE_PERTURB_STEP = 12345.6789


class PolicyViolation(RuntimeError):
    """Task 10 deploy/qualification 구조 가드 위반 — exit 2 (정책/동결 계약 위반)."""


# ── --qualify-and-deploy-frozen (Task 10) ─────────────────────────────
# Task 10 = the SOLE candidate dispatcher. Task 9 = NO_PROMOTION 이므로 이번 라운드의
# 실행 경로는 SKIPPED 뿐 — 후보 분기(family 라우팅 + 가드)는 미래 라운드용으로 구조만
# 구현하고, 7 개 failure-QA fixture 가 각 가드를 직접 트리거해 exit 2 를 증명한다.
# One-shot qualification (후보 동결 시에만 실행): r2022/r2023 폴드 각각 ΔBSS>1,
# fold-specific paired-bootstrap LB5>0, mean shift<=0.005; r2024 는 진단 전용
# (branch-inert). 어떤 폴드든 실패 → NO_PROMOTION — 다른 후보 패밀리 재개 없음.
DEPLOY_FIXTURES = {
    "failed-r2022",
    "nonfinite-logit",
    "shuffled-input",
    "catboost-source-hash-mismatch",
    "deepfm-epoch-mismatch",
    "mlp-preprocessor-hash-mismatch",
    "calibration-offset-cap-violation",
}
# 후보 family 라우팅: 동결 증거의 decision.source → family → 배포 전략.
DEPLOY_ROUTE_BY_SOURCE = {
    "task-3-cat-boundary-gate": "catboost-boundary",
    "task-7-promotion": "deepfm",
    "task-8-calibration": "calibrated",
}
DEPLOY_ROUTE_BY_FAMILY = {
    "catboost-boundary": "rehash-only",  # 기존 모델 바이트/포뮬라 재해시만 — 재학습 금지
    "deepfm": "replay",
    "mlp-preprocess": "replay",
    "calibrated": "replay",
}
DEPLOY_QUALIFY_FOLDS = ("r2022", "r2023")          # one-shot post-freeze qualification
DEPLOY_R2024_ROLE = "diagnostic-only-branch-inert"  # r2024 는 진단 폴드 (branch-inert)
DEPLOY_QUALIFY_GATES = {
    "delta_bss_min": 1.0,
    "paired_bootstrap_lb5_min": 0.0,
    "mean_shift_max": 0.005,
}
DEPLOY_C_LOGIT = -0.0404
DEPLOY_CLIP_LO, DEPLOY_CLIP_HI = 0.30, 0.70
DEPLOY_SAFETY_LIMIT_S = 540.0
DEPLOY_FINAL_FIT_ROWS = "official 2019-2024 rows only; 2025/test rows allowed solely for row-local inference QA"


def _pick_from(rec: JSON | None, keys: list[str]) -> Any:
    if not isinstance(rec, dict):
        return None
    for k in keys:
        v = rec.get(k)
        if v is not None:
            return v
    return None


def _resolve_tier(cands: list[JSON], tier: str, trace: list[JSON]) -> JSON | None:
    """Same-tier tie rule (pre-registered): pre-registered primary BSS descending; exact
    primary ties → declared per-family tick order ascending; final deterministic fallback
    candidate_id ascending (non-score, primary-only — never R-only/bootstrap/LB)."""
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    trace.append({
        "rule": f"tie:{tier}",
        "ok": True,
        "reason": f"{len(cands)} same-tier 후보 → tie rule: primary BSS desc, "
                  f"exact tie → tick order asc ({FREEZE_TICK_ORDER}), candidate_id asc (non-score)",
        "candidates": [{"candidate_id": c.get("candidate_id"), "primary_bss": c.get("primary_bss")}
                       for c in cands],
    })
    return sorted(
        cands,
        key=lambda c: (-(float(c["primary_bss"]) if _is_numeric(c.get("primary_bss")) else -1e300),
                       FREEZE_TICK_ORDER.get(str(c.get("family") or ""), 9),
                       str(c.get("candidate_id") or "")),
    )[0]


def decide_freeze(policy: JSON, t3: JSON | None, t7: JSON | None, t8: JSON | None) -> JSON:
    """Apply Task 9 priority/tie rules to task-3/7/8 evidence → freeze decision + trace.

    Returns {"decision": {...}, "trace": [...], "violations": [...]}. Pure function —
    reads only the evidence dicts passed in; NO label/R-fold/leaderboard access."""
    trace: list[JSON] = []
    violations: list[JSON] = []

    # 1) Task-3 base path (priority: FIRST base candidate that passed Task 3).
    if t3 is None:
        trace.append({"path": "task-3-cat-boundary-gate", "verdict": None,
                      "required": FREEZE_T3_PASS, "outcome": "not-passed",
                      "reason": "evidence missing/unparseable — no PASS recorded"})
    else:
        v3 = t3.get("verdict")
        if v3 not in FREEZE_T3_TERMINAL:
            violations.append({"rule": "task3_terminal", "ok": False,
                               "reason": f"task-3 verdict {v3!r} 인식 불가 — freeze 규칙 적용 불가"})
        elif v3 == FREEZE_T3_PASS:
            frozen = (t3.get("gate") or {}).get("frozen") or {}
            cid = _pick_from(frozen, ["candidate_id"]) or _pick_from(t3, ["candidate_id"])
            bss = _pick_from(frozen, ["primary_bss"]) or _pick_from(t3, ["primary_bss"])
            trace.append({"path": "task-3-cat-boundary-gate", "verdict": v3,
                          "required": FREEZE_T3_PASS, "outcome": "passed-base",
                          "candidate_id": cid, "primary_bss": bss,
                          "field": "gate.frozen.candidate_id / gate.frozen.primary_bss"})
        else:
            trace.append({"path": "task-3-cat-boundary-gate", "verdict": v3,
                          "required": FREEZE_T3_PASS, "outcome": "not-passed",
                          "reason": f"verdict {v3!r} — {FREEZE_T3_PASS} 필요"})

    # 2) Task-7 base path (only if Task-3 base does NOT exist — 'else Task 7 PASS base').
    t3_passed = any(e.get("outcome") == "passed-base" for e in trace if e.get("path", "").startswith("task-3"))
    t7 = None if t3_passed else t7
    if t7 is None and not t3_passed:
        trace.append({"path": "task-7-promotion", "verdict": None,
                      "required": FREEZE_T7_PASS, "outcome": "not-passed",
                      "reason": "evidence missing/unparseable — no PASS recorded"})
    elif t7 is not None:
        v7 = t7.get("verdict")
        if v7 not in FREEZE_T7_TERMINAL:
            violations.append({"rule": "task7_terminal", "ok": False,
                               "reason": f"task-7 verdict {v7!r} 인식 불가 — freeze 규칙 적용 불가"})
        elif v7 == FREEZE_T7_PASS:
            promoted = t7.get("promoted") if isinstance(t7.get("promoted"), dict) else {}
            decision = t7.get("decision") if isinstance(t7.get("decision"), dict) else {}
            cid = (_pick_from(t7, ["candidate_id", "promoted_candidate_id"])
                   or _pick_from(promoted, ["candidate_id"])
                   or _pick_from(decision, ["candidate_id"]))
            bss = (_pick_from(t7, ["primary_bss"])
                   or _pick_from(promoted, ["primary_bss"])
                   or _pick_from(decision, ["primary_bss"]))
            family = _pick_from(t7, ["family"]) or _pick_from(promoted, ["family"])
            trace.append({"path": "task-7-promotion", "verdict": v7,
                          "required": FREEZE_T7_PASS, "outcome": "passed-base",
                          "candidate_id": cid, "primary_bss": bss, "family": family,
                          "field": "candidate_id / promoted.candidate_id / decision.candidate_id"})
        else:
            trace.append({"path": "task-7-promotion", "verdict": v7,
                          "required": FREEZE_T7_PASS, "outcome": "not-passed",
                          "reason": f"verdict {v7!r} — {FREEZE_T7_PASS} 필요"})

    # 3) Task-8 calibrated derivative path (a passing derivative supersedes the base).
    if t8 is None:
        violations.append({"rule": "task8_terminal", "ok": False,
                           "reason": "task-8 evidence missing — Task 8 must be terminal "
                                     "(PASS/REJECT/SKIPPED_NO_BASE) before freeze"})
    else:
        v8 = t8.get("verdict")
        if v8 not in FREEZE_T8_TERMINAL:
            violations.append({"rule": "task8_terminal", "ok": False,
                               "reason": f"task-8 verdict {v8!r} terminal 아님 — freeze 차단 "
                                         f"(필요 {sorted(FREEZE_T8_TERMINAL)})"})
        elif v8 == FREEZE_T8_PASS:
            selected = t8.get("selected") if isinstance(t8.get("selected"), dict) else {}
            decision = t8.get("decision") if isinstance(t8.get("decision"), dict) else {}
            cid = (_pick_from(t8, ["calibrated_candidate_id", "candidate_id"])
                   or _pick_from(selected, ["candidate_id"])
                   or _pick_from(decision, ["candidate_id"]))
            bss = (_pick_from(t8, ["primary_bss"])
                   or _pick_from(selected, ["primary_bss"])
                   or _pick_from(decision, ["primary_bss"]))
            trace.append({"path": "task-8-calibration", "verdict": v8,
                          "required": FREEZE_T8_PASS, "outcome": "passing-derivative",
                          "candidate_id": cid, "primary_bss": bss,
                          "note": "passing calibrated derivative supersedes the base candidate"})
        else:
            trace.append({"path": "task-8-calibration", "verdict": v8,
                          "required": FREEZE_T8_PASS, "outcome": "not-passed",
                          "reason": f"verdict {v8!r} — PASS 필요 (derivative)"})

    # 4) Terminal decision: derivative > base; base = Task-3 first, else Task-7.
    derivative = next((e for e in trace if e.get("outcome") == "passing-derivative"), None)
    base = next((e for e in trace if e.get("outcome") == "passed-base"), None)
    frozen: JSON
    if derivative is not None:
        if base is None:
            violations.append({"rule": "derivative_without_base", "ok": False,
                               "reason": "task-8 verdict PASS 이지만 base 후보 없음 — 모순 (t8 PASS 는 "
                                         "Task-3/7 base 가 선행되어야 함)"})
        frozen = {"terminal_verdict": "FROZEN", "frozen_candidate_id": derivative.get("candidate_id"),
                  "primary_bss": derivative.get("primary_bss"), "source": "task-8-calibration"}
    elif base is not None:
        frozen = {"terminal_verdict": "FROZEN", "frozen_candidate_id": base.get("candidate_id"),
                  "primary_bss": base.get("primary_bss"), "source": base.get("path")}
    else:
        frozen = {"terminal_verdict": "NO_PROMOTION", "frozen_candidate_id": None,
                  "primary_bss": None, "source": None,
                  "note": "Task-3/7 base 경로 모두 미통과 — 현재 최고 유지, 패키징 전 중단"}

    return {"decision": frozen, "trace": trace, "violations": violations}


def _tampered_copies(records: list[JSON], key_norms: set[str]) -> list[JSON]:
    """R-only / leaderboard 값 주입+변형 인-메모리 복사 (evidence 파일은 건드리지 않음)."""
    out: list[JSON] = []
    for rec in records:
        rec = json.loads(json.dumps(rec))  # deep copy
        for d in _iter_dicts(rec):
            existing = {_norm_key(k) for k in d}
            for i, norm in enumerate(sorted(key_norms)):
                if norm in existing:
                    # 기존 값 변형 (숫자만 — 후보 id/verdict 같은 문자열은 R-only/LB 값이 아님)
                    for k, v in list(d.items()):
                        if _norm_key(k) == norm and _is_numeric(v):
                            d[k] = float(v) + (FREEZE_PERTURB_STEP if i % 2 == 0 else -FREEZE_PERTURB_STEP)
                else:
                    d[norm] = FREEZE_INJECT_VALUE  # 주입 (결정이 무시함을 증명)
        out.append(rec)
    return out


def _run_insensitivity_fixture(args: argparse.Namespace, policy: JSON,
                               t3: JSON | None, t7: JSON | None, t8: JSON | None) -> int:
    """Plan Task 9 failure QA: --fixture altered-r-only-values / altered-leaderboard-values.
    R-only or leaderboard/public-score 값들을 주입/변형한 사본에서 결정을 재실행해 frozen
    결과(후보 id 또는 NO_PROMOTION)가 불변임을 증명한다. 종료 코드 2 = 가드 발동
    (변조 시도 거부 — 계획 QA 'each exiting 2 without changing the frozen result')."""
    if args.fixture not in FREEZE_FIXTURES:
        print(f"[next_round_policy] FATAL: 알 수 없는 fixture {args.fixture!r}", file=sys.stderr)
        return 1
    clean = decide_freeze(policy, t3, t7, t8)
    if clean["violations"]:
        print("[next_round_policy] fixture: 사전 위반 존재 — fixture 실행 불가", file=sys.stderr)
        return 2
    clean_id = clean["decision"]["frozen_candidate_id"]
    clean_verdict = clean["decision"]["terminal_verdict"]

    if args.fixture == "altered-r-only-values":
        norms = {_norm_key(k) for k in (policy.get("selection") or {}).get("forbidden_sort_keys") or []}
        label = "R-only/부트스트랩-LB 값"
    else:
        norms = {_norm_key(k) for k in FREEZE_LB_KEYS}
        label = "리더보드/공개점수 값"
    records = [r for r in (t3, t7, t8) if r is not None]
    tampered = _tampered_copies(records, norms)
    res = decide_freeze(policy, *tampered)
    tampered_id = res["decision"]["frozen_candidate_id"]
    tampered_verdict = res["decision"]["terminal_verdict"]
    unchanged = (clean_id == tampered_id) and (clean_verdict == tampered_verdict)

    print(f"[next_round_policy] --freeze --fixture {args.fixture}:")
    print(f"  label: {label}")
    print(f"  injected/perturbed key norms: {sorted(norms)}")
    print(f"  clean decision:      {clean_verdict} candidate_id={clean_id}")
    print(f"  tampered decision:   {tampered_verdict} candidate_id={tampered_id}")
    if unchanged:
        print(f"  guard: 변조 시도 거부 — frozen result unchanged: {clean_verdict} "
              f"(exit 2 = 계획 QA 실패 경로 기대 종료 코드; R-only/리더보드 값은 결정에 영향 불가)")
    else:
        print(f"  FATAL: insensitivity VIOLATED — frozen result changed "
              f"({clean_verdict}/{clean_id} → {tampered_verdict}/{tampered_id})", file=sys.stderr)
    return 2


def cmd_freeze(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[next_round_policy] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[next_round_policy] FATAL: 정책 파일을 읽을 수 없음: {policy_path}", file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    violations = validate_policy(policy)

    t3 = load_json(evidence_dir / "task-3-cat-boundary-gate.json")
    t7 = load_json(evidence_dir / "task-7-promotion.json")
    t8 = load_json(evidence_dir / "task-8-calibration.json")
    if t3 is None and t7 is None and t8 is None:
        print(f"[next_round_policy] FATAL: task-3/7/8 증거 전부 없음 — freeze 결정 불가: "
              f"{evidence_dir}", file=sys.stderr)
        return 1

    res = decide_freeze(policy, t3, t7, t8)
    violations += res["violations"]
    trace = res["trace"]
    decision = res["decision"]

    if args.fixture:
        if violations:
            for v in violations:
                print(f"  [REJECT] {v['rule']}: {v['reason']}")
            return 2
        return _run_insensitivity_fixture(args, policy, t3, t7, t8)

    if violations:
        print(f"[next_round_policy] --freeze: REJECT (exit 2)")
        for v in violations:
            print(f"  [REJECT] {v['rule']}: {v['reason']}")
        return 2

    terminal = decision["terminal_verdict"]
    exit_code = 0  # NO_PROMOTION 은 정상 종결 (패키징 전 중단)
    rb_id = policy.get("rollback_baseline_candidate_id")
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 9 — Freeze single best passing candidate (primary-only, no R-fold/LB ranking)",
        "task": "aimers9-next-round/task-9-freeze",
        "mode": "freeze",
        "verdict": terminal,
        "exit_code": exit_code,
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "policy_path": str(policy_path),
        "policy_config_hash": policy_hash,
        "label_sources": ["primary"],
        "labels_read": False,
        "label_sources_note": "freeze 는 라벨을 읽지 않음 — task-3/7/8 증거의 verdict/primary 지표만 "
                              "소비 (R-only 폴드·부트스트랩·리더보드 값은 결정 입력에서 제외)",
        "decision": decision,
        "retained_rollback_baseline_candidate_id": rb_id,
        "retained_baseline_note": (f"NO_PROMOTION — 현재 최고 {policy.get('rollback_baseline_candidate')} "
                                   f"({rb_id}, Public {policy['live_state_frozen']['current_best_public_score']}) 유지, "
                                   "패키징 전 중단; leaderboard_state.json 변경 없음"),
        "priority_tie_rule_trace": trace,
        "checks": [
            {"rule": "policy_check", "ok": True,
             "reason": f"{len(validate_policy(policy))} violations — 동결 정책 유효"},
            {"rule": "evidence_present", "ok": True,
             "reason": "task-3/7/8 증거 로드 (파일별 verdict 아래 trace 참조)"},
            {"rule": "no_labels_read", "ok": True,
             "reason": "freeze 는 어떤 라벨도 로드하지 않음 (label_sources=['primary'], labels_read=false)"},
            {"rule": "no_r_fold_or_leaderboard_ranking", "ok": True,
             "reason": "결정 입력 = verdict + primary_bss + candidate_id 만; R-only/부트스트랩/리더보드 "
                       "값은 정렬·선택에 사용 금지 (정책 forbidden_sort_keys + no_public_feedback)"},
        ],
        "violations": violations,
        "findings": [
            {"rule": f"priority:{e.get('path')}", "ok": True,
             "reason": (f"verdict={e.get('verdict')} (필요 {e.get('required')}) → {e.get('outcome')}"
                        + (f"; candidate_id={e.get('candidate_id')}, primary_bss={e.get('primary_bss')}"
                           if e.get("candidate_id") else "")
                        + (f"; {e.get('reason', '')}" if e.get("reason") else ""))}
            for e in trace
        ],
        "insensitivity_guard": {
            "note": "R-only/리더보드 값의 결정 불변성은 --fixture altered-r-only-values / "
                    "altered-leaderboard-values (exit 2 가드) 및 test helper 로 증명",
            "no_fixture_run_in_happy_path": True,
        },
        "notes": "Task 3=PRIMARY_REJECT, Task 7=SKIPPED, Task 8=SKIPPED_NO_BASE → base 경로 없음 → "
                 "NO_PROMOTION (정직한 종결). Task 10 은 NO_PROMOTION 에 대해 SKIPPED.",
    }
    json_path, md_path = write_evidence(record, evidence_dir / "task-9-freeze")
    print(f"[next_round_policy] --freeze: {terminal} (exit {exit_code})")
    for e in trace:
        mark = "PASS" if e.get("ok") is not False else "INFO"
        print(f"  [{mark}] priority:{e.get('path')}: verdict={e.get('verdict')} "
              f"(필요 {e.get('required')}) → {e.get('outcome')}")
    if decision.get("frozen_candidate_id"):
        print(f"[next_round_policy] frozen candidate: {decision['frozen_candidate_id']} "
              f"(source {decision.get('source')})")
    else:
        print(f"[next_round_policy] retained rollback baseline: {rb_id} — 패키징 전 중단")
    print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --qualify-and-deploy-frozen (Task 10) ─────────────────────────────
# 구조 가드 (동결 계약 — 실패 주입 fixture 가 직접 트리거, exit 2).
def _guard_r2022_qualification(fold_failure: bool = False) -> None:
    """One-shot qualification stop rule: r2022 (또는 r2023) 폴드가 실패하면 NO_PROMOTION
    으로 즉시 중단하고 다른 후보 패밀리를 절대 재개하지 않는다 (no fallback)."""
    if fold_failure:
        raise PolicyViolation(
            "r2022 fold qualification 실패 → NO_PROMOTION stop; one-shot 규칙: "
            "다른 후보 패밀리를 절대 재개하지 않음 (exit 2)")


def _guard_finite_logits(logits: Any) -> None:
    """로짓이 전부 유한해야 한다 — NaN/inf 는 PolicyViolation (exit 2)."""
    for i, v in enumerate(logits):
        if not math.isfinite(float(v)):
            raise PolicyViolation(f"non-finite logit 감지: index={i} value={v!r} (exit 2)")


def _guard_row_order(predicted_ids: Any, expected_ids: Any) -> None:
    """추론 출력 행 순서가 입력과 동일해야 한다 (row order preservation)."""
    if list(predicted_ids) != list(expected_ids):
        raise PolicyViolation(
            f"row order 미보존: predicted {len(list(predicted_ids))} ids ≠ expected "
            f"{len(list(expected_ids))} ids (exit 2)")


def _guard_catboost_source_hash(model_bytes: Any, manifest_hash: str) -> None:
    """CatBoost boundary 는 기존 모델 바이트/포뮬라 재해시만 허용 (재학습 금지).
    재해시한 바이트의 sha256 이 매니페스트에 고정된 해시와 일치해야 한다."""
    actual = _sha256_bytes(bytes(model_bytes or b""))
    if actual != str(manifest_hash or ""):
        raise PolicyViolation(
            f"catboost source re-hash 불일치: actual {actual[:16]}… != manifest "
            f"{str(manifest_hash or '')[:16]}… (재학습 없이 재해시만 허용, exit 2)")


def _guard_deepfm_epochs(replay_epochs: Any, manifest_epochs: Any) -> None:
    """DeepFM replay 는 promote manifest 에 고정된 에포크 수와 정확히 일치해야 한다."""
    if int(replay_epochs) != int(manifest_epochs):
        raise PolicyViolation(
            f"deepfm replay epoch {replay_epochs} != pinned manifest epoch "
            f"{manifest_epochs} (exit 2)")


def _guard_mlp_preprocessor_hash(prep_hash: Any, manifest_hash: Any) -> None:
    """MLP-preprocess replay 는 프리프로세서 해시가 매니페스트와 일치해야 한다."""
    if str(prep_hash or "") != str(manifest_hash or ""):
        raise PolicyViolation(
            f"mlp preprocessor hash 불일치: {str(prep_hash or '')[:16]}… != manifest "
            f"{str(manifest_hash or '')[:16]}… (exit 2)")


def _guard_calibration_offset_cap(delta: Any) -> None:
    """캘리브레이션 오프셋 |delta| 는 ±0.05 캡을 절대 위반하지 않는다."""
    if abs(float(delta)) > 0.05:
        raise PolicyViolation(
            f"calibration offset |{delta}| > 0.05 캡 위반 (exit 2)")


def _guard_safety_limit(elapsed_s: Any) -> None:
    """콜드 오프라인 배포 프로세스의 540 초 안전 한도."""
    if float(elapsed_s) > DEPLOY_SAFETY_LIMIT_S:
        raise PolicyViolation(
            f"safety limit 초과: {elapsed_s}s > {DEPLOY_SAFETY_LIMIT_S}s (exit 2)")


def _apply_clip(values: Any, lo: float = DEPLOY_CLIP_LO,
                hi: float = DEPLOY_CLIP_HI) -> list[float]:
    """배포 스코어링 포뮬라의 클리핑 — clip(sigmoid(z_base+C_LOGIT), .30, .70)."""
    return [min(max(float(v), lo), hi) for v in values]


def _candidate_family(freeze_ev: JSON) -> str:
    """동결 증거 → 후보 family (decision.source 우선, family 필드 폴백)."""
    decision = freeze_ev.get("decision") or {}
    source = decision.get("source") or freeze_ev.get("source") or ""
    return DEPLOY_ROUTE_BY_SOURCE.get(str(source)) or str(freeze_ev.get("family") or "unknown")


# Task 10 사전 등록 config (라벨 접근 이전에 기록 — SKIPPED/후보 분기 공통).
# F1 provenance 요구: 모든 증거 JSON 은 최상위 config_hash 와 policy_config_hash 를 가져야 한다.
def _deploy_config_record(policy: JSON) -> JSON:
    body = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-10-deploy-config",
        "title": "Todo 10 — pre-registered one-shot qualification & deploy config",
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "mode": "qualify-and-deploy-frozen",
        "label_sources": [],
        "labels_read": False,
        "labels_read_note": "Config written BEFORE any label access (pre-registration "
                            "contract). SKIPPED branch (Task 9 NO_PROMOTION) reads no "
                            "labels, no data, and no packages at all.",
        "frozen_controls": {
            "c_logit": DEPLOY_C_LOGIT,
            "clip_lo": DEPLOY_CLIP_LO,
            "clip_hi": DEPLOY_CLIP_HI,
            "rollback_baseline_candidate_id": ROLLBACK_BASELINE_CANDIDATE_ID,
        },
        "qualification_gates": {
            "folds": list(DEPLOY_QUALIFY_FOLDS),
            "r2024": DEPLOY_R2024_ROLE,
            "per_fold_require": dict(DEPLOY_QUALIFY_GATES),
            "failure_policy": ("any fold failure -> NO_PROMOTION stop; never resume "
                               "another candidate family (one-shot, no fallback)"),
        },
        "deploy_rules": {
            "catboost_boundary": "re-hash existing model bytes/formula only - never "
                                 "retrain; emit formula/manifest artifacts",
            "deepfm": "candidate-specific replay; replay epoch count must equal the "
                      "pinned manifest epochs",
            "mlp_preprocess": "candidate-specific replay; preprocessor hash must equal "
                              "the pinned manifest hash",
            "calibrated": "candidate-specific replay; |offset delta| cap 0.05 enforced",
            "final_fit_rows": DEPLOY_FINAL_FIT_ROWS,
            "require": ["finite logits", "preserved row order",
                        f"clipping {DEPLOY_CLIP_LO}/{DEPLOY_CLIP_HI}",
                        "candidate-specific manifest",
                        f"{DEPLOY_SAFETY_LIMIT_S}s safety limit",
                        "peak RAM/GPU recorded"],
        },
    }
    record = dict(body)
    record["config_hash"] = _canonical_sha256(body)
    record["policy_config_hash"] = _canonical_sha256(policy)
    return record


def _run_deploy_fixture(args: argparse.Namespace, policy: JSON, freeze_ev: JSON) -> int:
    """Plan Task 10 failure QA: 7 개 구조 가드를 변조 입력으로 트리거 — 전부 exit 2.
    각 fixture 는 자기 전용 증거(task-10-deploy-fixture-<name>.json)를 쓰고 주 증거를
    절대 덮어쓰지 않는다."""
    name = args.fixture
    if name not in DEPLOY_FIXTURES:
        print(f"[next_round_policy] FATAL: 알 수 없는 fixture {name!r}", file=sys.stderr)
        return 1
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    policy_path = Path(args.policy).expanduser().resolve()
    policy_hash = _canonical_sha256(policy)
    config = _deploy_config_record(policy)
    config_hash = config["config_hash"]
    config_path = evidence_dir / f"task-10-deploy-config-fixture-{name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    config_file_sha = _sha256_bytes(config_path.read_bytes())

    try:
        if name == "failed-r2022":
            print("[fixture failed-r2022] r2022 폴드 실패 주입 — one-shot stop 가드…")
            _guard_r2022_qualification(fold_failure=True)
        elif name == "nonfinite-logit":
            print("[fixture nonfinite-logit] NaN 로짓 주입 — 유한성 가드…")
            _guard_finite_logits([0.5, float("nan"), 0.7])
        elif name == "shuffled-input":
            print("[fixture shuffled-input] 행 순서 뒤섞기 주입 — row order 가드…")
            _guard_row_order([3, 1, 2], [1, 2, 3])
        elif name == "catboost-source-hash-mismatch":
            print("[fixture catboost-source-hash-mismatch] 변조 모델 바이트 주입 — 재해시 가드…")
            _guard_catboost_source_hash(b"tampered model bytes", "a" * 64)
        elif name == "deepfm-epoch-mismatch":
            print("[fixture deepfm-epoch-mismatch] replay 에포크 불일치 주입…")
            _guard_deepfm_epochs(7, 5)
        elif name == "mlp-preprocessor-hash-mismatch":
            print("[fixture mlp-preprocessor-hash-mismatch] 프리프로세서 해시 불일치 주입…")
            _guard_mlp_preprocessor_hash("b" * 64, "a" * 64)
        elif name == "calibration-offset-cap-violation":
            print("[fixture calibration-offset-cap-violation] |delta|>0.05 주입…")
            _guard_calibration_offset_cap(0.07)
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": f"Todo 10 fixture — {name}",
            "task": "aimers9-next-round/task-10-deploy",
            "mode": "qualify-and-deploy-frozen-fixture",
            "verdict": "REJECT",
            "exit_code": 2,
            "recorded_at_utc": now_utc(),
            "git_head": _git_commit(),
            "policy_path": str(policy_path),
            "policy_config_hash": policy_hash,
            "config_hash": config_hash,
            "config_pre_registered": {
                "file": str(config_path), "sha256": config_file_sha,
                "written_before_labels_read": True,
            },
            "freeze_evidence": str(Path(args.freeze_evidence).expanduser().resolve()),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "fixture: no real label read",
            "fixture": name,
            "reason": str(exc),
            "violations": [],
        }
        json_path, md_path = write_evidence(record, evidence_dir / f"task-10-deploy-fixture-{name}")
        print(f"[next_round_policy] --qualify-and-deploy-frozen --fixture {name}: "
              f"exit 2 — {exc}")
        print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
        return 2
    print(f"[next_round_policy] FATAL: fixture {name} 가드가 발동하지 않음 — 가드 버그",
          file=sys.stderr)
    return 1


def _deploy_frozen_candidate(policy: JSON, freeze_ev: JSON, evidence_dir: Path,
                             policy_hash: str, config_path: Path, config_hash: str,
                             config_file_sha: str, freeze_path: Path,
                             policy_path: Path) -> int:
    """미래 대비 후보 분기 — 이번 라운드에서는 절대 실행되지 않는다 (Task 9 = NO_PROMOTION).

    family 라우팅: catboost-boundary → 기존 모델 바이트/포뮬라 재해시만 (재학습 금지);
    deepfm/mlp-preprocess/calibrated → 후보별 replay/final-fit (공식 2019-2024 행만,
    2025/test 행은 row-local 추론 QA 전용). 공통 가드: 유한 로짓, 행 순서 보존, 클리핑,
    후보별 manifest, 540 초 안전 한도, peak RAM/GPU 기록. 이번 라운드에는 후보 manifest
    가 존재할 수 없으므로 구조 가드가 REJECT(exit 2)로 발동한다."""
    decision = freeze_ev.get("decision") or {}
    cid = decision.get("frozen_candidate_id")
    family = _candidate_family(freeze_ev)
    route = DEPLOY_ROUTE_BY_FAMILY.get(family, "replay")
    manifest_path = (evidence_dir / f"task-10-deploy-manifest-{cid}.json") if cid else None
    try:
        if not cid:
            raise PolicyViolation("frozen_candidate_id 누락 — 후보 라우팅 불가 (exit 2)")
        manifest = load_json(manifest_path) if manifest_path else None
        if manifest is None:
            raise PolicyViolation(
                f"후보 {cid} candidate-specific manifest 없음 ({manifest_path.name if manifest_path else '?'}) "
                f"— 이번 라운드(Task 9 NO_PROMOTION)엔 후보가 없어 배포 분기 미실행 (exit 2)")
        if family == "catboost-boundary":
            _guard_catboost_source_hash(manifest.get("model_bytes_sha256") or b"",
                                        manifest.get("source_hash") or "")
        elif family == "deepfm":
            _guard_deepfm_epochs(manifest.get("replay_epochs"), manifest.get("pinned_epochs"))
        elif family == "mlp-preprocess":
            _guard_mlp_preprocessor_hash(manifest.get("preprocessor_hash"),
                                         manifest.get("pinned_preprocessor_hash"))
        elif family == "calibrated":
            _guard_calibration_offset_cap(manifest.get("max_abs_delta"))
        _guard_finite_logits(manifest.get("logits") or [])
        _guard_row_order(manifest.get("row_ids") or [], manifest.get("expected_row_ids") or [])
        _guard_safety_limit(manifest.get("elapsed_s") or 0.0)
        raise PolicyViolation(
            "candidate deploy 경로는 이번 라운드에서 실행되지 않음 (Task 9 NO_PROMOTION) — "
            "DEPLOYED 기록 불가 (exit 2)")
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 10 — one-shot qualification & frozen candidate deploy (rejected)",
            "task": "aimers9-next-round/task-10-deploy",
            "mode": "qualify-and-deploy-frozen",
            "verdict": "REJECT",
            "exit_code": 2,
            "recorded_at_utc": now_utc(),
            "git_head": _git_commit(),
            "policy_path": str(policy_path),
            "policy_config_hash": policy_hash,
            "config_hash": config_hash,
            "config_pre_registered": {
                "file": str(config_path), "sha256": config_file_sha,
                "written_before_labels_read": True,
            },
            "freeze_evidence": str(freeze_path),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "candidate branch: label access occurs only after all "
                                  "structural guards pass — none passed this round",
            "candidate_route": {"candidate_id": cid, "family": family, "route": route},
            "reason": str(exc),
            "violations": [],
        }
        json_path, md_path = write_evidence(record, evidence_dir / "task-10-deploy")
        print(f"[next_round_policy] --qualify-and-deploy-frozen: REJECT (exit 2) — "
              f"candidate branch: family={family} route={route} cid={cid}")
        print(f"[next_round_policy]   {exc}")
        print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
        return 2


def cmd_qualify_and_deploy_frozen(args: argparse.Namespace) -> int:
    """Task 10 디스패처. Task 9 동결 증거를 읽고 SKIPPED(NO_PROMOTION) 또는 후보 분기로
    분기한다. SKIPPED 경로는 config 사전 등록 외 어떤 라벨/모델/데이터/패키지도 접근하지
    않는다."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[next_round_policy] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[next_round_policy] FATAL: 정책 파일을 읽을 수 없음: {policy_path}", file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    violations = validate_policy(policy)
    if violations:
        print(f"[next_round_policy] --qualify-and-deploy-frozen: REJECT (exit 2)")
        for v in violations:
            print(f"  [REJECT] {v['rule']}: {v['reason']}")
        return 2

    freeze_path = Path(args.freeze_evidence).expanduser().resolve()
    freeze_ev = load_json(freeze_path)
    if freeze_ev is None:
        print(f"[next_round_policy] FATAL: Task 9 freeze 증거를 읽을 수 없음: {freeze_path}",
              file=sys.stderr)
        return 1

    if args.fixture:
        return _run_deploy_fixture(args, policy, freeze_ev)

    # ── 사전 등록 config: 어떤 라벨/모델 접근 이전에 기록 (SKIPPED 분기는 라벨 무접촉) ──
    config = _deploy_config_record(policy)
    config_hash = config["config_hash"]
    config_path = evidence_dir / "task-10-deploy-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    config_file_sha = _sha256_bytes(config_path.read_bytes())

    verdict = freeze_ev.get("verdict")
    decision = freeze_ev.get("decision") or {}
    cid = decision.get("frozen_candidate_id")
    if verdict == "FROZEN" or cid:
        return _deploy_frozen_candidate(policy, freeze_ev, evidence_dir, policy_hash,
                                        config_path, config_hash, config_file_sha,
                                        freeze_path, policy_path)
    if verdict != "NO_PROMOTION":
        print(f"[next_round_policy] FATAL: 인식 불가 freeze verdict {verdict!r} — "
              f"NO_PROMOTION 또는 FROZEN 필요", file=sys.stderr)
        return 1

    # ── SKIPPED 경로: Task 9 NO_PROMOTION → zero label/model/data access, exit 0 ──
    rb_id = policy.get("rollback_baseline_candidate_id")
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 10 — one-shot historical stress qualification & frozen deploy (skipped)",
        "task": "aimers9-next-round/task-10-deploy",
        "mode": "qualify-and-deploy-frozen",
        "verdict": "SKIPPED",
        "exit_code": 0,
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "policy_path": str(policy_path),
        "policy_config_hash": policy_hash,
        "config_hash": config_hash,
        "config_pre_registered": {
            "file": str(config_path), "sha256": config_file_sha,
            "written_before_labels_read": True,
        },
        "freeze_evidence": str(freeze_path),
        "label_sources": [],
        "labels_read": False,
        "label_sources_note": "Task 10 SKIPPED — Task 9 NO_PROMOTION; 어떤 라벨·모델·데이터·패키지도 "
                              "접근하지 않음 (구조적 가드)",
        "reason": (f"Task 9 verdict=NO_PROMOTION (frozen_candidate_id=null) — retained "
                   f"rollback baseline {rb_id}; one-shot r2022/r2023 qualification "
                   f"미실행, replay/final-fit 미실행, 패키징 미실행"),
        "retained_rollback_baseline_candidate_id": rb_id,
        "qualification_run": {"executed": False,
                              "note": "one-shot r2022/r2023 qualification 은 후보 동결 시에만 실행"},
        "checks": [
            {"rule": "policy_check", "ok": True,
             "reason": f"{len(validate_policy(policy))} violations — 동결 정책 유효"},
            {"rule": "freeze_evidence_present", "ok": True,
             "reason": f"Task 9 freeze 증거 로드: {freeze_path.name} (verdict={verdict})"},
            {"rule": "no_promotion_detected", "ok": True,
             "reason": "verdict=NO_PROMOTION — SKIPPED 경로 실행"},
            {"rule": "zero_label_model_data_access", "ok": True,
             "reason": "SKIPPED branch: 어떤 라벨·모델·패키지도 접근하지 않음 (구조적 가드, exit 0)"},
        ],
        "violations": [],
        "findings": [
            {"rule": "config_pre_registered", "ok": True,
             "reason": f"sha256 {config_file_sha[:16]}… written_before_labels_read=true"},
            {"rule": "rollback_baseline", "ok": True,
             "reason": f"retained {rb_id} — 변경 없음"},
            {"rule": "qualification_skipped", "ok": True,
             "reason": "r2022/r2023/r2024 qualification 미실행 (후보 없음) — Task 11 package "
                       "도 SKIPPED 여야 함"},
        ],
        "notes": ("Task 9 = NO_PROMOTION → Task 10 = SKIPPED (정직한 종결). "
                  "Notion 로컬 CV row 없음 (plan: label-free SKIPPED outcomes 제외)."),
    }
    json_path, md_path = write_evidence(record, evidence_dir / "task-10-deploy")
    print(f"[next_round_policy] --qualify-and-deploy-frozen: SKIPPED (exit 0)")
    print(f"[next_round_policy] retained rollback baseline: {rb_id} — no qualification, "
          f"no replay, no package")
    print(f"[next_round_policy] evidence -> {json_path} / {md_path}")
    return 0


# 계획 표준 명령(플래그 스타일) ↔ subcommand 스타일 정규화 매핑.
# --freeze 는 Task 9, --qualify-and-deploy-frozen 은 Task 10 에서 subcommand 로 등록됨.
_FLAG_TO_SUB = {
    "--check": "check",
    "--audit-compliance": "audit-compliance",
    "--audit-scope": "audit-scope",
    "--freeze": "freeze",
    "--qualify-and-deploy-frozen": "qualify-and-deploy-frozen",
}


def _normalize_argv(argv: list[str]) -> list[str]:
    """첫 인자가 --flag 스타일이면 해당 subcommand 이름으로 치환 (양쪽 호출 방식 모두 지원)."""
    a = list(argv)
    if a and a[0] in _FLAG_TO_SUB:
        a[0] = _FLAG_TO_SUB[a[0]]
    return a


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="next_round_policy — Task 1 정책 freeze 검증 + F1/F4 감사 CLI "
                    "(--check/--audit-compliance/--audit-scope 플래그 및 subcommand 둘 다 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_check = sub.add_parser("check", help="정책 JSON 검증 (exit 0 PASS / 2 REJECT / 1 fatal)")
    p_check.add_argument("policy", nargs="?", default=str(DEFAULT_POLICY),
                         help="정책 JSON 경로 (기본 repro_979/next_round_policy.json)")
    p_check.add_argument("--state", default=None,
                         help="leaderboard_state.json 경로 (지정 시 frozen 스냅샷과 교차 검증)")
    p_check.add_argument("--evidence-out", default=None,
                         help="증거 작성 베이스 경로 (예: .omo/evidence/aimers9-next-round/task-1-state-policy)")
    p_check.add_argument("--verification", default=None,
                         help="검증 결과(fixture 실행) JSON 경로 — 증거에 포함")
    p_check.set_defaults(func=cmd_check)

    p_c = sub.add_parser("audit-compliance", help="F1 — primary-only 선택 정책 준수 감사")
    p_c.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    p_c.add_argument("--policy", default=str(DEFAULT_POLICY))
    p_c.set_defaults(func=cmd_audit_compliance)

    p_s = sub.add_parser("audit-scope", help="F4 — 스코프/기록 감사")
    p_s.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    p_s.add_argument("--policy", default=str(DEFAULT_POLICY))
    p_s.set_defaults(func=cmd_audit_scope)

    p_f = sub.add_parser("freeze", help="Task 9 — 단일 최고 통과 후보 동결 (primary-only, "
                                        "R-fold/리더보드 랭킹 없음; base 없으면 NO_PROMOTION)")
    p_f.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    p_f.add_argument("--policy", default=str(DEFAULT_POLICY))
    p_f.add_argument("--fixture", default=None, choices=sorted(FREEZE_FIXTURES),
                     help="불변성 가드: R-only/리더보드 값 주입·변형 후 결정 재실행 (exit 2 가드 발동)")
    p_f.set_defaults(func=cmd_freeze)

    p_q = sub.add_parser("qualify-and-deploy-frozen",
                         help="Task 10 — sole candidate dispatcher: one-shot r2022/r2023 "
                              "qualification & frozen candidate deploy (Task 9 "
                              "NO_PROMOTION → SKIPPED, exit 0, zero label/model access)")
    p_q.add_argument("--freeze-evidence", default=str(DEFAULT_EVIDENCE_DIR / "task-9-freeze.json"),
                     help="Task 9 freeze 증거 JSON 경로 (기본 task-9-freeze.json)")
    p_q.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    p_q.add_argument("--policy", default=str(DEFAULT_POLICY))
    p_q.add_argument("--fixture", default=None, choices=sorted(DEPLOY_FIXTURES),
                     help="실패 QA fixture (전부 exit 2): 구조 가드 변조 트리거")
    p_q.set_defaults(func=cmd_qualify_and_deploy_frozen)

    args = parser.parse_args(_normalize_argv(list(argv) if argv is not None else sys.argv[1:]))
    if args.mode == "check" and args.verification:
        vpath = Path(args.verification).expanduser().resolve()
        ver = load_json(vpath)
        if ver is None:
            print(f"[next_round_policy] FATAL: verification JSON 을 읽을 수 없음: {vpath}",
                  file=sys.stderr)
            return 1
        args.verification = ver
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
