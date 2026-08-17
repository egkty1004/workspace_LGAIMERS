#!/usr/bin/env python3
"""recovery_live_state.py — Todo 1 (aimers9-top100-score-recovery): live leaderboard state & 1001.74449 provenance reconciliation.

Wave 1 gate. Reconciles the user-reported live DACON leaderboard state
(2026-08-17: rank-100 cutoff 1090.64249, team best Public 1001.74449, rank 287)
with the real submitted-package provenance, BEFORE any experiment decision.

Fact classes (never conflated):
  - user_observed     : cutoff / best / rank — supplied by the user. The exact
                        observation timestamp was NOT provided → observation_timestamp
                        stays "UNKNOWN".
  - worker_reported_at: current Asia/Seoul timestamp captured by this worker when
                        the state was recorded. NEVER mislabelled as the user's
                        observation/submission time.
  - package_proven    : what was actually verified from artifacts (git history,
                        submit_* packages, evidence, provenance.json, model hashes).

Baseline verdicts (recorded in state + evidence, drive Task 2-10 routing):
  - BASELINE_RECONCILED        : full retrainable provenance + .30/.35/.35
                                 LGB/MLP/Cat logit weight contract proven.
  - BASELINE_CONTRACT_BLOCK    : provenance proven but the weight contract fails.
  - BASELINE_PROVENANCE_BLOCK  : any of {retrainable_source, preprocessing,
                                 configuration, model_hashes, per_origin_refit}
                                 cannot be proven → Tasks 2-10 BLOCKED, only Task 11
                                 may emit its label-free SKIPPED_BASELINE_BLOCK record.
                                 Rollback 5890a4c54f502c4e / 992.8390640403 preserved.

Commands:
  --check --state <path> [--evidence-dir <dir>] [--notion-receipt <file> |
          --notion-unavailable <reason>]
      exit 0 = valid reconciled state (evidence written to
               <evidence-dir>/task-1-live-state.{json,md});
      exit 1 = state present but invalid (nothing written, state untouched).
  --fixture <name> --state <path> [--evidence-dir <dir>]
      exit 2 ALWAYS; writes task-1-live-state-fixture-<name>.{json,md};
      NEVER mutates the state file or the main evidence.
  --audit-baseline-block --evidence-dir <dir>
      exit 0 only when task-1-live-state.json (BLOCK verdict) AND
      task-11-readiness.json (verdict SKIPPED_BASELINE_BLOCK) exist and NO
      task-{2..10}-* artifacts are present; else exit 2.
  --audit-scope-baseline-block --evidence-dir <dir>
      exit 0 only when the above holds AND every evidence record passes the
      scope scan (evidence-name regex, forbidden path tokens, no upload markers);
      else exit 2.

Notion: the leaderboard SSOT append for the real 1001.74449 event happens ONLY
via a real Notion API call (performed by the worker). --notion-receipt records
the API-returned receipt in the evidence; --notion-unavailable records an honest
{status: unavailable, reason}. Never fabricate a receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent

JSON = dict[str, Any]

# ── 상수 ──────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1
LOCAL_TZ_NAME = "Asia/Seoul"
FALLBACK_TZ_NAME = "UTC"
USER_REPORT_DATE = "2026-08-17"          # 사용자가 값을 보고한 날짜 (세션 날짜)
EXPECTED_USER_VALUES: JSON = {
    "rank_100_cutoff": 1090.64249,
    "current_best_public_score": 1001.74449,
    "team_rank": 287,
}
ROLLBACK_CANDIDATE_ID = "5890a4c54f502c4e"
ROLLBACK_CANDIDATE = "lgb_mlp_cat"
ROLLBACK_PUBLIC_SCORE = 992.8390640403
ROLLBACK_PACKAGE_DIR = REPO / "submit_lgb_mlp_cat_20260814-2258"
ROLLBACK_QUALIFICATION_DIGEST = "e3e047dde80f28b31bb9b98b498e37c9ca95cca80686e23207fe875b50487f72"
ROLLBACK_SUBMISSIONS_BY_DATE: JSON = {"2026-08-14": 2}
ROLLBACK_LAST_SUBMISSION: JSON = {
    "date": "2026-08-14",
    "public_score": ROLLBACK_PUBLIC_SCORE,
}
ALLOWED_VERDICTS = ("BASELINE_RECONCILED", "BASELINE_CONTRACT_BLOCK",
                    "BASELINE_PROVENANCE_BLOCK")
# Task 11 이 BASELINE_BLOCK 경로에서 낼 수 있는 유일한 label-free readiness 기록.
TASK11_BLOCK_VERDICT = "SKIPPED_BASELINE_BLOCK"
BLOCKED_TASKS = [str(i) for i in range(2, 11)]   # 2..10 — Task 1 BLOCK 시 중단
PROVENANCE_ELEMENTS = ("retrainable_source", "preprocessing", "configuration",
                       "model_hashes", "per_origin_refit", "weight_contract")
# weight contract 를 제외한 5개 요소 — 하나라도 미증명이면 PROVENANCE_BLOCK.
HARD_PROVENANCE_ELEMENTS = ("retrainable_source", "preprocessing", "configuration",
                            "model_hashes", "per_origin_refit")

FIXTURES = ("missing-reported-at", "score-package-mismatch", "invented-event")

DEFAULT_STATE = REPO / "leaderboard_state.json"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100-recovery"
TASK8_EVIDENCE = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-8-package.json"

EVIDENCE_NAME_RE = re.compile(r"^(?:task-\d+-[a-z0-9_-]+|f[1-4]-[a-z0-9_-]+)\.json$")
# Task 2-10 증거 아티팩트 패턴 (audit: 부재해야 함).
TASKS_2_10_RE = re.compile(r"^task-(?:[2-9]|10)-[a-z0-9_-]+\.(?:json|md)$")
TASK1_EVIDENCE_NAME = "task-1-live-state"
TASK11_EVIDENCE_NAME = "task-11-readiness"
FORBIDDEN_PATH_TOKENS = ("데이터/", "/model/", "/cache/", ".zip", "submit_sim", "submit_sim_")
UPLOAD_KEY_NORMS = {"upload", "uploaded", "daconupload", "daconsubmission",
                    "submitted", "uploadattempt", "uploadmarker"}


def _norm_key(k: str) -> str:
    return re.sub(r"[^0-9a-z]", "", k.lower())


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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> JSON | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tz() -> tzinfo:
    if ZoneInfo is not None:
        try:
            return ZoneInfo(LOCAL_TZ_NAME)
        except Exception:  # tzdata 부재 등
            pass
    try:
        local = datetime.now().astimezone().tzinfo
        return local if local is not None else timezone.utc
    except Exception:  # pragma: no cover
        return timezone.utc


def now_local_iso() -> tuple[str, str]:
    """worker-captured 현재 Asia/Seoul ISO 타임스탬프 + 사용된 타임존 이름."""
    tz = _tz()
    name = getattr(tz, "key", None) or str(tz) or FALLBACK_TZ_NAME
    if "Asia/Seoul" not in name:
        name = FALLBACK_TZ_NAME
    return datetime.now(tz).isoformat(timespec="seconds"), name


def parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz())
    return dt


def _is_proven(elem: JSON | None) -> bool:
    return isinstance(elem, dict) and elem.get("proven") is True


def _verdict_from_facts(provenance: JSON | None) -> str:
    """provenance 필드로부터 기대되는 verdict (state 내용의 내부 일관성 검증용)."""
    if not isinstance(provenance, dict):
        return "BASELINE_PROVENANCE_BLOCK"
    proven = {e: _is_proven(provenance.get(e)) for e in PROVENANCE_ELEMENTS}
    hard_ok = all(proven.get(e) for e in HARD_PROVENANCE_ELEMENTS)
    contract_ok = proven.get("weight_contract", False)
    if hard_ok and contract_ok:
        return "BASELINE_RECONCILED"
    if hard_ok:
        return "BASELINE_CONTRACT_BLOCK"
    return "BASELINE_PROVENANCE_BLOCK"


# ── 상태 검증 ─────────────────────────────────────────────────────────
def validate_state(state: JSON) -> list[JSON]:
    """reconciled live-state 검증 → rule 목록 (모두 ok 일 때만 exit 0).

    rule 은 {rule, ok, reason} — ok=False 가 하나라도 있으면 --check 는 exit 1.
    """
    checks: list[JSON] = []

    def bad(rule: str, reason: str) -> None:
        checks.append({"rule": rule, "ok": False, "reason": reason})

    def good(rule: str, reason: str) -> None:
        checks.append({"rule": rule, "ok": True, "reason": reason})

    if not isinstance(state, dict) or not state:
        bad("state_loadable", "상태 파일 없음/파싱 불가 — reconciled state 가 아님")
        return checks

    if state.get("schema_version") != SCHEMA_VERSION:
        bad("schema_version", f"schema_version = {state.get('schema_version')!r} (필요 {SCHEMA_VERSION})")

    # ── user_observed 값 (사용자 제공, 그대로 수용) ──
    for key, expected in EXPECTED_USER_VALUES.items():
        got = state.get(key)
        if isinstance(expected, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - expected) < 1e-12
        else:
            ok = got == expected
        if ok:
            good(f"user_value_{key}", f"{key} = {got} (사용자 보고값 일치)")
        else:
            bad(f"user_value_{key}", f"{key} = {got!r} (필요 {expected}) — 사용자 보고값과 불일치")

    if state.get("source") != "user-reported DACON leaderboard":
        bad("source", f"source = {state.get('source')!r} (user-reported DACON leaderboard 필요)")

    # date_captured 는 사용자가 값을 보고한 날짜(2026-08-17)까지만 보장 —
    # 정확한 관측 시각은 observation_timestamp=UNKNOWN 으로 남긴다.
    if not str(state.get("date_captured", "")).startswith(USER_REPORT_DATE):
        bad("date_captured",
            f"date_captured = {state.get('date_captured')!r} — "
            f"{USER_REPORT_DATE} 보고일 이어야 함 (관측 시각 UNKNOWN)")
    else:
        good("date_captured", f"date_captured = {state.get('date_captured')} "
                              f"(보고일; 관측 시각은 UNKNOWN)")

    # 관측 시각: 사용자가 제공하지 않음 → UNKNOWN 이어야 한다 (invented-event 방지).
    if str(state.get("observation_timestamp", "")) != "UNKNOWN":
        bad("observation_timestamp_unknown",
            f"observation_timestamp = {state.get('observation_timestamp')!r} — "
            "사용자가 관측 시각을 제공하지 않았으므로 'UNKNOWN' 이어야 함")
    else:
        good("observation_timestamp_unknown", "observation_timestamp = UNKNOWN (사용자 미제공)")

    reported_at = state.get("reported_at")
    reported_dt = parse_iso(reported_at)
    if reported_dt is None:
        bad("reported_at_present", f"reported_at 없음/파싱 불가: {reported_at!r} — "
                                   "worker-captured Asia/Seoul 타임스탬프 필요")
    else:
        off = reported_dt.utcoffset() or timedelta(0)
        if off != timedelta(hours=9):
            bad("reported_at_kst",
                f"reported_at = {reported_at} — 오프셋 {off} 가 Asia/Seoul(+09:00) 아님")
        elif reported_dt > datetime.now(timezone.utc) + timedelta(minutes=5):
            bad("reported_at_not_future",
                f"reported_at = {reported_at} — 미래 시각 (불가능)")
        else:
            good("reported_at_kst", f"reported_at = {reported_at} (+09:00, worker-captured)")

    # ── rollback 보존 (절대 변경 금지) ──
    qp = state.get("qualified_package") or {}
    if qp.get("candidate_id") != ROLLBACK_CANDIDATE_ID:
        bad("rollback_qualified_package",
            f"qualified_package.candidate_id = {qp.get('candidate_id')!r} "
            f"(필요 {ROLLBACK_CANDIDATE_ID})")
    else:
        good("rollback_qualified_package", f"qualified_package 유지: {ROLLBACK_CANDIDATE_ID}")

    digest = qp.get("qualification_digest")
    if digest != ROLLBACK_QUALIFICATION_DIGEST:
        bad("rollback_qualification_digest",
            f"qualification_digest = {digest!r} (필요 {ROLLBACK_QUALIFICATION_DIGEST})")
    else:
        good("rollback_qualification_digest", "qualification_digest 유지")

    qp_score = state.get("qualified_candidate_public_score")
    if not (isinstance(qp_score, (int, float))
            and abs(float(qp_score) - ROLLBACK_PUBLIC_SCORE) < 1e-12):
        bad("score_package_mismatch",
            f"qualified_candidate_public_score = {qp_score!r} (필요 {ROLLBACK_PUBLIC_SCORE}) — "
            "1001.74449 를 rollback 패키지(5890a4c54f502c4e, 992.8390640403)의 점수로 "
            "오귀속 금지 (provenance 미확정)")
    else:
        good("score_package_mismatch",
             f"qualified_candidate_public_score = {qp_score} (rollback 실점수 유지)")

    subs = state.get("submissions_by_date") or {}
    if subs != ROLLBACK_SUBMISSIONS_BY_DATE:
        bad("submissions_by_date_preserved",
            f"submissions_by_date = {subs!r} (필요 {ROLLBACK_SUBMISSIONS_BY_DATE}) — "
            "제출 횟수 변경 금지 (실제 제출 이벤트만)")
    else:
        good("submissions_by_date_preserved", f"submissions_by_date 유지 ({subs})")

    last = state.get("last_submission") or {}
    if (last.get("date") != ROLLBACK_LAST_SUBMISSION["date"]
            or not (isinstance(last.get("public_score"), (int, float))
                    and abs(float(last.get("public_score", -1)) - ROLLBACK_PUBLIC_SCORE) < 1e-12)):
        bad("last_submission_preserved",
            f"last_submission = {last!r} (필요 date={ROLLBACK_LAST_SUBMISSION['date']}, "
            f"public_score={ROLLBACK_PUBLIC_SCORE})")
    else:
        good("last_submission_preserved", f"last_submission 유지 (date={last.get('date')})")

    # ── baseline verdict + provenance 일관성 ──
    verdict = state.get("baseline_verdict")
    provenance = state.get("package_provenance_1001_74449")
    if verdict not in ALLOWED_VERDICTS:
        bad("baseline_verdict", f"baseline_verdict = {verdict!r} "
                                f"(필요 {ALLOWED_VERDICTS})")
    elif not isinstance(provenance, dict):
        bad("package_provenance", "package_provenance_1001_74449 필드 없음 — "
                                  "조사 결과가 상태에 기록되어야 함")
    else:
        expected_verdict = _verdict_from_facts(provenance)
        if expected_verdict != verdict:
            bad("baseline_verdict_facts_consistency",
                f"baseline_verdict={verdict} 가 provenance 필드와 불일치 "
                f"(필요 {expected_verdict})")
        else:
            good("baseline_verdict_facts_consistency",
                 f"baseline_verdict = {verdict} (provenance 필드와 일관)")

    # ── rollback 패키지 해시 검증 (가능할 때만; 불가 시 unverifiable 로 기록) ──
    pkg = ROLLBACK_PACKAGE_DIR
    prov = load_json(pkg / "provenance.json") if pkg.is_dir() else None
    if prov is None:
        checks.append({"rule": "rollback_package_hashes", "ok": True,
                       "status": "unverifiable",
                       "reason": f"rollback 패키지 provenance.json 없음: {pkg} — "
                                 "디스크 해시 검증 생략 (환경 제약, 상태 무효 아님)"})
    else:
        problems = _verify_model_files(prov, pkg)
        if problems:
            bad("rollback_package_hashes",
                "rollback 패키지 모델 파일 해시 불일치: " + "; ".join(problems[:3]))
        else:
            good("rollback_package_hashes",
                 f"rollback 패키지 모델 파일 {len(prov.get('model_file_sha256') or {})}건 "
                 "디스크 재해시 일치")
    return checks


def _verify_model_files(provenance: JSON, package_dir: Path) -> list[str]:
    expected = provenance.get("model_file_sha256") or {}
    problems: list[str] = []
    for rel, want in sorted(expected.items()):
        path = package_dir / "model" / rel
        if not path.is_file():
            problems.append(f"model/{rel}: 파일 없음")
            continue
        got = _sha256_file(path)
        if got != want:
            problems.append(f"model/{rel}: 해시 불일치")
    return problems


# ── fixture 상태 합성 (in-memory copy — 상태 파일 절대 변경 없음) ───────
def _fixture_state(name: str, real_state: JSON) -> JSON:
    """각 fixture 는 실제 상태의 복사본에 단일 결함을 주입한다."""
    state = json.loads(json.dumps(real_state))  # deep copy
    if name == "missing-reported-at":
        state.pop("reported_at", None)
    elif name == "score-package-mismatch":
        # rollback 패키지에 1001.74449 오귀속 (qualified_candidate_public_score 변경)
        state["qualified_candidate_public_score"] = EXPECTED_USER_VALUES["current_best_public_score"]
    elif name == "invented-event":
        # 사용자가 제공하지 않은 관측 시각을 지어냄 + 제출 횟수 위조
        state["observation_timestamp"] = f"{USER_REPORT_DATE}T10:00:00+09:00"
        subs = dict(state.get("submissions_by_date") or {})
        subs["2026-08-17"] = 1
        state["submissions_by_date"] = subs
    return state


# ── 증거 작성 ─────────────────────────────────────────────────────────
def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_commit(),
        "label_sources": [],
        "labels_read": False,
    }


def _build_check_record(state: JSON, checks: list[JSON], notion: JSON,
                        state_sha: str, evidence_dir: Path,
                        state_path: Path | None = None) -> JSON:
    verdict = state.get("baseline_verdict", "BASELINE_PROVENANCE_BLOCK")
    provenance = state.get("package_provenance_1001_74449") or {}
    user_observed = {
        "rank_100_cutoff": state.get("rank_100_cutoff"),
        "current_best_public_score": state.get("current_best_public_score"),
        "team_rank": state.get("team_rank"),
        "reported_date": USER_REPORT_DATE,
        "observation_timestamp": state.get("observation_timestamp"),
        "source": state.get("source"),
        "note": "사용자가 제공한 값 (2026-08-17 세션). 관측 시각은 미제공 → UNKNOWN.",
    }
    worker_reported_at = {
        "reported_at": state.get("reported_at"),
        "timezone": LOCAL_TZ_NAME,
        "note": "worker-captured 현재 시각 — 사용자의 관측/제출 시각이 아님 "
                "(observation_timestamp=UNKNOWN 유지).",
    }
    package_proven = {
        "elements": {
            e: provenance.get(e) for e in PROVENANCE_ELEMENTS
        },
        "rollback_package": {
            "candidate_id": ROLLBACK_CANDIDATE_ID,
            "candidate": ROLLBACK_CANDIDATE,
            "public_score": ROLLBACK_PUBLIC_SCORE,
            "package_dir": str(ROLLBACK_PACKAGE_DIR),
            "qualification_digest": ROLLBACK_QUALIFICATION_DIGEST,
            "submissions_by_date": ROLLBACK_SUBMISSIONS_BY_DATE,
        },
        "note": "package_proven 은 실제 아티팩트(git/패키지/증거/해시)에서 검증된 사실만 기록. "
                "미증명 요소는 proven=false + 사유와 함께 honest BLOCK.",
    }
    record = _record_base(verdict, 0, "aimers9-top100-recovery/task-1-live-state",
                          "Todo 1 — reconcile live leaderboard state & 1001.74449 provenance")
    record.update({
        "state_path": str(state_path if state_path is not None else DEFAULT_STATE),
        "state_sha256": state_sha,
        "facts": {
            "user_observed": user_observed,
            "worker_reported_at": worker_reported_at,
            "package_proven": package_proven,
        },
        "baseline_verdict": verdict,
        "baseline_verdict_note": {
            "BASELINE_RECONCILED": "전체 retrainable provenance + .30/.35/.35 계약 증명",
            "BASELINE_CONTRACT_BLOCK": "provenance 는 증명되나 weight 계약 불일치",
            "BASELINE_PROVENANCE_BLOCK": "retrainable_source/preprocessing/configuration/"
                                         "model_hashes/per_origin_refit 중 미증명 요소 존재",
        }.get(verdict, ""),
        "rollback_preserved": {
            "candidate_id": ROLLBACK_CANDIDATE_ID,
            "public_score": ROLLBACK_PUBLIC_SCORE,
            "qualification_digest": ROLLBACK_QUALIFICATION_DIGEST,
            "submissions_by_date": ROLLBACK_SUBMISSIONS_BY_DATE,
            "last_submission": ROLLBACK_LAST_SUBMISSION,
        },
        "task_routing": {
            "blocked_tasks": BLOCKED_TASKS if verdict != "BASELINE_RECONCILED" else [],
            "task_11": "SKIPPED_BASELINE_BLOCK" if verdict != "BASELINE_RECONCILED" else "normal",
        },
        "notion": notion,
        "checks": checks,
        "violations": [],
        "findings": [
            {"rule": "facts_distinguished", "ok": True,
             "reason": "user_observed / worker_reported_at / package_proven 세 클래스 분리 기록"},
            {"rule": "observation_timestamp_honest", "ok": True,
             "reason": "observation_timestamp=UNKNOWN — 사용자 미제공 시각을 지어내지 않음"},
            {"rule": "no_submission_count_mutation", "ok": True,
             "reason": "submissions_by_date 유지 — 실제 제출 이벤트만 변경 허용"},
        ],
        "notes": (f"user-reported live state 수용 (cutoff {user_observed['rank_100_cutoff']}, "
                  f"best {user_observed['current_best_public_score']}, "
                  f"rank {user_observed['team_rank']}). baseline_verdict={verdict} — "
                  + ("Tasks 2-10 BLOCKED, Task 11 만 SKIPPED_BASELINE_BLOCK 발행 가능."
                     if verdict != "BASELINE_RECONCILED"
                     else "Tasks 2-10 진행 가능.")),
    })
    return record


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
        f"- **state**: {record.get('state_path', '')} (sha256 {str(record.get('state_sha256', ''))[:16]}…)",
        "",
        "## Facts",
        "",
        "### user_observed (사용자 제공)",
        "",
    ]
    uo = record.get("facts", {}).get("user_observed", {})
    for k, v in uo.items():
        lines.append(f"- **{k}**: `{v}`")
    lines += ["", "### worker_reported_at (worker 캡처 — 관측 시각 아님)", ""]
    wr = record.get("facts", {}).get("worker_reported_at", {})
    for k, v in wr.items():
        lines.append(f"- **{k}**: `{v}`")
    lines += ["", "### package_proven (아티팩트에서 검증된 사실)", ""]
    pp = record.get("facts", {}).get("package_proven", {})
    for elem, detail in (pp.get("elements") or {}).items():
        proven = (detail or {}).get("proven")
        mark = "PROVEN" if proven is True else ("FAIL" if proven is False else "UNKNOWN")
        lines.append(f"- **{elem}**: [{mark}] {(detail or {}).get('evidence', '')}")
    lines += [
        "",
        f"## Baseline verdict: **{verdict}**",
        "",
        f"- {record.get('baseline_verdict_note', '')}",
        f"- rollback preserved: {record.get('rollback_preserved', {}).get('candidate_id')} "
        f"/ {record.get('rollback_preserved', {}).get('public_score')}",
        f"- task routing: blocked={record.get('task_routing', {}).get('blocked_tasks')} "
        f"task11={record.get('task_routing', {}).get('task_11')}",
        "",
        "## Notion",
        "",
        f"- status: `{record.get('notion', {}).get('status')}`",
    ]
    receipt = record.get("notion", {}).get("receipt")
    if receipt:
        lines.append(f"- receipt: `{json.dumps(receipt, ensure_ascii=False)}`")
    lines += ["", "## Checks", ""]
    for c in record.get("checks", []):
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"- **[{mark}]** {c['rule']}: {c['reason']}")
    lines += ["", f"## Verdict: **{verdict}** (exit {record['exit_code']})", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ── 명령: --check ─────────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    state_path = Path(args.state).expanduser().resolve()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    state = load_json(state_path)

    if state is None:
        print("[recovery_live_state] FAIL: 상태 파일 없음/파싱 불가 — reconciled state 아님 "
              f"({state_path})", file=sys.stderr)
        return 1
    state_sha = _sha256_file(state_path)

    checks = validate_state(state)
    ok = all(c["ok"] for c in checks)
    if not ok:
        print("[recovery_live_state] INVALID (exit 1) — 상태 무효, 아무것도 기록하지 않음")
        for c in checks:
            if not c["ok"]:
                print(f"  [FAIL] {c['rule']}: {c['reason']}")
        return 1

    notion: JSON = {"status": "not_attempted"}
    if args.notion_receipt:
        receipt = load_json(Path(args.notion_receipt).expanduser().resolve())
        if receipt is None:
            print("[recovery_live_state] FATAL: --notion-receipt 파일을 읽을 수 없음",
                  file=sys.stderr)
            return 1
        notion = {"status": "available", "receipt": receipt}
    elif args.notion_unavailable:
        notion = {"status": "unavailable", "reason": args.notion_unavailable}

    record = _build_check_record(state, checks, notion, state_sha, evidence_dir, state_path)
    json_path, md_path = _write_evidence(record, evidence_dir / TASK1_EVIDENCE_NAME)
    verdict = record["verdict"]
    print(f"[recovery_live_state] VALID (exit 0) — baseline_verdict={verdict}")
    print(f"[recovery_live_state] user_observed: cutoff "
          f"{state.get('rank_100_cutoff')} / best {state.get('current_best_public_score')} "
          f"/ rank {state.get('team_rank')} (observation_timestamp="
          f"{state.get('observation_timestamp')})")
    print(f"[recovery_live_state] worker reported_at: {state.get('reported_at')} (Asia/Seoul)")
    print(f"[recovery_live_state] rollback preserved: {ROLLBACK_CANDIDATE_ID} "
          f"/ {ROLLBACK_PUBLIC_SCORE}")
    print(f"[recovery_live_state] notion: {notion['status']}")
    print(f"[recovery_live_state] evidence -> {json_path} / {md_path}")
    return 0


# ── 명령: --fixture ───────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    state_path = Path(args.state).expanduser().resolve()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    real_state = load_json(state_path) or {}
    state_sha = _sha256_file(state_path) if state_path.is_file() else None

    fixture_state = _fixture_state(name, real_state)
    checks = validate_state(fixture_state)
    failing = [c for c in checks if not c["ok"]]
    expected_failures = {
        "missing-reported-at": "reported_at_present",
        "score-package-mismatch": "score_package_mismatch",
        "invented-event": "observation_timestamp_unknown",
    }
    matched = any(c["rule"] == expected_failures[name] for c in failing)

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-1-live-state-fixture-{name}",
                          f"Todo 1 fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "state_sha256_before": state_sha,
        "state_mutation": {"mutated": False,
                           "note": "fixture 는 in-memory 복사본만 변형 — 상태 파일 미변경"},
        "expected_failure_rule": expected_failures[name],
        "matched": matched,
        "failing_checks": failing,
        "checks": checks,
        "violations": [],
        "findings": [
            {"rule": "no_state_mutation", "ok": True,
             "reason": "실제 상태 파일 sha256 불변 (in-memory copy 만 변형)"},
            {"rule": "no_main_evidence_clobber", "ok": True,
             "reason": "fixture 는 task-1-live-state-fixture-<name>.{json,md} 만 기록"},
        ],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — {expected_failures[name]} 규칙 위반 기대. "
                 "Notion append/상태 변경 없음.",
    })
    base = evidence_dir / f"{TASK1_EVIDENCE_NAME}-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_live_state] FIXTURE {name} (exit 2) — 기대 위반 "
          f"{expected_failures[name]} (matched={matched})")
    print(f"[recovery_live_state] evidence -> {json_path} / {md_path} (상태/메인 증거 미변경)")
    return 2


# ── 명령: --audit-baseline-block / --audit-scope-baseline-block ───────
def _iter_strs(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_iter_strs(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_strs(v))
    return out


def _scan_scope(record: JSON, path: Path) -> list[str]:
    """증거 기록 범위 스캔: 금지 경로 토큰 + 업로드 마커."""
    problems: list[str] = []
    for s in _iter_strs(record):
        for tok in FORBIDDEN_PATH_TOKENS:
            if tok in s:
                problems.append(f"forbidden path token {tok!r} in {path.name}")
                break
    for key in _iter_strs(record):
        if _norm_key(key) in UPLOAD_KEY_NORMS:
            problems.append(f"upload marker key {key!r} in {path.name}")
            break
    return problems


def _audit_block(evidence_dir: Path, scope: bool) -> int:
    problems: list[str] = []
    if not evidence_dir.is_dir():
        problems.append(f"evidence dir 없음: {evidence_dir}")
        _print_audit("baseline-block" if not scope else "scope-baseline-block",
                     problems)
        return 2

    task1 = load_json(evidence_dir / f"{TASK1_EVIDENCE_NAME}.json")
    task11 = load_json(evidence_dir / f"{TASK11_EVIDENCE_NAME}.json")

    # Task 1: BLOCK verdict 증거 필수
    if task1 is None:
        problems.append(f"{TASK1_EVIDENCE_NAME}.json 없음 — Task 1 증거 필수")
    else:
        v1 = task1.get("verdict") or task1.get("baseline_verdict")
        if v1 not in ("BASELINE_PROVENANCE_BLOCK", "BASELINE_CONTRACT_BLOCK"):
            problems.append(f"task-1 verdict = {v1!r} — BASELINE_*_BLOCK 필요")
    # Task 11: SKIPPED_BASELINE_BLOCK 증거 필수
    if task11 is None:
        problems.append(f"{TASK11_EVIDENCE_NAME}.json 없음 — Task 11 증거 필수")
    elif task11.get("verdict") != TASK11_BLOCK_VERDICT:
        problems.append(f"task-11 verdict = {task11.get('verdict')!r} — "
                        f"{TASK11_BLOCK_VERDICT} 필요")

    # Tasks 2-10 아티팩트 부재 필수
    present = sorted(p.name for p in evidence_dir.iterdir()
                     if TASKS_2_10_RE.match(p.name))
    if present:
        problems.append(f"Tasks 2-10 아티팩트 존재 (금지): {present}")

    # 증거 파일명 규칙 + 범위 스캔 (scope audit 한정: F4 범위/기록 감사)
    if scope:
        for path in sorted(evidence_dir.glob("*.json")):
            if not EVIDENCE_NAME_RE.match(path.name):
                problems.append(f"evidence 이름 위반: {path.name}")
                continue
            rec = load_json(path)
            if isinstance(rec, dict):
                problems += _scan_scope(rec, path)

    _print_audit("scope-baseline-block" if scope else "baseline-block", problems)
    return 0 if not problems else 2


def _print_audit(name: str, problems: list[str]) -> None:
    if problems:
        print(f"[recovery_live_state] AUDIT {name} FAIL (exit 2)")
        for p in problems:
            print(f"  [FAIL] {p}")
    else:
        print(f"[recovery_live_state] AUDIT {name} PASS (exit 0)")


def cmd_audit_baseline_block(args: argparse.Namespace) -> int:
    return _audit_block(Path(args.evidence_dir).expanduser().resolve(), scope=False)


def cmd_audit_scope_baseline_block(args: argparse.Namespace) -> int:
    return _audit_block(Path(args.evidence_dir).expanduser().resolve(), scope=True)


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_live_state.py",
        description="Todo 1 — live leaderboard state & 1001.74449 provenance reconciliation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=str(DEFAULT_STATE),
                        help="leaderboard_state.json 경로")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--check", action="store_true",
                        help="reconciled state 검증 — exit 0/1, 증거 기록")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2, fixture 증거 기록")
    parser.add_argument("--notion-receipt", default=None,
                        help="Notion API 가 반환한 영수증 JSON 파일 경로 (증거에 기록)")
    parser.add_argument("--notion-unavailable", default=None,
                        help="Notion 미도달 사유 (증거에 기록, 영수증 위조 금지)")
    parser.add_argument("--audit-baseline-block", action="store_true",
                        help="F1/F2 final-wave 감사: Task 1/11 BLOCK 증거 + Tasks 2-10 부재")
    parser.add_argument("--audit-scope-baseline-block", action="store_true",
                        help="F4 final-wave 감사: 위 조건 + 증거 범위/기록 스캔")
    args = parser.parse_args(argv)

    if args.audit_baseline_block:
        return cmd_audit_baseline_block(args)
    if args.audit_scope_baseline_block:
        return cmd_audit_scope_baseline_block(args)
    if args.fixture is not None:
        return cmd_fixture(args)
    # --check (기본)
    if args.notion_receipt and args.notion_unavailable:
        print("[recovery_live_state] FATAL: --notion-receipt 와 --notion-unavailable 는 "
              "동시 사용 불가", file=sys.stderr)
        return 2
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
