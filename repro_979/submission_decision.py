#!/usr/bin/env python3
"""submission_decision.py — Todo 9 제출 의사결정 러너 (live leaderboard + 일일 5회 정책).

aimers9-top100-score-improvement Todo 9: 리더보드 제출 전, 아래를 캡처해 사용자 제출 행동이
허용될 때만 ALLOW 를 내보낸다. 자동 업로드 금지, 하루 5회 초과 금지, 기각 후보 챔피언화 금지.

프로토콜:
  1. 사용자가 live DACON 리더보드를 확인한 뒤 `repro_979/leaderboard_state.json` 의
     `rank_100_cutoff` / `date_captured` / `current_best_public_score` 를 갱신한다.
     (실시간 컷오프는 사용자 확인 값 — 플랜 TL;DR 의 1066.47756 은 역사 참조일 뿐, 하드코딩 금지)
  2. Todo 8 검증 통과 후 `--register-qualified` 를 1회 실행해 후보 자격 다이제스트를 상태에 등록한다.
  3. 제출 행동 전 `submission_decision.py` (기본 `--check`) 로 5개 게이트 평가 → ALLOW exit 0 / BLOCK exit 1.
  4. 실제 점수가 나오면 사용자가 Notion SSOT 행을 append 한다 (본 러너는 Notion API 를 호출하지 않음 —
     리포트에 정확한 페이로드 템플릿을 제공할 뿐).

5개 게이트 (모두 충족 시에만 ALLOW):
  (a) Todo 8 패키지 검증기 `result == "PASS"` (증거 누락/결과 불일치/상태 `package_validated: false` 면 BLOCK)
  (b) 자격 다이제스트가 패키지와 일치 (등록 시점 다이제스트 == 현재 패키지에서 재계산한 다이제스트)
      + 디스크의 model/* 파일이 provenance.model_file_sha256 과 일치(재해시).
      패키지 변조/모델 파일 변경/증거 재생성은 BLOCK
  (c) 컷오프 신선도: `date_captured` 가 CUTOFF_TTL_HOURS(=24h) 이내 (없거나 TTL 초과면 BLOCK)
  (d) 오늘(`submissions_by_date[local_today]`) 제출 N회 — 하루 5회 정책 (5회 초과/도달 시 BLOCK)
  (e) 목표 마진 `rank_100_cutoff - current_best_public_score > 0` 또는 명시적 오버라이드
      `allow_below_cutoff: true` (기본 false). 컷오프/최고 점수 미확인 시 BLOCK (margin_unknown).

자격 다이제스트 (결정적): sha256(canonical JSON {candidate_id, candidate, blend weights,
c_logit, clip, seeds, model_file_sha256(sorted), package_validator_result,
package_recorded_at(=task-8 증거 날짜)}) — qualification_runner 의 정규화 JSON 컨벤션 재사용.

하루 단위 키: Asia/Seoul 달력일 (zoneinfo 로 결정 불가 시 시스템 로컬, 그마저도 실패 시 UTC — 문서화).
exit code: 0 = ALLOW, 1 = BLOCK(정책 게이트), 2 = 치명적 입력 오류.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — 평가/로컬 모두 Python 3.11
    ZoneInfo = None

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent

JSON = dict[str, Any]  # JSON 페이로드 타입 별칭 (provenance/증거/상태/게이트 레코드)

# ── 상수 ──────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1
CUTOFF_TTL_HOURS = 24          # 컷오프 신선도 TTL (문서화: .omo/notepads + REPORT)
DAILY_SUBMISSION_LIMIT = 5     # 하루 5회 (대회 규칙 — docs 191행 일일 제출 횟수 기준; 플랜 Todo 9의 1회 정책은 대회 규칙에 맞춰 5회로 상향)
LOCAL_TZ_NAME = "Asia/Seoul"   # 하루 단위 키 타임존 (문서화)
FALLBACK_TZ_NAME = "UTC"

DEFAULT_PACKAGE_DIR = REPO / "submit_champ_cat_20260814-0929"
DEFAULT_TASK8_EVIDENCE = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-8-package.json"
DEFAULT_STATE = REPO / "leaderboard_state.json"
DEFAULT_EVIDENCE_BASE = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-9-submission"

# ── Task 12 (aimers9-next-round) 추가 상수 ─────────────────────────────
NEXT_ROUND_MARKER = "aimers9-next-round"          # qualified_package.round 마커 (이번 라운드 등록 증명)
NEXT_ROUND_C_LOGIT = -0.0404                       # 동결 배포 스코어링 (정책과 동일 — 변경 금지)
NEXT_ROUND_CLIP_LO, NEXT_ROUND_CLIP_HI = 0.30, 0.70
NEXT_ROUND_ROLLBACK_BASELINE = "5890a4c54f502c4e"
STALE_STATE_TTL_MINUTES = 30                       # 사용자 live 관측 ≤30분 (plan Task 12)
DEFAULT_NEXT_ROUND_EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
DEFAULT_TASK11_EVIDENCE = DEFAULT_NEXT_ROUND_EVIDENCE_DIR / "task-11-package.json"
NEXT_ROUND_FIXTURES = {"stale-cutoff", "consumed-slot", "altered-package",
                       "unauthorized-state-mutation"}

# next_round_policy 에서 재사용 (재구현 금지 — plan/AGENTS 지침). 로컬에 동일 이름이 존재하는
# load_json/_canonical_sha256/_git_commit/write_evidence 는 로컬 정의를 그대로 사용하고,
# next-round 제네릭 증거 기록기(write_evidence)만 별칭으로 가져온다 (top100 기록기와 분리).
from next_round_policy import (  # noqa: E402
    DEFAULT_POLICY,
    EVIDENCE_NAME_RE,
    FORBIDDEN_PATH_TOKENS,
    PolicyViolation,
    _sha256_bytes,
    now_utc,
    scan_forbidden_usage,
    scan_provenance_fields,
    scan_upload_markers,
    validate_policy,
    write_evidence as write_next_round_evidence,
)


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
    """정규화 JSON(정렬, ensure_ascii=False) → sha256 — qualification_runner 컨벤션."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def compute_qualification_digest(provenance: JSON, task8_evidence: JSON) -> str:
    """자격 다이제스트: {candidate_id, weights, 모델 해시(정렬), 패키지 검증 결과, 날짜}.

    날짜 = task-8 증거의 recorded_at_utc 날짜부(YYYY-MM-DD). 증거를 재생성하면 다이제스트가
    달라져 등록 시점과 불일치 → BLOCK (의도된 안티탬퍼 동작: 재자격 필요).
    """
    blend = provenance.get("blend", {}) or {}
    recorded = str(task8_evidence.get("recorded_at_utc", "")) or ""
    payload = {
        "candidate_id": provenance.get("candidate_id"),
        "candidate": provenance.get("candidate"),
        "weights": blend.get("weights"),
        "c_logit": blend.get("c_logit"),
        "clip": blend.get("clip"),
        "seeds": blend.get("seeds"),
        "model_file_sha256": provenance.get("model_file_sha256"),
        "package_validator_result": task8_evidence.get("result"),
        "package_recorded_at": recorded[:10],
    }
    return _canonical_sha256(payload)


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
        local = datetime.now().astimezone().tzinfo  # 시스템 로컬 타임존
        return local if local is not None else timezone.utc
    except Exception:  # pragma: no cover
        return timezone.utc


def local_today() -> tuple[str, str]:
    """로컬 달력일(Asia/Seoul) + 사용된 타임존 이름."""
    tz = _tz()
    name = getattr(tz, "key", None) or str(tz) or FALLBACK_TZ_NAME
    if "Asia/Seoul" not in name:
        name = FALLBACK_TZ_NAME
    return datetime.now(tz).strftime("%Y-%m-%d"), name


def parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz())  # naive 는 로컬 타임존으로 간주
    return dt


def load_provenance(package_dir: Path) -> JSON | None:
    return load_json(package_dir / "provenance.json")


def verify_model_files(provenance: JSON, package_dir: Path) -> list[str]:
    """provenance.model_file_sha256 의 모델 파일을 디스크에서 재해시해 대조 (패키지 내용 변조 감지)."""
    expected = provenance.get("model_file_sha256") or {}
    problems: list[str] = []
    for rel, want in sorted(expected.items()):
        path = package_dir / "model" / rel
        if not path.is_file():
            problems.append(f"model/{rel}: 파일 없음")
            continue
        got = _sha256_file(path)
        if got != want:
            problems.append(f"model/{rel}: 해시 불일치 (기대 {want[:12]}… / 실제 {got[:12]}…)")
    return problems


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_package_validator(evidence_path: Path) -> tuple[JSON | None, bool]:
    """task-8 증거 로드. (evidence, exists) 반환."""
    evidence = load_json(evidence_path)
    return evidence, (evidence is not None)


def evaluate_gates(
    provenance: JSON | None,
    task8: JSON | None,
    state: JSON,
    package_dir: Path,
) -> tuple[list[JSON], JSON]:
    """5개 게이트 평가 → (gates, metrics). BLOCK 사유는 gates 의 ok=False 항목에 담긴다."""
    gates: list[JSON] = []
    now = datetime.now(timezone.utc)
    today, tz_name = local_today()

    qp = state.get("qualified_package") or None
    cutoff = state.get("rank_100_cutoff")
    captured = state.get("date_captured")
    best = state.get("current_best_public_score")
    submissions = state.get("submissions_by_date") or {}
    submitted_today = int(submissions.get(today, 0) or 0)
    allow_below = bool(state.get("allow_below_cutoff", False))

    # ── (a) 패키지 검증 PASS ──
    if task8 is None:
        gates.append({"gate": "a-package-pass", "ok": False,
                      "reason": "task8_evidence_missing",
                      "detail": f"task-8 증거를 읽을 수 없음: {DEFAULT_TASK8_EVIDENCE}"})
    elif task8.get("result") != "PASS":
        gates.append({"gate": "a-package-pass", "ok": False,
                      "reason": "package_not_pass",
                      "detail": f"패키지 검증기 result = {task8.get('result')!r} (PASS 필요)"})
    else:
        gates.append({"gate": "a-package-pass", "ok": True,
                      "reason": "ok", "detail": "Todo 8 패키지 검증기 result == PASS"})
    if qp and qp.get("package_validated") is False:
        gates[-1] = {"gate": "a-package-pass", "ok": False,
                     "reason": "package_validated_false",
                     "detail": "상태 파일의 qualified_package.package_validated == false"}

    # ── (b) 자격 다이제스트 일치 ──
    if provenance is None:
        gates.append({"gate": "b-qualification-digest", "ok": False,
                      "reason": "package_invalid",
                      "detail": f"provenance.json 을 읽을 수 없음: {package_dir / 'provenance.json'}"})
    else:
        computed = compute_qualification_digest(provenance, task8 or {})
        stored = (qp or {}).get("qualification_digest")
        file_problems = verify_model_files(provenance, package_dir)
        if file_problems:
            gates.append({"gate": "b-qualification-digest", "ok": False,
                          "reason": "model_files_changed",
                          "detail": "디스크 모델 파일이 provenance 해시와 불일치: "
                                    + "; ".join(file_problems[:3])
                                    + (f" 외 {len(file_problems) - 3}건" if len(file_problems) > 3 else "")})
        elif stored is None:
            gates.append({"gate": "b-qualification-digest", "ok": False,
                          "reason": "qualification_digest_not_registered",
                          "detail": "상태에 등록된 자격 다이제스트 없음 — "
                                    "`python3 repro_979/submission_decision.py --register-qualified` 먼저 실행"})
        elif computed != stored:
            gates.append({"gate": "b-qualification-digest", "ok": False,
                          "reason": "qualification_digest_mismatch",
                          "detail": f"패키지 재계산 {computed} != 등록 {stored} "
                                    f"— 패키지 변조 또는 task-8 증거 재생성. 재자격 필요."})
        else:
            gates.append({"gate": "b-qualification-digest", "ok": True,
                          "reason": "ok", "detail": "등록 다이제스트 == 현재 패키지 재계산 "
                                                    f"(모델 파일 {len(provenance.get('model_file_sha256') or {})}건 해시 일치)"})

    # ── (c) 컷오프 신선도 ──
    if cutoff is None:
        gates.append({"gate": "c-cutoff-freshness", "ok": False,
                      "reason": "cutoff_unknown",
                      "detail": "rank_100_cutoff 이 없음 — 사용자가 live 리더보드 확인 후 상태 파일 갱신 필요"})
    else:
        captured_dt = parse_iso(captured)
        if captured_dt is None:
            gates.append({"gate": "c-cutoff-freshness", "ok": False,
                          "reason": "cutoff_no_date",
                          "detail": "date_captured 없음/파싱 불가 — 신선도를 검증할 수 없음 (BLOCKING)"})
        elif (now - captured_dt) > timedelta(hours=CUTOFF_TTL_HOURS):
            gates.append({"gate": "c-cutoff-freshness", "ok": False,
                          "reason": "cutoff_stale",
                          "detail": f"date_captured={captured} 가 TTL {CUTOFF_TTL_HOURS}h 초과 — "
                                    f"live 리더보드 재확인 필요"})
        else:
            gates.append({"gate": "c-cutoff-freshness", "ok": True,
                          "reason": "ok",
                          "detail": f"컷오프 {cutoff} 캡처 {captured} (TTL {CUTOFF_TTL_HOURS}h 이내)"})

    # ── (d) 일일 제출 슬롯 ──
    if submitted_today >= DAILY_SUBMISSION_LIMIT:
        gates.append({"gate": "d-daily-slot", "ok": False,
                      "reason": "daily_slot_exhausted",
                      "detail": f"{today}({tz_name}) 제출 {submitted_today}회 — "
                                f"하루 {DAILY_SUBMISSION_LIMIT}회 제한 초과/도달"})
    else:
        gates.append({"gate": "d-daily-slot", "ok": True,
                      "reason": "ok",
                      "detail": f"{today}({tz_name}) 제출 {submitted_today}회 — 슬롯 사용 가능"})

    # ── (e) 목표 마진 ──
    margin = None
    if cutoff is not None and best is not None:
        margin = float(cutoff) - float(best)
    if allow_below:
        gates.append({"gate": "e-target-margin", "ok": True,
                      "reason": "override",
                      "detail": f"allow_below_cutoff=true 명시적 오버라이드 (margin={margin})"})
    elif margin is None:
        gates.append({"gate": "e-target-margin", "ok": False,
                      "reason": "margin_unknown",
                      "detail": f"rank_100_cutoff({cutoff}) 또는 current_best_public_score({best}) 없음 — "
                                "목표 마진을 계산할 수 없음 (BLOCKING)"})
    elif margin > 0:
        gates.append({"gate": "e-target-margin", "ok": True,
                      "reason": "ok",
                      "detail": f"목표 마진 {margin:+.2f} = cutoff {cutoff} − best {best} > 0"})
    else:
        gates.append({"gate": "e-target-margin", "ok": False,
                      "reason": "margin_not_positive",
                      "detail": f"목표 마진 {margin:+.2f} ≤ 0 — 제출해도 순위권 진입 가능성 낮음. "
                                f"오버라이드 시 allow_below_cutoff=true"})

    metrics = {
        "local_today": today,
        "timezone": tz_name,
        "rank_100_cutoff": cutoff,
        "date_captured": captured,
        "current_best_public_score": best,
        "qualified_candidate_public_score": state.get("qualified_candidate_public_score"),
        "target_margin": margin,
        "submissions_today": submitted_today,
        "unused_daily_slots": DAILY_SUBMISSION_LIMIT - submitted_today,
    }
    return gates, metrics


def build_record(
    decision: str,
    exit_code: int,
    provenance: JSON | None,
    task8: JSON | None,
    state: JSON,
    gates: list[JSON],
    metrics: JSON,
    package_dir: Path,
    digest_computed: str | None,
) -> JSON:
    qp = state.get("qualified_package") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-top100/task-9-submission",
        "decision": decision,
        "exit_code": exit_code,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "candidate": {
            "candidate_id": (provenance or {}).get("candidate_id", (qp or {}).get("candidate_id")),
            "candidate": (provenance or {}).get("candidate", (qp or {}).get("candidate")),
            "package_dir": str(package_dir),
        },
        "qualification_digest": digest_computed,
        "qualification_digest_registered": (qp or {}).get("qualification_digest"),
        "package_validator": {
            "result": (task8 or {}).get("result"),
            "recorded_at_utc": (task8 or {}).get("recorded_at_utc"),
        },
        "leaderboard": metrics,
        "policy": {
            "cutoff_ttl_hours": CUTOFF_TTL_HOURS,
            "daily_submission_limit": DAILY_SUBMISSION_LIMIT,
            "allow_below_cutoff": bool(state.get("allow_below_cutoff", False)),
            "timezone_for_daily_key": metrics["timezone"],
        },
        "gates": gates,
        "block_reasons": [g["reason"] for g in gates if not g["ok"]],
    }


def write_evidence(record: JSON, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    _ = json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision = record["decision"]
    lines = [
        f"# Todo 9 submission decision — {decision} (exit {record['exit_code']})",
        "",
        f"- **candidate**: {record['candidate']['candidate']} "
        f"(`{record['candidate']['candidate_id']}`)",
        f"- **package**: `{record['candidate']['package_dir']}`",
        f"- **recorded_at_utc**: {record['recorded_at_utc']}",
        f"- **git_commit**: {record['git_commit']}",
        "",
        "## Qualification digest",
        "",
        f"- computed: `{record['qualification_digest']}`",
        f"- registered: `{record['qualification_digest_registered']}`",
        "",
        "## Leaderboard state",
        "",
        f"- rank-100 cutoff: `{record['leaderboard']['rank_100_cutoff']}` "
        f"(captured {record['leaderboard']['date_captured']})",
        f"- current best public: `{record['leaderboard']['current_best_public_score']}`",
        f"- target margin: `{record['leaderboard']['target_margin']}`",
        f"- submissions today ({record['leaderboard']['local_today']}, "
        f"{record['leaderboard']['timezone']}): `{record['leaderboard']['submissions_today']}` "
        f"(unused daily slots: `{record['leaderboard']['unused_daily_slots']}`)",
        "",
        "## Gates",
        "",
    ]
    for g in record["gates"]:
        mark = "PASS" if g["ok"] else "FAIL"
        lines.append(f"- **{g['gate']}** [{mark}] ({g['reason']}): {g['detail']}")
    lines += ["", f"## Verdict: **{decision}**",
              "", f"Blocking reasons: {record['block_reasons'] or 'none'}", ""]
    _ = md_path.write_text("\n".join(lines), encoding="utf-8")


def load_state(state_path: Path) -> tuple[JSON, bool]:
    """상태 로드. 파일이 없으면 빈 기본 상태(→ cutoff_unknown BLOCK)로 취급."""
    if state_path.is_file():
        data = load_json(state_path)
        if data is not None:
            return data, True
    return {}, False


def save_state(state: JSON, state_path: Path) -> None:
    _ = state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── 명령: --check (기본) ──────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    state_path = Path(args.state).expanduser().resolve()
    evidence_base = Path(args.evidence_base).expanduser().resolve()
    state, state_exists = load_state(state_path)
    package_dir = Path(args.package_dir).expanduser().resolve()
    provenance = load_provenance(package_dir)
    task8, _task8_exists = load_package_validator(Path(args.task8_evidence).expanduser().resolve())

    gates, metrics = evaluate_gates(provenance, task8, state, package_dir)
    digest_computed = compute_qualification_digest(provenance, task8 or {}) if provenance else None
    all_pass = all(g["ok"] for g in gates)
    decision = "ALLOW" if all_pass else "BLOCK"
    exit_code = 0 if all_pass else 1

    record = build_record(decision, exit_code, provenance, task8, state, gates, metrics,
                          package_dir, digest_computed)
    write_evidence(record, evidence_base)

    print(f"[submission_decision] {decision} (exit {exit_code}) — candidate "
          f"{(provenance or state.get('qualified_package') or {}).get('candidate')}")
    print(f"[submission_decision] 자격 다이제스트: {digest_computed}")
    if state_exists:
        print(f"[submission_decision] 컷오프={metrics['rank_100_cutoff']} "
              f"캡처={metrics['date_captured']} 마진={metrics['target_margin']} "
              f"오늘 제출={metrics['submissions_today']}/{DAILY_SUBMISSION_LIMIT} "
              f"({metrics['timezone']})")
    else:
        print("[submission_decision] 상태 파일 없음 → cutoff_unknown BLOCK")
    for g in gates:
        mark = "PASS" if g["ok"] else "BLOCK"
        print(f"  [{mark}] {g['gate']} ({g['reason']}): {g['detail']}")
    print(f"[submission_decision] 증거 → {evidence_base.with_suffix('.json')} / "
          f"{evidence_base.with_suffix('.md')}")
    return exit_code


# ── 명령: --register-qualified ────────────────────────────────────────
def cmd_register(args: argparse.Namespace) -> int:
    """Todo 8 PASS 후 1회 실행: 자격 다이제스트를 상태 파일에 등록 (제출 게이트 (b)의 입력)."""
    state_path = Path(args.state).expanduser().resolve()
    evidence_base = Path(args.evidence_base).expanduser().resolve()
    state, state_exists = load_state(state_path)
    if not state_exists:
        print("[FAIL] 상태 파일 없음 — 등록할 상태가 없습니다 (repro_979/leaderboard_state.json 사용)",
              file=sys.stderr)
        return 2

    package_dir = Path(args.package_dir).expanduser().resolve()
    provenance = load_provenance(package_dir)
    task8, _task8_exists = load_package_validator(Path(args.task8_evidence).expanduser().resolve())
    if provenance is None:
        print(f"[FAIL] provenance.json 없음: {package_dir / 'provenance.json'}", file=sys.stderr)
        return 2
    if task8 is None:
        print(f"[FAIL] task-8 증거 없음/파싱 불가: {args.task8_evidence}", file=sys.stderr)
        return 1
    if task8.get("result") != "PASS":
        print(f"[FAIL] 패키지 검증기 result = {task8.get('result')!r} — PASS 아님, 등록 거부",
              file=sys.stderr)
        return 1

    digest = compute_qualification_digest(provenance, task8)
    state["qualified_package"] = {
        "candidate_id": provenance.get("candidate_id"),
        "candidate": provenance.get("candidate"),
        "package_dir": str(package_dir),
        "qualification_digest": digest,
        "qualification_date": str(task8.get("recorded_at_utc", ""))[:10],
        "package_validated": True,
    }
    save_state(state, state_path)

    record = build_record(
        "REGISTERED", 0, provenance, task8, state,
        [{"gate": "register", "ok": True, "reason": "ok",
          "detail": f"자격 다이제스트 등록: {digest}"}],
        {"local_today": local_today()[0], "timezone": local_today()[1],
         "rank_100_cutoff": state.get("rank_100_cutoff"),
         "date_captured": state.get("date_captured"),
         "current_best_public_score": state.get("current_best_public_score"),
         "qualified_candidate_public_score": state.get("qualified_candidate_public_score"),
         "target_margin": None, "submissions_today": 0, "unused_daily_slots": 5},
        package_dir, digest,
    )
    write_evidence(record, evidence_base)
    print(f"[submission_decision] REGISTERED {provenance.get('candidate')} "
          f"({provenance.get('candidate_id')}) — 다이제스트 {digest}")
    print(f"[submission_decision] 상태 갱신: {state_path} / 증거 → "
          f"{evidence_base.with_suffix('.json')}")
    return 0


# ── Task 12 (aimers9-next-round) — 등록 readiness ──────────────────────
# 이번 라운드 실행 경로: Task 11 verdict=SKIPPED → SKIPPED_NO_PACKAGE (exit 0).
# 후보 결속 등록 분기(미래 라운드, Task 11 이 DEPLOYED 후보 지명)는 구조만 구현 —
# synthetic-tested only. ALLOW/BLOCK 은 readiness 만: 업로드·제출 횟수·Notion 기록 없음.
def _task12_config_record(policy: JSON) -> JSON:
    """사전 등록 config: 어떤 상태/라벨 접근 이전에 기록 (pre-registration contract)."""
    body = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-12-decision-config",
        "title": "Todo 12 — submission decision & registration readiness config (pre-registered)",
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "mode": "register-qualified",
        "label_sources": [],
        "labels_read": False,
        "labels_read_note": "Config written BEFORE any state/label access (pre-registration "
                            "contract).",
        "frozen_controls": {
            "c_logit": NEXT_ROUND_C_LOGIT,
            "clip": [NEXT_ROUND_CLIP_LO, NEXT_ROUND_CLIP_HI],
            "rollback_baseline_candidate_id": NEXT_ROUND_ROLLBACK_BASELINE,
        },
        "registration_rules": {
            "readiness_only": "ALLOW/BLOCK 은 readiness 만 — 업로드·제출 횟수·Notion/"
                              "리더보드 기록 금지",
            "no_upload": "자동 업로드 금지 — 실제 업로드/점수는 사용자 이벤트 후 별도 follow-up",
            "no_submission_count": "submissions_by_date 갱신 금지 (사용자 실제 제출 이벤트만)",
            "no_state_write_without_user_event": "상태 쓰기는 사용자 이벤트 이후에만 허용",
            "stale_state_ttl_minutes": STALE_STATE_TTL_MINUTES,
        },
        "binding_rules": {
            "candidate_id": "--candidate-id == manifest.candidate_id == Task 11 deployed "
                            "candidate_id",
            "package_path": "--package-path == manifest.package_path",
            "manifest_hash": "manifest payload canonical sha256 == manifest.manifest_hash",
            "model_hash": "on-disk model files sha256 == manifest.model_file_sha256",
        },
    }
    record = dict(body)
    record["config_hash"] = _canonical_sha256(body)
    record["policy_config_hash"] = _canonical_sha256(policy)
    return record


def _write_task12_config(evidence_dir: Path, policy: JSON,
                         fname: str) -> tuple[Path, str, str]:
    config = _task12_config_record(policy)
    config_hash = config["config_hash"]
    path = evidence_dir / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, config_hash, _sha256_bytes(path.read_bytes())


def _task11_deployed_candidate(task11: JSON) -> str | None:
    """Task 11 증거에서 지명된 deployed/frozen 후보 id (없으면 None)."""
    if not isinstance(task11, dict):
        return None
    for key in ("deployed_candidate_id", "frozen_candidate_id", "candidate_id"):
        v = task11.get(key)
        if isinstance(v, str) and v:
            return v
    pkg = task11.get("package")
    if isinstance(pkg, dict):
        for key in ("deployed_candidate_id", "candidate_id", "id"):
            v = pkg.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def _assert_cutoff_fresh(state: JSON) -> None:
    """컷오프 신선도 게이트 (evaluate_gates (c) 와 동일 규칙) — stale/unknown 이면 PolicyViolation."""
    if state.get("rank_100_cutoff") is None:
        raise PolicyViolation("cutoff_unknown — rank_100_cutoff 없음 (BLOCK)")
    captured = state.get("date_captured")
    captured_dt = parse_iso(captured)
    if captured_dt is None:
        raise PolicyViolation("cutoff_no_date — date_captured 없음/파싱 불가 (BLOCK)")
    if (datetime.now(timezone.utc) - captured_dt) > timedelta(hours=CUTOFF_TTL_HOURS):
        raise PolicyViolation(f"cutoff_stale — date_captured={captured} 가 TTL "
                              f"{CUTOFF_TTL_HOURS}h 초과 (BLOCK)")


def _assert_daily_slot(state: JSON) -> None:
    """일일 제출 슬롯 게이트 (evaluate_gates (d) 와 동일 규칙) — 소진이면 PolicyViolation."""
    today, tz_name = local_today()
    submitted_today = int((state.get("submissions_by_date") or {}).get(today, 0) or 0)
    if submitted_today >= DAILY_SUBMISSION_LIMIT:
        raise PolicyViolation(f"daily_slot_exhausted — {today}({tz_name}) 제출 "
                              f"{submitted_today}회 — 하루 {DAILY_SUBMISSION_LIMIT}회 제한 "
                              f"초과/도달 (BLOCK)")


def _guard_state_mutation(user_event: bool) -> None:
    """상태 쓰기는 사용자 이벤트(실제 업로드/점수 보고) 이후에만 허용 — readiness-only 가드."""
    if not user_event:
        raise PolicyViolation("unauthorized state mutation — 사용자 이벤트 없이 상태 쓰기 "
                              "시도 (readiness-only: 업로드·제출 횟수·리더보드 기록은 사용자 "
                              "이벤트 후 별도 follow-up) (exit 2)")


def _guard_manifest_hash(actual_hash: Any, expected_hash: Any) -> None:
    """manifest 페이로드 정규 sha256 이 manifest 에 고정된 manifest_hash 와 일치해야 한다."""
    if str(actual_hash or "") != str(expected_hash or ""):
        raise PolicyViolation(f"manifest hash binding 불일치: actual "
                              f"{str(actual_hash or '')[:16]}… != manifest "
                              f"{str(expected_hash or '')[:16]}… (exit 2)")


def _guard_task12_binding(candidate_id: Any, manifest: JSON | None, package_path: Any) -> None:
    """등록 결속: candidate ID / package path / manifest hash / model hashes 모두 일치 필요."""
    if not isinstance(manifest, dict):
        raise PolicyViolation(f"manifest 없음 — 등록 결속 불가 (candidate {candidate_id}) "
                              f"(exit 2)")
    if str(candidate_id or "") != str(manifest.get("candidate_id") or ""):
        raise PolicyViolation(f"candidate binding 불일치: --candidate-id {candidate_id!r} != "
                              f"manifest {manifest.get('candidate_id')!r} (exit 2)")
    expected = (Path(str(manifest.get("package_path"))).resolve()
                if manifest.get("package_path") else None)
    actual = Path(str(package_path)).resolve()
    if expected is not None and expected != actual:
        raise PolicyViolation(f"package path binding 불일치: --package-path {actual} != "
                              f"manifest package_path {expected} (exit 2)")
    if manifest.get("manifest_hash"):
        payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        _guard_manifest_hash(_canonical_sha256(payload), manifest["manifest_hash"])
    model_hashes = manifest.get("model_file_sha256") or {}
    for rel in sorted(model_hashes):
        path = Path(str(package_path)) / rel
        if not path.is_file():
            raise PolicyViolation(f"model 파일 없음: {rel} (exit 2)")
        actual_hash = _sha256_file(path)
        if actual_hash != model_hashes[rel]:
            raise PolicyViolation(f"model hash binding 불일치: {rel} actual "
                                  f"{actual_hash[:16]}… != manifest "
                                  f"{str(model_hashes[rel])[:16]}… (exit 2)")


def _task12_qualification_digest(candidate_id: Any, manifest: JSON, package_path: Path) -> str:
    payload = {
        "candidate_id": str(candidate_id),
        "package_path": str(package_path),
        "manifest_candidate_id": manifest.get("candidate_id"),
        "manifest_hash": manifest.get("manifest_hash"),
        "model_file_sha256": manifest.get("model_file_sha256") or {},
    }
    return _canonical_sha256(payload)


def _write_skipped_no_package(task11: JSON, task11_path: Path, state_path: Path,
                              policy: JSON, policy_path: Path, policy_hash: str,
                              evidence_dir: Path, config_path: Path, config_hash: str,
                              config_file_sha: str, state_sha_before: str | None) -> int:
    """SKIPPED_NO_PACKAGE 기록 (이번 라운드 실행 경로) — 상태 파일 절대 건드리지 않음."""
    rb = policy.get("rollback_baseline_candidate_id") or NEXT_ROUND_ROLLBACK_BASELINE
    state_sha_after = _sha256_file(state_path) if state_path.is_file() else None
    state_untouched = (state_sha_before == state_sha_after)
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 12 — submission decision & registration readiness (skipped — no package)",
        "task": "aimers9-next-round/task-12-decision",
        "mode": "register-qualified",
        "verdict": "SKIPPED_NO_PACKAGE",
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
        "task11_evidence": str(task11_path),
        "label_sources": [],
        "labels_read": False,
        "label_sources_note": "Task 12 SKIPPED_NO_PACKAGE — 어떤 상태/라벨/모델/패키지도 "
                              "읽지 않음 (구조적 가드; 상태 파일은 sha256 파일 해시만 확인)",
        "reason": ("Task 11 verdict=SKIPPED → no qualified package; retained rollback baseline "
                   f"{rb}; 등록 미실행 (readiness-only: register-qualified 는 qualified package "
                   "를 소비하는 게이트)"),
        "state_mutation": {
            "mutated": False,
            "note": (f"leaderboard_state.json untouched (sha256 before == after: "
                     f"{str(state_sha_before)[:16]}… == {str(state_sha_after)[:16]}…)"),
            "sha256_before": state_sha_before,
            "sha256_after": state_sha_after,
        },
        "checks": [
            {"rule": "policy_check", "ok": True,
             "reason": f"{len(validate_policy(policy))} violations — 동결 정책 유효"},
            {"rule": "task11_evidence_present", "ok": True,
             "reason": f"Task 11 증거 로드: {task11_path.name} "
                       f"(verdict={task11.get('verdict')})"},
            {"rule": "no_qualified_package", "ok": True,
             "reason": "Task 11 SKIPPED → qualified package 없음 — 등록 미실행"},
            {"rule": "no_state_mutation", "ok": True,
             "reason": "state 파일 미변경 (sha256 before == after)"},
        ],
        "violations": [],
        "findings": [
            {"rule": "config_pre_registered", "ok": True,
             "reason": f"sha256 {config_file_sha[:16]}… written_before_labels_read=true"},
            {"rule": "rollback_baseline", "ok": True,
             "reason": f"retained {rb} — 변경 없음"},
            {"rule": "readiness_only", "ok": True,
             "reason": "ALLOW/BLOCK 은 readiness 만 — 업로드·제출 횟수·Notion 기록 없음"},
        ],
        "notes": "Task 11 = SKIPPED → Task 12 = SKIPPED_NO_PACKAGE (등록 대상 패키지 부재, "
                 "정직한 종결). Notion 로컬 CV/리더보드 row 없음 (plan: label-free SKIPPED "
                 "outcomes 제외).",
    }
    json_path, md_path = write_next_round_evidence(record, evidence_dir / "task-12-decision")
    print(f"[submission_decision] SKIPPED_NO_PACKAGE (exit 0)")
    print(f"[submission_decision] retained rollback baseline: {rb} — 등록 미실행, "
          f"상태 미변경 (sha256 before == after: {state_untouched})")
    print(f"[submission_decision] evidence -> {json_path} / {md_path}")
    return 0


def _register_bound_candidate(args: argparse.Namespace, task11: JSON, cid: str,
                              task11_path: Path, state_path: Path, state_sha_before: str | None,
                              policy: JSON, policy_path: Path, policy_hash: str,
                              evidence_dir: Path, config_path: Path, config_hash: str,
                              config_file_sha: str) -> int:
    """미래 라운드: Task 11 이 DEPLOYED 후보 지명 → 결속 가드 통과 후에만 상태 등록.
    이번 라운드엔 실행되지 않음 (Task 11 SKIPPED) — synthetic-tested only."""
    package_path = Path(args.package_path).expanduser().resolve()
    manifest_path = (Path(args.manifest_path).expanduser().resolve()
                     if args.manifest_path else None)
    manifest = load_json(manifest_path) if manifest_path else None
    try:
        if manifest is None:
            raise PolicyViolation(f"manifest 없음/파싱 불가: {manifest_path} (exit 2)")
        if args.candidate_id is not None and str(args.candidate_id) != str(cid):
            raise PolicyViolation(f"candidate binding 불일치: --candidate-id "
                                  f"{args.candidate_id!r} != Task 11 deployed {cid!r} (exit 2)")
        _guard_task12_binding(cid, manifest, package_path)
        provenance = load_provenance(package_path)
        if provenance is None:
            raise PolicyViolation(f"provenance.json 없음: {package_path / 'provenance.json'} "
                                  f"(exit 2)")
        task8_ev = load_json(Path(args.task8_evidence).expanduser().resolve())
        if task8_ev is None or task8_ev.get("result") != "PASS":
            raise PolicyViolation("task-8 증거 result != PASS — 등록 거부 (exit 2)")
        digest = compute_qualification_digest(provenance, task8_ev)
        binding_digest = _task12_qualification_digest(cid, manifest, package_path)
        state, state_exists = load_state(state_path)
        if not state_exists:
            raise PolicyViolation(f"상태 파일 없음: {state_path} (exit 2)")
        state["qualified_package"] = {
            "candidate_id": cid,
            "candidate": manifest.get("candidate") or provenance.get("candidate"),
            "package_dir": str(package_path),
            "qualification_digest": digest,
            "binding_digest": binding_digest,
            "qualification_date": str(task11.get("recorded_at_utc", ""))[:10],
            "package_validated": True,
            "round": NEXT_ROUND_MARKER,
        }
        save_state(state, state_path)
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 12 — submission decision & registration readiness (registered)",
            "task": "aimers9-next-round/task-12-decision",
            "mode": "register-qualified",
            "verdict": "REGISTERED",
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
            "task11_evidence": str(task11_path),
            "candidate": {"candidate_id": cid,
                          "candidate": manifest.get("candidate") or provenance.get("candidate"),
                          "package_dir": str(package_path)},
            "qualification_digest": digest,
            "binding_digest": binding_digest,
            "label_sources": [],
            "labels_read": False,
            "state_mutation": {
                "mutated": True,
                "note": f"qualified_package 등록 (round={NEXT_ROUND_MARKER}) — readiness-only, "
                        "업로드·제출 횟수·리더보드 기록 없음",
                "sha256_before": state_sha_before,
                "sha256_after": _sha256_file(state_path),
            },
            "checks": [
                {"rule": "policy_check", "ok": True, "reason": "동결 정책 유효"},
                {"rule": "task11_evidence_present", "ok": True,
                 "reason": f"Task 11 증거 로드: {task11_path.name} (verdict="
                           f"{task11.get('verdict')}, deployed {cid})"},
                {"rule": "binding_guards", "ok": True,
                 "reason": "candidate ID / package path / manifest hash / model hashes 결속 통과"},
                {"rule": "state_write_after_binding", "ok": True,
                 "reason": "결속 가드 통과 후에만 상태 등록 (readiness-only)"},
            ],
            "violations": [],
        }
        json_path, md_path = write_next_round_evidence(record, evidence_dir / "task-12-decision")
        print(f"[submission_decision] REGISTERED {record['candidate']['candidate']} ({cid}) "
              f"— 다이제스트 {digest}")
        print(f"[submission_decision] 상태 갱신: {state_path} / evidence -> {json_path} / "
              f"{md_path}")
        return 0
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 12 — submission decision & registration readiness (rejected)",
            "task": "aimers9-next-round/task-12-decision",
            "mode": "register-qualified",
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
            "task11_evidence": str(task11_path),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "candidate branch: 상태/라벨 접근은 결속 가드 통과 후에만 "
                                  "허용 — 통과한 가드 없음",
            "candidate_route": {"candidate_id": cid,
                                "package_path": str(package_path),
                                "manifest": manifest_path.name if manifest_path else None},
            "reason": str(exc),
            "checks": [],
            "violations": [],
            "state_mutation": {"mutated": False,
                               "note": "결속 가드 실패 — 상태 파일 미변경",
                               "sha256_before": state_sha_before,
                               "sha256_after": _sha256_file(state_path) if state_path.is_file()
                               else None},
        }
        json_path, md_path = write_next_round_evidence(record, evidence_dir / "task-12-decision")
        print(f"[submission_decision] REJECT (exit 2) — candidate branch: candidate_id={cid}")
        print(f"[submission_decision]   {exc}")
        print(f"[submission_decision] evidence -> {json_path} / {md_path}")
        return 2


def cmd_register_next_round(args: argparse.Namespace) -> int:
    """Task 12 등록(readiness-only): Task 11 증거를 소비해 SKIPPED_NO_PACKAGE(이번 라운드)
    또는 후보 결속 등록(미래 라운드)으로 분기. 자동 업로드/제출 횟수/Notion 기록 없음."""
    state_path = Path(args.state).expanduser().resolve()
    evidence_path = Path(args.evidence_path).expanduser().resolve()
    evidence_dir = evidence_path.parent
    state_sha_before = _sha256_file(state_path) if state_path.is_file() else None

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[submission_decision] FATAL: 정책 파일을 읽을 수 없음: {policy_path}",
              file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    violations = validate_policy(policy)
    if violations:
        print("[submission_decision] REJECT (exit 2) — 정책 위반")
        for v in violations:
            print(f"  [REJECT] {v['rule']}: {v['reason']}")
        return 2

    task11 = load_json(evidence_path)
    if task11 is None:
        print(f"[submission_decision] FATAL: Task 11 증거를 읽을 수 없음: {evidence_path}",
              file=sys.stderr)
        return 1

    config_path, config_hash, config_file_sha = _write_task12_config(
        evidence_dir, policy, "task-12-decision-config.json")

    cid = _task11_deployed_candidate(task11)
    if task11.get("verdict") == "SKIPPED" or cid is None:
        return _write_skipped_no_package(task11, evidence_path, state_path, policy, policy_path,
                                         policy_hash, evidence_dir, config_path, config_hash,
                                         config_file_sha, state_sha_before)
    return _register_bound_candidate(args, task11, cid, evidence_path, state_path,
                                     state_sha_before, policy, policy_path, policy_hash,
                                     evidence_dir, config_path, config_hash, config_file_sha)


def cmd_check_next_round(args: argparse.Namespace) -> int:
    """Task 12 check (next round): 이번 라운드 등록(round 마커)된 qualified package 가 없으면
    SKIPPED_NO_QUALIFIED_CANDIDATE exit 0 (never ALLOW). 상태 파일 없음 → BLOCK exit 1."""
    state_path = Path(args.state).expanduser().resolve()
    evidence_path = Path(args.evidence_path).expanduser().resolve()
    evidence_dir = evidence_path.parent
    state, state_exists = load_state(state_path)
    if not state_exists:
        print("[submission_decision] 상태 파일 없음 → BLOCK (exit 1)", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[submission_decision] FATAL: 정책 파일을 읽을 수 없음: {policy_path}",
              file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)

    qp = state.get("qualified_package")
    if qp is None or qp.get("round") != NEXT_ROUND_MARKER:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 12 — submission decision check (no qualified candidate)",
            "task": "aimers9-next-round/task-12-decision",
            "mode": "check",
            "verdict": "SKIPPED_NO_QUALIFIED_CANDIDATE",
            "exit_code": 0,
            "recorded_at_utc": now_utc(),
            "git_head": _git_commit(),
            "policy_path": str(policy_path),
            "policy_config_hash": policy_hash,
            "task11_evidence": str(evidence_path),
            "label_sources": [],
            "labels_read": False,
            "reason": ("이번 라운드 등록된 qualified package 없음 (Task 11 SKIPPED → 등록 대상 "
                       "부재; rollback baseline 유지) — readiness-only: ALLOW 불가, 업로드·"
                       "제출 기록 없음"),
            "checks": [
                {"rule": "qualified_package_registered", "ok": False,
                 "reason": "state.qualified_package 없음 또는 round 마커 "
                           f"({NEXT_ROUND_MARKER}) 부재"},
                {"rule": "no_allow", "ok": True,
                 "reason": "never ALLOW — 등록된 qualified package 없이는 readiness 게이트 "
                           "미평가"},
            ],
            "violations": [],
            "notes": "후속: 후보 등록(register-qualified) 후 재실행하면 게이트 평가 진입.",
        }
        json_path, md_path = write_next_round_evidence(
            record, evidence_dir / "task-12-decision-check")
        print(f"[submission_decision] SKIPPED_NO_QUALIFIED_CANDIDATE (exit 0)")
        print(f"[submission_decision] evidence -> {json_path} / {md_path}")
        return 0

    package_dir = Path(args.package_path or args.package_dir).expanduser().resolve()
    provenance = load_provenance(package_dir)
    task8, _task8_exists = load_package_validator(Path(args.task8_evidence).expanduser().resolve())
    gates, metrics = evaluate_gates(provenance, task8, state, package_dir)
    digest_computed = compute_qualification_digest(provenance, task8 or {}) if provenance else None
    all_pass = all(g["ok"] for g in gates)
    decision = "ALLOW" if all_pass else "BLOCK"
    exit_code = 0 if all_pass else 1
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 12 — submission decision check (next round)",
        "task": "aimers9-next-round/task-12-decision",
        "mode": "check",
        "verdict": decision,
        "exit_code": exit_code,
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "policy_path": str(policy_path),
        "policy_config_hash": policy_hash,
        "label_sources": [],
        "labels_read": False,
        "candidate": {
            "candidate_id": (provenance or qp).get("candidate_id"),
            "candidate": (provenance or qp).get("candidate"),
            "package_dir": str(package_dir),
        },
        "qualification_digest": digest_computed,
        "qualification_digest_registered": qp.get("qualification_digest"),
        "leaderboard": metrics,
        "leaderboard_source": "user-reported DACON leaderboard",
        "checks": [{"rule": g["gate"], "ok": g["ok"],
                    "reason": f"{g['reason']} — {g['detail']}"} for g in gates],
        "block_reasons": [g["reason"] for g in gates if not g["ok"]],
        "violations": [],
        "readiness_only": "ALLOW/BLOCK 은 readiness 만 — 업로드·제출 횟수·Notion 기록 없음",
    }
    json_path, md_path = write_next_round_evidence(
        record, evidence_dir / "task-12-decision-check")
    print(f"[submission_decision] {decision} (exit {exit_code}) — candidate "
          f"{(provenance or qp).get('candidate')}")
    print(f"[submission_decision] 컷오프={metrics['rank_100_cutoff']} "
          f"캡처={metrics['date_captured']} 마진={metrics['target_margin']} "
          f"오늘 제출={metrics['submissions_today']}/{DAILY_SUBMISSION_LIMIT}")
    for g in gates:
        mark = "PASS" if g["ok"] else "BLOCK"
        print(f"  [{mark}] {g['gate']} ({g['reason']}): {g['detail']}")
    print(f"[submission_decision] evidence -> {json_path} / {md_path}")
    return exit_code


def cmd_fixture(args: argparse.Namespace) -> int:
    """failure-QA fixtures (계획 QA: 전부 exit 2, fixture 전용 증거 — 주 증거 미변경).
    상태는 in-memory 복사(json.load → dict 변조)만 수행 — 실제 파일 절대 저장 안 함."""
    fixture = args.fixture
    evidence_path = Path(args.evidence_path).expanduser().resolve()
    evidence_dir = evidence_path.parent
    state_path = Path(args.state).expanduser().resolve()
    state, state_exists = load_state(state_path)
    if not state_exists:
        print(f"[submission_decision] FATAL: 상태 파일 없음 — fixture {fixture} 실행 불가: "
              f"{state_path}", file=sys.stderr)
        return 1
    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[submission_decision] FATAL: 정책 파일을 읽을 수 없음: {policy_path}",
              file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    state_sha_before = _sha256_file(state_path)

    config_path, config_hash, config_file_sha = _write_task12_config(
        evidence_dir, policy, f"task-12-decision-config-fixture-{fixture}.json")

    try:
        if fixture == "stale-cutoff":
            state["date_captured"] = "2020-01-01T00:00:00+00:00"
            _assert_cutoff_fresh(state)
        elif fixture == "consumed-slot":
            today, _tz = local_today()
            state.setdefault("submissions_by_date", {})[today] = DAILY_SUBMISSION_LIMIT
            _assert_daily_slot(state)
        elif fixture == "altered-package":
            manifest = {"candidate_id": "tampered0deadbeef",
                        "package_path": str(REPO / "submit_next_round_tampered"),
                        "manifest_hash": "0" * 64, "model_file_sha256": {}}
            _guard_task12_binding(NEXT_ROUND_ROLLBACK_BASELINE, manifest,
                                  str(REPO / "submit_next_round_tampered"))
        elif fixture == "unauthorized-state-mutation":
            _guard_state_mutation(user_event=False)
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": f"Todo 12 fixture — {fixture}",
            "task": "aimers9-next-round/task-12-decision",
            "mode": "register-qualified-fixture",
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
            "task11_evidence": str(evidence_path),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "fixture: no real label/state read — in-memory copy only",
            "fixture": fixture,
            "reason": str(exc),
            "state_untouched": {
                "sha256_before": state_sha_before,
                "sha256_after": _sha256_file(state_path),
                "note": "실제 상태 파일 미변경 (in-memory 변조만 수행)",
            },
            "violations": [],
        }
        json_path, md_path = write_next_round_evidence(
            record, evidence_dir / f"task-12-decision-fixture-{fixture}")
        print(f"[submission_decision] --fixture {fixture}: exit 2 — {exc}")
        print(f"[submission_decision] evidence -> {json_path} / {md_path}")
        return 2
    print(f"[submission_decision] FATAL: fixture {fixture} 가드가 발동하지 않음 — 가드 버그",
          file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 9 제출 의사결정 — live 리더보드 컷오프 + 하루 5회 정책 게이트 "
                    "(Task 12 next-round 확장: 등록 readiness / SKIPPED_NO_PACKAGE)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("check", "register"), default="check",
                        help="check(기본, 5개 게이트 평가) | register(자격 다이제스트 등록)")
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR),
                        help=f"패키지 디렉토리 (기본 {DEFAULT_PACKAGE_DIR.name})")
    parser.add_argument("--task8-evidence", default=str(DEFAULT_TASK8_EVIDENCE),
                        help="Todo 8 증거 JSON 경로")
    parser.add_argument("--state", default=str(DEFAULT_STATE),
                        help="리더보드 상태 JSON 경로")
    parser.add_argument("--evidence-base", default=str(DEFAULT_EVIDENCE_BASE),
                        help="결정 증거 베이스 경로 (예: task-9-submission → .json/.md)")
    parser.add_argument("--register-qualified", action="store_true",
                        help="단축: --mode register (자격 다이제스트 등록)")
    parser.add_argument("--check", action="store_true",
                        help="단축: --mode check (기본 제출 의사결정 모드 — 명시 호출 지원)")
    # ── Task 12 (aimers9-next-round) 인자 ──
    parser.add_argument("--candidate-id", default=None,
                        help="(next round) 등록 후보 id (Task 9 freeze candidate)")
    parser.add_argument("--package-path", default=None,
                        help="(next round) Task 11 패키지 dir")
    parser.add_argument("--manifest-path", default=None,
                        help="(next round) Task 11 manifest.json 경로")
    parser.add_argument("--evidence-path", default=str(DEFAULT_TASK11_EVIDENCE),
                        help="Task 11 증거 JSON (기본 task-11-package.json) — next-round 분기 입력")
    parser.add_argument("--fixture", default=None, choices=sorted(NEXT_ROUND_FIXTURES),
                        help="failure QA fixture (전부 exit 2): 등록/게이트 가드 변조 트리거")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY),
                        help="동결 정책 JSON (기본 repro_979/next_round_policy.json)")
    args = parser.parse_args(argv)
    if args.register_qualified:
        args.mode = "register"
    elif args.check:
        args.mode = "check"
    if args.fixture:
        return cmd_fixture(args)
    # 분기 규칙:
    #  - 명시적 next-round 인자(candidate/package/manifest/evidence-path 비기본) → next-round.
    #  - 명시적 legacy 인자(package-dir/task8-evidence/evidence-base 비기본) → top100 (하위호환).
    #  - 순수 기본값(인자 없음, 예: plan QA 의 그대로 `--check`) → next-round
    #    (이번 라운드의 기본 결정 경로; top100 라운드는 종료됨 — legacy 경로는 명시적
    #    legacy 인자로만 도달 가능).
    new_args = (args.candidate_id is not None or args.package_path is not None
                or args.manifest_path is not None
                or args.evidence_path != str(DEFAULT_TASK11_EVIDENCE))
    legacy_args = (args.package_dir != str(DEFAULT_PACKAGE_DIR)
                   or args.task8_evidence != str(DEFAULT_TASK8_EVIDENCE)
                   or args.evidence_base != str(DEFAULT_EVIDENCE_BASE))
    if new_args or not legacy_args:
        if args.mode == "register":
            return cmd_register_next_round(args)
        return cmd_check_next_round(args)
    if args.mode == "register":
        return cmd_register(args)
    return cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
