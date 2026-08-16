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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 9 제출 의사결정 — live 리더보드 컷오프 + 하루 5회 정책 게이트",
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
    args = parser.parse_args(argv)
    if args.register_qualified:
        args.mode = "register"
    elif args.check:
        args.mode = "check"
    if args.mode == "register":
        return cmd_register(args)
    return cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
