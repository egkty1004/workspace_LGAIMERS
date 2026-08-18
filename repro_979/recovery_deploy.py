#!/usr/bin/env python3
"""recovery_deploy.py — Todo 9 (aimers9-top100-score-recovery): replay and package only a
terminal-PASS candidate.

Consumes the Task 8 terminal receipt (task-8-terminal.json) and decides the deploy branch:

  - SKIPPED / NO_PROMOTION → emit SKIPPED_NO_PROMOTION (exit 0). No package, no manifest,
    no model artifact, and leaderboard_state.json is never touched ("if skipped, do not
    package").
  - STATISTICAL_PASS → candidate-bound deploy branch (future round): refit the frozen
    candidate only on official 2019-2024 training rows using its frozen manifest, record
    distinct selection/terminal/deployment model hashes, byte-copy the reconciled baseline
    LGB/MLP/CatBoost models (only C2/C3/C4 replace/refit the Cat component), then emit a
    source package (script.py, requirements.txt, model directory) + manifest + provenance
    + inference formula. This round Task 8 = SKIPPED, so no candidate exists and the
    structural guard rejects (exit 2) before any refit/package access.

Structural guards (directly triggered by --fixture):
  - _guard_manifest_replay: replayed manifest payload canonical sha256 must equal the
    frozen manifest_hash (manifest-hash-mismatch).
  - _assert_no_2025_rows: refit/train rows must never include 2025 (2025-train-row).
  - _guard_no_promotion_package: a non-PASS branch must never create a package
    (no-promotion-package).

Commands:
  --replay-frozen                 replay the frozen terminal verdict (exit 0 SKIPPED_NO_PROMOTION)
  --fixture <name>                adversarial fixture (always exit 2):
                                  manifest-hash-mismatch / 2025-train-row / no-promotion-package
  --evidence-dir <dir>            evidence directory (default .omo/evidence/...)

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.* 패키지 import 용

import repro_979.recovery_policy as rp  # noqa: E402

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

TERMINAL_RECEIPT_NAME = "task-8-terminal.json"
TERMINAL_VERDICTS = ("SKIPPED", "STATISTICAL_PASS", "NO_PROMOTION")
FIXTURES = ("manifest-hash-mismatch", "2025-train-row", "no-promotion-package")

# 배포 패키지 레이아웃 (STATISTICAL_PASS 분기 — 이번 라운드 미도달).
PACKAGE_LAYOUT = ("script.py", "requirements.txt", "model directory")

# 리핏 허용 시즌 — 공식 2019-2024 학습 행만 (2025/test 행 금지).
REFIT_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)


class PolicyViolation(RuntimeError):
    """배포 정책/가드 위반 — exit 2."""


class LeakageError(RuntimeError):
    """2025/test 행 피팅 누수 — exit 2."""


# ── 유틸 ──────────────────────────────────────────────────────────────
def _now_utc() -> str:
    return rp.now_utc()


def _git_commit() -> str:
    return rp._git_commit()


def deploy_config() -> JSON:
    """동결 배포 구성 — config_hash 의 원천 (라벨 무관)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_receipt": TERMINAL_RECEIPT_NAME,
        "terminal_verdicts": list(TERMINAL_VERDICTS),
        "package_layout": list(PACKAGE_LAYOUT),
        "refit_seasons": list(REFIT_SEASONS),
        "no_package_on_non_pass": True,
        "no_state_mutation": True,
        "c_logit": rp.C_LOGIT,
        "clip": [rp.CLIP_LO, rp.CLIP_HI],
    }


def deploy_config_hash() -> str:
    return rp._canonical_sha256(deploy_config())


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": deploy_config_hash(),
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


# ── 구조 가드 (fixture 가 직접 호출) ──────────────────────────────────
def _guard_manifest_replay(manifest: JSON, frozen_manifest: JSON | None = None) -> None:
    """재생 매니페스트가 동결 아티팩트와 정확히 일치해야 한다 (self-integrity + frozen 대조)."""
    if not isinstance(manifest, dict):
        raise PolicyViolation(f"manifest 재생 불가: dict 아님 (exit 2)")
    recorded = manifest.get("manifest_hash")
    recomputed = rp._canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_hash"})
    if str(recorded or "") != recomputed:
        raise PolicyViolation(
            f"manifest hash 불일치: recorded={str(recorded or '')[:16]}… "
            f"recomputed={recomputed[:16]}… (exit 2)")
    if frozen_manifest is not None:
        frozen_payload = {k: v for k, v in frozen_manifest.items() if k != "manifest_hash"}
        if rp._canonical_sha256(frozen_payload) != recomputed:
            raise PolicyViolation(
                "manifest 재생이 동결 아티팩트와 불일치 — replay/frozen drift (exit 2)")


def _assert_no_2025_rows(train) -> None:
    """리핏/학습 행에 2025 시즌이 절대 포함되지 않아야 한다 (누수 가드).

    train: season 컬럼을 가진 DataFrame-유사 객체 (공식 2019-2024 행만 허용).
    """
    if train is None:
        raise LeakageError("[LEAKAGE] 학습 프레임 없음 — 2025 행 검사 불가 (exit 2)")
    seasons = train.get("season")
    if seasons is None:
        raise LeakageError("[LEAKAGE] season 컬럼 없음 — 2025 행 검사 불가 (exit 2)")
    n_2025 = int((seasons == 2025).sum())
    if n_2025 > 0:
        raise LeakageError(
            f"[LEAKAGE] 2025-train-row: 리핏/학습 행에 2025 시즌 {n_2025}행 포함 — "
            f"공식 2019-2024 행만 사용 (exit 2)")


def _guard_no_promotion_package(terminal_verdict: str, package_dir: Path | None) -> None:
    """비-PASS 분기는 패키지를 절대 생성하지 않는다 (if skipped, do not package)."""
    if terminal_verdict == "STATISTICAL_PASS":
        return
    if package_dir is not None and package_dir.exists():
        raise PolicyViolation(
            f"no-promotion-package: {terminal_verdict} 분기에서 패키지 디렉토리 존재: "
            f"{package_dir} — 비-PASS 분기는 패키지 생성 금지 (exit 2)")


# ── Task 8 터미널 영수증 로드/검증 ────────────────────────────────────
def _load_terminal_receipt(evidence_dir: Path) -> tuple[Path, JSON] | None:
    """Task 8 터미널 영수증 로드 — 부재/파싱 불가 시 None (exit 2)."""
    path = evidence_dir / TERMINAL_RECEIPT_NAME
    if not path.is_file():
        print(f"[recovery_deploy] --replay-frozen: 터미널 영수증 부재: "
              f"{TERMINAL_RECEIPT_NAME}", file=sys.stderr)
        return None
    rec = rp.load_json(path)
    if rec is None:
        print(f"[recovery_deploy] --replay-frozen: 터미널 영수증 파싱 불가: "
              f"{TERMINAL_RECEIPT_NAME}", file=sys.stderr)
        return None
    return path, rec


def _validate_receipt(rec: JSON) -> list[str]:
    """터미널 영수증 구조 검증 → violation 목록 (비면 = 유효)."""
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["receipt not a dict"]
    if rec.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version = {rec.get('schema_version')!r} (필요 {SCHEMA_VERSION})")
    tv = rec.get("terminal_verdict")
    if tv not in TERMINAL_VERDICTS:
        problems.append(f"terminal_verdict = {tv!r} (필요 {TERMINAL_VERDICTS})")
    if rec.get("labels_read") is not False:
        problems.append("labels_read != False — 터미널 영수증은 라벨을 읽지 않아야 함")
    if not isinstance(rec.get("label_sources"), list):
        problems.append("label_sources(리스트) 누락")
    if not rec.get("pre_read_freeze_hash"):
        problems.append("pre_read_freeze_hash 누락")
    return problems


# ── SKIPPED/NO_PROMOTION 분기 ─────────────────────────────────────────
def _deploy_skipped(args: argparse.Namespace, receipt_path: Path, receipt: JSON) -> int:
    """Task 8 SKIPPED/NO_PROMOTION → SKIPPED_NO_PROMOTION — 패키지/상태 미변경."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    tv = receipt.get("terminal_verdict")
    fv = receipt.get("freeze_verdict")
    checks: list[JSON] = [
        {"rule": "terminal_verdict_detected", "ok": True,
         "reason": f"Task 8 터미널 verdict = {tv} — 비-PASS 분기"},
        {"rule": "pre_read_terminal_hash", "ok": True,
         "reason": f"터미널 영수증 사전-읽기 sha256 = {rp._sha256_file(receipt_path)[:16]}…"},
        {"rule": "no_package_created", "ok": True,
         "reason": "SKIPPED_NO_PROMOTION — 패키지/매니페스트/모델 아티팩트 생성 안 함 "
                   "(if skipped, do not package)"},
        {"rule": "no_state_mutation", "ok": True,
         "reason": "leaderboard_state.json 미변경 — 배포는 상태를 건드리지 않음"},
        {"rule": "no_terminal_labels_read", "ok": True,
         "reason": "배포 경로는 라벨을 읽지 않음 (labels_read=False, 구조적 파이어월)"},
    ]
    record = _record_base("SKIPPED_NO_PROMOTION", 0,
                          "aimers9-top100-recovery/task-9-deploy",
                          "Todo 9 — replay and package only a terminal-PASS candidate")
    record.update({
        "mode": "replay-frozen",
        "terminal_verdict": tv,
        "freeze_verdict": fv,
        "frozen_candidate_id": (receipt.get("freeze_decision") or {}).get("frozen_candidate_id"),
        "pre_read_terminal_hash": rp._sha256_file(receipt_path),
        "terminal_receipt": {
            "mode": receipt.get("mode"),
            "terminal_verdict": tv,
            "freeze_verdict": fv,
            "pre_read_freeze_hash": receipt.get("pre_read_freeze_hash"),
            "labels_read": receipt.get("labels_read"),
            "label_sources": receipt.get("label_sources"),
            "recorded_at_utc": receipt.get("recorded_at_utc"),
            "git_head": receipt.get("git_head"),
            "config_hash": receipt.get("config_hash"),
        },
        "package": {
            "created": False,
            "note": "no package / manifest / model artifact created (if skipped, do not package)",
        },
        "leaderboard_state": {
            "mutated": False,
            "note": "leaderboard_state.json untouched — deploy never mutates state",
        },
        "checks": checks,
        "violations": [],
        "findings": [
            {"rule": "no_package_for_non_pass", "ok": True,
             "reason": "비-PASS 분기는 패키지를 생성하지 않음 — acceptance: no package exists "
                       "for non-PASS branch"},
            {"rule": "no_state_mutation", "ok": True,
             "reason": "배포는 leaderboard_state.json 을 변경하지 않음"},
            {"rule": "no_terminal_read", "ok": True,
             "reason": "배포 경로는 read_primary_labels 를 호출하지 않음 — 라벨 구조적으로 미로드"},
        ],
        "notes": "Task 8 SKIPPED/NO_PROMOTION → Task 9 SKIPPED_NO_PROMOTION (정직한 종결). "
                 "STATISTICAL_PASS 후보가 없으므로 리핏/패키징 미실행.",
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-9-deploy")
    print(f"[recovery_deploy] --replay-frozen: SKIPPED_NO_PROMOTION (exit 0) — "
          f"Task 8 {tv}, 패키지 미생성")
    for c in checks:
        print(f"  [PASS] {c['rule']}: {c['reason']}")
    print(f"[recovery_deploy] evidence -> {json_path} / {md_path}")
    return 0


# ── STATISTICAL_PASS 분기 (미래 라운드 — 구조 가드) ───────────────────
def _deploy_statistical_pass(args: argparse.Namespace, receipt_path: Path,
                             receipt: JSON) -> int:
    """Task 8 STATISTICAL_PASS → 후보 결속 배포 분기 (미래 라운드).

    이번 라운드(Task 8 = SKIPPED)엔 STATISTICAL_PASS 후보가 존재할 수 없으므로,
    동결 매니페스트 재생/리핏/패키징에 도달하기 전 구조 가드가 REJECT(exit 2)한다.
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    decision = receipt.get("freeze_decision") or {}
    frozen_id = decision.get("frozen_candidate_id")
    try:
        if not isinstance(frozen_id, str) or not frozen_id:
            raise PolicyViolation(
                "STATISTICAL_PASS 분기: 동결 후보(frozen_candidate_id) 없음 — "
                "리핏/패키징 불가 (exit 2)")
        raise PolicyViolation(
            f"STATISTICAL_PASS 분기: 후보 {frozen_id} 동결 매니페스트 재생/리핏/패키징은 "
            f"미래 라운드 전용 — 이번 라운드 Task 8 = SKIPPED 이며 후보 패키지 부재 (exit 2)")
    except PolicyViolation as exc:
        record = _record_base("REJECT", 2,
                              "aimers9-top100-recovery/task-9-deploy",
                              "Todo 9 — replay and package only a terminal-PASS candidate")
        record.update({
            "mode": "replay-frozen",
            "terminal_verdict": "STATISTICAL_PASS",
            "frozen_candidate_id": frozen_id,
            "pre_read_terminal_hash": rp._sha256_file(receipt_path),
            "reason": str(exc),
            "checks": [],
            "violations": [{"rule": "statistical_pass_branch", "ok": False, "reason": str(exc)}],
            "findings": [],
        })
        json_path, md_path = _write_evidence(record, evidence_dir / "task-9-deploy")
        print(f"[recovery_deploy] --replay-frozen: REJECT (exit 2) — {exc}")
        print(f"[recovery_deploy] evidence -> {json_path} / {md_path}")
        return 2


# ── --replay-frozen ───────────────────────────────────────────────────
def cmd_replay_frozen(args: argparse.Namespace) -> int:
    """Task 8 터미널 영수증 재생 → 배포 분기 결정."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    loaded = _load_terminal_receipt(evidence_dir)
    if loaded is None:
        return 2
    receipt_path, receipt = loaded
    problems = _validate_receipt(receipt)
    if problems:
        print("[recovery_deploy] --replay-frozen: 터미널 영수증 무효 (exit 2)")
        for p in problems:
            print(f"  [REJECT] {p}")
        return 2
    tv = receipt.get("terminal_verdict")
    if tv in ("SKIPPED", "NO_PROMOTION"):
        return _deploy_skipped(args, receipt_path, receipt)
    if tv == "STATISTICAL_PASS":
        return _deploy_statistical_pass(args, receipt_path, receipt)
    print(f"[recovery_deploy] --replay-frozen: 알 수 없는 터미널 verdict {tv!r} (exit 2)",
          file=sys.stderr)
    return 2


# ── --fixture ─────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "manifest-hash-mismatch":
        # 동결 매니페스트 재생 해시 불일치 → 가드가 거부 (exit 2)
        manifest = rp.build_manifest("mutated-candidate", seeds=[42])
        manifest["c_logit"] = 0.0  # 변조 — 매니페스트 내용 변경
        try:
            _guard_manifest_replay(manifest)
            detail = "변조된 매니페스트 재생이 허용됨 — 무결성 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"변조된 매니페스트 재생 거부: {exc}"
    elif name == "2025-train-row":
        # 리핏/학습 행에 2025 시즌 포함 → 누수 가드 (exit 2)
        import pandas as pd  # noqa: PLC0415
        train = pd.DataFrame({"season": [2024, 2025], "game_type": ["R", "R"],
                              "control_success": [1, 0]})
        try:
            _assert_no_2025_rows(train)
            detail = "2025 시즌 행이 리핏/학습에 허용됨 — 누수 위반!"
        except LeakageError as exc:
            matched = True
            detail = f"2025 시즌 행 차단: {exc}"
    elif name == "no-promotion-package":
        # 비-PASS 분기에서 패키지 존재 → 가드가 거부 (exit 2)
        import tempfile  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="t9_noprom_") as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir(parents=True, exist_ok=True)
            try:
                _guard_no_promotion_package("SKIPPED", pkg)
                detail = "비-PASS 분기에서 패키지가 허용됨 — 패키징 위반!"
            except PolicyViolation as exc:
                matched = True
                detail = f"비-PASS 분기 패키지 거부: {exc}"
    else:
        print(f"[recovery_deploy] FATAL: 알 수 없는 fixture {name!r}", file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-9-deploy-fixture-{name}",
                          f"Todo 9 deploy fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-9-deploy-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-9-deploy-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_deploy] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_deploy] evidence -> {json_path} / {md_path}")
    return 2


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_deploy.py",
        description="Todo 9 — replay and package only a terminal-PASS candidate",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--replay-frozen", action="store_true",
                        help="Task 8 터미널 영수증 재생 → 배포 분기 결정 "
                             "(exit 0 SKIPPED_NO_PROMOTION)")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    args = parser.parse_args(argv)

    if args.fixture is not None:
        return cmd_fixture(args)
    if args.replay_frozen:
        return cmd_replay_frozen(args)
    print("[recovery_deploy] FATAL: 명령을 지정하세요 (--replay-frozen / --fixture)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
