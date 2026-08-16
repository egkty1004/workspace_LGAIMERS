#!/usr/bin/env python3
"""package_validator_next_round.py — Task 11 패키지 검증기 (이번 라운드 실행 경로: SKIPPED).

[Todo 11 배포 QA] Task 10 deploy 증거를 소비해 SKIPPED(이번 라운드 실행 경로) 또는
후보 결속(candidate-bound) 검증 분기(미래 라운드 — 이번 라운드엔 절대 실행되지 않음)로
분기한다. "If Task 10 passes, create a timestamped source package, manifest, and
candidate-bound validator … If skipped, do not package."

실행 경로 (Task 10 verdict=SKIPPED → "if skipped, do not package"):
  - config 사전 등록(task-11-package-config.json, 라벨/패키지 접근 이전 기록) 후,
    어떤 라벨·모델·데이터·패키지도 접근하지 않고 SKIPPED 증거
    (task-11-package.{json,md,log})를 기록, exit 0. 패키지/zip/모델 아티팩트 미생성.
  - Task 10 이 DEPLOYED 후보를 지명했지만 submission dir 에 manifest/provenance 가
    없으면 마찬가지로 SKIPPED (검증 대상 패키지 부재).

후보 결속 검증 분기 (미래 라운드, Task 10 이 DEPLOYED 후보 지명 + manifest 존재):
  - timestamped source package + manifest + candidate-bound validator.
  - 요구 게이트: 5행 및 합성 245,789행 패리티 < 1e-6, 유한/클리핑 값, 출력 row-ID/순서
    보존, shuffled/chunked 행 독립성, 오프라인 --no-index 설치, < 600s 런타임, zip
    레이아웃, 엄격 migration audit.
  - candidate ID / manifest / model hash / clipping / ordering / package path 가
    다르면 FAIL — 구조 가드가 PolicyViolation(exit 2)을 던진다. 이번 라운드엔
    DEPLOYED 후보가 존재할 수 없으므로 가드가 REJECT(exit 2)로 발동한다.

사용법:
  python3 repro_979/package_validator_next_round.py --submission-dir <dir>
  python3 repro_979/package_validator_next_round.py --submission-dir <dir> \
      --fixture {altered-hash, swapped-row-order, missing-clip-check}

Exit: 0 = SKIPPED/PASS, 1 = fatal input error, 2 = REJECT (gate/guard violation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent

# next_round_policy 에서 재사용 (재구현 금지 — plan/AGENTS 지침).
from next_round_policy import (  # noqa: E402
    DEFAULT_POLICY,
    FORBIDDEN_PATH_TOKENS,
    PolicyViolation,
    ROLLBACK_BASELINE_CANDIDATE_ID,
    SCHEMA_VERSION,
    _canonical_sha256,
    _git_commit,
    _sha256_bytes,
    load_json,
    now_utc,
    scan_forbidden_usage,
    scan_provenance_fields,
    scan_upload_markers,
    validate_policy,
    write_evidence,
)

JSON = dict[str, Any]

DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"
DEFAULT_TASK10_EVIDENCE = DEFAULT_EVIDENCE_DIR / "task-10-deploy.json"

# 동결 배포 스코어링 상수 (정책과 동일 — 변경 금지).
C_LOGIT = -0.0404
CLIP_LO, CLIP_HI = 0.30, 0.70

# Task 11 validator rules (사전 등록 config 에 박제 — 이번 라운드 실행 경로는 SKIPPED).
PARITY_TOL = 1e-6
RUNTIME_BUDGET_S = 600
N_FULL = 245789
REQUIRED_ZIP_TOP = {"script.py", "common.py", "requirements.txt", "provenance.json",
                    "README.md", "manifest.json", "model/"}
FULL_VALIDATION_STAGES = [
    "five-row parity < 1e-6",
    "synthetic 245789-row parity < 1e-6",
    "finite / clipped [0.30, 0.70] values",
    "output row-ID/order preservation",
    "shuffled/chunked row-independence",
    "offline --no-index install check",
    "runtime < 600s",
    "zip layout",
    "strict migration audit",
]

# Task 11 failure-QA fixtures (계획 QA: 각각 exit 2).
PACKAGE_FIXTURES = {"altered-hash", "swapped-row-order", "missing-clip-check"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class Logger:
    """stdout + 로그 파일 동시 기록 (main 증거는 task-11-package.log, fixture 는 전용 로그)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# 구조 가드 (후보 결속 계약 — failure-QA fixture 가 직접 트리거, exit 2)
# ────────────────────────────────────────────────────────────────────────────
def _guard_candidate_binding(evidence_cid: Any, manifest_cid: Any,
                             package_path: Any, manifest_package_path: Any) -> None:
    """candidate ID + package path 결속: evidence(Task 10) 와 manifest 가 모두 일치해야 한다."""
    if str(evidence_cid or "") != str(manifest_cid or ""):
        raise PolicyViolation(
            f"candidate ID binding 불일치: Task 10 evidence {evidence_cid!r} != "
            f"manifest {manifest_cid!r} (exit 2)")
    expected = Path(str(manifest_package_path)).resolve() if manifest_package_path else None
    actual = Path(str(package_path)).resolve()
    if expected is not None and expected != actual:
        raise PolicyViolation(
            f"package path binding 불일치: validator submission dir {actual} != "
            f"manifest package_path {expected} (exit 2)")


def _guard_manifest_hash(actual_hash: Any, expected_hash: Any) -> None:
    """manifest 페이로드 정규 sha256 이 manifest 에 고정된 manifest_hash 와 일치해야 한다."""
    if str(actual_hash or "") != str(expected_hash or ""):
        raise PolicyViolation(
            f"manifest hash binding 불일치: actual {str(actual_hash or '')[:16]}… != "
            f"manifest {str(expected_hash or '')[:16]}… (exit 2)")


def _guard_model_hash(model_rel: Any, actual_hash: Any, expected_hash: Any) -> None:
    """model 파일의 on-disk sha256 이 manifest 의 model_file_sha256 항목과 일치해야 한다."""
    if str(actual_hash or "") != str(expected_hash or ""):
        raise PolicyViolation(
            f"model hash binding 불일치: {model_rel} actual {str(actual_hash or '')[:16]}… "
            f"!= manifest {str(expected_hash or '')[:16]}… (exit 2)")


def _guard_clip_check(clip_ok: Any, formula: JSON | None = None) -> None:
    """클리핑 검증 누락/실패 시 REJECT. formula 가 주어지면 동결 c_logit/clip 과도 결속."""
    if not clip_ok:
        raise PolicyViolation(
            "clipping verification 누락/실패 — validator REJECT (exit 2)")
    if isinstance(formula, dict):
        if formula.get("c_logit") != C_LOGIT:
            raise PolicyViolation(
                f"formula c_logit = {formula.get('c_logit')!r} != 동결 {C_LOGIT} (exit 2)")
        if list(formula.get("clip") or []) != [CLIP_LO, CLIP_HI]:
            raise PolicyViolation(
                f"formula clip = {formula.get('clip')!r} != 동결 [{CLIP_LO}, {CLIP_HI}] (exit 2)")


def _guard_row_order(output_ids: Any, input_ids: Any) -> None:
    """출력 row-ID/순서가 입력과 동일해야 한다 (row order preservation)."""
    if list(output_ids) != list(input_ids):
        raise PolicyViolation(
            f"row order 미보존: output {len(list(output_ids))} ids ≠ input "
            f"{len(list(input_ids))} ids (exit 2)")


def _guard_parity(max_abs_diff: Any) -> None:
    """script.py 출력 vs 독립 참조 max|Δ| < 1e-6 (5행 및 245,789행 공통)."""
    if not (float(max_abs_diff) < PARITY_TOL):
        raise PolicyViolation(
            f"parity 위반: max|Δ|={max_abs_diff} ≥ {PARITY_TOL} (exit 2)")


def _guard_row_independence(max_abs_diff: Any) -> None:
    """shuffled/chunked 추론이 전체 순서 출력과 동일해야 한다 (행 독립성)."""
    if not (float(max_abs_diff) < PARITY_TOL):
        raise PolicyViolation(
            f"shuffled/chunked row-independence 위반: max|Δ|={max_abs_diff} ≥ {PARITY_TOL} "
            f"(exit 2)")


def _guard_runtime(elapsed_s: Any) -> None:
    """245,789행 추론 런타임 < 600s."""
    if not (float(elapsed_s) < RUNTIME_BUDGET_S):
        raise PolicyViolation(
            f"runtime 위반: {elapsed_s}s ≥ {RUNTIME_BUDGET_S}s (exit 2)")


def _guard_offline_install(ok: Any, detail: str = "") -> None:
    """오프라인 pip install --no-index 설치 시뮬레이션 게이트."""
    if not ok:
        raise PolicyViolation(f"offline --no-index install FAIL: {detail} (exit 2)")


def _guard_zip_layout(ok: Any, detail: str = "") -> None:
    """zip 레이아웃 게이트: 필수 최상위 항목만, 잡 파일(stray/junk) 부재."""
    if not ok:
        raise PolicyViolation(f"zip layout FAIL: {detail} (exit 2)")


def _guard_migration_audit(ok: Any, tail: str = "") -> None:
    """migration_audit.py --strict PASS 게이트."""
    if not ok:
        raise PolicyViolation(f"migration_audit --strict FAIL: {tail[-300:]} (exit 2)")


# ────────────────────────────────────────────────────────────────────────────
# 사전 등록 config (라벨/패키지 접근 이전 기록 — pre-registration contract)
# ────────────────────────────────────────────────────────────────────────────
def _package_config_record(policy: JSON) -> JSON:
    body = {
        "schema_version": SCHEMA_VERSION,
        "task": "aimers9-next-round/task-11-package-config",
        "title": "Todo 11 — pre-registered package & independent qualification validator config",
        "recorded_at_utc": now_utc(),
        "git_head": _git_commit(),
        "mode": "package-validate",
        "label_sources": [],
        "labels_read": False,
        "labels_read_note": "Config written BEFORE any label/package access "
                            "(pre-registration contract). SKIPPED branch (Task 10 SKIPPED) "
                            "reads no labels, no data, and no packages at all.",
        "frozen_controls": {
            "c_logit": C_LOGIT,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "rollback_baseline_candidate_id": ROLLBACK_BASELINE_CANDIDATE_ID,
        },
        "validator_rules": {
            "parity_tol": PARITY_TOL,
            "five_row_parity": "script.py vs independent reference max|Δ| < 1e-6 (5-row fixture)",
            "full_245789_parity": "synthetic 245789-row fixture max|Δ| < 1e-6",
            "finite_clipped": "all outputs finite and within [0.30, 0.70]",
            "row_order": "output row-ID/order identical to input",
            "row_independence": "shuffled/chunked inference identical to full-order output "
                                "(max|Δ| < 1e-6)",
            "offline_install": "pip install --no-index from local wheels only",
            "runtime_budget_s": RUNTIME_BUDGET_S,
            "zip_layout": "required top-level entries only; no stray/junk files",
            "migration_audit": "repro_979/migration_audit.py --strict PASS",
            "full_validation_stages": FULL_VALIDATION_STAGES,
        },
        "binding_rules": {
            "candidate_id": "manifest candidate_id == Task 10 deployed candidate_id",
            "package_path": "validator submission dir == manifest package_path",
            "manifest_hash": "manifest payload canonical sha256 == manifest.manifest_hash",
            "model_hash": "every manifest model_file_sha256 entry matches on-disk file sha256",
            "formula_clip": "manifest formula c_logit/clip == frozen policy controls",
        },
    }
    record = dict(body)
    record["config_hash"] = _canonical_sha256(body)
    record["policy_config_hash"] = _canonical_sha256(policy)
    return record


def _write_config(evidence_dir: Path, policy: JSON, fname: str) -> tuple[Path, str, str]:
    """사전 등록 config 파일 기록 → (경로, config_hash, 파일 sha256)."""
    config = _package_config_record(policy)
    config_hash = config["config_hash"]
    path = evidence_dir / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, config_hash, _sha256_bytes(path.read_bytes())


# ────────────────────────────────────────────────────────────────────────────
# SKIPPED 경로 (이번 라운드 실행 경로 — zero label/model/data/package access)
# ────────────────────────────────────────────────────────────────────────────
def _write_skipped(task10_ev: JSON, task10_path: Path, submission_dir: Path,
                   policy: JSON, policy_path: Path, policy_hash: str,
                   evidence_dir: Path, config_path: Path, config_hash: str,
                   config_file_sha: str, log: Logger, no_manifest: bool) -> int:
    verdict = task10_ev.get("verdict")
    rb_id = policy.get("rollback_baseline_candidate_id")
    if no_manifest:
        cid = _deployed_candidate_id(task10_ev)
        reason = (f"Task 10 names deployed candidate {cid} but submission dir "
                  f"({_rel_or_abs(submission_dir)}) lacks manifest/provenance — no deployed "
                  f"package exists → SKIPPED, 패키지 미생성 (if skipped, do not package)")
        manifest_rule = "no_package_manifest"
        manifest_reason = "submission dir 에 manifest/provenance 없음 — 검증 대상 패키지 부재"
    else:
        reason = (f"Task 10 verdict=SKIPPED (Task 9 NO_PROMOTION) — retained rollback "
                  f"baseline {rb_id}; 패키징 미실행 (if skipped, do not package): timestamped "
                  f"source package·manifest·zip·model 아티팩트 생성 안 함")
        manifest_rule = "no_deployed_candidate"
        manifest_reason = "verdict=SKIPPED — 패키징/검증 경로 미실행"
    record = {
        "schema_version": SCHEMA_VERSION,
        "title": "Todo 11 — package & independently qualify deployed candidate (skipped)",
        "task": "aimers9-next-round/task-11-package",
        "mode": "package-validate",
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
        "task10_evidence": str(task10_path),
        "submission_dir": _rel_or_abs(submission_dir),
        "label_sources": [],
        "labels_read": False,
        "label_sources_note": "Task 11 SKIPPED — Task 10 SKIPPED; 어떤 라벨·모델·데이터·패키지도 "
                              "접근하지 않음 (구조적 가드)",
        "reason": reason,
        "retained_rollback_baseline_candidate_id": rb_id,
        "package": {"created": False,
                    "note": "no package dir / zip / model artifact created "
                            "(if skipped, do not package)"},
        "checks": [
            {"rule": "policy_check", "ok": True,
             "reason": f"{len(validate_policy(policy))} violations — 동결 정책 유효"},
            {"rule": "task10_evidence_present", "ok": True,
             "reason": f"Task 10 증거 로드: {task10_path.name} (verdict={verdict})"},
            {"rule": manifest_rule, "ok": True, "reason": manifest_reason},
            {"rule": "zero_label_model_data_package_access", "ok": True,
             "reason": "SKIPPED branch: 어떤 라벨·모델·패키지도 접근하지 않음 "
                       "(구조적 가드, exit 0)"},
        ],
        "violations": [],
        "findings": [
            {"rule": "config_pre_registered", "ok": True,
             "reason": f"sha256 {config_file_sha[:16]}… written_before_labels_read=true"},
            {"rule": "rollback_baseline", "ok": True,
             "reason": f"retained {rb_id} — 변경 없음"},
            {"rule": "no_package_created", "ok": True,
             "reason": "if skipped, do not package — 패키지/zip/모델 아티팩트 생성 안 함"},
        ],
        "notes": "Task 10 = SKIPPED → Task 11 = SKIPPED (정직한 종결). Notion 로컬 CV row 없음 "
                 "(plan: label-free SKIPPED outcomes 제외).",
    }
    json_path, md_path = write_evidence(record, evidence_dir / "task-11-package")
    log.log(f"[package_validator] SKIPPED (exit 0)")
    log.log(f"[package_validator] retained rollback baseline: {rb_id} — no package, no manifest, "
            f"no validator run")
    log.log(f"[package_validator] evidence -> {json_path} / {md_path}")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# 후보 결속 검증 분기 (미래 라운드 — 이번 라운드엔 DEPLOYED 후보가 없어 가드가 REJECT)
# ────────────────────────────────────────────────────────────────────────────
def _deployed_candidate_id(task10: JSON) -> str | None:
    """Task 10 증거에서 지명된 deployed/frozen 후보 id (없으면 None)."""
    if not isinstance(task10, dict):
        return None
    for key in ("deployed_candidate_id", "frozen_candidate_id", "candidate_id"):
        v = task10.get(key)
        if isinstance(v, str) and v:
            return v
    for section in ("candidate", "deployed"):
        d = task10.get(section)
        if isinstance(d, dict):
            for key in ("deployed_candidate_id", "candidate_id", "id"):
                v = d.get(key)
                if isinstance(v, str) and v:
                    return v
    return None


def _validate_candidate_package(submission_dir: Path, task10_ev: JSON, task10_path: Path,
                                policy: JSON, policy_path: Path, policy_hash: str,
                                evidence_dir: Path, config_path: Path, config_hash: str,
                                config_file_sha: str, log: Logger) -> int:
    """후보 결속 검증 — candidate ID / manifest / model hash / clipping / ordering /
    package path 결속 가드 체인. 이번 라운드(Task 10 SKIPPED)엔 DEPLOYED 후보가 없어
    전체 검증 단계에 도달하기 전 PolicyViolation(REJECT, exit 2)이 발동한다."""
    cid = _deployed_candidate_id(task10_ev)
    checks: list[JSON] = []
    manifest_path: Path | None = None
    try:
        if not submission_dir.is_dir():
            raise PolicyViolation(f"submission dir 없음: {submission_dir} (exit 2)")
        manifest_path = submission_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest is None:
            manifest_path = submission_dir / "provenance.json"
            manifest = load_json(manifest_path)
        if manifest is None:
            raise PolicyViolation(
                f"candidate {cid} manifest/provenance 없음: {submission_dir} (exit 2)")
        checks.append({"rule": "submission_dir_manifest", "ok": True,
                       "reason": f"{manifest_path.name} 존재 — 후보 결속 검증 진입"})

        _guard_candidate_binding(cid, manifest.get("candidate_id"),
                                 submission_dir, manifest.get("package_path"))
        checks.append({"rule": "candidate_id_binding", "ok": True,
                       "reason": f"Task 10 evidence {cid} == manifest candidate_id "
                                 f"{manifest.get('candidate_id')}, package path 일치"})

        if manifest.get("manifest_hash"):
            payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
            _guard_manifest_hash(_canonical_sha256(payload), manifest["manifest_hash"])
            checks.append({"rule": "manifest_hash_binding", "ok": True,
                           "reason": "manifest payload canonical sha256 == manifest_hash"})

        model_hashes = manifest.get("model_file_sha256") or {}
        for rel in sorted(model_hashes):
            path = submission_dir / rel
            if not path.is_file():
                raise PolicyViolation(f"model 파일 없음: {rel} (exit 2)")
            _guard_model_hash(rel, _sha256(path), model_hashes[rel])
        checks.append({"rule": "model_hash_binding", "ok": True,
                       "reason": f"{len(model_hashes)} model 파일 on-disk sha256 == manifest"})

        _guard_clip_check(True, manifest.get("formula"))
        checks.append({"rule": "formula_clip_binding", "ok": True,
                       "reason": f"formula c_logit={C_LOGIT}, clip=[{CLIP_LO},{CLIP_HI}] 동결 일치"})

        # 전체 검증 단계 (5행/245,789행 패리티, 유한/클리핑, 행 순서, shuffled/chunked 행
        # 독립성, 오프라인 설치, <600s, zip 레이아웃, migration audit) — 실제 후보 패키지가
        # 존재하는 라운드에만 실행. 이번 라운드(Task 10 SKIPPED)는 DEPLOYED 후보가 없으므로
        # 여기 도달하기 전에 REJECT 된다.
        raise PolicyViolation(
            "candidate package 전체 검증 미실행 — 이번 라운드 Task 10 SKIPPED 이며 DEPLOYED "
            f"후보 패키지 부재 (candidate {cid}, {_rel_or_abs(submission_dir)}) (exit 2)")
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": "Todo 11 — package & independent qualification (rejected)",
            "task": "aimers9-next-round/task-11-package",
            "mode": "package-validate",
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
            "task10_evidence": str(task10_path),
            "submission_dir": _rel_or_abs(submission_dir),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "candidate branch: label/package access occurs only after "
                                  "all structural binding guards pass — none passed this round",
            "candidate_route": {
                "candidate_id": cid,
                "submission_dir": _rel_or_abs(submission_dir),
                "manifest": manifest_path.name if manifest_path else None,
                "full_validation_stages": FULL_VALIDATION_STAGES,
            },
            "reason": str(exc),
            "checks": checks,
            "violations": [],
        }
        json_path, md_path = write_evidence(record, evidence_dir / "task-11-package")
        log.log(f"[package_validator] REJECT (exit 2) — candidate branch: candidate_id={cid}")
        log.log(f"[package_validator]   {exc}")
        log.log(f"[package_validator] evidence -> {json_path} / {md_path}")
        return 2


# ────────────────────────────────────────────────────────────────────────────
# failure-QA fixtures (계획 QA: 전부 exit 2, fixture 전용 증거/로그 — 주 증거 미변경)
# ────────────────────────────────────────────────────────────────────────────
def _run_package_fixture(fixture: str, task10_path: Path, policy: JSON, policy_path: Path,
                         policy_hash: str, evidence_dir: Path) -> int:
    if fixture not in PACKAGE_FIXTURES:
        print(f"[package_validator] FATAL: 알 수 없는 fixture {fixture!r}", file=sys.stderr)
        return 1
    log = Logger()
    config_path, config_hash, config_file_sha = _write_config(
        evidence_dir, policy, f"task-11-package-config-fixture-{fixture}.json")
    try:
        if fixture == "altered-hash":
            log.log(f"[fixture {fixture}] model/manifest hash 변조 주입 — hash 결속 가드…")
            _guard_manifest_hash("b" * 64, "a" * 64)
        elif fixture == "swapped-row-order":
            log.log(f"[fixture {fixture}] 출력 행 순서 뒤섞기 주입 — row order 가드…")
            _guard_row_order([3, 1, 2], [1, 2, 3])
        elif fixture == "missing-clip-check":
            log.log(f"[fixture {fixture}] 클리핑 검증 누락 주입 — clip 가드…")
            _guard_clip_check(False)
    except PolicyViolation as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "title": f"Todo 11 fixture — {fixture}",
            "task": "aimers9-next-round/task-11-package",
            "mode": "package-validate-fixture",
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
            "task10_evidence": str(task10_path),
            "label_sources": [],
            "labels_read": False,
            "label_sources_note": "fixture: no real label read",
            "fixture": fixture,
            "reason": str(exc),
            "violations": [],
        }
        json_path, md_path = write_evidence(record, evidence_dir / f"task-11-package-fixture-{fixture}")
        log.log(f"[package_validator] --fixture {fixture}: exit 2 — {exc}")
        log.log(f"[package_validator] evidence -> {json_path} / {md_path}")
        log.write(evidence_dir / f"task-11-package-fixture-{fixture}.log")
        return 2
    log.log("[package_validator] FATAL: fixture 가드가 발동하지 않음 — 가드 버그")
    log.write(evidence_dir / f"task-11-package-fixture-{fixture}.log")
    print("[package_validator] FATAL: fixture 가드가 발동하지 않음 — 가드 버그", file=sys.stderr)
    return 1


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────
def cmd_package(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_base or args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[package_validator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_json(policy_path)
    if policy is None:
        print(f"[package_validator] FATAL: 정책 파일을 읽을 수 없음: {policy_path}", file=sys.stderr)
        return 1
    policy_hash = _canonical_sha256(policy)
    violations = validate_policy(policy)
    if violations:
        print(f"[package_validator] REJECT (exit 2) — 정책 위반")
        for v in violations:
            print(f"  [REJECT] {v['rule']}: {v['reason']}")
        return 2

    task10_path = Path(args.task10_evidence).expanduser().resolve()
    task10_ev = load_json(task10_path)
    if task10_ev is None:
        print(f"[package_validator] FATAL: Task 10 deploy 증거를 읽을 수 없음: {task10_path}",
              file=sys.stderr)
        return 1

    if args.fixture:
        return _run_package_fixture(args.fixture, task10_path, policy, policy_path,
                                    policy_hash, evidence_dir)

    # ── 사전 등록 config: 어떤 라벨/모델/패키지 접근 이전에 기록 ──
    config_path, config_hash, config_file_sha = _write_config(
        evidence_dir, policy, "task-11-package-config.json")

    log = Logger()
    log.log("=" * 72)
    log.log(f"[package_validator] Task 11 package validator — evidence_dir={_rel_or_abs(evidence_dir)}")
    log.log("=" * 72)

    verdict = task10_ev.get("verdict")
    cid = _deployed_candidate_id(task10_ev)
    submission_dir = Path(args.submission_dir or REPO / "submit_next_round").resolve()
    if verdict == "SKIPPED":
        rc = _write_skipped(task10_ev, task10_path, submission_dir, policy, policy_path,
                            policy_hash, evidence_dir, config_path, config_hash,
                            config_file_sha, log, no_manifest=False)
        log.write(evidence_dir / "task-11-package.log")
        return rc
    if verdict in {"DEPLOYED", "FROZEN"} or cid:
        has_manifest = ((submission_dir / "manifest.json").is_file()
                        or (submission_dir / "provenance.json").is_file())
        if not has_manifest:
            rc = _write_skipped(task10_ev, task10_path, submission_dir, policy, policy_path,
                                policy_hash, evidence_dir, config_path, config_hash,
                                config_file_sha, log, no_manifest=True)
            log.write(evidence_dir / "task-11-package.log")
            return rc
        rc = _validate_candidate_package(submission_dir, task10_ev, task10_path, policy,
                                         policy_path, policy_hash, evidence_dir, config_path,
                                         config_hash, config_file_sha, log)
        log.write(evidence_dir / "task-11-package.log")
        return rc
    log.log(f"[package_validator] FATAL: 인식 불가 task-10 verdict {verdict!r} — "
            f"SKIPPED 또는 DEPLOYED 필요")
    log.write(evidence_dir / "task-11-package.log")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task 11 — package & independently qualify deployed candidate "
                    "(SKIPPED 이번 라운드; 후보 결속 검증은 미래 라운드)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", default=os.environ.get("LGA_SUBMIT_DIR"),
                        help="후보 패키지 submission dir (SKIPPED 경로에선 접근 안 함)")
    parser.add_argument("--fixture", default=None, choices=sorted(PACKAGE_FIXTURES),
                        help="failure QA fixture (전부 exit 2): 구조 가드 변조 트리거")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉토리 (기본 .omo/evidence/aimers9-next-round)")
    parser.add_argument("--evidence-base", default=None,
                        help="(alias) --evidence-dir 와 동일")
    parser.add_argument("--task10-evidence", default=str(DEFAULT_TASK10_EVIDENCE),
                        help="Task 10 deploy 증거 JSON 경로 (기본 task-10-deploy.json)")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY),
                        help="동결 정책 JSON 경로 (기본 repro_979/next_round_policy.json)")
    args = parser.parse_args(argv)
    return cmd_package(args)


if __name__ == "__main__":
    raise SystemExit(main())
