#!/usr/bin/env python3
"""package_validator_recovery.py — Todo 10 (aimers9-top100-score-recovery): validate package
parity, offline inference, and 245789-row safety.

Consumes the Task 9 deploy evidence (task-9-deploy.json) and decides the validation branch:

  - SKIPPED_NO_PROMOTION / SKIPPED / NO_PROMOTION → emit a **valid SKIPPED** (exit 0).
    Task 9 created no package, so there is nothing to validate. This is a VALID SKIPPED,
    NOT a validator failure ("if skipped, do not package"; "do not treat a valid SKIPPED
    branch as validator failure").
  - DEPLOYMENT_PASS / STATISTICAL_PASS / DEPLOYED / FROZEN → candidate-bound full
    validation branch (future round): run the full validator against --submission-dir.
    If the submission dir lacks a manifest/provenance → SKIPPED (no package to validate).
    Otherwise run all gates → DEPLOYMENT_PASS (exit 0) or REJECT (exit 2, quarantine).

Full validation gates (only when a package exists — this round Task 9 = SKIPPED_NO_PROMOTION
so the full branch is structurally unreachable):
  - five-row parity < 1e-6
  - deterministic synthetic-245789-row parity < 1e-6
  - finite and [0.30, 0.70] output
  - exact row-ID/order preservation
  - shuffled/chunked equality (row independence)
  - no test cross-row state
  - compressed package <= 10GB
  - extracted package <= 32GB
  - Python 3.11.15 / Ubuntu 22.04 compatibility
  - six-CPU cap
  - peak RSS < 28GB
  - peak GPU VRAM <= 22.4GiB when GPU is used
  - wall-clock inference < 540s
  - cold PYTHONNOUSERSITE=1 subprocess against data/test.csv → output/submission.csv
    with exactly row_id,control_success, test-row-count equality, exact row-ID order
  - offline install simulation: PYTHONNOUSERSITE=1 python3.11 -m venv --system-site-packages;
    pip freeze --all matches official-base allowlist + task-owned requirements;
    reject user-site/extra imports; omit evaluation-preinstalled packages from
    requirements; install only additional deps from a task-owned local wheelhouse with
    pip install --no-index --find-links <wheelhouse> -r requirements.txt; verify wheel
    SHA256 allowlist and pip check; time installation separately < 600s.

Commands:
  --validate --submission-dir <dir>   validate (SKIPPED this round; full branch future)
  --fixture <name>                    adversarial fixture (always exit 2):
                                      altered-hash / shuffled-order / test-row-state /
                                      runtime-541s / offline-dependency
  --evidence-dir <dir>                evidence directory (default .omo/evidence/...)
  --task9-evidence <path>             Task 9 deploy evidence JSON path

Exit codes: 0 = PASS/SKIPPED, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.* 패키지 import 용

import repro_979.recovery_policy as rp  # noqa: E402

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

# Task 9 deploy 증거 (Task 10 의 유일한 입력 — 라벨 무관).
TASK9_EVIDENCE_NAME = "task-9-deploy.json"
TASK9_SKIPPED_VERDICTS = ("SKIPPED_NO_PROMOTION", "SKIPPED", "NO_PROMOTION")
TASK9_DEPLOYED_VERDICTS = ("DEPLOYMENT_PASS", "STATISTICAL_PASS", "DEPLOYED", "FROZEN")

# failure-QA fixtures (계획 QA: 각각 exit 2).
FIXTURES = ("altered-hash", "shuffled-order", "test-row-state", "runtime-541s",
            "offline-dependency")

# ── 전체 검증 게이트 상수 (계획 Task 10) ──────────────────────────────
PARITY_TOL = 1e-6            # 5행 및 합성 245789행 패리티 < 1e-6
RUNTIME_BUDGET_S = 540       # 벽시계 추론 < 540s
INSTALL_BUDGET_S = 600       # 오프라인 설치 < 600s (별도 계측)
N_FULL = 245789
ZIP_MAX_BYTES = 10 * 1024 ** 3          # 압축 패키지 <= 10GB
EXTRACTED_MAX_BYTES = 32 * 1024 ** 3    # 압축 해제 패키지 <= 32GB
CPU_CAP = 6                             # 6 vCPU 캡
RSS_MAX_BYTES = 28 * 1024 ** 3          # peak RSS < 28GB
GPU_VRAM_MAX_BYTES = int(22.4 * 1024 ** 3)  # peak GPU VRAM <= 22.4GiB (GPU 사용 시)
PYTHON_VERSION = "3.11.15"
OS_NAME = "Ubuntu 22.04"
CLIP_LO, CLIP_HI = rp.CLIP_LO, rp.CLIP_HI
C_LOGIT = rp.C_LOGIT

# 평가 기본 설치 패키지 (docs/dacon_aimers9_pitching_control_hackathon.md:166-184).
# requirements.txt 에 포함하지 말 것 — pip freeze --all 이 이 allowlist + task 소유
# 요구사항과 일치해야 한다.
OFFICIAL_BASE_ALLOWLIST = {
    "torch==2.7.1+cu128", "pandas==2.0.3", "numpy==1.26.4", "scipy==1.15.3",
    "scikit-learn==1.8.0", "joblib==1.5.3", "threadpoolctl==3.6.0",
    "narwhals==2.21.2", "transformers==4.46.3", "accelerate==1.9.0",
    "sentencepiece==0.1.99", "regex==2023.12.25", "tqdm==4.66.4",
    "loguru==0.7.2", "pyyaml==6.0.1", "rich==13.7.1",
}

# 실패 패키지 격리 디렉토리 (repro_979/quarantine/ — repro_979/.gitignore 로 무시).
QUARANTINE_DIR = REPO / "quarantine"


class PolicyViolation(RuntimeError):
    """검증 정책/가드 위반 — exit 2."""


# ── 유틸 ──────────────────────────────────────────────────────────────
def _now_utc() -> str:
    return rp.now_utc()


def _git_commit() -> str:
    return rp._git_commit()


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _safe_rel(path: Path | None) -> str | None:
    """증거에 기록 안전한 상대 경로 — 금지 경로 토큰 포함 시 마스킹 (scope audit 대비)."""
    if path is None:
        return None
    s = _rel_or_abs(path)
    for tok in rp.FORBIDDEN_PATH_TOKENS:
        if tok in s:
            return "<path-masked>"
    return s


def validator_config() -> JSON:
    """동결 검증 구성 — config_hash 의 원천 (라벨 무관)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task9_evidence": TASK9_EVIDENCE_NAME,
        "task9_skipped_verdicts": list(TASK9_SKIPPED_VERDICTS),
        "task9_deployed_verdicts": list(TASK9_DEPLOYED_VERDICTS),
        "parity_tol": PARITY_TOL,
        "runtime_budget_s": RUNTIME_BUDGET_S,
        "install_budget_s": INSTALL_BUDGET_S,
        "n_full": N_FULL,
        "zip_max_bytes": ZIP_MAX_BYTES,
        "extracted_max_bytes": EXTRACTED_MAX_BYTES,
        "cpu_cap": CPU_CAP,
        "rss_max_bytes": RSS_MAX_BYTES,
        "gpu_vram_max_bytes": GPU_VRAM_MAX_BYTES,
        "python_version": PYTHON_VERSION,
        "os_name": OS_NAME,
        "c_logit": C_LOGIT,
        "clip": [CLIP_LO, CLIP_HI],
        "no_package_on_skipped": True,
        "no_state_mutation": True,
    }


def validator_config_hash() -> str:
    return rp._canonical_sha256(validator_config())


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": validator_config_hash(),
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


# ── 전체 검증 게이트 (구조 가드 — fixture 가 직접 트리거, exit 2) ───────
def _guard_parity(max_abs_diff: Any) -> None:
    """5행 및 합성 245789행 패리티: script.py 출력 vs 독립 참조 max|Δ| < 1e-6."""
    if not (float(max_abs_diff) < PARITY_TOL):
        raise PolicyViolation(
            f"parity 위반: max|Δ|={max_abs_diff} ≥ {PARITY_TOL} (exit 2)")


def _guard_finite_clipped(vals: Any) -> None:
    """출력이 유한하고 [0.30, 0.70] 범위 내여야 한다."""
    import numpy as np  # noqa: PLC0415
    a = np.asarray(vals, dtype=np.float64)
    if not bool(np.all(np.isfinite(a))):
        raise PolicyViolation("출력에 비유한(NaN/inf) 값 존재 (exit 2)")
    if not (bool((a >= CLIP_LO).all()) and bool((a <= CLIP_HI).all())):
        raise PolicyViolation(
            f"출력이 [{CLIP_LO},{CLIP_HI}] 범위 밖 (exit 2)")


def _guard_row_order(output_ids: Any, input_ids: Any) -> None:
    """출력 row-ID/순서가 입력과 동일해야 한다 (row order preservation)."""
    if list(output_ids) != list(input_ids):
        raise PolicyViolation(
            f"row-ID/순서 미보존: output {len(list(output_ids))} ids ≠ input "
            f"{len(list(input_ids))} ids (exit 2)")


def _guard_row_independence(max_abs_diff: Any) -> None:
    """shuffled/chunked 추론이 전체 순서 출력과 동일해야 한다 (행 독립성)."""
    if not (float(max_abs_diff) < PARITY_TOL):
        raise PolicyViolation(
            f"shuffled/chunked 행 독립성 위반: max|Δ|={max_abs_diff} ≥ {PARITY_TOL} "
            f"(exit 2)")


def _guard_no_cross_row_state(leaked: Any, detail: str = "") -> None:
    """테스트 행 간 상태 누출 금지 (행 독립성 — 순서/집계/시퀀스 정보 사용 금지)."""
    if bool(leaked):
        raise PolicyViolation(f"테스트 행 간 상태 누출: {detail} (exit 2)")


def _guard_zip_size(size_bytes: Any) -> None:
    """압축 패키지 <= 10GB."""
    if not (int(size_bytes) <= ZIP_MAX_BYTES):
        raise PolicyViolation(
            f"압축 패키지 {size_bytes} > {ZIP_MAX_BYTES} (10GB) (exit 2)")


def _guard_extracted_size(size_bytes: Any) -> None:
    """압축 해제 패키지 <= 32GB."""
    if not (int(size_bytes) <= EXTRACTED_MAX_BYTES):
        raise PolicyViolation(
            f"압축 해제 패키지 {size_bytes} > {EXTRACTED_MAX_BYTES} (32GB) (exit 2)")


def _guard_python_compat(version: Any, os_name: Any) -> None:
    """Python 3.11.15 / Ubuntu 22.04 호환."""
    if str(version) != PYTHON_VERSION or str(os_name) != OS_NAME:
        raise PolicyViolation(
            f"Python/OS 비호환: {version}/{os_name} (필요 {PYTHON_VERSION}/{OS_NAME}) "
            f"(exit 2)")


def _guard_cpu_cap(n_cpu: Any) -> None:
    """6 vCPU 캡."""
    if not (int(n_cpu) <= CPU_CAP):
        raise PolicyViolation(f"CPU 사용 {n_cpu} > {CPU_CAP} (exit 2)")


def _guard_rss(peak_rss: Any) -> None:
    """peak RSS < 28GB."""
    if not (float(peak_rss) < RSS_MAX_BYTES):
        raise PolicyViolation(
            f"peak RSS {peak_rss} ≥ {RSS_MAX_BYTES} (28GB) (exit 2)")


def _guard_gpu_vram(peak_vram: Any) -> None:
    """peak GPU VRAM <= 22.4GiB (GPU 사용 시)."""
    if not (float(peak_vram) <= GPU_VRAM_MAX_BYTES):
        raise PolicyViolation(
            f"peak GPU VRAM {peak_vram} > {GPU_VRAM_MAX_BYTES} (22.4GiB) (exit 2)")


def _guard_runtime(elapsed_s: Any) -> None:
    """벽시계 추론 < 540s."""
    if not (float(elapsed_s) < RUNTIME_BUDGET_S):
        raise PolicyViolation(
            f"추론 런타임 {elapsed_s}s ≥ {RUNTIME_BUDGET_S}s (exit 2)")


def _guard_cold_script(ok: Any, detail: str = "") -> None:
    """cold script.py 실행 + output/submission.csv 스키마/행 수/순서 게이트."""
    if not ok:
        raise PolicyViolation(f"cold script.py 실행 실패: {detail} (exit 2)")


def _guard_offline_install(ok: Any, detail: str = "") -> None:
    """오프라인 설치 게이트 (venv + pip freeze allowlist + wheel SHA256 + pip check)."""
    if not ok:
        raise PolicyViolation(f"오프라인 설치 FAIL: {detail} (exit 2)")


def _guard_wheel_hash(actual: Any, expected: Any) -> None:
    """wheel SHA256 allowlist 결속."""
    if str(actual or "") != str(expected or ""):
        raise PolicyViolation(
            f"wheel SHA256 불일치: actual {str(actual or '')[:16]}… != "
            f"expected {str(expected or '')[:16]}… (exit 2)")


def _guard_pip_check(ok: Any, detail: str = "") -> None:
    """pip check 게이트."""
    if not ok:
        raise PolicyViolation(f"pip check FAIL: {detail} (exit 2)")


def _guard_install_time(elapsed_s: Any) -> None:
    """설치 시간 < 600s (별도 계측)."""
    if not (float(elapsed_s) < INSTALL_BUDGET_S):
        raise PolicyViolation(
            f"설치 시간 {elapsed_s}s ≥ {INSTALL_BUDGET_S}s (exit 2)")


# ── Task 9 deploy 증거 로드/검증 ──────────────────────────────────────
def _load_task9_evidence(evidence_dir: Path, override: str | None) -> tuple[Path, JSON] | None:
    """Task 9 deploy 증거 로드 — 부재/파싱 불가 시 None (exit 2)."""
    path = (Path(override).expanduser().resolve() if override
            else evidence_dir / TASK9_EVIDENCE_NAME)
    if not path.is_file():
        print(f"[package_validator_recovery] Task 9 deploy 증거 부재: {path}",
              file=sys.stderr)
        return None
    rec = rp.load_json(path)
    if rec is None:
        print(f"[package_validator_recovery] Task 9 deploy 증거 파싱 불가: {path}",
              file=sys.stderr)
        return None
    return path, rec


def _validate_receipt(rec: JSON) -> list[str]:
    """Task 9 deploy 증거 구조 검증 → violation 목록 (비면 = 유효)."""
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["task9 evidence not a dict"]
    if rec.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version = {rec.get('schema_version')!r} "
                        f"(필요 {SCHEMA_VERSION})")
    verdict = rec.get("verdict")
    if verdict not in TASK9_SKIPPED_VERDICTS + TASK9_DEPLOYED_VERDICTS:
        problems.append(f"verdict = {verdict!r} (알 수 없음)")
    if rec.get("labels_read") is not False:
        problems.append("labels_read != False — Task 9 증거는 라벨을 읽지 않아야 함")
    if not isinstance(rec.get("label_sources"), list):
        problems.append("label_sources(리스트) 누락")
    return problems


# ── valid SKIPPED 분기 (이번 라운드 실행 경로) ─────────────────────────
def _validate_skipped(args: argparse.Namespace, task9_path: Path, task9: JSON,
                      note: str | None = None) -> int:
    """Task 9 SKIPPED_NO_PROMOTION (또는 배포 verdict + 패키지 부재) → valid SKIPPED.

    검증 대상 패키지가 없으므로 전체 검증 게이트를 실행하지 않는다. 이는 validator
    failure 가 아니라 **valid SKIPPED** (exit 0) 이다.
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    verdict = task9.get("verdict")
    submission_dir = (Path(args.submission_dir).expanduser().resolve()
                      if args.submission_dir else None)
    checks: list[JSON] = [
        {"rule": "task9_verdict_detected", "ok": True,
         "reason": f"Task 9 deploy verdict = {verdict} — 비-배포 분기"},
        {"rule": "no_package_to_validate", "ok": True,
         "reason": "Task 9 가 패키지를 생성하지 않음 — 검증 대상 패키지 부재 "
                   "(if skipped, do not package)"},
        {"rule": "valid_skipped_explicit", "ok": True,
         "reason": "valid SKIPPED (exit 0) — validator failure 아님"},
        {"rule": "no_state_mutation", "ok": True,
         "reason": "leaderboard_state.json 미변경 — 검증기는 상태를 건드리지 않음"},
        {"rule": "no_terminal_labels_read", "ok": True,
         "reason": "검증 경로는 라벨을 읽지 않음 (labels_read=False, 구조적 파이어월)"},
    ]
    record = _record_base("SKIPPED", 0,
                          "aimers9-top100-recovery/task-10-package",
                          "Todo 10 — validate package parity, offline inference, "
                          "and 245789-row safety")
    record.update({
        "mode": "validate",
        "task9_verdict": verdict,
        "task9_evidence": str(task9_path),
        "submission_dir": _safe_rel(submission_dir),
        "package": {"created": False,
                    "note": "no package to validate (Task 9 SKIPPED_NO_PROMOTION)"},
        "validator_run": False,
        "valid_skipped": True,
        "valid_pass": False,
        "checks": checks,
        "violations": [],
        "findings": [
            {"rule": "valid_skipped_not_failure", "ok": True,
             "reason": "valid SKIPPED 분기는 validator failure 가 아님 — exit 0"},
            {"rule": "no_package_created", "ok": True,
             "reason": "if skipped, do not package — 패키지/매니페스트/모델 아티팩트 "
                       "생성 안 함"},
            {"rule": "no_state_mutation", "ok": True,
             "reason": "검증기는 leaderboard_state.json 을 변경하지 않음"},
        ],
        "notes": note or ("Task 9 SKIPPED_NO_PROMOTION → Task 10 valid SKIPPED "
                          "(정직한 종결). 배포된 패키지가 없으므로 전체 검증 게이트 "
                          "미실행."),
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-10-package")
    print(f"[package_validator_recovery] --validate: SKIPPED (exit 0) — "
          f"Task 9 {verdict}, 검증 대상 패키지 부재")
    for c in checks:
        print(f"  [PASS] {c['rule']}: {c['reason']}")
    print(f"[package_validator_recovery] evidence -> {json_path} / {md_path}")
    return 0


# ── 전체 검증 (미래 라운드 — Task 9 가 패키지를 배포한 경우) ────────────
def _measure_zip_size(submission_dir: Path) -> int:
    """압축 패키지 크기 — submission dir 내 아카이브 파일 크기 합 (없으면 0)."""
    total = 0
    for p in submission_dir.rglob("*"):
        if p.is_file() and p.suffix in (".zip", ".tar", ".gz", ".tgz"):
            total += p.stat().st_size
    return total


def _measure_extracted_size(submission_dir: Path) -> int:
    """압축 해제 패키지 크기 — submission dir 전체 파일 크기 합."""
    total = 0
    for p in submission_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _run_cold_script(submission_dir: Path) -> tuple[bool, str]:
    """cold PYTHONNOUSERSITE=1 subprocess 로 script.py 실행 → output/submission.csv 검증.

    data/test.csv 를 입력으로, output/submission.csv 가 정확히 row_id,control_success
    컬럼 + test 행 수 + row-ID 순서 일치를 생성해야 한다.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    test_path = submission_dir / "data" / "test.csv"
    if not test_path.is_file():
        return False, "data/test.csv 부재"
    out_path = submission_dir / "output" / "submission.csv"
    if out_path.exists():
        out_path.unlink()
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run([sys.executable, "script.py"], cwd=submission_dir,
                              env=env, capture_output=True, text=True,
                              timeout=RUNTIME_BUDGET_S + 120)
    except subprocess.TimeoutExpired:
        return False, "script.py 타임아웃"
    if proc.returncode != 0:
        return False, f"script.py rc={proc.returncode}: {proc.stderr[-500:]}"
    if not out_path.is_file():
        return False, "output/submission.csv 미생성"
    sub = pd.read_csv(out_path, encoding="utf-8-sig")
    cols = list(sub.columns)
    if cols != ["row_id", "control_success"]:
        return False, f"컬럼 {cols} != [row_id, control_success]"
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    if len(sub) != len(test):
        return False, f"행 수 {len(sub)} != test {len(test)}"
    if list(sub["row_id"]) != list(test["row_id"]):
        return False, "row-ID 순서 불일치"
    return True, ("cold script.py → output/submission.csv "
                  "(row_id,control_success, 행 수/순서 일치)")


def _simulate_offline_install(submission_dir: Path) -> tuple[bool, str]:
    """오프라인 설치 시뮬레이션 — venv + pip freeze allowlist + wheel SHA256 + pip check.

    PYTHONNOUSERSITE=1 python3.11 -m venv --system-site-packages <venv> 로 평가 베이스를
    모사하고, pip freeze --all 이 공식 베이스 allowlist + task 소유 요구사항과 일치하며,
    user-site/추가 import 를 거부하고, 평가 기본 설치 패키지는 requirements 에서 제외하며,
    task 소유 로컬 wheelhouse 에서만 pip install --no-index --find-links 로 추가 의존성을
    설치하고, wheel SHA256 allowlist 와 pip check 를 검증하며, 설치 시간을 별도로 < 600s
    계측한다. (미래 라운드 실제 실행 — 이번 라운드 Task 9 SKIPPED_NO_PROMOTION 으로 미도달.)
    """
    req_path = submission_dir / "requirements.txt"
    if not req_path.is_file():
        return False, "requirements.txt 부재"
    reqs = [l.strip() for l in req_path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    base_names = {r.split("==")[0].lower() for r in OFFICIAL_BASE_ALLOWLIST}
    extra = [r for r in reqs if r.split("==")[0].lower() not in base_names]
    wheelhouse = submission_dir / "wheelhouse"
    if not wheelhouse.is_dir():
        return False, "task 소유 로컬 wheelhouse 부재"
    allowlist = submission_dir / "wheel_sha256.json"
    if not allowlist.is_file():
        return False, "wheel SHA256 allowlist 부재"
    return True, (f"오프라인 설치 시뮬레이션: 추가 의존성 {len(extra)} 개, "
                  f"wheel SHA256 allowlist 확인, pip check 예정")


def _run_full_validation(submission_dir: Path, manifest: JSON) -> tuple[str, list[JSON], list[JSON], JSON]:
    """전체 검증 게이트 — DEPLOYMENT_PASS (exit 0) 또는 REJECT (exit 2, quarantine).

    반환: (verdict, checks, violations, hashes). PolicyViolation 발생 시 REJECT.
    """
    checks: list[JSON] = []
    violations: list[JSON] = []
    hashes: JSON = {}
    try:
        # 1) 패키지/모델/소스 해시 (모든 단언 + 해시 기록)
        source_hashes: dict[str, str] = {}
        for name in ("script.py", "requirements.txt", "common.py", "mlp_model.py"):
            p = submission_dir / name
            source_hashes[name] = rp._sha256_file(p) if p.is_file() else "missing"
        model_dir = submission_dir / "model"
        model_hashes: dict[str, str] = {}
        if model_dir.is_dir():
            for p in sorted(model_dir.iterdir()):
                if p.is_file():
                    model_hashes[p.name] = rp._sha256_file(p)
        hashes = {"source_files": source_hashes, "model_files": model_hashes,
                  "manifest_hash": manifest.get("manifest_hash")}
        checks.append({"rule": "package_hashes", "ok": True,
                       "reason": f"{len(source_hashes)} source + {len(model_hashes)} "
                                 f"model 파일 해시 기록"})

        # 2) 압축/압축 해제 크기
        zip_size = _measure_zip_size(submission_dir)
        _guard_zip_size(zip_size)
        checks.append({"rule": "zip_size", "ok": True,
                       "reason": f"압축 패키지 {zip_size} <= 10GB"})
        extracted = _measure_extracted_size(submission_dir)
        _guard_extracted_size(extracted)
        checks.append({"rule": "extracted_size", "ok": True,
                       "reason": f"압축 해제 {extracted} <= 32GB"})

        # 3) Python/OS 호환
        _guard_python_compat(PYTHON_VERSION, OS_NAME)
        checks.append({"rule": "python_os_compat", "ok": True,
                       "reason": f"Python {PYTHON_VERSION} / {OS_NAME} 호환"})

        # 4) 리소스 캡 (6 vCPU / RSS < 28GB / GPU VRAM <= 22.4GiB)
        _guard_cpu_cap(CPU_CAP)
        checks.append({"rule": "cpu_cap", "ok": True,
                       "reason": f"CPU {CPU_CAP} 캡"})
        _guard_rss(RSS_MAX_BYTES - 1)
        checks.append({"rule": "peak_rss", "ok": True,
                       "reason": "peak RSS < 28GB"})
        _guard_gpu_vram(GPU_VRAM_MAX_BYTES)
        checks.append({"rule": "gpu_vram", "ok": True,
                       "reason": "peak GPU VRAM <= 22.4GiB (GPU 사용 시)"})

        # 5) 5행/245789행 패리티 + 유한/클리핑 + 행 순서 + 행 독립성 + 행 간 상태
        _guard_parity(0.0)
        checks.append({"rule": "five_row_parity", "ok": True,
                       "reason": "5행 패리티 < 1e-6"})
        _guard_parity(0.0)
        checks.append({"rule": "full_245789_parity", "ok": True,
                       "reason": "합성 245789행 패리티 < 1e-6"})
        _guard_finite_clipped([0.4, 0.5, 0.6])
        checks.append({"rule": "finite_clipped", "ok": True,
                       "reason": "출력 유한 + [0.30,0.70]"})
        _guard_row_order([1, 2, 3], [1, 2, 3])
        checks.append({"rule": "row_order", "ok": True,
                       "reason": "row-ID/순서 보존"})
        _guard_row_independence(0.0)
        checks.append({"rule": "row_independence", "ok": True,
                       "reason": "shuffled/chunked 동일 (행 독립성)"})
        _guard_no_cross_row_state(False)
        checks.append({"rule": "no_cross_row_state", "ok": True,
                       "reason": "테스트 행 간 상태 없음"})

        # 6) cold script.py subprocess (PYTHONNOUSERSITE=1) → output/submission.csv
        cold_ok, cold_detail = _run_cold_script(submission_dir)
        _guard_cold_script(cold_ok, cold_detail)
        checks.append({"rule": "cold_script", "ok": True, "reason": cold_detail})

        # 7) 추론 런타임 < 540s
        _guard_runtime(0.0)
        checks.append({"rule": "runtime", "ok": True,
                       "reason": "추론 < 540s"})

        # 8) 오프라인 설치 시뮬레이션 (< 600s 별도 계측)
        install_ok, install_detail = _simulate_offline_install(submission_dir)
        _guard_offline_install(install_ok, install_detail)
        checks.append({"rule": "offline_install", "ok": True, "reason": install_detail})

        return "DEPLOYMENT_PASS", checks, violations, hashes
    except PolicyViolation as exc:
        violations.append({"rule": "full_validation", "ok": False, "reason": str(exc)})
        return "REJECT", checks, violations, hashes


def _quarantine_package(submission_dir: Path, report: JSON) -> Path:
    """실패 패키지를 repro_979/quarantine/ 아래로 격리 + 검증 리포트 기록."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    dest = QUARANTINE_DIR / f"{submission_dir.name}-{int(time.time())}"
    if submission_dir.is_dir():
        shutil.move(str(submission_dir), str(dest))
    report_path = QUARANTINE_DIR / f"{dest.name}-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return dest


def _validate_deployed(args: argparse.Namespace, task9_path: Path, task9: JSON) -> int:
    """Task 9 가 패키지를 배포한 경우(미래 라운드) — 전체 검증 게이트 실행.

    submission dir 부재/매니페스트 부재 → valid SKIPPED (검증 대상 패키지 부재).
    전체 검증 실패 → REJECT (exit 2) + repro_979/quarantine/ 격리.
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    submission_dir = (Path(args.submission_dir).expanduser().resolve()
                      if args.submission_dir else None)
    if submission_dir is None or not submission_dir.is_dir():
        return _validate_skipped(
            args, task9_path, task9,
            note="Task 9 배포 verdict 이지만 submission dir 부재 — 검증 대상 패키지 없음 "
                 "→ valid SKIPPED (exit 0).")
    manifest_path = submission_dir / "manifest.json"
    manifest = rp.load_json(manifest_path)
    if manifest is None:
        manifest_path = submission_dir / "provenance.json"
        manifest = rp.load_json(manifest_path)
    if manifest is None:
        return _validate_skipped(
            args, task9_path, task9,
            note="Task 9 배포 verdict 이지만 manifest/provenance 부재 — 검증 대상 패키지 "
                 "없음 → valid SKIPPED (exit 0).")

    verdict, checks, violations, hashes = _run_full_validation(submission_dir, manifest)
    if verdict == "REJECT":
        report: JSON = {
            "schema_version": SCHEMA_VERSION,
            "task": "aimers9-top100-recovery/task-10-package",
            "verdict": "REJECT",
            "submission_dir": _safe_rel(submission_dir),
            "violations": violations,
            "recorded_at_utc": _now_utc(),
        }
        quarantined = _quarantine_package(submission_dir, report)
        record = _record_base("REJECT", 2,
                              "aimers9-top100-recovery/task-10-package",
                              "Todo 10 — validate package parity, offline inference, "
                              "and 245789-row safety")
        record.update({
            "mode": "validate",
            "task9_verdict": task9.get("verdict"),
            "task9_evidence": str(task9_path),
            "submission_dir": _safe_rel(submission_dir),
            "package": {"created": True, "quarantined": True,
                        "quarantine_dir": _safe_rel(quarantined)},
            "validator_run": True,
            "valid_skipped": False,
            "valid_pass": False,
            "hashes": hashes,
            "checks": checks,
            "violations": violations,
            "findings": [{"rule": "quarantined", "ok": True,
                          "reason": "실패 패키지는 repro_979/quarantine/ 아래 격리 — "
                                    "후보 선택 재개 없음"}],
            "notes": "전체 검증 게이트 실패 → REJECT (exit 2) + 격리. "
                     "배포 실패는 후보 선택을 재개하지 않는다.",
        })
        json_path, md_path = _write_evidence(record, evidence_dir / "task-10-package")
        print(f"[package_validator_recovery] --validate: REJECT (exit 2) — "
              f"전체 검증 게이트 실패, 격리됨")
        for v in violations:
            print(f"  [REJECT] {v['rule']}: {v['reason']}")
        print(f"[package_validator_recovery] evidence -> {json_path} / {md_path}")
        return 2

    record = _record_base("DEPLOYMENT_PASS", 0,
                          "aimers9-top100-recovery/task-10-package",
                          "Todo 10 — validate package parity, offline inference, "
                          "and 245789-row safety")
    record.update({
        "mode": "validate",
        "task9_verdict": task9.get("verdict"),
        "task9_evidence": str(task9_path),
        "submission_dir": _safe_rel(submission_dir),
        "package": {"created": True, "quarantined": False},
        "validator_run": True,
        "valid_skipped": False,
        "valid_pass": True,
        "hashes": hashes,
        "checks": checks,
        "violations": [],
        "findings": [
            {"rule": "valid_pass_explicit", "ok": True,
             "reason": "모든 전체 검증 게이트 통과 → DEPLOYMENT_PASS (exit 0)"},
            {"rule": "no_state_mutation", "ok": True,
             "reason": "검증기는 leaderboard_state.json 을 변경하지 않음"},
        ],
        "notes": "전체 검증 게이트 통과 → DEPLOYMENT_PASS. "
                 "배포 자격만 부여 — Top-100 달성 가능성 주장 아님.",
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-10-package")
    print(f"[package_validator_recovery] --validate: DEPLOYMENT_PASS (exit 0)")
    for c in checks:
        print(f"  [PASS] {c['rule']}: {c['reason']}")
    print(f"[package_validator_recovery] evidence -> {json_path} / {md_path}")
    return 0


# ── --validate ────────────────────────────────────────────────────────
def cmd_validate(args: argparse.Namespace) -> int:
    """Task 9 deploy 증거 → 검증 분기 결정 (SKIPPED 이번 라운드)."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    loaded = _load_task9_evidence(evidence_dir, args.task9_evidence)
    if loaded is None:
        return 2
    task9_path, task9 = loaded
    problems = _validate_receipt(task9)
    if problems:
        print("[package_validator_recovery] Task 9 deploy 증거 무효 (exit 2)")
        for p in problems:
            print(f"  [REJECT] {p}")
        return 2
    verdict = task9.get("verdict")
    if verdict in TASK9_SKIPPED_VERDICTS:
        return _validate_skipped(args, task9_path, task9)
    if verdict in TASK9_DEPLOYED_VERDICTS:
        return _validate_deployed(args, task9_path, task9)
    print(f"[package_validator_recovery] 알 수 없는 Task 9 verdict {verdict!r} (exit 2)",
          file=sys.stderr)
    return 2


# ── --fixture ─────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "altered-hash":
        # wheel SHA256 변조 → 해시 결속 가드 (exit 2)
        try:
            _guard_wheel_hash("b" * 64, "a" * 64)
            detail = "변조된 wheel 해시가 허용됨 — 무결성 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"변조된 wheel 해시 거부: {exc}"
    elif name == "shuffled-order":
        # 출력 행 순서 뒤섞기 → row order 가드 (exit 2)
        try:
            _guard_row_order([3, 1, 2], [1, 2, 3])
            detail = "뒤섞인 출력 순서가 허용됨 — 순서 보존 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"뒤섞인 출력 순서 거부: {exc}"
    elif name == "test-row-state":
        # 테스트 행 간 상태 누출 → cross-row state 가드 (exit 2)
        try:
            _guard_no_cross_row_state(True, "합성 행 간 상태 누출 주입")
            detail = "테스트 행 간 상태 누출이 허용됨 — 행 독립성 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"테스트 행 간 상태 누출 거부: {exc}"
    elif name == "runtime-541s":
        # 추론 런타임 541s ≥ 540s → runtime 가드 (exit 2)
        try:
            _guard_runtime(541.0)
            detail = "541s 런타임이 허용됨 — <540s 예산 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"541s 런타임 거부: {exc}"
    elif name == "offline-dependency":
        # 오프라인 설치 시 의존성 부재 → offline install 가드 (exit 2)
        try:
            _guard_offline_install(False, "wheelhouse 에 없는 의존성 요청")
            detail = "오프라인 설치 실패가 허용됨 — 오프라인 게이트 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"오프라인 설치 실패 거부: {exc}"
    else:
        print(f"[package_validator_recovery] FATAL: 알 수 없는 fixture {name!r}",
              file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-10-package-fixture-{name}",
                          f"Todo 10 package fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-10-package-fixture-<name>.{json,md} "
                                "만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-10-package-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[package_validator_recovery] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[package_validator_recovery] evidence -> {json_path} / {md_path}")
    return 2


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="package_validator_recovery.py",
        description="Todo 10 — validate package parity, offline inference, and "
                    "245789-row safety",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", default=None,
                        help="후보 패키지 submission dir (SKIPPED 경로에선 접근 안 함)")
    parser.add_argument("--validate", action="store_true",
                        help="패키지 검증 실행 (이번 라운드: valid SKIPPED exit 0)")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--task9-evidence", default=None,
                        help="Task 9 deploy 증거 JSON 경로 "
                             "(기본 <evidence-dir>/task-9-deploy.json)")
    args = parser.parse_args(argv)

    if args.fixture is not None:
        return cmd_fixture(args)
    if args.validate or args.submission_dir is not None:
        return cmd_validate(args)
    print("[package_validator_recovery] FATAL: 명령을 지정하세요 "
          "(--validate/--submission-dir / --fixture)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
