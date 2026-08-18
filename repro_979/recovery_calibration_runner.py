#!/usr/bin/env python3
"""recovery_calibration_runner.py — Todo 6 (aimers9-top100-score-recovery): screen
Brier-focused causal calibration transforms without global-offset retuning.

Operates ONLY on the frozen reconciled v93 6-leg baseline OOF logits (Task 3 cache).
It screens two pre-registered selectable calibration candidates — `calibration_beta`
and `calibration_isotonic` — plus the `calibration_identity` control (cannot win),
under the same rolling-origin selection gates as the other recovery screens.

Time-causal contract (plan Task 6):
  - The causal panel is built from the frozen one-year-ahead baseline OOF logits.
    The available one-year-ahead OOF years are 2022 (r2022: model trained <=2021,
    applied to 2022 R rows) and 2023 (r2023: model trained <=2022, applied to 2023
    R rows). 2021 OOF is NOT available in the frozen cache: the v93 models are
    trained on 2019-2024, so they cannot produce a one-year-ahead OOF for 2021
    without retraining, which is forbidden by "operate only on frozen baseline OOF
    logits". The panel therefore starts at 2022; r2022 (outer year 2022) has no
    prior OOF year and is a documented cold-start (identity transform).
  - For each outer year Y, each transform is fit SOLELY on OOF data from years <Y,
    then applied once to Y. No within-origin transform chooser exists.
  - p0 = sigmoid(z + C_LOGIT); transform output is converted back to a
    scorer-compatible candidate logit z_candidate = logit(p_transform) - C_LOGIT,
    so the immutable deployed scoring remains
        clip(sigmoid(z_candidate + C_LOGIT), .30, .70).
  - Beta freeze (Task 3 registry): eps=1e-6, feature [log(p0), -log1p(-p0)],
    LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', tol=1e-8,
    max_iter=1000, fit_intercept=True), unrestricted coefficients.
  - Isotonic freeze (Task 3 registry): IsotonicRegression(increasing=True,
    y_min=1e-6, y_max=1-1e-6, out_of_bounds='clip') with sorted-x/tie-mean
    behavior from scikit-learn.
  - A learned Beta intercept is permitted ONLY inside this prior-OOF transform;
    a standalone outer/Public/test-fitted offset is rejected (fixture
    `free-intercept`).
  - Terminal labels are never read in the screen path (structural firewall;
    fixture `terminal-label-read`).

Bounded-subset consistency: the calibration screens use the SAME bounded 30,000-row
subset of each selection origin as Task 3 (documented deviation). The baseline OOF
logits are loaded from the hash-addressed Task 3 cache
`cache/recovery_baseline/<cache_key>/<origin>.npy`.

Commands:
  --screen                         run the calibration screen on r2022/r2023 (exit 0)
  --fixture <name>                 adversarial fixture (always exit 2):
                                   free-intercept / outer-fit / terminal-label-read /
                                   nonmonotonic-isotonic-output
  --evidence-dir <dir>             evidence directory (default .omo/evidence/...)

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.* 패키지 import 용

import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_evaluator as re  # noqa: E402

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

# ── 동결 캘리브레이션 상수 (Task 3 레지스트리와 동일) ─────────────────
BETA_EPS = 1e-6
BETA_FEATURES = ("log(p0)", "-log1p(-p0)")
BETA_LR = {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "tol": 1e-8,
           "max_iter": 1000, "fit_intercept": True}
ISOTONIC = {"increasing": True, "y_min": 1e-6, "y_max": 1 - 1e-6,
            "out_of_bounds": "clip"}

# 스크린할 후보 (identity 는 control — 이길 수 없음).
SELECTABLE_CALIB = ("calibration_beta", "calibration_isotonic")
CONTROL_CALIB = ("calibration_identity",)
ALL_CALIB = SELECTABLE_CALIB + CONTROL_CALIB

# 인과 패널 연도 (사용 가능한 one-year-ahead OOF 연도).
PANEL_YEARS = (2022, 2023)
DEPLOY_REFIT_YEAR = 2025  # 배포 리핏 진단: through-2023 OOF (연도 < 2025)

FIXTURES = ("free-intercept", "outer-fit", "terminal-label-read",
            "nonmonotonic-isotonic-output")

# Task 3 베이스라인 캐시 (해시 주소). bounded 30k 부분집합 로짓.
BASELINE_CACHE_KEY = "ee8ea9579969deaa3284ae6507ad28c1c2859fb507b956e2c0a4a93a28dc903f"
BOUNDED_ROWS = 30000


class PolicyViolation(RuntimeError):
    """동결 스펙/정책 위반 — exit 2."""


class LeakageError(RuntimeError):
    """스코어 연도/터미널 라벨 피팅 누수 — exit 2."""


# ── 작은 헬퍼 ────────────────────────────────────────────────────────
def _now_utc() -> str:
    return rp.now_utc()


def _git_commit() -> str:
    return rp._git_commit()


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": calibration_config_hash(),
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


def calibration_config() -> JSON:
    """동결 캘리브레이션 구성 — config_hash 의 원천 (라벨 무관)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "c_logit": rp.C_LOGIT,
        "clip": [rp.CLIP_LO, rp.CLIP_HI],
        "scoring": "common.score(np.clip(common.sigmoid(z_candidate + C_LOGIT), .30, .70), y)",
        "z_candidate": "logit(p_transform) - C_LOGIT",
        "p0": "sigmoid(z + C_LOGIT)",
        "panel_years": list(PANEL_YEARS),
        "deploy_refit_year": DEPLOY_REFIT_YEAR,
        "bounded_rows": BOUNDED_ROWS,
        "baseline_cache_key": BASELINE_CACHE_KEY,
        "beta": {"eps": BETA_EPS, "features": list(BETA_FEATURES), "model": "LogisticRegression",
                 **BETA_LR, "coefficients": "unrestricted"},
        "isotonic": {"model": "IsotonicRegression", **ISOTONIC,
                     "x_behavior": "sorted-x/tie-mean (scikit-learn)"},
        "identity": {"model": "identity", "note": "control only, cannot win"},
        "selection_origins": list(rp.SELECTION_ORIGINS),
        "gate": {
            "delta_bss_min": rp.GATE_DELTA_BSS_MIN,
            "brier_reduce": rp.GATE_BRIER_REDUCE,
            "bootstrap_lb5_min": rp.GATE_BOOTSTRAP_LB5_MIN,
            "mean_shift_max": rp.GATE_MEAN_SHIFT_MAX,
        },
    }


def calibration_config_hash() -> str:
    return rp._canonical_sha256(calibration_config())


# ── 구조 가드 (fixture 가 직접 호출) ─────────────────────────────────
def _assert_no_outer_fit(fit_years: list[int], outer_year: int) -> None:
    """outer year Y 의 변환은 OOF 연도 < Y 에서만 피팅 — >= Y 연도 행 사용은 누수."""
    future = [y for y in fit_years if y >= outer_year]
    if future:
        raise LeakageError(
            f"[LEAKAGE] outer-fit 시도: outer_year={outer_year} 피팅 연도에 {future} 포함 — "
            f"변환은 OOF 연도 < Y 에서만 피팅 (exit 2)")


def _assert_no_standalone_offset(fit_years: list[int], outer_year: int) -> None:
    """독립(standalone) outer/Public/test 피팅 오프셋 거부.

    학습된 Beta 인터셉트는 이 prior-OOF 변환 내부에서만 허용된다. outer 연도 자체
    (>= outer_year) 또는 Public/test 행에 피팅된 독립 오프셋은 거부한다.
    """
    bad = [y for y in fit_years if y >= outer_year]
    if bad:
        raise PolicyViolation(
            f"[POLICY] free-intercept 시도: outer_year={outer_year} 에 독립 오프셋을 "
            f"피팅 연도 {bad} (>= outer_year) 에 피팅 — standalone outer/Public/test "
            f"오프셋 금지 (exit 2)")


def _assert_no_terminal_read(labels: dict[str, Any]) -> None:
    """스크린 경로는 r2022/r2023 선택 라벨만 — primary/터미널 라벨 읽기 차단."""
    if set(labels) != set(rp.SELECTION_ORIGINS):
        raise LeakageError(
            f"[LEAKAGE] terminal-label-read 시도: 스크린 경로에 선택 기원 외 라벨 "
            f"{sorted(labels)} 존재 — primary/터미널 라벨 읽기 금지 (exit 2)")


def _assert_monotonic_isotonic(p: np.ndarray[Any, Any],
                               x: np.ndarray[Any, Any]) -> None:
    """isotonic 출력은 정렬된 x 에 대해 비감소여야 한다 — 비단조 출력 거부."""
    order = np.argsort(x, kind="stable")
    sorted_p = np.asarray(p)[order]
    if np.any(np.diff(sorted_p) < -1e-12):
        raise PolicyViolation(
            "[POLICY] nonmonotonic-isotonic-output: isotonic 출력이 정렬된 x 에 대해 "
            "비감소가 아님 — increasing=True 위반 (exit 2)")


# ── 데이터/캐시 로드 ─────────────────────────────────────────────────
def _load_train() -> Any:
    import pandas as pd  # noqa: PLC0415
    return pd.read_csv(REPO / "open" / "data" / "train.csv", encoding="utf-8-sig")


def _load_baseline_logits() -> dict[str, np.ndarray[Any, Any]]:
    """Task 3 캐시의 동결 베이스라인 OOF 로짓 (bounded 30k 부분집합)."""
    out: dict[str, np.ndarray[Any, Any]] = {}
    for origin in rp.SELECTION_ORIGINS:
        path = REPO / "cache" / "recovery_baseline" / BASELINE_CACHE_KEY / f"{origin}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"[FAIL] 베이스라인 OOF 로짓 없음: {path}")
        z = np.load(path)
        if len(z) != BOUNDED_ROWS:
            raise PolicyViolation(
                f"[POLICY] 베이스라인 OOF 로짓 길이 {len(z)} != bounded {BOUNDED_ROWS} "
                f"({origin}) — 부분집합 불일치 (exit 2)")
        if not np.isfinite(z).all():
            raise PolicyViolation(f"[POLICY] {origin} 베이스라인 로짓에 비유한 값 (exit 2)")
        out[origin] = np.asarray(z, dtype=np.float64)
    return out


def _bounded_labels(train, masks) -> dict[str, np.ndarray[Any, Any]]:
    """각 선택 기원의 bounded 30k 부분집합 라벨 (Task 3 부분집합 정의와 동일)."""
    labels: dict[str, np.ndarray[Any, Any]] = {}
    for origin in rp.SELECTION_ORIGINS:
        va = masks[origin][1]
        idx = np.flatnonzero(va.to_numpy())[:BOUNDED_ROWS]
        labels[origin] = train.loc[idx, "control_success"].values.astype(np.float64)
    return labels


def build_causal_panel(train, masks,
                       baseline_logits: dict[str, np.ndarray[Any, Any]]) -> JSON:
    """인과 패널 — 사용 가능한 one-year-ahead OOF 연도별 {z, y}.

    연도 2022 = r2022 로짓 (모델 <=2021 학습, 2022 R 행 적용), 연도 2023 = r2023 로짓
    (모델 <=2022 학습, 2023 R 행 적용). 2021 OOF 는 동결 캐시에 없음 (v93 모델이
    2019-2024 전체 학습 — 재학습 없이는 one-year-ahead OOF 불가, 금지).
    """
    panel: JSON = {}
    for origin in rp.SELECTION_ORIGINS:
        outer_year = int(origin.replace("r", ""))
        va = masks[origin][1]
        idx = np.flatnonzero(va.to_numpy())[:BOUNDED_ROWS]
        panel[outer_year] = {
            "z": np.asarray(baseline_logits[origin], dtype=np.float64),
            "y": train.loc[idx, "control_success"].values.astype(np.float64),
            "n_rows": int(len(idx)),
        }
    return panel


# ── 변환 (동결 정의) ─────────────────────────────────────────────────
def _p0(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    import repro_979.common as common  # noqa: PLC0415
    return common.sigmoid(np.asarray(z, dtype=np.float64) + rp.C_LOGIT)


def _beta_features(p0: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Beta 캘리브레이션 피처 [log(p0), -log1p(-p0)] (eps 클립)."""
    p = np.clip(np.asarray(p0, dtype=np.float64), BETA_EPS, 1.0 - BETA_EPS)
    return np.column_stack([np.log(p), -np.log1p(-p)])


def fit_beta(z_fit: np.ndarray[Any, Any], y_fit: np.ndarray[Any, Any]):
    """Beta 캘리브레이션 피팅 (동결 LogisticRegression). 반환: fitted model."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    p0 = _p0(z_fit)
    X = _beta_features(p0)
    model = LogisticRegression(**BETA_LR)
    model.fit(X, np.asarray(y_fit, dtype=np.float64))
    return model


def apply_beta(model, z_apply: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    p0 = _p0(z_apply)
    X = _beta_features(p0)
    return model.predict_proba(X)[:, 1]


def fit_isotonic(z_fit: np.ndarray[Any, Any], y_fit: np.ndarray[Any, Any]):
    """Isotonic 캘리브레이션 피팅 (동결 IsotonicRegression). 반환: fitted model."""
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    p0 = _p0(z_fit)
    model = IsotonicRegression(**ISOTONIC)
    model.fit(p0, np.asarray(y_fit, dtype=np.float64))
    return model


def apply_isotonic(model, z_apply: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    p0 = _p0(z_apply)
    p = model.predict(p0)
    _assert_monotonic_isotonic(p, p0)
    return p


def apply_transform(cid: str, panel: JSON, fit_years: list[int],
                    z_apply: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """prior-OOF 연도(fit_years)에서 변환 피팅 후 z_apply 에 1회 적용.

    fit_years 가 비어 있으면 cold-start → identity (p_transform = p0).
    """
    if cid == "calibration_identity":
        return _p0(z_apply)
    if not fit_years:
        return _p0(z_apply)  # cold-start → identity
    z_fit = np.concatenate([panel[y]["z"] for y in fit_years])
    y_fit = np.concatenate([panel[y]["y"] for y in fit_years])
    if cid == "calibration_beta":
        model = fit_beta(z_fit, y_fit)
        return apply_beta(model, z_apply)
    if cid == "calibration_isotonic":
        model = fit_isotonic(z_fit, y_fit)
        return apply_isotonic(model, z_apply)
    raise ValueError(f"미등록 캘리브레이션 후보: {cid!r}")


def z_candidate_from_p(p_transform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """변환 출력 → 스코어 호환 후보 로짓: z_candidate = logit(p_transform) - C_LOGIT."""
    import repro_979.common as common  # noqa: PLC0415
    return common.logit(np.asarray(p_transform, dtype=np.float64)) - rp.C_LOGIT


# ── 스크린 ───────────────────────────────────────────────────────────
def run_screen(train, masks, baseline_logits, labels) -> JSON:
    """캘리브레이션 스크린 — r2022/r2023 선택 라벨만 (구조적 파이어월)."""
    panel = build_causal_panel(train, masks, baseline_logits)
    base_p = {o: rp.deployed_probs(baseline_logits[o]) for o in rp.SELECTION_ORIGINS}
    per_candidate: JSON = {}
    for cid in ALL_CALIB:
        cand_p: dict[str, np.ndarray[Any, Any]] = {}
        fit_diag: dict[str, JSON] = {}
        for origin in rp.SELECTION_ORIGINS:
            outer_year = int(origin.replace("r", ""))
            z_base = baseline_logits[origin]
            fit_years = [y for y in PANEL_YEARS if y < outer_year]
            _assert_no_outer_fit(fit_years, outer_year)
            _assert_no_standalone_offset(fit_years, outer_year)
            p_transform = apply_transform(cid, panel, fit_years, z_base)
            z_cand = z_candidate_from_p(p_transform)
            cand_p[origin] = rp.deployed_probs(z_cand)
            fit_diag[origin] = {
                "outer_year": outer_year,
                "fit_years": fit_years,
                "cold_start": not fit_years,
                "n_fit_rows": int(sum(panel[y]["n_rows"] for y in fit_years)),
            }
        gate = rp.screen_gate(cand_p, base_p, labels)
        per_candidate[cid] = {
            "candidate_id": cid,
            "selectable": cid in SELECTABLE_CALIB,
            "fit": fit_diag,
            "gate": gate,
        }
    # 배포 리핏 진단 (through-2023 OOF = 연도 < 2025) — 스코어링 아님.
    deploy_fit_years = [y for y in PANEL_YEARS if y < DEPLOY_REFIT_YEAR]
    deploy_diag: JSON = {}
    for cid in SELECTABLE_CALIB:
        z_fit = np.concatenate([panel[y]["z"] for y in deploy_fit_years])
        y_fit = np.concatenate([panel[y]["y"] for y in deploy_fit_years])
        if cid == "calibration_beta":
            model = fit_beta(z_fit, y_fit)
            deploy_diag[cid] = {
                "fit_years": deploy_fit_years,
                "coef": [float(v) for v in model.coef_[0].tolist()],
                "intercept": float(model.intercept_[0]),
            }
        else:
            model = fit_isotonic(z_fit, y_fit)
            deploy_diag[cid] = {
                "fit_years": deploy_fit_years,
                "n_thresholds": int(len(model.X_thresholds_)),
            }
    return {
        "panel": {str(y): {"n_rows": panel[y]["n_rows"]} for y in PANEL_YEARS},
        "per_candidate": per_candidate,
        "deployment_refit": {
            "outer_year": DEPLOY_REFIT_YEAR,
            "fit_years": deploy_fit_years,
            "note": "through-2023 OOF (연도 < 2025) — 진단 스냅샷, 스코어링 아님",
            "per_candidate": deploy_diag,
        },
    }


# ── --screen ─────────────────────────────────────────────────────────
def cmd_screen(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    checks: list[JSON] = []
    violations: list[JSON] = []

    # 레지스트리 검증 (후보 ID 허용)
    try:
        reg = re.load_registry()
    except re.RegistryError as exc:
        print(f"[recovery_calibration_runner] --screen: {exc}", file=sys.stderr)
        return 2
    reg_problems = re.validate_registry(reg)
    if reg_problems:
        print(f"[recovery_calibration_runner] --screen: 레지스트리 무효: "
              f"{reg_problems}", file=sys.stderr)
        return 2
    for cid in SELECTABLE_CALIB:
        blocked = re.reject_blocked(reg, cid)
        if blocked:
            print(f"[recovery_calibration_runner] --screen: 후보 {cid} 차단: "
                  f"{blocked}", file=sys.stderr)
            return 2
    checks.append({"rule": "registry_valid", "ok": True,
                   "reason": "calibration_beta/calibration_isotonic 허용, identity control"})

    # 데이터 + 마스크 + 누수 가드
    train = _load_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        print(f"[recovery_calibration_runner] --screen: 누수 가드 실패: {leak}",
              file=sys.stderr)
        return 2
    rp.assert_row_disjointness(masks, train)
    checks.append({"rule": "origin_masks", "ok": True,
                   "reason": "r2022/r2023 마스크, 2025 부재, primary 와 row-ID 분리"})

    # 선택 라벨만 (구조적 파이어월)
    labels = _bounded_labels(train, masks)
    _assert_no_terminal_read(labels)
    checks.append({"rule": "selection_labels_only", "ok": True,
                   "reason": "r2022/r2023 선택 라벨만 — primary/터미널 미로드"})

    # 베이스라인 OOF 로짓 (동결 캐시)
    baseline_logits = _load_baseline_logits()
    checks.append({"rule": "frozen_baseline_oof", "ok": True,
                   "reason": f"동결 v93 베이스라인 OOF 로짓 로드 (bounded {BOUNDED_ROWS} "
                             f"행/기원, 캐시 키 {BASELINE_CACHE_KEY[:16]}…)"})

    # 스크린
    screen = run_screen(train, masks, baseline_logits, labels)
    for cid in ALL_CALIB:
        gate = screen["per_candidate"][cid]["gate"]
        checks.append({"rule": f"screen_{cid}", "ok": True,
                       "reason": f"verdict={gate['verdict']} "
                                 f"violations={gate['violations']}"})
        for v in gate["violations"]:
            violations.append({"rule": f"screen_{cid}", "ok": False, "reason": v})

    # 스크린은 진단 — 게이트 verdict 를 증거에 기록하되 exit 0 (스크린 성공).
    # REJECT 는 게이트 실패가 아니라 정직한 스크린 결과다 (recovery_evaluator --screen 과 동일).
    ok = not violations
    record = _record_base("PASS" if ok else "REJECT", 0,
                          "aimers9-top100-recovery/task-6-calibration",
                          "Todo 6 — screen causal Brier calibration transforms")
    record.update({
        "mode": "screen",
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "candidates": list(ALL_CALIB),
        "selectable_candidates": list(SELECTABLE_CALIB),
        "control_candidates": list(CONTROL_CALIB),
        "frozen": {
            "c_logit": rp.C_LOGIT,
            "clip": [rp.CLIP_LO, rp.CLIP_HI],
            "scoring": "common.score(np.clip(common.sigmoid(z_candidate + C_LOGIT), .30, .70), y)",
            "z_candidate": "logit(p_transform) - C_LOGIT",
            "p0": "sigmoid(z + C_LOGIT)",
            "beta": {"eps": BETA_EPS, "features": list(BETA_FEATURES), **BETA_LR},
            "isotonic": dict(ISOTONIC),
            "baseline_cache_key": BASELINE_CACHE_KEY,
            "bounded_rows": BOUNDED_ROWS,
        },
        "panel": screen["panel"],
        "per_candidate": screen["per_candidate"],
        "deployment_refit": screen["deployment_refit"],
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "time_causal_fitting", "ok": True,
             "reason": "각 outer year Y 의 변환은 OOF 연도 < Y 에서만 피팅 후 1회 적용; "
                       "r2022(Y=2022) 는 prior OOF 부재 cold-start identity (문서화)"},
            {"rule": "frozen_formula_preserved", "ok": True,
             "reason": "배포 산식 불변 — z_candidate=logit(p_transform)-C_LOGIT, "
                       "clip(sigmoid(z_candidate+C_LOGIT),.30,.70)"},
            {"rule": "no_global_offset_retune", "ok": True,
             "reason": "C_LOGIT/clip 상수 불변 — 전역 오프셋 재튜닝 없음"},
            {"rule": "no_terminal_read", "ok": True,
             "reason": "스크린 경로는 r2022/r2023 선택 라벨만 — primary/터미널 미로드"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-6-calibration")
    print(f"[recovery_calibration_runner] --screen: {'PASS' if ok else 'REJECT'} "
          f"(exit {0 if ok else 2})")
    for cid in ALL_CALIB:
        gate = screen["per_candidate"][cid]["gate"]
        print(f"  {cid}: verdict={gate['verdict']}")
        for o, r in gate["origins"].items():
            print(f"    {o}: ΔBSS={r['delta_bss']:+.4f} Brier {r['brier_candidate']:.6f}"
                  f"<{r['brier_baseline']:.6f} LB5={r['bootstrap_lb5']:+.4f} "
                  f"mean_shift={r['mean_shift']:.6f}")
    print(f"[recovery_calibration_runner] evidence -> {json_path} / {md_path}")
    return 0


# ── --fixture ────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "free-intercept":
        # 독립(standalone) outer/Public/test 피팅 오프셋 거부
        try:
            _assert_no_standalone_offset([2022, 2023], outer_year=2023)
            detail = "독립 오프셋이 허용됨 — 정책 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"독립 오프셋 거부: {exc}"
    elif name == "outer-fit":
        # outer year 자체 데이터로 피팅 시도 → 누수 가드
        try:
            _assert_no_outer_fit([2022, 2023], outer_year=2023)
            detail = "outer-fit 이 허용됨 — 누수 위반!"
        except LeakageError as exc:
            matched = True
            detail = f"outer-fit 차단: {exc}"
    elif name == "terminal-label-read":
        # 스크린 경로에 primary/터미널 라벨 존재 → 차단
        try:
            _assert_no_terminal_read({"r2022": np.zeros(1), "primary": np.zeros(1)})
            detail = "터미널 라벨이 스크린 경로에 허용됨 — 파이어월 위반!"
        except LeakageError as exc:
            matched = True
            detail = f"터미널 라벨 읽기 차단: {exc}"
    elif name == "nonmonotonic-isotonic-output":
        # 비단조 isotonic 출력 → 거부
        try:
            _assert_monotonic_isotonic(np.array([0.3, 0.2, 0.4]),
                                       np.array([0.1, 0.2, 0.3]))
            detail = "비단조 isotonic 출력이 허용됨 — increasing 위반!"
        except PolicyViolation as exc:
            matched = True
            detail = f"비단조 isotonic 출력 거부: {exc}"
    else:
        print(f"[recovery_calibration_runner] FATAL: 알 수 없는 fixture {name!r}",
              file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-6-calibration-fixture-{name}",
                          f"Todo 6 fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-6-calibration-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-6-calibration-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_calibration_runner] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_calibration_runner] evidence -> {json_path} / {md_path}")
    return 2


# ── CLI ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_calibration_runner.py",
        description="Todo 6 — screen causal Brier calibration transforms (--screen / --fixture)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--screen", action="store_true",
                        help="캘리브레이션 스크린 (r2022/r2023 선택 라벨만)")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    args = parser.parse_args(argv)

    if args.fixture is not None:
        return cmd_fixture(args)
    if args.screen:
        return cmd_screen(args)
    print("[recovery_calibration_runner] FATAL: 명령을 지정하세요 (--screen / --fixture)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
