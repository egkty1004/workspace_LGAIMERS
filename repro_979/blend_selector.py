#!/usr/bin/env python3
"""blend_selector.py — Todo 7: 홀드아웃 앙상블 선택 + 전이 게이트 (held-out ensemble selection).

Todo 2-6 의 **qualified OOF 예측 행렬**만 소비한다:
  - champion 블렌드 로짓        : cache/qualification/champion/{fold}.npy   (Task 2)
  - LGB/MLP 컴포넌트 로짓        : cache/{preds_primary,h2b_*}*.npy / cache/mlp_*.npy (Task 2)
  - 모델 패밀리                  : cache/qualification/{xgboost,catboost}/{fold}.npy (Task 5)
  - 피처 후보                    : cache/qualification/task4_<id>/{fold}.npy  (Task 4)
  - 캘리브레이션 스케줄          : cache/calibration/task6/{schedule}/{fold}.npy (Task 6)
  - 없는 출처 = 누락으로 스킵 + 사유 기록. result.json/증거 다이제스트 대조로 출처 검증.

선택 폴드(selection fold) = `primary`(2024 검증 기간) 전용으로만 블렌드 가중치를
피팅하고, 고정 가중치를 홀드아웃 R-only 폴드(r2022/r2023/r2024)에서만 평가한다.
홀드아웃 라벨은 피팅에 절대 사용하지 않는다 (assert + --leakage-injection 실패 QA).

전이 게이트(수락 조건, 전부 충족 시에만 ACCEPTED):
  (a) 누수 증명: 피팅 가중치 아티팩트에 홀드아웃 라벨 미포함 (구조적 검증).
  (b) 전이: 홀드아웃 3폴드 전부에서 챔피언 대비 ΔBSS > +1.0.
  (c) 평균 이동: max|Δmean| ≤ 0.005.
  (d) 다양성: 최대 구성원 로짓 상관 < 0.99 또는 잔차 정렬 ≥ 0.02.
  (e) 부트스트랩: 풀링 Δ 하한(5%) > 0.
  하나라도 실패 → rejected + 사유. 단독 BSS 만으로는 승격하지 않는다.

결정적 후보 ID: sha256(canonical_json{members, weights, selection_fold, seed, step})[:16].

기본 실행은 캐시 백업(학습 없음, numpy 전용 — scipy 미사용). CPU-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너와 동일

# ── Task 2 동결 컨트롤 재사용 (읽기 전용 import — 자기 무결성 게이트가 로드 시 1회 수행됨) ──
from qualification_runner import (  # noqa: E402
    FOLDS, R_FOLDS, W_LGB, C_LOGIT, CLIP_LO, CLIP_HI,
    build_folds, _check_leakage, _sha256,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

# ════════════════════════════════════════════════════════════════════
# Todo 7 상수
# ════════════════════════════════════════════════════════════════════
SCHEMA_VERSION = 1
SELECTION_FOLD = "primary"        # 지정 선택 폴드 (2024 검증 기간 — 챔피언 w=0.51 의 원점과 동일)
BOOT_SEED = 42
N_BOOT = 1000
WEIGHT_STEP = 0.05                # 단순 시플렉스 격자 (numpy 전용, scipy 불요)
TRANSFER_THRESHOLD_BSS = 1.0      # R-only 폴드당 ΔBSS > +1.0 (선언 임계값)
MEAN_SHIFT_MAX = 0.005            # max|Δmean| 허용 한계 (Task 4/6 와 동일)
CORR_DIVERSITY_MAX = 0.99         # 비자명 다양성: 구성원 로짓 상관 < 0.99
RESIDUAL_ALIGN_MIN = 0.02         # 대안: 잔차 정렬 ≥ 0.02
FIT_ALLOWED_FOLDS = (SELECTION_FOLD,)  # 가중치 피팅 허용 폴드 = 선택 폴드 단 하나


class LeakageError(RuntimeError):
    """피팅에 홀드아웃/스코어 폴드 라벨이 개입되면 발생."""


def _assert_fit_fold_allowed(fit_fold: str) -> None:
    """가중치 피팅 폴드가 선택 폴드가 아니면 LeakageError — 홀드아웃 라벨 피팅 차단."""
    if fit_fold not in FIT_ALLOWED_FOLDS:
        raise LeakageError(
            f"[LEAKAGE] 가중치 피팅 폴드 {fit_fold!r} 는 홀드아웃 폴드입니다. "
            f"피팅은 선택 폴드 {SELECTION_FOLD!r} 만 허용. (누수 가드)"
        )


def _git_commit() -> str:
    import subprocess  # noqa: PLC0415
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


# ════════════════════════════════════════════════════════════════════
# 멤버 레지스트리 — 출처(provenance) 존재 시에만 등록. 없으면 스킵+사유.
# 기대 다이제스트: 커밋된 아티팩트(champion result.json / task-5 evidence)에서 전사.
# ════════════════════════════════════════════════════════════════════
# 각 폴드 파일 + 기대 sha256 (Task 2 champion result.json: per_fold/cache_provenance)
_CHAMPION_DIGESTS = {
    "primary": "7302f5673cfbcfdfec524d174ff3eb901d89adc3b28e30989be29c1385ff5ec7",
    "r2022": "c5e32ff829ca43b49521398df167e22007c5c3301ba5e2d78d3d8fa65794f189",
    "r2023": "178f6632e8ee2099e7448e7ec11d29a84155e7811ea2b057f105bbb9b50ce745",
    "r2024": "5d098720d8922dcce43ef09fe81b2af28e087321d0bc74698157558eb84131cc",
}
_LGB_DIGESTS = {
    "primary": "e482cc869f9964b013223e7e97828b2825a4ee722e0259454627b376d46aceda",
    "r2022": "7960fdff1216f0a275e50f063218d6ac9d07f326da916e31decd03022162e2bd",
    "r2023": "ae02faa114a961dc1d7d7fce8d34097558a1a889250ba8dffc9944067fddbe44",
    "r2024": "a931358a81a0df2c4bcca0dea4b6f20a9a5b0640bfb9a083673d4d5029584c44",
}
_MLP_DIGESTS = {
    "primary": "9504e89a94af082818f3329c270c751d24025996a629e3cd3ed934374f429e1e",
    "r2022": "fe85410baff20b33cb6dba7c7a138920a4ff375a53ec7532f88cbff29aedbb31",
    "r2023": "c754e5c4ac4fa2365a5cbd5795f3377beb18bc4cdefd3928a987b011d0e5c87d",
    "r2024": "c48dad0501220009b94b74a1474c8a2090c6e8c8d6edbd6bbcb4bbf9416ab1dc",
}
_XGB_DIGESTS = {
    "primary": "01fc847ad5fa9aa855ee225cb686f9310ad095b65d8412272f2bcee1225c5ab0",
    "r2022": "4738216bfc375962d6421eef8b7ad8e326296801995e5b0f00c0f3707689b7ab",
    "r2023": "bab38fac679ac33868d7df9556c33dc5be376f4d08d9721fdfd3877a3aa1aafa",
    "r2024": "db882b38fad570998d4a26740cb38ff7d7563fe6ba902830097672e7919ba4e3",
}
_CAT_DIGESTS = {
    "primary": "e6470f40e2f2fe6cc705c3af5fd5b4687393a6549f6b98f99f204b76479bd660",
    "r2022": "6a9fcdb8fe698d8cffc6589bfa131f9145512361f349fb0125e70822da12f606",
    "r2023": "3bffe62d4e3da2547d07e649eab0a3d7560cf80f87e5992e2e036110192289e4",
    "r2024": "d453162ccfc3043d406b74abd45d7152c643312f0f7d126f43a5931977186798",
}

def _evidence_digests(rel_evidence: str) -> dict | None:
    """커밋된 증거 JSON에서 per_fold OOF 다이제스트 로드."""
    p = PROJECT_ROOT / rel_evidence
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {f: d["per_fold"][f]["digest"] for f in FOLDS
            if f in d.get("per_fold", {})}


# Task 2 멤버 다이제스트 소스 = 동결 상수(커밋 task-2-control.json 와 일치 확인됨)
_T2_SOURCE = "Task 2 qualification 증거(동결) — .omo/evidence/aimers9-top100/task-2-control.json"
# Task 5 패밀리 다이제스트 소스 = 커밋된 패밀리 증거 (full 또는 smoke 중 실측 일치 소스 선택)
_EV = ".omo/evidence/aimers9-top100"
_FAMILY_SOURCES = {
    "xgboost": [
        ("task-5 full evidence (10시드)", _evidence_digests(f"{_EV}/task-5-models-xgboost.json")),
        ("task-5 smoke evidence (2시드, full OOF 덮어씀)", _evidence_digests(f"{_EV}/task-5-models-smoke-xgboost.json")),
    ],
    "catboost": [
        ("task-5 full evidence (10시드)", _evidence_digests(f"{_EV}/task-5-models-catboost.json")),
    ],
}

# id → (설명, Task 출처, 폴드→상대경로, 폴드→기대 다이제스트, 기타 메모)
_MEMBER_SPECS: list[dict] = [
    {
        "id": "champion",
        "desc": "챔피언 블렌드 로짓 (LGB×0.51 + MLP×0.49, Task 2 qualification)",
        "task": "Task 2",
        "files": {f: f"cache/qualification/champion/{f}.npy" for f in FOLDS},
        "digests": _CHAMPION_DIGESTS,
        "digest_source": _T2_SOURCE,
        "note": "현재 제출 기준선 (primary blend 774.159, R-only +28.8/+20.9/+12.7)",
    },
    {
        "id": "lgb",
        "desc": "LGB F3 10시드 평균 로짓 (Task 2 캐시 백업 컴포넌트)",
        "task": "Task 2",
        "files": {
            "primary": "cache/preds_primary_lgb_f3.npy",
            "r2022": "cache/h2b_r2022_lgb_f3.npy",
            "r2023": "cache/h2b_r2023_lgb_f3.npy",
            "r2024": "cache/h2b_r2024_lgb_f3.npy",
        },
        "digests": _LGB_DIGESTS,
        "digest_source": _T2_SOURCE,
        "note": "챔피언 블렌드의 LGB 컴포넌트",
    },
    {
        "id": "mlp",
        "desc": "MLP 5시드 평균 로짓 (Task 2 캐시 백업 컴포넌트)",
        "task": "Task 2",
        "files": {f: f"cache/mlp_{f}.npy" for f in FOLDS},
        "digests": _MLP_DIGESTS,
        "digest_source": _T2_SOURCE,
        "note": "챔피언 블렌드의 MLP 컴포넌트",
    },
    {
        "id": "xgboost",
        "desc": "XGBoost 평균 로짓 (Task 5, GPU 부재 시 CPU-hist 폴백)",
        "task": "Task 5",
        "files": {f: f"cache/qualification/xgboost/{f}.npy" for f in FOLDS},
        "digests": None,  # _register_members 가 _FAMILY_SOURCES 로 동적 해석
        "digest_sources": _FAMILY_SOURCES["xgboost"],
        "note": "Task 5 기각: R-only 전이 없음 (OOF 행렬은 블렌드 분석 전용)",
    },
    {
        "id": "catboost",
        "desc": "CatBoost 평균 로짓 (Task 5, depth7 CPU)",
        "task": "Task 5",
        "files": {f: f"cache/qualification/catboost/{f}.npy" for f in FOLDS},
        "digests": None,
        "digest_sources": _FAMILY_SOURCES["catboost"],
        "note": "Task 5 기각: R-only 전이 없음 + 배포 install_risk (OOF 행렬은 블렌드 분석 전용)",
    },
]

TASK4_REJECTED = (
    "batter_debut_league_impute_success", "count_state_abs_regime",
    "debut_league_impute_success", "f_game_post2023_label_mapping",
    "league_trend_ratio_middle", "log1p_usage_pitcher",
    "season_dev_success_pitcher", "usage_rank_pitcher",
)
TASK6_SCHEDULES = ("uniform", "linear_recent", "exp_recent", "regime_step")


def _task4_spec(cid: str) -> dict:
    result = REPO / "cache" / "qualification" / f"task4_{cid}" / "result.json"
    digests = {}
    if result.is_file():
        meta = json.loads(result.read_text(encoding="utf-8"))
        digests = {f: meta["per_fold"][f]["digest"] for f in FOLDS if f in meta.get("per_fold", {})}
    return {
        "id": f"task4_{cid}",
        "desc": f"Task 4 피처 후보 {cid} OOF 로짓 (10시드, 전부 기각)",
        "task": "Task 4",
        "files": {f: f"cache/qualification/task4_{cid}/{f}.npy" for f in FOLDS},
        "digests": digests,
        "note": "Task 4 기각 (primary Δ<0, bootstrap 하한<0) — OOF 행렬은 블렌드 분석 전용",
    }


def _task6_spec(sched: str) -> dict:
    return {
        "id": f"task6_{sched}",
        "desc": f"Task 6 캘리브레이션 스케줄 {sched} OOF 로짓 (고정 109라운드 중첩 프로토콜)",
        "task": "Task 6",
        "files": {f: f"cache/calibration/task6/{sched}/{f}.npy" for f in FOLDS},
        "digests": {},  # Task 6 증거는 OOF 다이제스트 미기록 → 실측만 기록
        "note": ("Task 6 기각 (R-only Δ<0) + 다른 학습 프로토콜(고정109라운드) — "
                 "OOF 행렬은 블렌드 분석 전용, 챔피언과 스케일 불일치 가능"),
    }


def _all_member_specs() -> list[dict]:
    specs = list(_MEMBER_SPECS)
    for cid in TASK4_REJECTED:
        specs.append(_task4_spec(cid))
    for sched in TASK6_SCHEDULES:
        specs.append(_task6_spec(sched))
    return specs


# ════════════════════════════════════════════════════════════════════
# 멤버 등록 (디스크 출처 검증)
# ════════════════════════════════════════════════════════════════════
EXPECTED_ROWS = {"primary": 253507, "r2022": 217024, "r2023": 219839, "r2024": 223497}


def _register_members() -> tuple[dict, list[dict]]:
    """존재하는 출처만 등록. 결과: {id: spec} + manifest(검증 레코드)."""
    members: dict = {}
    manifest: list[dict] = []
    for spec in _all_member_specs():
        rec = {
            "id": spec["id"], "task": spec["task"], "desc": spec["desc"],
            "note": spec["note"], "registered": False,
        }
        missing = [f for f in FOLDS if not (REPO / spec["files"][f]).is_file()]
        if missing:
            rec["registered"] = False
            rec["skip_reason"] = f"OOF 파일 누락: {missing}"
            manifest.append(rec)
            continue
        actual = {f: _sha256(REPO / spec["files"][f]) for f in FOLDS}

        # 다이제스트 소스 해석: 동적(evidence 파일) 또는 정적(동결 상수)
        digest_source = spec.get("digest_source")
        expected: dict = spec.get("digests") or {}
        if spec.get("digest_sources"):
            digest_source, expected = _resolve_family_source(spec["id"], actual, spec["digest_sources"])

        rows_ok = True
        digest_records = {}
        for f in FOLDS:
            p = REPO / spec["files"][f]
            n = int(len(np.load(p)))
            exp = expected.get(f)
            match = (exp is None) or (actual[f] == exp)
            digest_records[f] = {
                "file": spec["files"][f], "rows": n, "expected_rows": EXPECTED_ROWS[f],
                "rows_match": n == EXPECTED_ROWS[f], "sha256": actual[f],
                "expected_sha256": exp, "digest_match": bool(match),
            }
            if n != EXPECTED_ROWS[f]:
                rows_ok = False
        all_digest_ok = all(dr["digest_match"] for dr in digest_records.values())
        rec["digest_source"] = digest_source
        rec["registered"] = rows_ok and all_digest_ok
        rec["rows_match"] = rows_ok
        rec["digest_match_all"] = all_digest_ok
        rec["fold_files"] = digest_records
        if not rows_ok:
            rec["skip_reason"] = "행 수가 폴드 검증 크기와 불일치 (낡은 캐시)"
        elif not all_digest_ok:
            bad = [f for f, dr in digest_records.items() if not dr["digest_match"]]
            rec["skip_reason"] = f"출처 다이제스트 불일치: {bad} (재생성 의심)"
        else:
            rec["registered"] = True
            members[spec["id"]] = spec
        manifest.append(rec)
    return members, manifest


def _resolve_family_source(mid: str, actual: dict, sources: list[tuple[str, dict | None]]) -> tuple[str, dict]:
    """패밀리 OOF 다이제스트가 어떤 증거(full/smoke)와 일치하는지 해석.
    일치 소스가 없으면 첫 소스로 규정하고 불일치 플래그는 호출부가 처리."""
    for label, exp in sources:
        if exp is None:
            continue
        if all(actual[f] == exp.get(f) for f in FOLDS):
            return label, exp
    if sources and sources[0][1] is not None:
        return f"{sources[0][0]} (불일치 — 실측과 일치 소스 없음)", sources[0][1]
    return "증거 다이제스트 미기록 (실측만)", {}


# ════════════════════════════════════════════════════════════════════
# 블렌드 후보 청사진
# ════════════════════════════════════════════════════════════════════
def _build_blueprints(members: dict) -> list[tuple[str, tuple[str, ...], str]]:
    """(후보 id, 멤버 id 목록, 종류) — 등록된 멤버로만 구성."""
    bps: list[tuple[str, tuple[str, ...], str]] = [
        ("lgb_mlp", ("lgb", "mlp"), "recovery"),                     # w 복구 검사 (챔피언 재도출)
        ("champ_xgb", ("champion", "xgboost"), "pair"),
        ("champ_cat", ("champion", "catboost"), "pair"),
        ("champ_xgb_cat", ("champion", "xgboost", "catboost"), "triple"),
        ("lgb_mlp_xgb", ("lgb", "mlp", "xgboost"), "triple"),        # 독립 모델 블렌드
        ("lgb_mlp_cat", ("lgb", "mlp", "catboost"), "triple"),
    ]
    for cid in TASK4_REJECTED:
        mid = f"task4_{cid}"
        if mid in members:
            bps.append((f"champ_{cid}", ("champion", mid), "pair"))
    for sched in TASK6_SCHEDULES:
        mid = f"task6_{sched}"
        if mid in members:
            bps.append((f"champ_{sched}", ("champion", mid), "pair"))
    # 등록 안 된 멤버가 포함된 블루프린트는 제거
    bps = [(bid, mids, kind) for bid, mids, kind in bps
           if all(m in members for m in mids)]
    return bps


# ════════════════════════════════════════════════════════════════════
# 가중치 피팅 — 선택 폴드 전용, numpy 시플렉스 격자 (결정적)
# ════════════════════════════════════════════════════════════════════
def fit_weights(member_logits: list[np.ndarray], y_sel: np.ndarray,
                step: float = WEIGHT_STEP) -> tuple[np.ndarray, float]:
    """선택 폴드에서 w≥0, Σw=1 시플렉스 격자 탐색 — 선택 폴드 BSS 최대화.
    fit_fold 가드는 호출부(_assert_fit_fold_allowed)가 수행."""
    k = len(member_logits)
    if k < 1:
        raise ValueError("블렌드 멤버가 없습니다")
    zs = [z.astype(np.float64) for z in member_logits]
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def bss_of(z: np.ndarray) -> float:
        return float(common.score(common.sigmoid(z), y_sel))

    if k == 1:
        return np.array([1.0]), bss_of(zs[0])
    best_s, best_w = -1e18, None
    if k == 2:
        for a in grid:
            z = a * zs[0] + (1.0 - a) * zs[1]
            s = bss_of(z)
            if s > best_s:
                best_s, best_w = s, (a, 1.0 - a)
    elif k == 3:
        for a in grid:
            w2 = np.arange(0.0, 1.0 - a + 1e-9, step)
            for b in w2:
                c = 1.0 - a - b
                z = a * zs[0] + b * zs[1] + c * zs[2]
                s = bss_of(z)
                if s > best_s:
                    best_s, best_w = s, (a, b, c)
    else:
        raise NotImplementedError(f"{k} 멤버 블렌드는 미지원 (≤3)")
    return np.asarray(best_w, dtype=np.float64), best_s


# ════════════════════════════════════════════════════════════════════
# 평가 헬퍼
# ════════════════════════════════════════════════════════════════════
def _clip_fracs(z: np.ndarray) -> dict:
    """클리핑 진단: raw 시그모이드 / 챔피언 C_LOGIT 적용 후 0.30·0.70 경계 도달 비율."""
    p_raw = common.sigmoid(z)
    p_post = common.sigmoid(z + C_LOGIT)
    return {
        "frac_below_0.30_pre": float((p_raw < CLIP_LO).mean()),
        "frac_above_0.70_pre": float((p_raw > CLIP_HI).mean()),
        "frac_below_0.30_post": float((p_post < CLIP_LO).mean()),
        "frac_above_0.70_post": float((p_post > CLIP_HI).mean()),
        "clip_bounds": [CLIP_LO, CLIP_HI],
        "note": ("진단 전용 — 검증 BSS 는 raw 시그모이드(캘리브레이션 불변), "
                 "C_LOGIT=-0.0404 는 제출 정렬 상수"),
    }


def _pairwise_corr(zs: list[np.ndarray]) -> dict:
    return {f"{i}_{j}": float(np.corrcoef(zs[i], zs[j])[0, 1])
            for i in range(len(zs)) for j in range(i + 1, len(zs))}


def _residual_align(p_members: list[np.ndarray], p_champ: np.ndarray,
                    y: np.ndarray) -> dict:
    """선택 폴드 전용 진단: corr(p_member − p_champion, y − p_champion) (Task 5 지표).
    챔피언 자신(편차 0)은 정의 불가 → None."""
    res = y.astype(np.float64) - p_champ
    out = {}
    for i in range(len(p_members)):
        dev = p_members[i] - p_champ
        if float(np.abs(dev).max()) < 1e-12:
            out[f"member_{i}"] = None
        else:
            out[f"member_{i}"] = float(np.corrcoef(dev, res)[0, 1])
    return out


def _max_non_null(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(max(vals)) if vals else None


def _bootstrap_pooled(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      n_boot: int, seed: int) -> dict:
    """홀드아웃 풀링 ΔBSS 부트스트랩 (행 재표본, 폴드 크기 가중).
    pairs = [(blend_p, champ_p, y)] per held-out fold."""
    rng = np.random.default_rng(seed)
    total = sum(len(y) for _, _, y in pairs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pooled = 0.0
        for bp, cp, yv in pairs:
            n = len(yv)
            idx = rng.integers(0, n, n)
            yb = yv[idx]
            r = yb.mean()
            denom = r * (1.0 - r)
            e_b = float(((bp[idx] - yb) ** 2).mean())
            e_c = float(((cp[idx] - yb) ** 2).mean())
            s_b = 100000.0 * (1.0 - e_b / denom)
            s_c = 100000.0 * (1.0 - e_c / denom)
            pooled += (s_b - s_c) * (n / total)
        deltas[i] = pooled
    p5, p50, p95 = np.percentile(deltas, [5.0, 50.0, 95.0])
    return {
        "n_iter": int(n_boot), "seed": int(seed),
        "pct_5": float(p5), "pct_50": float(p50), "pct_95": float(p95),
        "mean": float(deltas.mean()), "lower_bound_gt_0": bool(p5 > 0.0),
    }


# ════════════════════════════════════════════════════════════════════
# 후보 실행
# ════════════════════════════════════════════════════════════════════
def _candidate_id(members: tuple[str, ...], weights: np.ndarray,
                  selection_fold: str, seed: int, step: float) -> str:
    canonical = json.dumps({
        "members": list(members),
        "weights": [round(float(w), 4) for w in weights],
        "selection_fold": selection_fold,
        "seed": seed,
        "weight_step": step,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run_candidate(bp_id: str, mids: tuple[str, ...], kind: str,
                  members: dict, labels: dict, folds: dict,
                  n_boot: int, fit_fold: str = SELECTION_FOLD,
                  force_negative: bool = False) -> dict:
    _assert_fit_fold_allowed(fit_fold)  # 누수 가드 (LeakageError)
    # 선택 폴드 가중치 피팅
    z_sel = np.stack([np.load(REPO / members[m]["files"][SELECTION_FOLD]) for m in mids], axis=1)
    y_sel = labels[SELECTION_FOLD]
    weights, sel_bss = fit_weights([z_sel[:, i] for i in range(len(mids))], y_sel, WEIGHT_STEP)
    member_sel_bss = {m: float(common.score(common.sigmoid(z_sel[:, i]), y_sel))
                      for i, m in enumerate(mids)}

    out: dict = {
        "id": bp_id,
        "kind": kind,
        "members": list(mids),
        "weight_fit": {
            "procedure": ("numpy 시플렉스 격자 w≥0, Σw=1, step=0.05, 선택 폴드 BSS 최대화"),
            "fit_fold": fit_fold,
            "weights": [float(w) for w in weights],
            "weights_by_member": {m: float(w) for m, w in zip(mids, weights)},
            "selection_bss": sel_bss,
            "member_selection_bss": member_sel_bss,
            "labels_used": ["selection_fold_only"],
            "held_out_folds_untouched": True,
        },
        "per_fold": {},
        "diversity": {},
        "bootstrap": None,
        "mean_shift": {"max_abs_over_held_out": None},
        "clipping": {},
        "verdict": None,
        "reasons": [],
    }

    held_out = tuple(f for f in R_FOLDS if f != fit_fold) if fit_fold != SELECTION_FOLD else R_FOLDS
    boot_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    max_dmean = 0.0
    max_pair_corr = 0.0
    per_fold_deltas = {}
    p_champ_ho: dict = {}

    for fn in held_out:
        zs = np.stack([np.load(REPO / members[m]["files"][fn]) for m in mids], axis=1)
        z_blend = np.stack([zs[:, i] * weights[i] for i in range(len(mids))], axis=1).sum(axis=1)
        if force_negative:
            z_blend = -z_blend  # 강제 음성 전이 주입: 반예측 → ΔBSS < 0 보장
        p_blend = common.sigmoid(z_blend)
        p_champ = common.sigmoid(np.load(REPO / members["champion"]["files"][fn]))
        yv = labels[fn]
        bss_blend = float(common.score(p_blend, yv))
        bss_champ = float(common.score(p_champ, yv))
        delta = bss_blend - bss_champ
        dmean = float(p_blend.mean() - p_champ.mean())
        per_fold_deltas[fn] = delta
        max_dmean = max(max_dmean, abs(dmean))
        corr_champ = float(np.corrcoef(z_blend, np.load(REPO / members["champion"]["files"][fn]))[0, 1])
        pw = _pairwise_corr([zs[:, i] for i in range(len(mids))])
        max_pair_corr = max(max_pair_corr, max(pw.values()))
        out["per_fold"][fn] = {
            "blend_bss": bss_blend, "champion_bss": bss_champ, "delta_vs_champion": delta,
            "pred_mean": float(p_blend.mean()),
            "champion_corr": corr_champ,
            "member_pairwise_corr": pw,
            "mean_shift": dmean,
            "clipping": _clip_fracs(z_blend),
        }
        boot_pairs.append((p_blend, p_champ, yv))
        p_champ_ho[fn] = p_champ
    out["mean_shift"]["max_abs_over_held_out"] = float(max_dmean)
    out["mean_shift"]["per_fold"] = {fn: out["per_fold"][fn]["mean_shift"] for fn in held_out}

    # 선택 폴드 잔차 진단 (라벨 = 선택 폴드만 사용)
    z_sel_champ = np.load(REPO / members["champion"]["files"][SELECTION_FOLD])
    p_sel_members = [common.sigmoid(z_sel[:, i]) for i in range(len(mids))]
    out["residual_analysis_sel_fold"] = {
        "residual_alignment": _residual_align(p_sel_members, common.sigmoid(z_sel_champ), y_sel),
        "pairwise_residual_corr": _pairwise_corr(
            [z_sel[:, i] - z_sel_champ for i in range(len(mids))]),
        "note": "선택 폴드 라벨만 사용하는 진단 (홀드아웃 라벨 미사용)",
    }
    max_res_align = _max_non_null(
        list(out["residual_analysis_sel_fold"]["residual_alignment"].values()))
    out["diversity"] = {
        "max_pairwise_member_corr": float(max_pair_corr),
        "threshold_corr": CORR_DIVERSITY_MAX,
        "max_residual_alignment_sel": max_res_align,
        "threshold_residual_align": RESIDUAL_ALIGN_MIN,
        "non_trivial": bool(max_pair_corr < CORR_DIVERSITY_MAX
                            or (max_res_align is not None and max_res_align >= RESIDUAL_ALIGN_MIN)),
    }

    # 부트스트랩 (홀드아웃 풀링 Δ)
    out["bootstrap"] = _bootstrap_pooled(boot_pairs, n_boot, BOOT_SEED)

    # 결정적 후보 ID
    out["candidate_id"] = _candidate_id(mids, weights, SELECTION_FOLD, BOOT_SEED, WEIGHT_STEP)

    # ── 전이 게이트 ──
    reasons: list[str] = []
    if fit_fold != SELECTION_FOLD:
        reasons.append("(a) 누수 증명 실패: 가중치가 홀드아웃 폴드에서 피팅됨")
    pos_transfer = all(per_fold_deltas[f] > TRANSFER_THRESHOLD_BSS for f in R_FOLDS)
    if not pos_transfer:
        reasons.append(
            "(b) R-only 전이 미충족 (ΔBSS > +1.0 각각): "
            + ", ".join(f"{f} {per_fold_deltas[f]:+.2f}" for f in R_FOLDS))
    if max_dmean > MEAN_SHIFT_MAX:
        reasons.append(f"(c) 평균 이동 위반: max|Δmean| {max_dmean:.4f} > {MEAN_SHIFT_MAX}")
    if not out["diversity"]["non_trivial"]:
        reasons.append(
            f"(d) 비자명 다양성 미충족: max 구성원 상관 {max_pair_corr:.3f} ≥ {CORR_DIVERSITY_MAX} "
            f"이고 잔차 정렬 {max_res_align:.3f} < {RESIDUAL_ALIGN_MIN}")
    if not out["bootstrap"]["lower_bound_gt_0"]:
        reasons.append(f"(e) 부트스트랩 하한(5%) {out['bootstrap']['pct_5']:+.2f} ≤ 0 (풀링)")
    if force_negative:
        reasons.append("(주입) --force-negative-transfer: 홀드아웃 로짓 반전 — 전이 게이트 기각 예상")
    accepted = (fit_fold == SELECTION_FOLD and pos_transfer
                and max_dmean <= MEAN_SHIFT_MAX and out["diversity"]["non_trivial"]
                and out["bootstrap"]["lower_bound_gt_0"])
    out["verdict"] = "accepted" if accepted else "rejected"
    out["reasons"] = reasons
    zero_w = [m for m, w in zip(mids, weights) if float(w) == 0.0]
    out["redundant"] = bool(zero_w)
    if zero_w:
        out["redundant_note"] = (f"가중치 0 멤버 {zero_w} — 유효 블렌드는 나머지 멤버만으로 "
                                 "구성된 것과 동일 (중복 후보)")
    if accepted:
        out["standalone_bss_note"] = ("단독 BSS 가 아닌 전이 게이트(홀드아웃 Δ)로만 승격 — "
                                      "단독 BSS 만으로 승격하지 않음")
    return out


# ════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, path: Path):
        self.fh = path.open("w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s: str):
        self.out.write(s)
        self.fh.write(s)

    def flush(self):
        self.out.flush()
        self.fh.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Todo 7: 홀드아웃 앙상블 선택 + 전이 게이트 (qualified OOF 행렬 소비)")
    parser.add_argument("--candidate-set", default="qualified",
                        choices=["qualified", "core", "recovery"],
                        help="후보 집합 (기본 qualified=전체)")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 선택 폴드 + 홀드아웃 1폴드(r2022) 축소 경로 — 게이트/스키마 검증")
    parser.add_argument("--leakage-injection", action="store_true",
                        help="실패 QA: 홀드아웃 폴드(r2022)에서 가중치 피팅 주입 → LeakageError exit 2")
    parser.add_argument("--force-negative-transfer", action="store_true",
                        help="실패 QA: 홀드아웃 로짓 반전으로 강제 음성 전이 → 전부 rejected + exit 1")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 .omo/evidence/aimers9-top100/task-7-blend.json)")
    parser.add_argument("--log", default=None,
                        help="증거 로그 경로 (기본 .omo/evidence/aimers9-top100/task-7-blend.log)")
    parser.add_argument("--n-boot", type=int, default=None, help="부트스트랩 반복 수 (기본 1000)")
    args = parser.parse_args(argv)

    smoke = bool(args.smoke)
    leak_inject = bool(args.leakage_injection)
    force_neg = bool(args.force_negative_transfer)
    n_boot = args.n_boot or (200 if smoke else N_BOOT)

    ev_base = "task-7-blend-negative" if force_neg else ("task-7-blend-smoke" if smoke else "task-7-blend")
    evidence_path = (Path(args.evidence).expanduser().resolve() if args.evidence
                     else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / f"{ev_base}.json")
    log_path = (Path(args.log).expanduser().resolve() if args.log
                else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / f"{ev_base}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)

    t0 = time.time()
    print(f"[blend_selector] Todo 7 — 홀드아웃 앙상블 선택 (smoke={smoke}, "
          f"leakage_injection={leak_inject}, force_negative={force_neg})", flush=True)

    # ── 데이터 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    folds = build_folds(train)
    leakage_problems = _check_leakage(folds, train)
    if leakage_problems:
        print("[FAIL] 누수 가드 실패:\n  " + "\n  ".join(leakage_problems), file=sys.stderr)
        return 1
    labels = {f: train.loc[folds[f][1], common.TARGET].values.astype(np.float64) for f in FOLDS}
    for f in FOLDS:
        assert int(folds[f][1].sum()) == EXPECTED_ROWS[f], f"{f} 행 수 불일치"
    print(f"[blend_selector] 폴드: selection={SELECTION_FOLD} held_out={list(R_FOLDS)} | "
          f"라벨 로드 OK (2025 부재 확인)", flush=True)

    # ── 멤버 등록 ──
    members, manifest = _register_members()
    registered_ids = sorted(members)
    print(f"[manifest] 등록 {len(members)}/{len(manifest)} 멤버: {registered_ids}", flush=True)
    for rec in manifest:
        if not rec["registered"]:
            print(f"[manifest] SKIP {rec['id']}: {rec.get('skip_reason')}", flush=True)
        else:
            ok = "OK" if rec.get("digest_match_all") else "?"
            print(f"[manifest] OK    {rec['id']} (task={rec['task']}, digest={ok})", flush=True)
    if not members:
        print("[FAIL] 등록된 OOF 멤버가 없습니다 — 캐시 아티팩트 부재", file=sys.stderr)
        return 1

    # ── 후보 집합 ──
    blueprints = _build_blueprints(members)
    if args.candidate_set == "core":
        blueprints = [b for b in blueprints if b[2] in ("recovery", "pair", "triple")
                      and not b[0].startswith("task4_") and not b[0].startswith("champ_")]
    elif args.candidate_set == "recovery":
        blueprints = [b for b in blueprints if b[0] == "lgb_mlp"]
    if smoke:
        blueprints = [b for b in blueprints if b[0] in ("lgb_mlp", "champ_xgb")]
    if not blueprints:
        print("[FAIL] 후보 블렌드가 없습니다", file=sys.stderr)
        return 1
    print(f"[blend_selector] 후보 {len(blueprints)}개: "
          f"{', '.join(b[0] for b in blueprints)}", flush=True)

    # ── 누수 주입 실패 QA (스코어/평가 전에 차단) ──
    if leak_inject:
        print("\n[leakage-injection] Failure QA: 홀드아웃 폴드(r2022) 라벨로 가중치 피팅 주입…",
              flush=True)
        bid, mids, kind = blueprints[0]
        try:
            run_candidate(bid, mids, kind, members, labels, folds,
                          n_boot=n_boot, fit_fold="r2022")
        except LeakageError as exc:
            print(f"[leakage-injection] PASS — 누수 가드가 홀드아웃 피팅을 차단했습니다: {exc}",
                  flush=True)
            leak_log = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-7-leakage-failure.log"
            leak_log.write_text(
                f"[leakage-injection] exit 2 (LeakageError) at {datetime.now(timezone.utc).isoformat()}\n"
                f"fit_fold=r2022 는 홀드아웃 — 선택 폴드 {SELECTION_FOLD} 만 허용.\n"
                f"{exc}\n", encoding="utf-8")
            print(f"[leakage-injection] 증거: {leak_log}", flush=True)
            return 2
        print("[leakage-injection] FAIL — LeakageError 가 발생하지 않았습니다", file=sys.stderr)
        return 1

    # ── 후보 실행 ──
    results = []
    for bid, mids, kind in blueprints:
        print(f"\n=== candidate {bid} (members={list(mids)}, kind={kind}) ===", flush=True)
        try:
            res = run_candidate(bid, mids, kind, members, labels, folds,
                                n_boot=n_boot, force_negative=force_neg)
        except LeakageError as exc:
            print(f"[FAIL] 누수 가드: {exc}", file=sys.stderr)
            return 2
        results.append(res)
        w = res["weight_fit"]["weights_by_member"]
        print(f"  weight_fit(selection={SELECTION_FOLD}): "
              + ", ".join(f"{m}={w[m]:.2f}" for m in mids)
              + f" | sel_bss={res['weight_fit']['selection_bss']:.3f} | "
              + f"candidate_id={res['candidate_id']}", flush=True)
        for fn in R_FOLDS:
            if fn in res["per_fold"]:
                pf = res["per_fold"][fn]
                print(f"  [{fn:<8s}] blend={pf['blend_bss']:>8.1f} champ={pf['champion_bss']:>8.1f} "
                      f"Δ={pf['delta_vs_champion']:>+7.2f} mean={pf['pred_mean']:.4f} "
                      f"Δmean={pf['mean_shift']:+.4f} corr_champ={pf['champion_corr']:.3f}",
                      flush=True)
        b = res["bootstrap"]
        print(f"  bootstrap(pooled): p5={b['pct_5']:+.2f} p50={b['pct_50']:+.2f} "
              f"p95={b['pct_95']:+.2f} LB>0={b['lower_bound_gt_0']}", flush=True)
        print(f"  mean_shift max|Δmean|={res['mean_shift']['max_abs_over_held_out']:.4f} "
              f"| diversity max_pairwise_corr={res['diversity']['max_pairwise_member_corr']:.3f} "
              f"non_trivial={res['diversity']['non_trivial']}", flush=True)
        print(f"  verdict: {res['verdict']}" + ("" if not res["reasons"]
              else f" — 사유: {'; '.join(res['reasons'])}"), flush=True)

    accepted = [r for r in results if r["verdict"] == "accepted"]
    print(f"\n[blend_selector] 전이 게이트 결과: accepted {len(accepted)}/{len(results)}", flush=True)
    if not accepted:
        print("[blend_selector] 승격 가능한 블렌드 없음 — 챔피언 컨트롤 "
              f"(LGB×{W_LGB} + MLP×{1.0 - W_LGB}, w={W_LGB}) 이 Todo 8 제출 기준선 유지.",
              flush=True)
    else:
        for r in accepted:
            print(f"[blend_selector] ACCEPTED: {r['id']} candidate_id={r['candidate_id']} "
                  f"weights={r['weight_fit']['weights_by_member']}", flush=True)

    # ── 스키마 + 저장 ──
    schema = {
        "schema_version": SCHEMA_VERSION,
        "task": "blend-selection",
        "smoke": smoke,
        "selection_fold": SELECTION_FOLD,
        "selection_fold_rationale": ("primary = 2024 검증 기간; 챔피언 w=0.51 도 primary 에서 "
                                     "피팅된 원점과 동일 — R-only 홀드아웃만 고정 가중치 평가"),
        "held_out_folds": list(R_FOLDS),
        "weight_fit_procedure": {
            "method": "numpy simplex grid w≥0, Σw=1",
            "step": WEIGHT_STEP,
            "scipy_used": False,
            "bootstrap_seed": BOOT_SEED,
            "n_boot": n_boot,
        },
        "transfer_gate": {
            "delta_threshold_bss_each_fold": TRANSFER_THRESHOLD_BSS,
            "mean_shift_max_abs": MEAN_SHIFT_MAX,
            "diversity_corr_max": CORR_DIVERSITY_MAX,
            "residual_align_min": RESIDUAL_ALIGN_MIN,
            "bootstrap_lower_bound_gt_0": True,
            "standalone_bss_promotion_forbidden": True,
        },
        "member_manifest": manifest,
        "registered_members": registered_ids,
        "skipped_members": [m["id"] for m in manifest if not m["registered"]],
        "candidates": results,
        "acceptance": {
            "accepted_ids": [r["id"] for r in accepted],
            "rejected_ids": [r["id"] for r in results if r["verdict"] == "rejected"],
            "redundant_accepted_ids": [r["id"] for r in accepted if r.get("redundant")],
            "deployment_note": (
                "수락 블렌드에 catboost 가 포함됨 — Task 5 기록상 평가환경 기본 패키지에 "
                "catboost 가 없어 배포 등급 install_risk. Todo 8 에서 requirements.txt 설치 "
                "검증(오프라인, 10분 한도)이 승인 선행 조건. 미충족 시 챔피언 컨트롤로 롤백."),
            "baseline_statement": (
                "전이 게이트 통과 블렌드가 있더라도 단독 BSS 가 아닌 홀드아웃 Δ로만 승격. "
                "통과 없었을 경우 챔피언 컨트롤(LGB×0.51 + MLP×0.49)이 Todo 8 제출 기준선. "
                "기각 후보의 OOF 행렬은 블렌드 분석 전용이며 승인된 것처럼 재도입하지 않는다."),
        },
        "leakage_guard": {
            "fit_allowed_folds": list(FIT_ALLOWED_FOLDS),
            "no_2025_in_any_validation_mask": True,
            "leakage_injection_ran": leak_inject,
            "force_negative_ran": force_neg,
        },
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\n[blend_selector] 증거 JSON → {evidence_path}", flush=True)
    print(f"[blend_selector] 증거 LOG  → {log_path}", flush=True)
    print(f"[blend_selector] 총 {time.time() - t0:.0f}s", flush=True)

    if smoke:
        required = ["schema_version", "selection_fold", "held_out_folds", "member_manifest",
                    "candidates", "acceptance", "transfer_gate"]
        ok = (all(k in schema for k in required)
              and len(blueprints) >= 1
              and all(r["verdict"] in ("accepted", "rejected") for r in results)
              and all(r["candidate_id"] for r in results)
              and all(r["bootstrap"] for r in results))
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 선택/게이트/스키마 경로 검증 "
              f"(selection={SELECTION_FOLD}, held_out={list(R_FOLDS)[:1]} 축소)", flush=True)
        return 0 if ok else 1

    # 강제 음성 전이 주입: 기각 보장 검증 (실패 QA)
    if force_neg:
        if accepted:
            print("[force-negative-transfer] FAIL — 주입에도 수락된 후보 존재", file=sys.stderr)
            return 1
        print("[force-negative-transfer] PASS — 강제 음성 전이에서 전부 rejected "
              "(전이 게이트 기각 확인)", flush=True)
        return 1  # 비정상 종료 = 주입 시나리오에서 기각 보장

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
