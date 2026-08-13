#!/usr/bin/env python3
"""qualification_runner.py — 챔피언 프리즈 컨트롤 + 공용 시계열 OOF 자격(qualification) 계약.

Todo 2 (aimers9-top100-score-improvement): 챔피언(979.31) 제어를 읽기전용으로 동결하고,
정확한 10시드(42..51) × 4폴드(primary/r2022/r2023/r2024) 시계열 OOF를 단일 러너/스키마로
평가한다. 이 스키마는 이후 Todo 4-7(피처 게이팅, 모델 패밀리, 캘리브레이션, 블렌드)의
입력 계약이 된다.

계약:
  - 폴드 마스크 = screen_all_10seed.py:build_folds 와 정확히 동일
    (primary: season<=2023 → season==2024; rYYYY: season<=YYYY-1 & 'R' → season==YYYY & 'R').
  - 로짓 = LGB × MLP 블렌드, z = W_LGB*z_lgb + (1-W_LGB)*z_mlp (W_LGB=0.51 동결).
  - 검증 BSS는 raw 시그모이드(common.score(common.sigmoid(z), y))로 측정.
    C_LOGIT은 제출 정렬 상수(챔피언 캘리브레이션) — 검증 BSS에는 미적용, 변경 금지.
  - --candidate 로 후보 선택. 미등록 id/다이제스트는 스코어링 전에 비정상 종료.
  - 무결성 게이트: 후보 구성 다이제스트(자기 무결성) + (champion) 챔피언 파일 sha256 동결 대조
    + 피처/MLP-cat 계약(동결 아티팩트 READ) 대조. 조작(예: W_LGB 0.51→0.52) 또는 미등록
    --candidate 는 스코어링 전에 실패. --allow-tampered 이스케이프 해치 없음.
  - 기본(캐시 백업): repro_979/cache 의 기존 로짓 아티팩트 사용 — 이는 문서화된 e6c 수치
    (experiments/e6c_blend_results.json)의 정확한 원본. 캐시 길이는 각 폴드 검증 행 수와
    대조(낡은 캐시 가드). --rerun-train: 10시드 × 4폴드 처음부터 재학습(CPU 전용, 느림 —
    이후 장시간 실행용. 본 Todo 인수 기준 아님).
  - --smoke: 시드 [42,43]으로 폴드/다이제스트/무결성/스키마 전체 경로 검증 후 PASS 출력.
    기본 동작에 영향 없음 (screen_all_10seed.py --smoke 미러).

누수 가드(비협상): 폴드 검증 마스크가 2025 시즌을 절대 포함하지 않음을 자체 확인.
스키마: schema_version / candidate / per_fold{bss,pred_mean,digest,n_rows,config_digest} /
챔피언 상대 지표(문서화 대비 delta + 선언 공차) / 10시드 집계 / 캐시 출처 / 환경 버전 / git 해시.

출처 참고: script.py(V4), screen_all_10seed.py, e6c_blend_folds.py, check_champion_provenance.py,
migration_audit.py(sha256 방식), experiments/e6c_blend_results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("LGAIMERS_ROOT", str(PROJECT_ROOT))

# ════════════════════════════════════════════════════════════════════
# 동결 챔피언 컨트롤 (읽기 전용 — 절대 변경 금지)
# ════════════════════════════════════════════════════════════════════
W_LGB = 0.51
C_LOGIT = -0.0404
CLIP_LO, CLIP_HI = 0.30, 0.70
SEEDS = tuple(range(42, 52))  # 10시드 42..51
FOLDS = ("primary", "r2022", "r2023", "r2024")
R_FOLDS = ("r2022", "r2023", "r2024")
CHAMPION_DIR = PROJECT_ROOT / "team_member_materials" / "GIHO" / "submit979_extract"

SCHEMA_VERSION = 1
DECLARED_TOLERANCE_BSS = 1.0  # 문서화 수치 대비 허용차(±BSS) — 캐시 아티팩트 재현 검증용
FORBIDDEN_FEATURES = ("asof_n_bucket", "score_diff_binary")

# 챔피언 피처 계약 — model/train_meta.json 에서 READ(재계산 아님). 동결 목록(49).
CHAMPION_FEATURES = (
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "platoon", "count_state",
)

# MLP 범주 계약 — model/mlp_meta.json 에서 READ(9). 동결 목록.
MLP_CATS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "top_bottom",
    "game_type", "base_state", "platoon", "count_state",
)
# LGB 범주 계약 — model/train_meta.json cats(5).
LGB_CATS = ("top_bottom", "game_type", "base_state", "platoon", "count_state")

# 챔피언 파일 sha256 동결 — migration_audit.py 와 동일 방식(script.py + model/* 전체 24파일).
CHAMPION_FILES_SHA256 = {
    "script.py": "e02ae9df6325e38ecce748143abf69a09ee5a300123b2b78643f579e5060d76f",
    "model/f3_s42.txt": "0b5a63cacf39ae4c0f93fa34c54d4e1bc2ad5bc0175a71a143e39036efd5de7f",
    "model/f3_s43.txt": "d4e8dcf4ad95b52ecb86a49cf02950cbf2050737d9db44713d992c67744b4722",
    "model/f3_s44.txt": "676bb19e737caf8f0de1ed99bf7e9d22762485e04204600222e0bc2c6ba7f808",
    "model/f3_s45.txt": "7e270e3332eaa236be16770b4cadf0e6ac0da84241516bb834ca919ca120ee20",
    "model/f3_s46.txt": "1076d4f19818d1150918799e383d59e0262c3c4641773a15310f70d4d6533d98",
    "model/f3_s47.txt": "a6775da634885b31e99db19e4fc4442012b0518e581f68e19e03af382681bffb",
    "model/f3_s48.txt": "9813b4c7027af3987988b73c18bf5069a3d4dfa94f92682d5ea8480ea66d3322",
    "model/f3_s49.txt": "067180e0dff0c703b9bc87fda3bb548cc256ca23d5827343584b94532f22520f",
    "model/f3_s50.txt": "ddb75cff7e4e3fa75791fbd444c6c8edebce5ea7de5f3d755b24225c0d534de1",
    "model/f3_s51.txt": "a5b27e653dbe9b6f733dcc10b098072eb8e30da3b36569b511fbe57f00ce46cd",
    "model/mlp_meta.json": "4892f1b5ce43cd57aef7b9c13a98adf8cc15ec285177aae7acb5a634d355c5e8",
    "model/mlp_prep.pkl": "ac7bc2887b26e905692cd650898fbe70c2f29f494af425929a905b3b980d4e71",
    "model/mlp_s42.pt": "5a1bf9cd457f7cbaf0916b78bef199a2111ba02c93ab9c42ad6848a6f3dadbc5",
    "model/mlp_s43.pt": "d6e493cb0433c19ed94cab462b014da0f8a35f7fe6dc833b6fc82069c413759d",
    "model/mlp_s44.pt": "cd95ccd3b7e55837442451a607615c4509cc866a3c4da853f29d9026f515f545",
    "model/mlp_s45.pt": "27ffccf65c0c6106b612dc58dcaf63e7990bd43522b4809f7567e3dc88a842f5",
    "model/mlp_s46.pt": "9a5002fc84da55c51dc26478a77ee4bfe0184ded730e85c9658e20328ab8bc11",
    "model/mlp_s47.pt": "025d55e33cfb3b957750056891f1ed16b92ba8444078fa3addd2301192084883",
    "model/mlp_s48.pt": "1f1becab5aca63b86f934fd0f88b43a6df70381b8759b45228269ff3bac1ac3e",
    "model/mlp_s49.pt": "f46f345d4ae0b5113f04d4b033675ca120b041a66d8679b23e89a74a090dbe86",
    "model/mlp_s50.pt": "87c76700101b49860769cd12e25e368ca22a76e519f2d7a915d7a5d2970037b3",
    "model/mlp_s51.pt": "018cb20d1ebb3783bab98f2f300cff27aafabeda5641498b8b3cbf25aff04f39",
    "model/train_meta.json": "60ee08356525b55faa2db4aab64ad4fdf4c69fb23f42759b4f5e87f2ed928f6a",
}

# 문서화된 챔피언 수치 — experiments/e6c_blend_results.json (w_lgb=0.51, LGB 10시드 × MLP 5시드
# 아티팩트 기준). delta 비교의 기준값.
DOCUMENTED = {
    "primary_blend": 774.1586482109209,
    "primary_gain": 44.47870005647178,
    "primary_lgb": 729.6799481544491,
    "r2022": {"lgb": 578.4091008043557, "gain_fixed": 28.81996499751733},
    "r2023": {"lgb": 550.9528062504487, "gain_fixed": 20.877822340092962},
    "r2024": {"lgb": 711.3974940101131, "gain_fixed": 12.74579370166282},
}


def _config_dict() -> dict:
    """동결 컨트롤 전체의 정규화 표현 — 구성 다이제스트의 원천."""
    return {
        "schema_version": SCHEMA_VERSION,
        "w_lgb": W_LGB,
        "c_logit": C_LOGIT,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "r_folds": list(R_FOLDS),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "champion_features": list(CHAMPION_FEATURES),
        "mlp_cats": list(MLP_CATS),
        "lgb_cats": list(LGB_CATS),
        "champion_files_sha256": dict(sorted(CHAMPION_FILES_SHA256.items())),
        "documented": DOCUMENTED,
        "tolerance_bss": DECLARED_TOLERANCE_BSS,
    }


def _config_digest() -> str:
    """동결 컨트롤 구성 다이제스트 (정규화 JSON → sha256)."""
    return hashlib.sha256(
        json.dumps(_config_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


EXPECTED_CONFIG_DIGEST = "6a8a42fa5c031f1a0e221227b0b34f8ea1b7f7ffa388db3c5cab62e289259ac2"

# ── 자기 무결성 게이트 (모듈 로드 시 1회; 표준 라이브러리만 사용 — 데이터/경로 무관) ──
_SELF_INTEGRITY_OK = _config_digest() == EXPECTED_CONFIG_DIGEST
if not _SELF_INTEGRITY_OK:
    _msg = (
        "[FAIL] 챔피언 컨트롤 구성 다이제스트 불일치: computed="
        f"{_config_digest()} expected={EXPECTED_CONFIG_DIGEST}. "
        "동결 상수(W_LGB/C_LOGIT/시드/폴드/피처/해시/문서화 수치)가 조작되었습니다. "
        "스코어링 전에 중단합니다."
    )
    print(_msg, file=sys.stderr)
    raise SystemExit(1)

os.chdir(REPO)  # common.py 상대 경로(DATA_DIR/CACHE_DIR) 기준 — 기존 러너(e6c_blend_folds)와 동일

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

# ── 후보 레지스트리 (Todo 4-7 이 확장; 미등록 id는 스코어링 전 거부) ──
CANDIDATES = {
    "champion": {
        "id": "champion",
        "description": "Frozen 979.31 champion — LGB F3(10시드)×MLP(5시드 e6c 아티팩트) "
                       "blend w=0.51, C_LOGIT=-0.0404 (제출 정렬용, 검증 BSS 무적용)",
        "features_source": "model/train_meta.json",
        "mlp_cats_source": "model/mlp_meta.json",
        "lgb_cache": {
            "primary": "cache/preds_primary_lgb_f3.npy",
            "r2022": "cache/h2b_r2022_lgb_f3.npy",
            "r2023": "cache/h2b_r2023_lgb_f3.npy",
            "r2024": "cache/h2b_r2024_lgb_f3.npy",
        },
        "mlp_cache": {
            "primary": "cache/mlp_primary.npy",
            "r2022": "cache/mlp_r2022.npy",
            "r2023": "cache/mlp_r2023.npy",
            "r2024": "cache/mlp_r2024.npy",
        },
        "lgb_seeds": list(range(42, 52)),  # 문서화 10시드 평균 로짓 (캐시 아티팩트)
        "mlp_seeds": list(range(42, 47)),  # 문서화 5시드(42..46) 평균 로짓 (e6c 아티팩트)
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _environment() -> dict:
    import torch  # noqa: PLC0415
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": __import__("lightgbm").__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def build_folds(train: pd.DataFrame) -> dict:
    """시계열 폴드 — screen_all_10seed.py:build_folds 와 정확히 동일."""
    is_r = train["game_type"] == "R"
    return {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & is_r, (train["season"] == 2022) & is_r),
        "r2023": ((train["season"] <= 2022) & is_r, (train["season"] == 2023) & is_r),
        "r2024": ((train["season"] <= 2023) & is_r, (train["season"] == 2024) & is_r),
    }


def _check_leakage(folds: dict, train: pd.DataFrame) -> list[str]:
    """누수 가드: 검증 마스크에 2025 시즌 행이 절대 없어야 한다."""
    problems = []
    for fn, (_tr, va) in folds.items():
        if int(va.sum()) == 0:
            problems.append(f"{fn}: 검증 마스크가 비어 있음")
        if (train.loc[va, "season"] == 2025).any():
            problems.append(f"{fn}: 검증 마스크에 2025 시즌 포함 — 누수!")
    return problems


def _check_champion_files() -> list[str]:
    """챔피언 파일 24종 sha256 을 동결 컨트롤과 대조 (스코어링 전)."""
    problems = []
    for rel, expected in CHAMPION_FILES_SHA256.items():
        path = CHAMPION_DIR / rel
        if not path.is_file():
            problems.append(f"champion {rel}: 파일 없음")
            continue
        actual = _sha256(path)
        if actual != expected:
            problems.append(f"champion {rel}: 기대 {expected} / 실제 {actual}")
    return problems


def _check_feature_contract(candidate_id: str) -> tuple[bool, str]:
    """챔피언 피처(49)/MLP-cat(9)/LGB-cat(5) 계약 — 동결 아티팩트에서 READ 후 동결 목록과 대조."""
    problems = []
    train_meta_path = CHAMPION_DIR / "model" / "train_meta.json"
    if not train_meta_path.is_file():
        return False, f"train_meta.json 없음: {train_meta_path}"
    train_meta = json.loads(train_meta_path.read_text(encoding="utf-8"))
    feats = tuple(train_meta.get("features", []))
    if feats != CHAMPION_FEATURES:
        problems.append(
            f"features {len(feats)}개가 동결 계약(49)과 불일치: "
            f"same={[f for f, e in zip(feats, CHAMPION_FEATURES) if f == e][:5]}... "
            f"n={len(feats)}")
    forbidden = [f for f in FORBIDDEN_FEATURES if f in feats]
    if forbidden:
        problems.append(f"train_meta에 금지 피처 포함: {forbidden}")
    lgb_cats = tuple(train_meta.get("cats", []))
    if lgb_cats != LGB_CATS:
        problems.append(f"LGB cats {lgb_cats} != 동결 {LGB_CATS}")

    mlp_meta_path = CHAMPION_DIR / "model" / "mlp_meta.json"
    if not mlp_meta_path.is_file():
        return False, f"mlp_meta.json 없음: {mlp_meta_path}"
    mlp_meta = json.loads(mlp_meta_path.read_text(encoding="utf-8"))
    mlp_cats = tuple(mlp_meta.get("cats", []))
    if mlp_cats != MLP_CATS:
        problems.append(f"MLP cats {mlp_cats} != 동결 {MLP_CATS}")

    detail = (f"피처 {len(feats)}개 / LGB cats {len(lgb_cats)} / MLP cats {len(mlp_cats)} "
              f"— 동결 아티팩트 READ 후 계약 대조")
    return (not problems), detail


def _load_cached(folds: dict, train: pd.DataFrame, candidate: dict) -> tuple[dict, dict, list]:
    """캐시 백업: LGB/MLP 폴드 로짓 로드. 길이는 검증 행 수와 대조(낡은 캐시 가드)."""
    z_lgb, z_mlp, provenance = {}, {}, []
    for fn in FOLDS:
        n_va = int(folds[fn][1].sum())
        for comp, key, store in (("lgb", "lgb_cache", z_lgb), ("mlp", "mlp_cache", z_mlp)):
            rel = candidate[key][fn]
            path = REPO / rel
            if not path.is_file():
                raise RuntimeError(f"[FAIL] 캐시 로짓 없음: {rel} (--rerun-train 필요)")
            arr = np.load(path)
            ok = int(len(arr)) == n_va
            provenance.append({
                "fold": fn, "component": comp, "file": rel,
                "sha256": _sha256(path), "rows": int(len(arr)), "n_va": n_va,
                "row_match": ok,
            })
            if not ok:
                raise RuntimeError(
                    f"[FAIL] 캐시 로짓 길이 불일치: {rel} rows={len(arr)} != "
                    f"fold {fn} 검증 행 수 {n_va} (낡은 캐시 의심)")
            store[fn] = arr
    return z_lgb, z_mlp, provenance


def _retrain_lgb(train: pd.DataFrame, folds: dict, seeds: list[int]) -> dict:
    """LGB 재학습 (10시드 로짓 평균) — screen_all_10seed.run_config 패턴, 동결 피처 계약."""
    import lightgbm as lgb  # noqa: PLC0415
    feats = list(CHAMPION_FEATURES)
    out = {}
    for fn, (tr_m, va_m) in folds.items():
        X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
        X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
        zs = []
        for seed in seeds:
            params = dict(common.PARAMS)
            params["seed"] = seed
            dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=list(LGB_CATS))
            dva = lgb.Dataset(X_va, y_va, categorical_feature=list(LGB_CATS), reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            zs.append(common.logit(model.predict(X_va, num_iteration=model.best_iteration)))
        out[fn] = np.mean(zs, axis=0)
        print(f"  [retrain lgb] {fn}: seeds={seeds} done", flush=True)
    return out


def _retrain_mlp(train: pd.DataFrame, folds: dict, seeds: list[int]) -> dict:
    """MLP 재학습 (10시드 로짓 평균) — e6c_blend_folds.py EntityMLP/train_seed CPU 패턴.
    캐시 백업과 달리 학습 중 UNK dropout 을 포함한 e6c OOF 패턴 그대로 재현."""
    from e6c_blend_folds import (build_fold, train_seed, DEVICE)  # noqa: PLC0415
    print(f"  [retrain mlp] device: {DEVICE} (CPU 전용 여기선 느림)", flush=True)
    feats = list(CHAMPION_FEATURES)
    cats = [c for c in MLP_CATS if c in feats]
    out = {}
    for fn, (tr_m, va_m) in folds.items():
        (Xn_tr, Xc_tr, Xn_va, Xc_va, _ytr, _yv, cat_vocab, n_num) = build_fold(
            train, feats, tr_m, va_m, cats)
        zs = []
        for seed in seeds:
            _b, z = train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, _ytr, _yv, cat_vocab, n_num, seed)
            zs.append(z)
        out[fn] = np.mean(zs, axis=0)
        print(f"  [retrain mlp] {fn}: seeds={seeds} done", flush=True)
    return out


def _run_fold(fn: str, z_lgb: np.ndarray, z_mlp: np.ndarray, yv: np.ndarray,
              n_va: int, candidate: str) -> dict:
    """한 폴드: 블렌드 로짓 → 저장 + BSS/평균/다이제스트."""
    if len(z_lgb) != n_va or len(z_mlp) != n_va:
        raise RuntimeError(f"{fn}: 로짓 길이({len(z_lgb)}/{len(z_mlp)}) != 검증 행 수({n_va})")
    z_blend = W_LGB * z_lgb + (1.0 - W_LGB) * z_mlp
    out_dir = REPO / "cache" / "qualification" / candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fn}.npy"
    np.save(out_path, z_blend.astype(np.float64))
    p_blend = common.sigmoid(z_blend)
    return {
        "n_rows": int(n_va),
        "bss": float(common.score(p_blend, yv)),
        "pred_mean": float(p_blend.mean()),
        "digest": _sha256(out_path),
        "config_digest": EXPECTED_CONFIG_DIGEST,
        "logits_path": f"cache/qualification/{candidate}/{fn}.npy",
        "lgb_bss": float(common.score(common.sigmoid(z_lgb), yv)),
        "mlp_bss": float(common.score(common.sigmoid(z_mlp), yv)),
        "blend_bss": float(common.score(p_blend, yv)),
        "blend_gain_vs_lgb": float(common.score(p_blend, yv) - common.score(common.sigmoid(z_lgb), yv)),
        "corr_lgb_mlp": float(np.corrcoef(z_lgb, z_mlp)[0, 1]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="챔피언 프리즈 컨트롤 + 공용 시계열 OOF 자격 계약 러너")
    parser.add_argument("--candidate", default="champion",
                        help="후보 id (기본 champion; 미등록 id 는 스코어링 전 거부)")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크: 시드 [42,43]으로 폴드/다이제스트/무결성/스키마 경로 검증 후 PASS 출력")
    parser.add_argument("--rerun-train", action="store_true",
                        help="10시드 × 4폴드 처음부터 재학습 (CPU 전용, 느림 — 기본은 캐시 백업)")
    parser.add_argument("--evidence", default=None,
                        help="증거 JSON 경로 (기본 .omo/evidence/aimers9-top100/task-2-control.json)")
    args = parser.parse_args(argv)

    t0 = time.time()
    candidate_id = args.candidate
    smoke = bool(args.smoke)
    rerun = bool(args.rerun_train)
    seeds = list(SEEDS[:2]) if smoke else list(SEEDS)

    evidence_path = (
        Path(args.evidence).expanduser().resolve() if args.evidence
        else PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-2-control.json"
    )

    # ── 1) 후보 레지스트리 검증 (스코어링 전) ──
    if candidate_id not in CANDIDATES:
        print(f"[FAIL] 미등록 후보 id: {candidate_id!r} — 등록된 후보: {sorted(CANDIDATES)}",
              file=sys.stderr)
        return 1
    candidate = CANDIDATES[candidate_id]

    # ── 2) 무결성 게이트 (스코어링 전) ──
    checks: list[dict] = [{
        "check": "config-digest", "ok": bool(_SELF_INTEGRITY_OK),
        "detail": f"동결 구성 다이제스트 일치 ({EXPECTED_CONFIG_DIGEST})",
    }]
    if not _SELF_INTEGRITY_OK:
        print("[FAIL] 구성 다이제스트 게이트 실패 (동결 상수 조작)", file=sys.stderr)
        return 1
    if candidate_id == "champion":
        file_problems = _check_champion_files()
        checks.append({
            "check": "champion-files-sha256", "ok": not file_problems,
            "detail": (f"{len(CHAMPION_FILES_SHA256)}파일 전부 동결 해시 일치"
                       if not file_problems else "; ".join(file_problems)),
        })
        if file_problems:
            print("[FAIL] 챔피언 파일 sha256 게이트 실패 — 스코어링 전 중단:\n  "
                  + "\n  ".join(file_problems), file=sys.stderr)
            return 1
    feat_ok, feat_detail = _check_feature_contract(candidate_id)
    checks.append({"check": "feature-contract", "ok": feat_ok, "detail": feat_detail})
    if not feat_ok:
        print(f"[FAIL] 피처/MLP-cat 계약 게이트 실패: {feat_detail}", file=sys.stderr)
        return 1

    # ── 3) 데이터 로드 + 폴드 + 누수 가드 ──
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    missing_feats = [f for f in CHAMPION_FEATURES if f not in train.columns]
    if missing_feats:
        print(f"[FAIL] 학습 데이터에 동결 피처 부재: {missing_feats}", file=sys.stderr)
        return 1
    folds = build_folds(train)
    leakage_problems = _check_leakage(folds, train)
    checks.append({
        "check": "leakage-guard", "ok": not leakage_problems,
        "detail": ("검증 마스크에 2025 시즌 없음 (누수 가드 통과)"
                   if not leakage_problems else "; ".join(leakage_problems)),
    })
    if leakage_problems:
        print("[FAIL] 누수 가드 실패:\n  " + "\n  ".join(leakage_problems), file=sys.stderr)
        return 1

    # ── 4) 폴드 로짓 획득 (캐시 백업 or 재학습) ──
    mode = "retrain" if rerun else "cache"
    cache_provenance: list[dict] = []
    if rerun:
        print(f"[qualification] {candidate_id} 재학습 모드: {len(seeds)}시드 × 4폴드 (CPU, 느림)",
              flush=True)
        z_lgb_all = _retrain_lgb(train, folds, seeds)
        z_mlp_all = _retrain_mlp(train, folds, seeds)
        lgb_seeds = list(seeds)
        mlp_seeds = list(seeds)
    else:
        print(f"[qualification] {candidate_id} 캐시 백업 모드 (문서화 e6c 수치의 원본 아티팩트)",
              flush=True)
        z_lgb_all, z_mlp_all, cache_provenance = _load_cached(folds, train, candidate)
        lgb_seeds = list(candidate["lgb_seeds"])
        mlp_seeds = list(candidate["mlp_seeds"])

    # ── 5) 폴드별 평가 ──
    per_fold: dict[str, dict] = {}
    for fn in FOLDS:
        va_m = folds[fn][1]
        yv = train.loc[va_m, common.TARGET].values
        per_fold[fn] = _run_fold(fn, z_lgb_all[fn], z_mlp_all[fn], yv,
                                 int(va_m.sum()), candidate_id)
        row = per_fold[fn]
        print(f"  [{fn:<8s}] lgb={row['lgb_bss']:>8.1f} mlp={row['mlp_bss']:>8.1f} "
              f"blend={row['blend_bss']:>8.1f} gain={row['blend_gain_vs_lgb']:>+7.1f} "
              f"mean={row['pred_mean']:.4f} n={row['n_rows']}", flush=True)

    # ── 6) 챔피언 상대 지표 + 공차 ──
    blend_gains = {fn: per_fold[fn]["blend_gain_vs_lgb"] for fn in FOLDS}
    deltas = {
        "primary_blend": per_fold["primary"]["blend_bss"] - DOCUMENTED["primary_blend"],
        "primary_gain": blend_gains["primary"] - DOCUMENTED["primary_gain"],
    }
    for fn in R_FOLDS:
        deltas[f"{fn}_gain"] = blend_gains[fn] - DOCUMENTED[fn]["gain_fixed"]
    max_abs_delta = max(abs(v) for v in deltas.values())
    within_tolerance = max_abs_delta <= DECLARED_TOLERANCE_BSS
    champion_relative = {
        "documented_primary_blend": DOCUMENTED["primary_blend"],
        "repro_primary_blend": per_fold["primary"]["blend_bss"],
        "delta_primary_blend": deltas["primary_blend"],
        "documented_primary_gain": DOCUMENTED["primary_gain"],
        "repro_primary_gain": blend_gains["primary"],
        "delta_primary_gain": deltas["primary_gain"],
        "r_only": {fn: {
            "documented_gain": DOCUMENTED[fn]["gain_fixed"],
            "repro_gain": blend_gains[fn],
            "delta": deltas[f"{fn}_gain"],
        } for fn in R_FOLDS},
        "declared_tolerance_bss": DECLARED_TOLERANCE_BSS,
        "max_abs_delta_bss": max_abs_delta,
        "within_tolerance": bool(within_tolerance),
    }

    # ── 7) 10시드 집계 ──
    seed_aggregation = {
        "lgb_seeds": lgb_seeds,
        "mlp_seeds": mlp_seeds,
        "aggregation": "mean of per-seed logits",
        "note": ("캐시 백업: LGB는 문서화 10시드 평균, MLP는 문서화 5시드(42..46) 평균 로짓 "
                 "(experiments/e6c_blend_results.json 원본 아티팩트). "
                 "--rerun-train 시 LGB/MLP 모두 요청 시드 평균."),
    }

    # ── 8) 스키마 + 저장 ──
    schema = {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate_id,
        "candidate_description": candidate["description"],
        "mode": mode,
        "smoke": smoke,
        "seeds_requested": list(seeds),
        "folds": list(FOLDS),
        "r_folds": list(R_FOLDS),
        "w_lgb": W_LGB,
        "c_logit": C_LOGIT,
        "clip": [CLIP_LO, CLIP_HI],
        "c_logit_note": ("제출 정렬용 동결 상수 — 검증 BSS는 raw 시그모이드로 측정(캘리브레이션 불변)"),
        "control": {
            "config_digest": EXPECTED_CONFIG_DIGEST,
            "config_digest_matches_frozen": True,
            "feature_contract": {"source": "model/train_meta.json", "n_features": len(CHAMPION_FEATURES),
                                 "forbidden_present": False},
            "mlp_cats": {"source": "model/mlp_meta.json", "n_cats": len(MLP_CATS)},
            "champion_files_checked": len(CHAMPION_FILES_SHA256),
        },
        "integrity": {"gate": "PASS", "checks": checks},
        "leakage_guard": {
            "no_2025_in_any_validation_mask": True,
            "train_seasons": sorted(int(s) for s in train["season"].unique().tolist()),
        },
        "per_fold": per_fold,
        "seed_aggregation": seed_aggregation,
        "champion_relative": champion_relative,
        "cache_provenance": cache_provenance,
        "environment": _environment(),
        "git_commit": _git_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.time() - t0,
    }

    result_path = REPO / "cache" / "qualification" / candidate_id / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    print(f"\n[qualification] {candidate_id}: 4폴드 스키마 저장 → {result_path}", flush=True)
    print(f"[qualification] {candidate_id}: 증거 JSON → {evidence_path}", flush=True)
    print(f"[qualification] primary blend={per_fold['primary']['blend_bss']:.3f} "
          f"(문서화 {DOCUMENTED['primary_blend']:.3f}, Δ{deltas['primary_blend']:+.3f}) | "
          f"공차 ±{DECLARED_TOLERANCE_BSS:.1f} → "
          f"{'OK' if within_tolerance else 'FAIL'}", flush=True)
    print(f"[qualification] R-only gains: " + ", ".join(
        f"{fn} {blend_gains[fn]:+.1f} (Δ{deltas[f'{fn}_gain']:+.3f})" for fn in R_FOLDS),
        flush=True)

    # ── 9) 스모크 판정 ──
    if smoke:
        required = ["schema_version", "candidate", "per_fold", "integrity", "leakage_guard",
                    "champion_relative", "seed_aggregation", "cache_provenance",
                    "environment", "git_commit"]
        ok = (
            len(seeds) == 2
            and all(fn in per_fold for fn in FOLDS)
            and all(k in schema for k in required)
            and schema["integrity"]["gate"] == "PASS"
            and schema["leakage_guard"]["no_2025_in_any_validation_mask"]
        )
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — 폴드/다이제스트/무결성/스키마 경로 "
              f"검증 완료 (seeds={seeds}, mode={mode})", flush=True)
        return 0 if ok else 1

    return 0 if within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
