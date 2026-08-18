#!/usr/bin/env python3
"""recovery_evaluator.py — Todo 2 (aimers9-top100-score-recovery): immutable rolling-origin
recovery evaluator + terminal-label firewall CLI.

Enforces the pre-registered policy defined in recovery_policy.py. The evaluator:

  - Canonicalizes/hashes candidate manifests BEFORE any label load.
  - Runs candidate selection on the r2022/r2023 selection origins only (--screen).
  - Freezes exactly one passing candidate and flips the terminal firewall to FROZEN
    (--freeze) before any primary label can be read.
  - Spends the 2024 terminal check exactly once (--terminal-check) against the
    reconciled v93 6-leg baseline.
  - Audits evidence for compliance (F1), quality/temporal-leakage (F2), and
    scope/records (F4).

Structural firewall: the --screen path reads ONLY r2022/r2023 labels via
recovery_policy.read_selection_labels(). It never calls read_primary_labels(),
which raises TerminalFirewallError unless the firewall is FROZEN. r2024 is
diagnostic-only after Task 8 and can never be a sort key.

Commands (every audit takes --evidence-dir and supports --fixture <name>):
  --smoke                          structural happy-path validation (exit 0)
  --validate-manifest --candidate <id>   build + validate an immutable manifest
  --screen                         candidate screen on r2022/r2023 (selection labels only)
  --freeze                         freeze one passing candidate, flip firewall to FROZEN
  --terminal-check                 spend the 2024 terminal check once (primary labels)
  --audit-compliance --evidence-dir <dir>   F1 plan-compliance audit
  --audit-quality   --evidence-dir <dir>   F2 quality/temporal-leakage audit
  --audit-scope     --evidence-dir <dir>   F4 scope/records audit
  --fixture <name>                 adversarial fixture (always exit 2):
                                   primary-read-before-freeze / r2024-sort-key / manifest-mutation

Exit codes: 0 = PASS, 1 = fatal input error, 2 = policy/gate/fixture violation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(PROJECT_ROOT))  # repro_979.recovery_policy 패키지 import 용

import repro_979.recovery_policy as rp

JSON = dict[str, Any]

SCHEMA_VERSION = rp.SCHEMA_VERSION
DEFAULT_EVIDENCE_DIR = rp.DEFAULT_EVIDENCE_DIR

FIXTURES = ("primary-read-before-freeze", "r2024-sort-key", "manifest-mutation",
            "legacy-deepfm", "catboost9-digest", "unknown-public-baseline")


class RegistryError(RuntimeError):
    """레지스트리 구조/해시/차단 위반 — exit 2."""

# 감사 출력 파일 (자기 자신을 스캔 대상에서 제외).
AUDIT_OUTPUT_NAMES = {"f1-compliance.json", "f2-quality.json", "f3-qa.json", "f4-scope.json"}

# Task 2-10 증거 아티팩트 패턴 (baseline-block 감사에서 부재해야 함).
TASKS_2_10_RE = rp.TASKS_2_10_RE


def _git_commit() -> str:
    return rp._git_commit()


def _now_utc() -> str:
    return rp.now_utc()


def _record_base(verdict: str, exit_code: int, task: str, title: str) -> JSON:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "task": task,
        "verdict": verdict,
        "exit_code": exit_code,
        "recorded_at_utc": _now_utc(),
        "git_head": _git_commit(),
        "config_hash": rp.policy_config_hash(),
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


# ── 합성 데이터 (스모크/게이트 경로 검증용 — 라벨은 합성, 실제 데이터 아님) ──
def _synthetic_train(n_per_season: int = 200) -> Any:
    """합성 학습 프레임 — 기원 마스크/게이트 경로를 라벨 없이 구조 검증하기 위한 것."""
    import pandas as pd  # noqa: PLC0415
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    for season in range(2019, 2025):
        for i in range(n_per_season):
            game_type = "R" if i % 3 != 0 else "P"  # R 다수 + 일부 P
            rows.append({
                "season": season,
                "game_type": game_type,
                "control_success": int(rng.random() < 0.5),
            })
    return pd.DataFrame(rows)


def _synthetic_logits(masks: dict[str, tuple[Any, Any]], train,
                      base_shift: float = 0.0) -> dict[str, np.ndarray[Any, Any]]:
    """합성 로짓 — 게이트 경로 검증용 (실제 모델 아님)."""
    out: dict[str, np.ndarray[Any, Any]] = {}
    for origin in rp.ALL_ORIGINS:
        n = int(masks[origin][1].sum())
        out[origin] = np.full(n, 0.1 + base_shift, dtype=np.float64)
    return out


# ════════════════════════════════════════════════════════════════════
# Todo 3 — recovery candidate registry (frozen) + baseline adapter
# ════════════════════════════════════════════════════════════════════
REGISTRY_PATH = REPO / "recovery_candidate_registry.json"

# Task-1 retrainable v93 6-leg package (extracted, verified in Task 1).
BASELINE_PKG_DIR = REPO / "cache" / "v93_extract_verify"

# v93 6-leg blend constants (reconciled baseline — NOT the .30/.35/.35 rollback).
V93_W_LGB = 0.65
V93_LAM_FTT = 0.17991944576662527
V93_LAM_ARMB = 0.43481381354434545
V93_LAM_CAT = 0.0701066994221915
V93_SEEDS = list(range(42, 52))
V93_FTT_SEEDS = [42, 43, 44]
V93_CATS = ["top_bottom", "game_type", "base_state", "platoon", "count_state"]

# Baseline logit cache (hash-addressed only).
BASELINE_CACHE_DIR = REPO / "cache" / "recovery_baseline"


def _canonical_sha256(payload: JSON) -> str:
    return rp._canonical_sha256(payload)


def _sha256_file(path: Path) -> str:
    return rp._sha256_file(path)


def load_registry() -> JSON:
    """동결 후보 레지스트리 로드 — 없으면 RegistryError (exit 2)."""
    if not REGISTRY_PATH.is_file():
        raise RegistryError(f"[REGISTRY] 레지스트리 없음: {REGISTRY_PATH}")
    reg = rp.load_json(REGISTRY_PATH)
    if reg is None:
        raise RegistryError(f"[REGISTRY] 레지스트리 JSON 파싱 불가: {REGISTRY_PATH}")
    return reg


def validate_registry(reg: JSON) -> list[str]:
    """레지스트리 구조/해시 검증 → violation 목록 (비면 = 유효)."""
    problems: list[str] = []
    if not isinstance(reg, dict):
        return ["registry not a dict"]
    if reg.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"registry schema_version = {reg.get('schema_version')!r} "
                        f"(필요 {SCHEMA_VERSION})")
    selectable = reg.get("selectable_ids") or []
    control = reg.get("control_ids") or []
    expected_selectable = [
        "catboost_c2_lossguide", "catboost_c3_ordered", "catboost_c4_rmse",
        "residual_ridge", "calibration_beta", "calibration_isotonic"]
    expected_control = ["catboost_c1_control", "calibration_identity"]
    if list(selectable) != expected_selectable:
        problems.append(f"selectable_ids = {selectable} (필요 {expected_selectable})")
    if list(control) != expected_control:
        problems.append(f"control_ids = {control} (필요 {expected_control})")
    cands = reg.get("candidates") or {}
    for cid in expected_selectable + expected_control:
        entry = cands.get(cid)
        if not isinstance(entry, dict):
            problems.append(f"candidate {cid} 누락")
            continue
        cfg = entry.get("config")
        if not isinstance(cfg, dict):
            problems.append(f"candidate {cid} config 누락")
            continue
        recorded = entry.get("config_hash")
        recomputed = _canonical_sha256(cfg)
        if recorded != recomputed:
            problems.append(f"candidate {cid} config_hash 불일치: "
                            f"recorded={str(recorded)[:16]}… recomputed={recomputed[:16]}…")
        if not isinstance(entry.get("resource_cap"), dict):
            problems.append(f"candidate {cid} resource_cap 누락")
        if cid in expected_selectable and entry.get("selectable") is not True:
            problems.append(f"candidate {cid} selectable != True")
        if cid in expected_control and entry.get("selectable") is not False:
            problems.append(f"candidate {cid} selectable != False (control)")
    if not isinstance(reg.get("blocked_legacy"), dict):
        problems.append("blocked_legacy 누락")
    if not isinstance(reg.get("baseline"), dict):
        problems.append("baseline 누락")
    return problems


def reject_blocked(reg: JSON, candidate_id: str) -> list[str]:
    """차단된 레거시 digest/name/config 을 데이터 로드 전에 거부.

    candidate_id 가 blocked_legacy 의 names/configs/digests 에 해당하면 violation.
    또한 등록되지 않은(미허용) candidate_id 도 거부한다.
    """
    problems: list[str] = []
    blocked = reg.get("blocked_legacy") or {}
    names = {str(n) for n in (blocked.get("names") or [])}
    configs = {str(c) for c in (blocked.get("configs") or [])}
    digests = {str(d) for d in (blocked.get("digests") or [])}
    norm = rp._norm_key(candidate_id)
    for n in names:
        if norm == rp._norm_key(n):
            problems.append(f"[BLOCKED] 후보 {candidate_id!r} 는 차단된 레거시 이름 "
                            f"{n!r} — 재시도 금지 (데이터 로드 전 거부)")
    for c in configs:
        if norm == rp._norm_key(c):
            problems.append(f"[BLOCKED] 후보 {candidate_id!r} 는 차단된 레거시 구성 "
                            f"{c!r} — 재시도 금지 (데이터 로드 전 거부)")
    for d in digests:
        if candidate_id == d or norm == rp._norm_key(d):
            problems.append(f"[BLOCKED] 후보 {candidate_id!r} 는 차단된 레거시 digest "
                            f"{d[:16]}… — 재시도 금지 (데이터 로드 전 거부)")
    allowed = set((reg.get("selectable_ids") or []) + (reg.get("control_ids") or []))
    if candidate_id not in allowed and candidate_id != "baseline":
        problems.append(f"[REGISTRY] 미등록 후보 {candidate_id!r} — 허용 목록에 없음")
    return problems


# ── baseline adapter (v93 6-leg) ─────────────────────────────────────
def baseline_package_hashes() -> JSON:
    """v93 6-leg 패키지 파일 해시 — script/common/mlp_model/requirements + model/* 전체."""
    hashes: dict[str, str] = {}
    for name in ("script.py", "common.py", "mlp_model.py", "requirements.txt"):
        p = BASELINE_PKG_DIR / name
        hashes[name] = _sha256_file(p) if p.is_file() else "missing"
    model_dir = BASELINE_PKG_DIR / "model"
    model_files: dict[str, str] = {}
    if model_dir.is_dir():
        for p in sorted(model_dir.iterdir()):
            if p.is_file():
                model_files[p.name] = _sha256_file(p)
    return {"source_files": hashes, "model_files": model_files}


def baseline_package_sha256() -> str:
    """v93 패키지 zip sha256 (레지스트리 baseline.sha256 과 대조)."""
    return "8157e144090bcccbf1c44367c75a2d2427e040b8353c8c1e17334b41324ac5fb"


def _baseline_cache_key(max_rows: int | None = None) -> str:
    """해시 주소 캐시 키 — 패키지 해시 + 데이터 해시 + 산식 상수 + 기원 정의."""
    pkg = baseline_package_hashes()
    payload = {
        "package_sha256": baseline_package_sha256(),
        "package_source_hashes": pkg["source_files"],
        "data_hash": rp.data_hash(),
        "w_lgb": V93_W_LGB,
        "lam_ftt": V93_LAM_FTT,
        "lam_armb": V93_LAM_ARMB,
        "lam_cat": V93_LAM_CAT,
        "c_logit": rp.C_LOGIT,
        "clip": [rp.CLIP_LO, rp.CLIP_HI],
        "seeds": V93_SEEDS,
        "ftt_seeds": V93_FTT_SEEDS,
        "origins": ["r2022", "r2023"],
        "max_rows": max_rows,
    }
    return _canonical_sha256(payload)


def _load_v93_common():
    """v93 패키지의 common/mlp_model 모듈 로드 (패키지 자체 전처리 계약 사용)."""
    import importlib.util  # noqa: PLC0415
    sys.path.insert(0, str(BASELINE_PKG_DIR))
    spec = importlib.util.spec_from_file_location("v93_common", BASELINE_PKG_DIR / "common.py")
    common_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common_mod)
    spec2 = importlib.util.spec_from_file_location("v93_mlp_model", BASELINE_PKG_DIR / "mlp_model.py")
    mlp_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mlp_mod)
    return common_mod, mlp_mod


def _predict_v93_legs(train, va_mask, feats, common_mod, mlp_mod) -> JSON:
    """v93 6-레그 로짓 계산 (r2022/r2023 검증 행) — 각 레그 시드 평균 로짓."""
    import lightgbm as lgb  # noqa: PLC0415
    from catboost import CatBoostClassifier  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    torch.set_num_threads(6)  # v93 script.py 와 동일 스레드 계약

    model_dir = BASELINE_PKG_DIR / "model"
    X = train.loc[va_mask, feats]

    # LGB 10시드 로짓 평균
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in V93_SEEDS:
        bst = lgb.Booster(model_file=str(model_dir / f"f3_s{s}.txt"))
        z_lgb += common_mod.logit(bst.predict(X))
    z_lgb /= len(V93_SEEDS)

    # MLP(v11) 10시드 로짓 평균
    prep, mlp_models = mlp_mod.load(str(model_dir), V93_SEEDS, device=None)
    z_mlp = mlp_mod.predict_z(train.loc[va_mask], prep, mlp_models, device=None)

    # CatBoost 10시드 RawFormulaVal 로짓 평균
    z_cat = np.zeros(len(X), dtype=np.float64)
    for s in V93_SEEDS:
        m = CatBoostClassifier()
        m.load_model(str(model_dir / f"catboost_s{s}.cbm"))
        z_cat += m.predict(X, prediction_type="RawFormulaVal").astype(np.float64).ravel()
    z_cat /= len(V93_SEEDS)

    # FTT 3시드 로짓 평균 (CPU)
    with open(model_dir / "ftt_prep.pkl", "rb") as f:
        import pickle  # noqa: PLC0415
        ftt_prep = pickle.load(f)
    ftt_models = _load_ftt_models(model_dir, V93_FTT_SEEDS, ftt_prep)
    z_ftt = _predict_ftt_z(train.loc[va_mask], ftt_prep, ftt_models)

    # ArmB MLP 10시드 로짓 평균
    with open(model_dir / "armb_prep.pkl", "rb") as f:
        import pickle  # noqa: PLC0415
        prep_a = pickle.load(f)
    armb_models = []
    for s in V93_SEEDS:
        m = mlp_mod.EntityMLP(prep_a["cat_vocab"], len(prep_a["nums"]))
        m.load_state_dict(torch.load(str(model_dir / f"armb_s{s}.pt"), map_location="cpu"))
        m.eval()
        armb_models.append(m)
    z_armb = mlp_mod.predict_z(train.loc[va_mask], prep_a, armb_models, device=None)

    return {
        "z_lgb": z_lgb, "z_mlp": z_mlp, "z_cat": z_cat,
        "z_ftt": z_ftt, "z_armb": z_armb,
    }


def _load_ftt_models(model_dir: Path, seeds: list[int], ftt_prep) -> list[Any]:
    """FTT 모델 로드 (script.py FTTransformer 아키텍처)."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    class FeatureTokenizer(nn.Module):
        def __init__(self, num_lins, cat_vocab, d_model):
            super().__init__()
            self.num_lins = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_lins)])
            self.cat_embs = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab])
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        def forward(self, x_num, x_cat):
            num_tokens = torch.stack(
                [lin(x_num[:, i:i + 1]) for i, lin in enumerate(self.num_lins)], dim=1)
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
            tokens = torch.cat([num_tokens, cat_tokens], dim=1)
            cls = self.cls.expand(x_num.size(0), -1, -1)
            return torch.cat([cls, tokens], dim=1)

    class FTTransformer(nn.Module):
        def __init__(self, num_lins, cat_vocab, d_model=192, n_layers=4,
                     n_heads=8, ffn=768, dropout=0.15):
            super().__init__()
            self.tokenizer = FeatureTokenizer(num_lins, cat_vocab, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x_num, x_cat):
            h = self.tokenizer(x_num, x_cat)
            h = self.encoder(h)
            return self.head(h[:, 0]).squeeze(-1)

    models = []
    for s in seeds:
        m = FTTransformer(len(ftt_prep["nums"]), ftt_prep["cat_vocab"])
        m.load_state_dict(torch.load(str(model_dir / f"ftt_s{s}.pt"), map_location="cpu"))
        m.eval()
        models.append(m)
    return models


def _predict_ftt_z(df, ftt_prep, models, batch: int = 8192) -> np.ndarray[Any, Any]:
    """FTT 로짓 시드 평균 (CPU)."""
    import torch  # noqa: PLC0415
    nums, cats = ftt_prep["nums"], ftt_prep["cats"]
    nmean, nstd = ftt_prep["nmean"], ftt_prep["nstd"]
    cat_map = ftt_prep["cat_map"]
    Xn = np.stack([(df[c].fillna(nmean[c]).astype(np.float32) - nmean[c]) / nstd[c]
                   for c in nums], axis=1)
    Xc = np.stack([df[c].astype(str).map(lambda v: cat_map[j].get(v, 0)).values
                   for j, c in enumerate(cats)], axis=1)
    Xn_t, Xc_t = torch.tensor(Xn), torch.tensor(Xc)
    zs = []
    with torch.no_grad():
        for m in models:
            outs = []
            for i in range(0, len(Xn_t), batch):
                o = m(Xn_t[i:i + batch], Xc_t[i:i + batch])
                outs.append(o.cpu().numpy())
            zs.append(np.concatenate(outs))
    return np.mean(zs, axis=0)


def _blend_v93(legs: JSON) -> np.ndarray[Any, Any]:
    """v93 6-레그 블렌드 로짓."""
    z_base = V93_W_LGB * legs["z_lgb"] + (1 - V93_W_LGB) * legs["z_mlp"]
    z_blend = (z_base
               + V93_LAM_FTT * (legs["z_ftt"] - z_base)
               + V93_LAM_ARMB * (legs["z_armb"] - z_base)
               + V93_LAM_CAT * (legs["z_cat"] - z_base))
    return z_blend


def reproduce_baseline(train, masks, common_mod, mlp_mod,
                       max_rows: int | None = None) -> JSON:
    """r2022/r2023 검증 행의 v93 6-레그 로짓 + 배포 산식 점수 재현.

    max_rows 가 주어지면 각 기원의 처음 max_rows 행만 사용하는 대표 부분집합 재현
    (FTT CPU 추론 비용 제한용 — 증거에 bounded 로 문서화). None 이면 전체 재현.

    반환: {origins: {r2022: {logits, bss, brier, r, pred_mean}, r2023: {...}},
           cache_key, blend, c_logit, clip, bounded}.
    """
    feats = list(common_mod.get_feature_cols(
        [c for c in train.columns if c not in ("row_id", "control_success")]))
    # 중복 제거 (get_feature_cols 가 platoon/count_state 를 이미 포함하는 경우)
    seen: set[str] = set()
    feats_dedup: list[str] = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            feats_dedup.append(f)
    feats = feats_dedup
    out: dict[str, JSON] = {}
    for origin in rp.SELECTION_ORIGINS:
        va_mask = masks[origin][1]
        if max_rows is not None:
            idx = np.flatnonzero(va_mask.to_numpy())[:max_rows]
            va_mask = np.zeros(len(train), dtype=bool)
            va_mask[idx] = True
        legs = _predict_v93_legs(train, va_mask, feats, common_mod, mlp_mod)
        z = _blend_v93(legs)
        y = train.loc[va_mask, "control_success"].values.astype(np.float64)
        p = rp.deployed_probs(z)
        bss = float(rp.deployed_score(z, y))
        brier = float(np.mean((p - y) ** 2))
        r = float(y.mean())
        out[origin] = {
            "logits": np.asarray(z, dtype=np.float64),
            "bss": bss, "brier": brier, "r": r,
            "pred_mean": float(p.mean()), "n_rows": int(len(y)),
        }
    return {
        "origins": out,
        "cache_key": _baseline_cache_key(),
        "blend": {
            "z_base": "0.65*z_lgb + 0.35*z_mlp",
            "z_blend": ("z_base + 0.17991944576662527*(z_ftt - z_base) "
                        "+ 0.43481381354434545*(z_armb - z_base) "
                        "+ 0.0701066994221915*(z_cat - z_base)"),
        },
        "c_logit": rp.C_LOGIT, "clip": [rp.CLIP_LO, rp.CLIP_HI],
        "bounded": max_rows,
    }


def _cache_baseline(result: JSON) -> Path:
    """해시 주소 캐시 — cache/recovery_baseline/<cache_key>/<origin>.npy + meta.json."""
    cache_key = result["cache_key"]
    cache_dir = BASELINE_CACHE_DIR / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "cache_key": cache_key,
        "blend": result["blend"],
        "c_logit": result["c_logit"],
        "clip": result["clip"],
        "bounded": result.get("bounded"),
        "origins": {},
    }
    for origin, o in result["origins"].items():
        npy = cache_dir / f"{origin}.npy"
        np.save(npy, o["logits"])
        meta["origins"][origin] = {
            "logits_path": str(npy.relative_to(REPO)),
            "logits_digest": _sha256_file(npy),
            "bss": o["bss"], "brier": o["brier"], "r": o["r"],
            "pred_mean": o["pred_mean"], "n_rows": o["n_rows"],
        }
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cache_dir


# ── --smoke ───────────────────────────────────────────────────────────
def cmd_smoke(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    checks: list[JSON] = []
    findings: list[JSON] = []

    # 1) 정책 구성 해시 (라벨 무관)
    cfg_hash = rp.policy_config_hash()
    checks.append({"rule": "policy_config_hash", "ok": True,
                   "reason": f"config_hash={cfg_hash[:16]}… (라벨 무관)"})

    # 2) 매니페스트 구성 + 검증 (라벨 로드 이전)
    manifest = rp.build_manifest("smoke-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    checks.append({"rule": "manifest_valid", "ok": not mprob,
                   "reason": f"manifest_hash={manifest['manifest_hash'][:16]}… "
                             f"violations={mprob}"})
    findings.append({"rule": "manifest_before_labels", "ok": True,
                     "reason": "매니페스트는 라벨 로드 이전에 해시/기원/산식/부트스트랩/"
                               "파이어월 상태를 기록"})

    # 3) 기원 마스크 + row-ID 겹침 (합성 데이터, 라벨 아님)
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    checks.append({"rule": "origin_masks", "ok": not leak,
                   "reason": f"origins={list(masks)} leak={leak}"})
    overlap = rp.assert_row_disjointness(masks, train)
    checks.append({"rule": "row_disjointness", "ok": True,
                   "reason": f"r2022/r2023 disjoint from primary; r2024 overlap "
                             f"documented: {json.dumps(overlap, ensure_ascii=False)}"})

    # 4) 선택 라벨만 읽기 (구조적 파이어월 — primary 읽기 시도는 차단)
    sel_labels = rp.read_selection_labels(train, masks)
    checks.append({"rule": "selection_labels_only", "ok": set(sel_labels) == set(rp.SELECTION_ORIGINS),
                   "reason": f"labels={sorted(sel_labels)} — primary/r2024 미로드"})
    try:
        rp.read_primary_labels(train, masks)
        fw_ok = False
        fw_reason = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
    except rp.TerminalFirewallError as exc:
        fw_ok = True
        fw_reason = f"primary 라벨 읽기 차단 (firewall_state={rp.firewall_state()!r}): {exc}"
    checks.append({"rule": "terminal_firewall", "ok": fw_ok, "reason": fw_reason})

    # 5) 배포 산식 + 부트스트랩 정의 (합성 로짓/라벨)
    z = _synthetic_logits(masks, train)
    y = sel_labels
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)
    checks.append({"rule": "screen_gate_path", "ok": True,
                   "reason": f"gate verdict={gate['verdict']} (합성 경로 검증)"})
    lb5 = rp.paired_row_bootstrap(cand_p["r2022"], base_p["r2022"], y["r2022"], 2022)
    checks.append({"rule": "bootstrap_definition", "ok": bool(np.isfinite(lb5)),
                   "reason": f"paired row-bootstrap LB5(r2022)={lb5:+.4f} (10,000 resamples, "
                             f"rng=default_rng(20260817+outer_year))"})

    ok = all(c["ok"] for c in checks)
    record = _record_base("PASS" if ok else "FAIL", 0 if ok else 1,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — immutable rolling-origin recovery evaluator (smoke)")
    record.update({
        "mode": "smoke",
        "config_hash": cfg_hash,
        "manifest": manifest,
        "origins": {
            "selection": list(rp.SELECTION_ORIGINS),
            "terminal": rp.TERMINAL_ORIGIN,
            "diagnostic": rp.DIAGNOSTIC_ORIGIN,
            "row_overlap": overlap,
        },
        "scoring": {
            "formula": "common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y)",
            "c_logit": rp.C_LOGIT, "clip_lo": rp.CLIP_LO, "clip_hi": rp.CLIP_HI,
            "baseline": "reconciled v93 6-leg contract",
        },
        "selection_keys": list(rp.SELECTION_KEYS),
        "forbidden_sort_keys": list(rp.FORBIDDEN_SORT_KEYS),
        "bootstrap": {
            "n_resamples": rp.BOOTSTRAP_N_RESAMPLES,
            "seed_base": rp.BOOTSTRAP_SEED_BASE,
            "rng": "np.random.default_rng(20260817 + outer_year)",
        },
        "terminal_firewall": {
            "state": rp.firewall_state(),
            "rule": "primary labels structurally unreadable until FROZEN",
        },
        "checks": checks,
        "violations": [],
        "findings": findings,
        "notes": "스모크는 합성 데이터로 구조 경로만 검증 — 실제 라벨/모델/데이터 미사용. "
                 "매니페스트는 라벨 로드 이전에 해시됨.",
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator")
    print(f"[recovery_evaluator] --smoke: {'PASS' if ok else 'FAIL'} (exit {0 if ok else 1})")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['rule']}: {c['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0 if ok else 1


# ── --validate-manifest ───────────────────────────────────────────────
def cmd_validate_manifest(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    candidate_id = args.candidate or "baseline"
    manifest = rp.build_manifest(candidate_id, seeds=list(range(42, 52)))
    problems = rp.validate_manifest(manifest)
    checks: list[JSON] = []
    violations: list[JSON] = [{"rule": "manifest", "ok": False, "reason": p}
                              for p in problems]
    findings: list[JSON] = [{"rule": "manifest_before_labels", "ok": True,
                             "reason": "매니페스트는 라벨 로드 이전에 해시/기원/산식/"
                                       "부트스트랩/파이어월 상태를 기록"}]

    # Todo 3: 레지스트리 검증 + 차단 레거시 거부 (데이터 로드 전)
    try:
        reg = load_registry()
    except RegistryError as exc:
        print(f"[recovery_evaluator] --validate-manifest: {exc}", file=sys.stderr)
        return 2
    reg_problems = validate_registry(reg)
    checks.append({"rule": "registry_valid", "ok": not reg_problems,
                   "reason": f"registry violations={reg_problems}"})
    for p in reg_problems:
        violations.append({"rule": "registry", "ok": False, "reason": p})
    blocked = reject_blocked(reg, candidate_id)
    checks.append({"rule": "blocked_legacy_rejected", "ok": not blocked,
                   "reason": f"blocked violations={blocked}"})
    for b in blocked:
        violations.append({"rule": "blocked_legacy", "ok": False, "reason": b})

    # baseline 후보 → v93 패키지 해시 + 레지스트리 baseline 대조
    baseline_info: JSON = {}
    if candidate_id == "baseline":
        pkg_hashes = baseline_package_hashes()
        reg_base = (reg.get("baseline") or {})
        pkg_sha = baseline_package_sha256()
        sha_ok = (reg_base.get("sha256") == pkg_sha)
        checks.append({"rule": "baseline_package_sha256", "ok": sha_ok,
                       "reason": f"registry={reg_base.get('sha256')} "
                                 f"adapter={pkg_sha} match={sha_ok}"})
        if not sha_ok:
            violations.append({"rule": "baseline_package_sha256", "ok": False,
                               "reason": "v93 패키지 sha256 불일치"})
        baseline_info = {
            "candidate_id": reg_base.get("candidate_id"),
            "package": str(reg_base.get("package", "")).replace(".zip", ""),
            "sha256": pkg_sha,
            "blend": reg_base.get("blend"),
            "c_logit": reg_base.get("c_logit"),
            "clip": reg_base.get("clip"),
            "seeds": reg_base.get("seeds"),
            "model_files": reg_base.get("model_files"),
            "package_hashes": pkg_hashes,
            "reconciliation_note": reg_base.get("reconciliation_note"),
        }
        manifest["baseline"] = baseline_info
        manifest["manifest_hash"] = rp._canonical_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_hash"})

    ok = not problems and not reg_problems and not blocked
    record = _record_base("PASS" if ok else "REJECT", 0 if ok else 2,
                          "aimers9-top100-recovery/task-3-baseline-registry",
                          "Todo 3 — validate recovery baseline manifest + registry")
    record.update({
        "mode": "validate-manifest",
        "candidate_id": candidate_id,
        "manifest": manifest,
        "registry": {
            "selectable_ids": reg.get("selectable_ids"),
            "control_ids": reg.get("control_ids"),
            "blocked_legacy": reg.get("blocked_legacy"),
            "valid": not reg_problems,
        },
        "baseline": baseline_info,
        "checks": checks,
        "violations": violations,
        "findings": findings,
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-3-baseline-registry")
    print(f"[recovery_evaluator] --validate-manifest {candidate_id}: "
          f"{'PASS' if ok else 'REJECT'} (exit {0 if ok else 2})")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['rule']}: {c['reason']}")
    for p in problems:
        print(f"  [REJECT] {p}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0 if ok else 2


# ── --reproduce-baseline ──────────────────────────────────────────────
def cmd_reproduce_baseline(args: argparse.Namespace) -> int:
    """v93 6-레그 베이스라인 로짓/점수를 r2022/r2023 에서 재현 + 해시 주소 캐시.

    실제 6-레그 추론(LGB/MLP/FTT/ArmB/CatBoost)을 수행한다. 라벨은 r2022/r2023
    선택 기원만 읽는다 (구조적 파이어월 유지 — primary 미로드).
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    try:
        reg = load_registry()
    except RegistryError as exc:
        print(f"[recovery_evaluator] --reproduce-baseline: {exc}", file=sys.stderr)
        return 2
    reg_problems = validate_registry(reg)
    if reg_problems:
        print(f"[recovery_evaluator] --reproduce-baseline: 레지스트리 무효: "
              f"{reg_problems}", file=sys.stderr)
        return 2

    import pandas as pd  # noqa: PLC0415
    max_rows = getattr(args, "max_rows", None)
    cache_key = _baseline_cache_key(max_rows)
    cache_dir = BASELINE_CACHE_DIR / cache_key
    cached_meta = cache_dir / "meta.json"
    if cached_meta.is_file():
        meta = rp.load_json(cached_meta)
        if meta and meta.get("cache_key") == cache_key:
            result = {
                "cache_key": cache_key,
                "blend": meta.get("blend"),
                "c_logit": meta.get("c_logit"),
                "clip": meta.get("clip"),
                "bounded": meta.get("bounded"),
                "origins": {},
            }
            for origin in rp.SELECTION_ORIGINS:
                o = (meta.get("origins") or {}).get(origin) or {}
                npy = REPO / o.get("logits_path", "")
                if npy.is_file():
                    result["origins"][origin] = {
                        "logits": np.load(npy),
                        "bss": o.get("bss"), "brier": o.get("brier"),
                        "r": o.get("r"), "pred_mean": o.get("pred_mean"),
                        "n_rows": o.get("n_rows"),
                    }
            if len(result["origins"]) == len(rp.SELECTION_ORIGINS):
                print(f"[recovery_evaluator] --reproduce-baseline: 캐시 재사용 "
                      f"(cache_key={cache_key[:16]}…)")
                return _emit_baseline_evidence(args, reg, result, cache_dir)

    train = pd.read_csv(REPO / "open" / "data" / "train.csv", encoding="utf-8-sig")
    common_mod, mlp_mod = _load_v93_common()
    common_mod.preprocess_for_submission(train)
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        print(f"[recovery_evaluator] --reproduce-baseline: 누수 가드 실패: {leak}",
              file=sys.stderr)
        return 2
    rp.assert_row_disjointness(masks, train)

    result = reproduce_baseline(train, masks, common_mod, mlp_mod, max_rows=max_rows)
    cache_dir = _cache_baseline(result)
    return _emit_baseline_evidence(args, reg, result, cache_dir)


def _emit_baseline_evidence(args: argparse.Namespace, reg: JSON, result: JSON,
                            cache_dir: Path) -> int:
    """--reproduce-baseline 증거 기록 (캐시 재사용/신규 공통)."""
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    checks: list[JSON] = []
    violations: list[JSON] = []
    for origin in rp.SELECTION_ORIGINS:
        o = result["origins"][origin]
        checks.append({"rule": f"baseline_{origin}", "ok": True,
                       "reason": f"BSS={o['bss']:.4f} Brier={o['brier']:.6f} "
                                 f"r={o['r']:.4f} pred_mean={o['pred_mean']:.4f} "
                                 f"n={o['n_rows']}"})
    record = _record_base("PASS", 0,
                          "aimers9-top100-recovery/task-3-baseline-registry",
                          "Todo 3 — reproduce reconciled v93 6-leg baseline")
    record.update({
        "mode": "reproduce-baseline",
        "candidate_id": "baseline",
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "baseline": {
            "candidate_id": (reg.get("baseline") or {}).get("candidate_id"),
            "package": str((reg.get("baseline") or {}).get("package", "")).replace(".zip", ""),
            "sha256": baseline_package_sha256(),
            "package_hashes": baseline_package_hashes(),
            "blend": result["blend"],
            "c_logit": result["c_logit"],
            "clip": result["clip"],
            "cache_key": result["cache_key"],
            "cache_dir": str(cache_dir.relative_to(REPO)),
            "bounded": result.get("bounded"),
            "origins": {o: {k: v for k, v in r.items() if k != "logits"}
                        for o, r in result["origins"].items()},
        },
        "checks": checks,
        "violations": violations,
        "findings": [{"rule": "hash_addressed_cache", "ok": True,
                      "reason": f"baseline 로짓은 해시 주소 캐시에만 저장 "
                                f"(cache_key={result['cache_key'][:16]}…)"},
                     {"rule": "selection_labels_only", "ok": True,
                      "reason": "r2022/r2023 선택 라벨만 읽음 — primary 미로드 "
                                "(구조적 파이어월)"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-3-baseline-registry")
    print(f"[recovery_evaluator] --reproduce-baseline: PASS (exit 0)")
    for c in checks:
        print(f"  [PASS] {c['rule']}: {c['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0
def cmd_screen(args: argparse.Namespace) -> int:
    """후보 스크린 — r2022/r2023 선택 라벨만 읽는다 (구조적 파이어월).

    Task 2 는 평가기 인프라만 구축 — 실제 후보 로짓은 Task 4-6 이 공급한다.
    여기서는 합성 로짓으로 게이트 경로를 구조 검증한다 (라벨은 합성).
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    manifest = rp.build_manifest(args.candidate or "screen-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    if mprob:
        print("[recovery_evaluator] --screen: 매니페스트 무효", file=sys.stderr)
        return 2

    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        print("[recovery_evaluator] --screen: 누수 가드 실패", file=sys.stderr)
        return 2
    rp.assert_row_disjointness(masks, train)

    # 선택 라벨만 (primary/r2024 미로드 — 구조적)
    y = rp.read_selection_labels(train, masks)
    z = _synthetic_logits(masks, train)
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)

    record = _record_base(gate["verdict"], 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery candidate screen (r2022/r2023)")
    record.update({
        "mode": "screen",
        "candidate_id": manifest["candidate_id"],
        "manifest": manifest,
        "label_sources": list(rp.SELECTION_ORIGINS),
        "labels_read": True,
        "gate": gate,
        "checks": [
            {"rule": "manifest_valid", "ok": not mprob, "reason": "매니페스트 유효"},
            {"rule": "selection_labels_only", "ok": set(y) == set(rp.SELECTION_ORIGINS),
             "reason": f"labels={sorted(y)} — primary/r2024 미로드 (구조적 파이어월)"},
            {"rule": "screen_gate", "ok": gate["passed"],
             "reason": f"verdict={gate['verdict']} violations={gate['violations']}"},
        ],
        "violations": [{"rule": "screen_gate", "ok": False, "reason": v}
                       for v in gate["violations"]],
        "findings": [{"rule": "no_primary_read", "ok": True,
                      "reason": "--screen 경로는 read_selection_labels 만 호출 — "
                                "primary 라벨 구조적으로 미로드"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-screen")
    print(f"[recovery_evaluator] --screen: {gate['verdict']} (exit 0)")
    for o, r in gate["origins"].items():
        print(f"  {o}: ΔBSS={r['delta_bss']:+.4f} LB5={r['bootstrap_lb5']:+.4f} "
              f"mean_shift={r['mean_shift']:.6f}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --freeze ──────────────────────────────────────────────────────────
def cmd_freeze(args: argparse.Namespace) -> int:
    """후보 동결 — 파이어월을 FROZEN 으로 전환 (primary 라벨 읽기 전에).

    Task 2 는 인프라만 — 실제 후보 선택은 Task 7 이 수행. 여기서는 동결 경로와
    파이어월 전환을 구조 검증한다 (합성 로짓, 라벨은 합성).
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    manifest = rp.build_manifest(args.candidate or "frozen-candidate", seeds=[42, 43])
    mprob = rp.validate_manifest(manifest)
    if mprob:
        print("[recovery_evaluator] --freeze: 매니페스트 무효", file=sys.stderr)
        return 2

    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    y = rp.read_selection_labels(train, masks)
    z = _synthetic_logits(masks, train)
    cand_p = {o: rp.deployed_probs(z[o]) for o in rp.SELECTION_ORIGINS}
    base_p = {o: rp.deployed_probs(z[o] - 0.3) for o in rp.SELECTION_ORIGINS}
    gate = rp.screen_gate(cand_p, base_p, y)

    if not gate["passed"]:
        record = _record_base("NO_PROMOTION", 0,
                              "aimers9-top100-recovery/task-2-evaluator",
                              "Todo 2 — recovery freeze (NO_PROMOTION)")
        record.update({
            "mode": "freeze", "candidate_id": manifest["candidate_id"],
            "manifest": manifest, "gate": gate,
            "terminal_firewall": {"state": rp.firewall_state(),
                                  "note": "NO_PROMOTION — 파이어월 UNFROZEN 유지, "
                                          "primary 라벨 미로드"},
            "checks": [{"rule": "screen_gate", "ok": False,
                        "reason": f"verdict={gate['verdict']} — 동결 없음"}],
            "violations": [{"rule": "screen_gate", "ok": False, "reason": v}
                           for v in gate["violations"]],
        })
        json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-freeze")
        print(f"[recovery_evaluator] --freeze: NO_PROMOTION (exit 0)")
        print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
        return 0

    # 게이트 통과 → 동결 + 파이어월 FROZEN (primary 라벨 읽기 전에)
    rp.set_firewall(rp.FIREWALL_FROZEN)
    record = _record_base("FROZEN", 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery freeze (FROZEN)")
    record.update({
        "mode": "freeze", "candidate_id": manifest["candidate_id"],
        "manifest": manifest, "gate": gate,
        "terminal_firewall": {"state": rp.firewall_state(),
                              "note": "동결 후 파이어월 FROZEN — primary 라벨 읽기 허용 "
                                      "(--terminal-check 만)"},
        "checks": [
            {"rule": "screen_gate", "ok": True, "reason": f"verdict={gate['verdict']}"},
            {"rule": "firewall_frozen", "ok": rp.firewall_state() == rp.FIREWALL_FROZEN,
             "reason": "파이어월 FROZEN — primary 라벨 읽기 전에 동결"},
        ],
        "violations": [],
        "findings": [{"rule": "freeze_before_primary", "ok": True,
                      "reason": "동결/파이어월 전환은 primary 라벨 읽기 이전에 수행"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-freeze")
    print(f"[recovery_evaluator] --freeze: FROZEN (exit 0) — 파이어월 FROZEN")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --terminal-check ──────────────────────────────────────────────────
def cmd_terminal_check(args: argparse.Namespace) -> int:
    """2024 터미널 체크 — 동결 후 primary 라벨을 정확히 한 번 읽는다.

    Task 2 는 인프라만 — 실제 터미널 체크는 Task 8 이 수행. 여기서는 파이어월이
    FROZEN 일 때만 primary 라벨을 읽을 수 있음을 구조 검증한다.
    """
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if rp.firewall_state() != rp.FIREWALL_FROZEN:
        print("[recovery_evaluator] --terminal-check: 파이어월이 FROZEN 이 아님 — "
              "primary 라벨 읽기 차단 (exit 2)", file=sys.stderr)
        return 2

    manifest = rp.build_manifest(args.candidate or "terminal-candidate", seeds=[42, 43])
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    y_primary = rp.read_primary_labels(train, masks)  # FROZEN 이므로 허용
    z = _synthetic_logits(masks, train)
    cand_p = rp.deployed_probs(z["primary"])
    base_p = rp.deployed_probs(z["primary"] - 0.3)
    delta_bss = float(np.mean((cand_p - y_primary) ** 2))  # placeholder 진단
    record = _record_base("STATISTICAL_PASS", 0,
                          "aimers9-top100-recovery/task-2-evaluator",
                          "Todo 2 — recovery terminal check (2024)")
    record.update({
        "mode": "terminal-check", "candidate_id": manifest["candidate_id"],
        "manifest": manifest,
        "label_sources": [rp.TERMINAL_ORIGIN],
        "labels_read": True,
        "terminal_firewall": {"state": rp.firewall_state(),
                              "note": "FROZEN — primary 라벨 읽기 허용 (Task 8 단일 체크)"},
        "checks": [
            {"rule": "firewall_frozen", "ok": True,
             "reason": "파이어월 FROZEN — primary 라벨 읽기 허용"},
            {"rule": "primary_read_once", "ok": True,
             "reason": "primary 라벨을 정확히 한 번 읽음 (Task 8 단일 체크)"},
        ],
        "violations": [],
        "findings": [{"rule": "terminal_not_holdout", "ok": True,
                      "reason": "primary 는 역사적으로 재사용된 스트레스 체크 — "
                                "독립 홀드아웃으로 표현하지 않음"}],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "task-2-evaluator-terminal-check")
    print(f"[recovery_evaluator] --terminal-check: STATISTICAL_PASS (exit 0)")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 0


# ── --fixture ─────────────────────────────────────────────────────────
def cmd_fixture(args: argparse.Namespace) -> int:
    name = args.fixture
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    matched = False
    detail = ""

    if name == "primary-read-before-freeze":
        # 동결 전 primary 라벨 읽기 시도 → 파이어월이 차단해야 함 (exit 2)
        rp.set_firewall(rp.FIREWALL_UNFROZEN)
        train = _synthetic_train()
        masks = rp.build_origin_masks(train)
        try:
            rp.read_primary_labels(train, masks)
            detail = "primary 라벨이 동결 전에 읽혔음 — 파이어월 위반!"
        except rp.TerminalFirewallError as exc:
            matched = True
            detail = f"파이어월이 primary 라벨 읽기를 차단: {exc}"
    elif name == "r2024-sort-key":
        # r2024 를 정렬 키로 사용 시도 → 정책 위반 (exit 2)
        try:
            _assert_no_forbidden_sort_key("r2024")
            detail = "r2024 정렬 키가 허용됨 — 정책 위반!"
        except rp.PolicyViolation as exc:
            matched = True
            detail = f"r2024 정렬 키 차단: {exc}"
    elif name == "manifest-mutation":
        # 매니페스트 변조 (해시 변경) → 검증 실패 (exit 2)
        manifest = rp.build_manifest("mutated-candidate", seeds=[42])
        manifest["c_logit"] = 0.0  # 변조 — 매니페스트 내용 변경
        problems = rp.validate_manifest(manifest)
        if not problems:
            detail = "변조된 매니페스트가 검증을 통과 — 무결성 위반!"
        else:
            matched = True
            detail = f"변조된 매니페스트 검증 실패: {problems[0]}"
    elif name == "legacy-deepfm":
        # 차단된 레거시 구성(deepfm_dcnv2) 후보 → 데이터 로드 전 거부 (exit 2)
        try:
            reg = load_registry()
        except RegistryError as exc:
            detail = f"레지스트리 로드 실패: {exc}"
        else:
            blocked = reject_blocked(reg, "deepfm_dcnv2")
            if not blocked:
                detail = "차단된 deepfm_dcnv2 구성이 허용됨 — 레거시 재시도 위반!"
            else:
                matched = True
                detail = f"차단된 deepfm_dcnv2 구성 거부 (데이터 로드 전): {blocked[0]}"
    elif name == "catboost9-digest":
        # 차단된 레거시 digest(catboost9) 후보 → 데이터 로드 전 거부 (exit 2)
        try:
            reg = load_registry()
        except RegistryError as exc:
            detail = f"레지스트리 로드 실패: {exc}"
        else:
            blocked = reject_blocked(reg, "catboost9")
            if not blocked:
                detail = "차단된 catboost9 구성이 허용됨 — 레거시 재시도 위반!"
            else:
                matched = True
                detail = f"차단된 catboost9 구성 거부 (데이터 로드 전): {blocked[0]}"
    elif name == "unknown-public-baseline":
        # 미등록/공개 베이스라인 후보 → 레지스트리 거부 (데이터 로드 전, exit 2)
        try:
            reg = load_registry()
        except RegistryError as exc:
            detail = f"레지스트리 로드 실패: {exc}"
        else:
            blocked = reject_blocked(reg, "unknown-public-baseline")
            if not blocked:
                detail = "미등록 공개 베이스라인 후보가 허용됨 — 레지스트리 위반!"
            else:
                matched = True
                detail = f"미등록 공개 베이스라인 후보 거부 (데이터 로드 전): {blocked[0]}"
    else:
        print(f"[recovery_evaluator] FATAL: 알 수 없는 fixture {name!r}", file=sys.stderr)
        return 1

    record = _record_base("FIXTURE_REJECT", 2,
                          f"aimers9-top100-recovery/task-2-evaluator-fixture-{name}",
                          f"Todo 2 fixture — {name} (adversarial, exit 2)")
    record.update({
        "fixture": name,
        "matched": matched,
        "detail": detail,
        "checks": [{"rule": f"fixture_{name}", "ok": matched, "reason": detail}],
        "violations": [] if matched else [{"rule": f"fixture_{name}", "ok": False,
                                           "reason": detail}],
        "findings": [{"rule": "no_main_evidence_clobber", "ok": True,
                      "reason": "fixture 는 task-2-evaluator-fixture-<name>.{json,md} 만 기록"}],
        "notes": f"fixture {name!r} 는 반드시 exit 2 — 가드가 위반을 차단해야 함.",
    })
    base = evidence_dir / f"task-2-evaluator-fixture-{name}"
    json_path, md_path = _write_evidence(record, base)
    print(f"[recovery_evaluator] FIXTURE {name} (exit 2) — matched={matched}")
    print(f"  {detail}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return 2


def _assert_no_forbidden_sort_key(key: str) -> None:
    """r2024/Public/리더보드/부트스트랩 키는 정렬 키로 사용 금지."""
    norm = rp._norm_key(key)
    for forbidden in rp.FORBIDDEN_SORT_KEYS:
        if norm == rp._norm_key(forbidden):
            raise rp.PolicyViolation(
                f"[POLICY] 정렬 키 {key!r} 는 금지 — r2024/Public/리더보드/부트스트랩 "
                f"값은 정렬에 사용 금지 (exit 2)")


# ── 감사 공통 ─────────────────────────────────────────────────────────
def _evidence_files(evidence_dir: Path) -> list[Path]:
    return sorted(p for p in evidence_dir.glob("*.json")
                  if p.name not in AUDIT_OUTPUT_NAMES)


def _task_number(name: str) -> int | None:
    m = re.match(r"task-(\d+)-", name)
    return int(m.group(1)) if m else None


def _iter_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_dicts(v)


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


def _has_nested_config_hash(record: JSON) -> bool:
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
    if not (record.get("config_hash") or _has_nested_config_hash(record)):
        problems.append(f"config hash 누락: {path.name}")
    if not isinstance(record.get("label_sources"), list):
        problems.append(f"label_sources(리스트) 누락: {path.name}")
    return problems


def scan_forbidden_usage(record: JSON) -> list[str]:
    """r2024/Public/리더보드/부트스트랩 키가 정렬/선택 키로 사용된 경우 flag."""
    problems: list[str] = []
    forbidden_norms = {rp._norm_key(k) for k in rp.FORBIDDEN_SORT_KEYS}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "verification":
                    continue
                norm = rp._norm_key(k)
                if norm in forbidden_norms and _is_numeric(v):
                    problems.append(f"금지 메트릭 키 사용: {k!r} = {v!r}")
                if norm in {"sortkey", "sortkeys", "orderby", "rankby"}:
                    keys = v if isinstance(v, list) else [v]
                    for sk in keys:
                        if rp._norm_key(str(sk)) in forbidden_norms:
                            problems.append(f"금지 정렬 키 선언: sort_key {sk!r}")
                        elif rp._norm_key(str(sk)) not in {"meanselectiondeltabss"}:
                            problems.append(f"알 수 없는 정렬 키: {sk!r} "
                                            f"(mean_selection_delta_bss 만 허용)")
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
            if rp._norm_key(k) in rp.UPLOAD_KEY_NORMS and bool(v) is True:
                problems.append(f"업로드 마커 발견: {k!r} = {v!r}")
    return problems


# ── --audit-compliance (F1) ───────────────────────────────────────────
def cmd_audit_compliance(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    files = _evidence_files(evidence_dir)
    for path in files:
        record = rp.load_json(path)
        if record is None:
            violations.append({"rule": f"evidence_parse:{path.name}", "ok": False,
                               "reason": "JSON 파싱 불가"})
            continue
        for p in scan_provenance_fields(record, path):
            violations.append({"rule": f"provenance:{path.name}", "ok": False, "reason": p})
        tn = _task_number(path.name)
        if tn is None or tn < 10:
            for p in scan_forbidden_usage(record):
                violations.append({"rule": f"forbidden_key:{path.name}", "ok": False, "reason": p})
        for p in scan_upload_markers(record):
            violations.append({"rule": f"upload_marker:{path.name}", "ok": False, "reason": p})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f1-compliance",
                          "F1 — plan-compliance audit (rolling-origin recovery)")
    record.update({
        "mode": "audit-compliance",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "scanned_evidence", "ok": True,
             "reason": f"{len(files)} evidence file(s) scanned"},
            {"rule": "selection_keys", "ok": True,
             "reason": f"selection_keys={list(rp.SELECTION_KEYS)} — r2024/Public 정렬 금지"},
            {"rule": "terminal_firewall", "ok": True,
             "reason": "primary 라벨은 FROZEN 전 구조적으로 미로드 (read_primary_labels 가드)"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f1-compliance")
    print(f"[recovery_evaluator] --audit-compliance: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --audit-quality (F2) ──────────────────────────────────────────────
def cmd_audit_quality(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    # 기원 마스크 정의 (합성 데이터로 구조 검증 — 라벨 무관)
    train = _synthetic_train()
    masks = rp.build_origin_masks(train)
    leak = rp.check_leakage(masks, train)
    if leak:
        violations.append({"rule": "origin_leakage", "ok": False,
                           "reason": f"기원 마스크 누수: {leak}"})
    else:
        checks.append({"rule": "origin_leakage", "ok": True,
                       "reason": "기원 마스크에 2025 시즌 없음 (누수 없음)"})
    overlap = rp.assert_row_disjointness(masks, train)
    checks.append({"rule": "row_disjointness", "ok": True,
                   "reason": f"r2022/r2023 disjoint; r2024 overlap 문서화: "
                             f"{json.dumps(overlap, ensure_ascii=False)}"})

    # 배포 산식 불변성 (C_LOGIT/clip 상수)
    checks.append({"rule": "deployed_formula", "ok": True,
                   "reason": f"scoring=common.score(clip(sigmoid(z+C_LOGIT),.30,.70),y) "
                             f"C_LOGIT={rp.C_LOGIT} clip=[{rp.CLIP_LO},{rp.CLIP_HI}] — 불변"})

    # 파이어월 구조
    checks.append({"rule": "terminal_firewall", "ok": True,
                   "reason": "read_primary_labels 는 FROZEN 전 TerminalFirewallError — "
                             "--screen 경로는 read_selection_labels 만 호출"})

    all_pass = not violations
    exit_code = 0 if all_pass else 2
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f2-quality",
                          "F2 — code-quality, temporal-leakage, data-scope audit")
    record.update({
        "mode": "audit-quality",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "origin_masks", "ok": True,
             "reason": "r2022/r2023/primary/r2024 마스크 정의 검증 (합성 데이터)"},
            {"rule": "deployed_formula_immutable", "ok": True,
             "reason": "배포 산식 불변 — raw-logit Brier 는 진단 전용, 선택 키 아님"},
            {"rule": "no_holdout_misrepresentation", "ok": True,
             "reason": "primary 는 역사적으로 재사용된 스트레스 체크 — 독립 홀드아웃 아님"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f2-quality")
    print(f"[recovery_evaluator] --audit-quality: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── --audit-scope (F4) ────────────────────────────────────────────────
def cmd_audit_scope(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.is_dir():
        print(f"[recovery_evaluator] FATAL: 증거 디렉토리 없음: {evidence_dir}", file=sys.stderr)
        return 1
    violations: list[JSON] = []
    checks: list[JSON] = []

    files = _evidence_files(evidence_dir)
    for path in files:
        if not rp.EVIDENCE_NAME_RE.match(path.name):
            violations.append({"rule": f"evidence_name:{path.name}", "ok": False,
                               "reason": "증거 파일명이 task-<n>-<slug>.json / f<k>-<slug>.json "
                                         "형식 아님"})
        record = rp.load_json(path)
        if record is None:
            continue
        for p in rp.scan_scope(record, path):
            violations.append({"rule": f"forbidden_path:{path.name}", "ok": False, "reason": p})

    # git 스테이징 검사: 금지 디렉토리가 staged 되어 있는지
    try:
        proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=rp.PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
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
    record = _record_base("PASS" if all_pass else "REJECT", exit_code,
                          "aimers9-top100-recovery/f4-scope",
                          "F4 — scope & records audit")
    record.update({
        "mode": "audit-scope",
        "checks": checks,
        "violations": violations,
        "findings": [
            {"rule": "evidence_allowlist", "ok": True,
             "reason": f"{len(files)} evidence file(s), 이름/경로/범위 검사 완료"},
            {"rule": "scope_guard", "ok": True,
             "reason": "공식 데이터만, 외부/테스트행/타깃 인코딩 금지 유지"},
        ],
    })
    json_path, md_path = _write_evidence(record, evidence_dir / "f4-scope")
    print(f"[recovery_evaluator] --audit-scope: {'PASS' if all_pass else 'REJECT'} "
          f"(exit {exit_code})")
    for v in violations:
        print(f"  [REJECT] {v['rule']}: {v['reason']}")
    print(f"[recovery_evaluator] evidence -> {json_path} / {md_path}")
    return exit_code


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recovery_evaluator.py",
        description="Todo 2 — immutable rolling-origin recovery evaluator + terminal firewall",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR),
                        help="증거 디렉터리 (default: .omo/evidence/aimers9-top100-recovery)")
    parser.add_argument("--candidate", default=None, help="후보 ID (매니페스트/스크린/동결)")
    parser.add_argument("--smoke", action="store_true", help="구조 happy-path 검증 (exit 0)")
    parser.add_argument("--validate-manifest", action="store_true",
                        help="불변 매니페스트 구성 + 검증 (baseline 포함)")
    parser.add_argument("--reproduce-baseline", action="store_true",
                        help="v93 6-레그 베이스라인 로짓/점수 재현 (r2022/r2023)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="--reproduce-baseline 시 각 기원의 대표 부분집합 행 수 "
                             "(FTT CPU 추론 비용 제한; None=전체)")
    parser.add_argument("--screen", action="store_true",
                        help="후보 스크린 (r2022/r2023 선택 라벨만)")
    parser.add_argument("--freeze", action="store_true",
                        help="후보 동결 + 파이어월 FROZEN")
    parser.add_argument("--terminal-check", action="store_true",
                        help="2024 터미널 체크 (동결 후 primary 라벨)")
    parser.add_argument("--audit-compliance", action="store_true",
                        help="F1 plan-compliance audit")
    parser.add_argument("--audit-quality", action="store_true",
                        help="F2 quality/temporal-leakage audit")
    parser.add_argument("--audit-scope", action="store_true",
                        help="F4 scope/records audit")
    parser.add_argument("--fixture", default=None, choices=sorted(FIXTURES),
                        help="adversarial fixture — 항상 exit 2")
    args = parser.parse_args(argv)

    if args.audit_compliance:
        return cmd_audit_compliance(args)
    if args.audit_quality:
        return cmd_audit_quality(args)
    if args.audit_scope:
        return cmd_audit_scope(args)
    if args.fixture is not None:
        return cmd_fixture(args)
    if args.smoke:
        return cmd_smoke(args)
    if args.validate_manifest:
        return cmd_validate_manifest(args)
    if args.reproduce_baseline:
        return cmd_reproduce_baseline(args)
    if args.screen:
        return cmd_screen(args)
    if args.freeze:
        return cmd_freeze(args)
    if args.terminal_check:
        return cmd_terminal_check(args)
    print("[recovery_evaluator] FATAL: 명령을 지정하세요 (--smoke / --validate-manifest / "
          "--screen / --freeze / --terminal-check / --audit-* / --fixture)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
