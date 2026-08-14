#!/usr/bin/env python3
"""
[Todo 8] champ_cat 제출 패키지 provenance.json 생성기 (2026-08-14)

provenance.json: 후보 ID / 블렌드 가중치 / 구성원 / model/* sha256 /
챔피언 롤백 참조(GIHO extract 경로+해시) / 구성 다이제스트.

사용법:
  python3 repro_979/gen_package_provenance.py --submission-dir repro_979/submit_champ_cat_<ts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
GIHO = PROJECT_ROOT / "team_member_materials" / "GIHO" / "submit979_extract"

CANDIDATE_ID = "691a2947b2b1879c"   # Todo 7 champ_cat (챔피언×0.85 + catboost×0.15)
W_LGB = 0.51
C_LOGIT = -0.0404
W_CAT = 0.15
W_CHAMP = 1.0 - W_CAT
SEEDS = list(range(42, 52))
CLIP = (0.30, 0.70)
CAT_FEATURES = ["top_bottom", "game_type", "base_state", "platoon", "count_state"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(d: Path, skip: tuple[str, ...] = ("catboost_full_train_manifest.json",)) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(d.rglob("*")):
        if not path.is_file():
            continue
        if any(s in path.parts for s in skip):
            continue
        out[path.relative_to(d).as_posix()] = _sha256(path)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True)
    args = parser.parse_args(argv)
    submit_dir = Path(args.submission_dir)
    model_dir = submit_dir / "model"
    if not model_dir.is_dir():
        print(f"[FAIL] model dir 없음: {model_dir}", file=sys.stderr)
        return 1

    config = {
        "candidate": "champ_cat",
        "candidate_id": CANDIDATE_ID,
        "weights": {"champion": W_CHAMP, "catboost": W_CAT},
        "champion_members": {"lgb": W_LGB, "mlp": round(1 - W_LGB, 4)},
        "c_logit": C_LOGIT,
        "clip": list(CLIP),
        "seeds": list(SEEDS),
        "cat_features": CAT_FEATURES,
        "catboost_iterations": 300,
        "model_dir_rel": "model",
    }
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    provenance = {
        "schema_version": 1,
        "task": "aimers9-top100/task-8-package",
        "candidate_id": CANDIDATE_ID,
        "candidate": "champ_cat",
        "blend": {
            "space": "logit",
            "weights": {"champion": W_CHAMP, "catboost": W_CAT},
            "formula": "p = clip(sigmoid(0.85*(0.51*z_lgb + 0.49*z_mlp) + 0.15*z_catboost + C_LOGIT), 0.30, 0.70)",
            "champion_members": {"lgb_weight": W_LGB, "mlp_weight": round(1 - W_LGB, 4)},
            "c_logit": C_LOGIT,
            "clip": list(CLIP),
            "seeds": list(SEEDS),
        },
        "members": {
            "champion": {
                "desc": "챔피언 컨트롤 (Task 2 동결, GIHO extract 모델 byte-copy)",
                "lgb": {"files": [f"f3_s{s}.txt" for s in SEEDS], "source": str(GIHO / "model")},
                "mlp": {"files": [f"mlp_s{s}.pt" for s in SEEDS] + ["mlp_prep.pkl", "mlp_meta.json"],
                        "source": str(GIHO / "model")},
            },
            "catboost": {
                "desc": "CatBoost full-data 10시드 (Task 8 재학습, 시즌 2019-2024, 피처 49/cats 5, depth 7, lr 0.05, 300 iter)",
                "files": [f"catboost_s{s}.cbm" for s in SEEDS],
                "source": "repro_979/deploy_train_catboost_full.py",
            },
        },
        "model_file_sha256": file_hashes(model_dir),
        "champion_rollback": {
            "path": str(GIHO),
            "note": "챔피언 V4 패키지 (byte-frozen, 979.31 실측). catboost 미설치 시 즉시 롤백.",
            "requirements": "lightgbm==4.7.0",
            "file_sha256": file_hashes(GIHO, skip=("__pycache__",)),
        },
        "config_digest": config_digest,
        "config": config,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = submit_dir / "provenance.json"
    out_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] {out_path} (model hashes={len(provenance['model_file_sha256'])}, "
          f"rollback hashes={len(provenance['champion_rollback']['file_sha256'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
