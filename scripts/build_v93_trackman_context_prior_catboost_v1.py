#!/usr/bin/env python3
"""Deterministic future package builder for the v93 TrackMan CatBoost leg.

The builder is inert until explicitly invoked with trained C1 artifacts.  It
starts from the pinned original v93 archive, verifies every tracked asset,
preserves LGB/MLP/FTT/ArmB bytes, replaces only CatBoost models/prep, embeds the
reviewed aggregate lookup, and patches the original inference script so only
the CatBoost leg sees the 56-feature frame.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import experiment_v93_trackman_context_prior_catboost_v1 as experiment


class PackageError(RuntimeError):
    """Fail-closed package authority, parity, or build error."""


CATBOOST_MODEL_NAMES = tuple(f"catboost_s{seed}.cbm" for seed in range(42, 52))
REPLACEABLE_MODEL_NAMES = frozenset((*CATBOOST_MODEL_NAMES, "catboost_prep.pkl"))
LOOKUP_NAME = "trackman_crosswalk_free_context_priors_v2_lookup.json"
PROVENANCE_NAME = "v93_trackman_context_prior_catboost_v1.json"
ZIP_TIMESTAMP = (2025, 1, 1, 0, 0, 0)


def _read_task3(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = repo_root / config["authority"]["v93_task3_evidence_path"]
    if experiment.sha256_file(path) != config["authority"]["v93_task3_evidence_sha256"]:
        raise PackageError("Task-3 v93 evidence drift")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_original_hashes(repo_root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    evidence = _read_task3(repo_root, config)
    package_hashes = evidence.get("baseline", {}).get("package_hashes", {})
    source = package_hashes.get("source_files")
    models = package_hashes.get("model_files")
    if not isinstance(source, dict) or not isinstance(models, dict):
        raise PackageError("Task-3 package identity map missing")
    if len(models) != 51:
        raise PackageError("Task-3 v93 model count drift")
    return {"source_files": dict(source), "model_files": dict(models)}


def fixed_asset_hashes(root: Path, expected: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected_hash in expected["source_files"].items():
        if name == "script.py":
            continue
        path = root / name
        actual = experiment.sha256_file(path)
        if actual != expected_hash:
            raise PackageError(f"fixed source asset drift: {name}")
        hashes[name] = actual
    for name, expected_hash in expected["model_files"].items():
        if name in REPLACEABLE_MODEL_NAMES:
            continue
        path = root / "model" / name
        actual = experiment.sha256_file(path)
        if actual != expected_hash:
            raise PackageError(f"fixed model/prep asset drift: {name}")
        hashes[f"model/{name}"] = actual
    return dict(sorted(hashes.items()))


def validate_candidate_artifacts(
    candidate_model_dir: Path,
    candidate_prep: Path,
    lookup_json: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if experiment.sha256_file(lookup_json) != config["authority"]["trackman_lookup_sha256"]:
        raise PackageError("candidate lookup differs from reviewed v2 artifact")
    expected_files = set(CATBOOST_MODEL_NAMES)
    actual_models = {path.name for path in candidate_model_dir.iterdir() if path.is_file()}
    if actual_models != expected_files:
        raise PackageError(f"candidate model set drift: {sorted(actual_models ^ expected_files)}")
    try:
        prep = pickle.loads(candidate_prep.read_bytes())
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise PackageError(f"candidate CatBoost prep cannot be read: {exc}") from exc
    expected_prep = experiment.candidate_prep_contract(config)
    expected_features = expected_prep["features"]
    mismatches = [key for key, expected in expected_prep.items() if prep.get(key) != expected]
    if mismatches:
        raise PackageError(f"candidate prep contract drift: {mismatches}")
    return {
        "model_sha256": {name: experiment.sha256_file(candidate_model_dir / name)
                          for name in sorted(expected_files)},
        "prep_sha256": experiment.sha256_file(candidate_prep),
        "lookup_sha256": experiment.sha256_file(lookup_json),
        "feature_count": len(expected_features),
    }


INFERENCE_HELPERS = r'''
TM_LOOKUP_PATH = os.path.join(MODEL_DIR, "trackman_crosswalk_free_context_priors_v2_lookup.json")
TM_LOOKUP_SHA256 = "27747c22e0ff25e86040f5825667f8d9c0b8d7e840037ad9df87072781160515"
TM_FEATURES = [
    "tm_cf_p_fastball", "tm_cf_p_breaking", "tm_cf_p_offspeed", "tm_cf_p_other",
    "tm_cf_entropy_norm4", "tm_cf_support", "tm_cf_context_vs_global_tv",
]
TM_SUPPORT_THRESHOLD = 100


def _tm_integer(value, low, high):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("invalid main context")
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("invalid main context")
    integer = int(numeric)
    if integer < low or integer > high:
        raise ValueError("invalid main context")
    return integer


def _load_tm_lookup():
    with open(TM_LOOKUP_PATH, "rb") as handle:
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != TM_LOOKUP_SHA256:
        raise RuntimeError("TrackMan aggregate lookup SHA-256 drift")
    lookup = json.loads(payload.decode("utf-8"))
    if lookup.get("contract_version") != "aimers9-trackman-crosswalk-free-context-priors-v2":
        raise RuntimeError("TrackMan aggregate lookup contract drift")
    return lookup


def _tm_payload_record(payload):
    probabilities = payload.get("probabilities")
    if probabilities is None:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan]
    values = [float(value) for value in probabilities]
    if len(values) != 4 or not all(np.isfinite(values)) or abs(sum(values) - 1.0) > 1e-9:
        raise RuntimeError("invalid TrackMan prior simplex")
    support = int(payload.get("support", 0))
    entropy = float(payload["entropy"])
    tv = float(payload["context_vs_global_tv"])
    if support <= 0 or not np.isfinite(entropy) or not np.isfinite(tv):
        raise RuntimeError("invalid TrackMan prior summary")
    return [*values, entropy, float(support), tv]


def _tm_row_features(row, lookup):
    season = _tm_integer(row["season"], 2019, 2025)
    balls = _tm_integer(row["balls_before"], 0, 3)
    strikes = _tm_integer(row["strikes_before"], 0, 2)
    outs = _tm_integer(row["outs_before"], 0, 2)
    season_lookup = lookup.get("seasons", {}).get(str(season))
    if not isinstance(season_lookup, dict):
        raise RuntimeError("missing frozen TrackMan lookup season")
    l0 = season_lookup.get("L0", {}).get(f"{balls}:{strikes}:{outs}")
    l1 = season_lookup.get("L1", {}).get(f"{balls}:{strikes}")
    global_payload = season_lookup.get("global", {}).get("GLOBAL")
    if not isinstance(global_payload, dict) or int(global_payload.get("support", 0)) == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan]
    if isinstance(l0, dict) and int(l0.get("support", 0)) >= TM_SUPPORT_THRESHOLD:
        return _tm_payload_record(l0)
    if isinstance(l1, dict) and int(l1.get("support", 0)) >= TM_SUPPORT_THRESHOLD:
        return _tm_payload_record(l1)
    return _tm_payload_record(global_payload)


def add_tm_context_prior_features(df, lookup):
    output = df.copy()
    values = [_tm_row_features(row, lookup) for row in df[["season", "balls_before", "strikes_before", "outs_before"]].to_dict("records")]
    matrix = np.asarray(values, dtype=np.float64)
    for index, column in enumerate(TM_FEATURES):
        output[column] = matrix[:, index].astype(np.float32)
    return output
'''


def render_candidate_script(original: str) -> str:
    """Patch only imports, CatBoost feature construction, and lookup helpers."""
    if original.count("CAT_THREADS = 6\n") != 1:
        raise PackageError("original v93 inference thread contract drift")
    if original.count("import os\nimport pickle\nimport time\n") != 1:
        raise PackageError("original v93 import anchor drift")
    script = original.replace(
        "import os\nimport pickle\nimport time\n",
        "import hashlib\nimport json\nimport os\nimport pickle\nimport time\n",
        1,
    )
    marker = "\n\ndef main():\n"
    if script.count(marker) != 1:
        raise PackageError("original v93 main anchor drift")
    script = script.replace(marker, "\n" + INFERENCE_HELPERS + marker, 1)
    old = '''    feats = common.get_feature_cols(test.columns)

    common.preprocess_for_submission(test)
    X = test[feats].copy()
    print(f"features: {len(feats)}", flush=True)
'''
    new = '''    base_feats = common.get_feature_cols(test.columns)

    common.preprocess_for_submission(test)
    X = test[base_feats].copy()
    tm_lookup = _load_tm_lookup()
    cat_frame = add_tm_context_prior_features(test, tm_lookup)
    cat_feats = list(base_feats) + list(TM_FEATURES)
    if len(base_feats) != 49 or len(cat_feats) != 56:
        raise RuntimeError("CatBoost feature contract drift")
    print(f"fixed-leg features: {len(base_feats)}; CatBoost features: {len(cat_feats)}", flush=True)
'''
    if script.count(old) != 1:
        raise PackageError("original v93 feature-frame anchor drift")
    script = script.replace(old, new, 1)
    old_call = "    z_cat = predict_cat_z(test, feats, cats, MODEL_DIR, SEEDS)\n"
    new_call = "    z_cat = predict_cat_z(cat_frame, cat_feats, cats, MODEL_DIR, SEEDS)\n"
    if script.count(old_call) != 1:
        raise PackageError("original v93 CatBoost call anchor drift")
    script = script.replace(old_call, new_call, 1)
    forbidden = ("trackman_history.csv", "pitcher_trackman_id", "test.groupby(", "test.merge(")
    if any(token in script for token in forbidden):
        raise PackageError("rendered inference script violates crosswalk/test-independence scope")
    if "z_cat = predict_cat_z(cat_frame, cat_feats" not in script or "z_lgb += common.logit(bst.predict(X))" not in script:
        raise PackageError("CatBoost-only feature-frame separation not proven")
    return script


def deterministic_zip(source_root: Path, output_zip: Path) -> None:
    if output_zip.exists():
        raise PackageError("output ZIP exists; refusing overwrite")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*"), key=lambda value: value.as_posix()):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def build_package(
    repo_root: str | Path,
    config_path: str | Path,
    v93_archive: str | Path,
    candidate_model_dir: str | Path,
    candidate_prep: str | Path,
    lookup_json: str | Path,
    output_zip: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = experiment.load_config(config_path)
    archive_path = Path(v93_archive).resolve()
    model_dir = Path(candidate_model_dir).resolve()
    prep_path = Path(candidate_prep).resolve()
    lookup_path = Path(lookup_json).resolve()
    output = Path(output_zip).resolve()
    experiment._outside_repo(output, root)
    if output.exists():
        raise PackageError("output ZIP exists; refusing overwrite")
    if experiment.sha256_file(archive_path) != config["authority"]["v93_archive_sha256"]:
        raise PackageError("source v93 archive SHA-256 drift")
    expected = expected_original_hashes(root, config)
    candidate = validate_candidate_artifacts(model_dir, prep_path, lookup_path, config)
    with tempfile.TemporaryDirectory(prefix="aimers9-v93-tm-cat-") as directory:
        staging = Path(directory) / "package"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging)
        before = fixed_asset_hashes(staging, expected)
        for name in CATBOOST_MODEL_NAMES:
            shutil.copyfile(model_dir / name, staging / "model" / name)
        shutil.copyfile(prep_path, staging / "model" / "catboost_prep.pkl")
        shutil.copyfile(lookup_path, staging / "model" / LOOKUP_NAME)
        original_script = (staging / "script.py").read_text(encoding="utf-8")
        (staging / "script.py").write_text(render_candidate_script(original_script), encoding="utf-8")
        provenance = {
            "contract_version": experiment.CONTRACT_VERSION,
            "parent_archive_sha256": config["authority"]["v93_archive_sha256"],
            "lookup_sha256": candidate["lookup_sha256"],
            "candidate_models": candidate["model_sha256"],
            "candidate_prep_sha256": candidate["prep_sha256"],
            "fixed_asset_hashes": before,
            "catboost_feature_count": 56,
            "fixed_leg_feature_count": 49,
            "raw_trackman_scan": False,
            "test_distribution_statistics": False,
            "inference_threads_max": 6,
        }
        (staging / "model" / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        after = fixed_asset_hashes(staging, expected)
        if before != after:
            raise PackageError("fixed LGB/MLP/FTT/ArmB asset bytes changed")
        deterministic_zip(staging, output)
    return {
        "zip_path": str(output),
        "zip_sha256": experiment.sha256_file(output),
        "zip_size": output.stat().st_size,
        "fixed_assets_preserved": True,
        "fixed_asset_hash": experiment.canonical_hash(after),
        "candidate": candidate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=experiment.PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=experiment.CONFIG_PATH)
    parser.add_argument("--v93-archive", type=Path, required=True)
    parser.add_argument("--candidate-model-dir", type=Path, required=True)
    parser.add_argument("--candidate-prep", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_package(args.repo_root, args.config, args.v93_archive,
                               args.candidate_model_dir, args.candidate_prep,
                               args.lookup_json, args.output_zip)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (PackageError, experiment.ExperimentError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
