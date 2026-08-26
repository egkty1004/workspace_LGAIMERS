#!/usr/bin/env python3
"""Matched v93-style CatBoost experiment for crosswalk-free TrackMan priors.

The module is a leaderboard-track experiment, not recovery policy.  Its static
command is data-free.  The future ``screen`` command is deliberately explicit:
it trains matched C0/C1 CatBoost arms, seals both outer logits, and only then
reads r2022/r2023 outer labels.  B0 is the hash-addressed original v93 bounded
composite; no primary/r2024/2024 target is available to this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro_979 import common  # noqa: E402
from scripts import audit_trackman_crosswalk_free_context_priors_v2 as tm_v2  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "v93_trackman_context_prior_catboost_v1.json"
CONTRACT_VERSION = "aimers9-v93-trackman-context-prior-catboost-v1"
ROW_ID = "row_id"
TARGET = "control_success"
DERIVED_BASE = ("platoon", "count_state")
ORIGINS = ("r2022", "r2023")


class ExperimentError(RuntimeError):
    """Fail-closed authority, parity, firewall, or execution error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_hash(values: Sequence[Any]) -> str:
    return canonical_hash([str(value) for value in values])


def float_array_hash(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ExperimentError("array hash requires a finite one-dimensional float64 array")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _git_sha(repo_root: Path) -> str:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             check=True, capture_output=True, text=True)
    return process.stdout.strip()


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"cannot load runtime config: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("contract_version") != CONTRACT_VERSION:
        raise ExperimentError("wrong experiment contract version")
    base = list(config.get("base_features") or [])
    tm = list(config.get("trackman_features") or [])
    cats = list(config.get("categorical_features") or [])
    if len(base) != 49 or len(set(base)) != 49:
        raise ExperimentError("base CatBoost contract must contain 49 unique features")
    if tuple(tm) != tuple(tm_v2.FEATURE_COLUMNS):
        raise ExperimentError("TrackMan feature order differs from reviewed v2 authority")
    if len(tm) != 7 or set(base).intersection(tm):
        raise ExperimentError("C1 must add exactly seven distinct TrackMan features")
    if "__backoff_level" in base + tm or config.get("forbidden_model_features") != ["__backoff_level"]:
        raise ExperimentError("audit-only backoff level entered the model contract")
    if cats != ["top_bottom", "game_type", "base_state", "platoon", "count_state"]:
        raise ExperimentError("categorical feature contract drift")
    if config["geometry"]["origins"] != list(ORIGINS):
        raise ExperimentError("only r2022/r2023 are allowed")
    if config["geometry"]["raw_cat_panel"] != "full_outer_validation":
        raise ExperimentError("raw CatBoost diagnostic must use the full outer panel")
    if config["geometry"]["deployed_composite_panel"] != "first_30000_true_positions_in_source_order":
        raise ExperimentError("B0/C0/C1 composite geometry drift")
    if int(config["geometry"]["bounded_rows"]) != 30_000:
        raise ExperimentError("bounded composite row budget drift")
    expected_split = (
        "numpy.random.RandomState(12345).choice(n_rows,size=int(n_rows*0.05),"
        "replace=False);sorted;complement_fit"
    )
    if config["catboost"]["es_split"]["algorithm"] != expected_split:
        raise ExperimentError("exp95 deterministic split algorithm drift")
    if int(config["catboost"]["es_split"]["seed"]) != 12345:
        raise ExperimentError("exp95 split seed drift")
    if config["catboost"]["seeds"] != list(range(42, 52)):
        raise ExperimentError("v93 seed contract drift")
    expected_composite = {
        "w_lgb": 0.65,
        "lambda_ftt": 0.17991944576662527,
        "lambda_armb": 0.43481381354434545,
        "lambda_cat": 0.0701066994221915,
    }
    if any(float(config["composite"][key]) != value
           for key, value in expected_composite.items()):
        raise ExperimentError("v93 blend weight drift")
    if float(config["composite"]["c_logit"]) != -0.0461645795229729:
        raise ExperimentError("v93 deployed offset drift")
    if config["composite"]["clip"] != [0.3, 0.7]:
        raise ExperimentError("v93 clipping drift")
    if config["recovery_promotion"] is not False:
        raise ExperimentError("experiment must remain outside recovery promotion")
    if config["package"]["inference_threads_max"] != 6:
        raise ExperimentError("package inference thread contract drift")


def _expected_model_hashes(repo_root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    evidence_path = repo_root / config["authority"]["v93_task3_evidence_path"]
    if sha256_file(evidence_path) != config["authority"]["v93_task3_evidence_sha256"]:
        raise ExperimentError("tracked v93 Task-3 evidence hash drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    model_hashes = evidence.get("baseline", {}).get("package_hashes", {}).get("model_files")
    if not isinstance(model_hashes, dict) or len(model_hashes) != 51:
        raise ExperimentError("v93 Task-3 model identity map is invalid")
    return {str(name): str(value) for name, value in model_hashes.items()}


def verify_static_authorities(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    authority = config["authority"]
    checks = {
        "task3_evidence_sha256": sha256_file(root / authority["v93_task3_evidence_path"]),
        "trackman_v2_runner_sha256": sha256_file(root / "scripts" / "audit_trackman_crosswalk_free_context_priors_v2.py"),
        "trackman_v2_config_sha256": sha256_file(root / "configs" / "trackman_crosswalk_free_context_priors_v2.json"),
    }
    expected = {
        "task3_evidence_sha256": authority["v93_task3_evidence_sha256"],
        "trackman_v2_runner_sha256": authority["trackman_v2_runner_sha256"],
        "trackman_v2_config_sha256": authority["trackman_v2_config_sha256"],
    }
    mismatches = [name for name in checks if checks[name] != expected[name]]
    _expected_model_hashes(root, config)
    if mismatches:
        raise ExperimentError(f"static authority drift: {mismatches}")
    return {"passed": True, "actual": checks, "expected": expected}


def _zip_member_hash(archive: zipfile.ZipFile, name: str) -> str:
    try:
        return sha256_bytes(archive.read(name))
    except KeyError as exc:
        raise ExperimentError(f"v93 archive member missing: {name}") from exc


def verify_external_authorities(
    repo_root: str | Path,
    config: Mapping[str, Any],
    v93_archive: str | Path,
    lookup_json: str | Path,
    baseline_cache_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    archive_path = Path(v93_archive).resolve()
    lookup_path = Path(lookup_json).resolve()
    cache_dir = Path(baseline_cache_dir).resolve()
    authority = config["authority"]
    if sha256_file(archive_path) != authority["v93_archive_sha256"]:
        raise ExperimentError("v93 source archive SHA-256 drift")
    if sha256_file(lookup_path) != authority["trackman_lookup_sha256"]:
        raise ExperimentError("reviewed TrackMan lookup SHA-256 drift")
    expected_models = _expected_model_hashes(root, config)
    with zipfile.ZipFile(archive_path) as archive:
        prep_hash = _zip_member_hash(archive, "model/catboost_prep.pkl")
        if prep_hash != authority["v93_catboost_prep_sha256"]:
            raise ExperimentError("v93 CatBoost prep SHA-256 drift")
        for name, expected_hash in expected_models.items():
            if _zip_member_hash(archive, f"model/{name}") != expected_hash:
                raise ExperimentError(f"v93 model asset hash drift: {name}")
        prep = pickle.loads(archive.read("model/catboost_prep.pkl"))
    if list(prep.get("features") or []) != list(config["base_features"]):
        raise ExperimentError("recovered CatBoost 49-feature order drift")
    if list(prep.get("cats") or []) != list(config["categorical_features"]):
        raise ExperimentError("recovered CatBoost categorical contract drift")
    recovered = dict(prep.get("params") or {})
    frozen = config["catboost"]
    for key in ("allow_writing_files", "depth", "iterations", "l2_leaf_reg",
                "learning_rate", "loss_function", "thread_count", "verbose"):
        if recovered.get(key) != frozen["params"].get(key):
            raise ExperimentError(f"recovered exp95 parameter drift: {key}")
    if recovered.get("early_stopping_rounds") != frozen["early_stopping_rounds"]:
        raise ExperimentError("recovered exp95 early stopping drift")
    if recovered.get("use_best_model") is not frozen["use_best_model"]:
        raise ExperimentError("recovered exp95 use_best_model drift")
    if prep.get("es_seed") != frozen["es_split"]["seed"] or prep.get("es_frac") != frozen["es_split"]["es_fraction"]:
        raise ExperimentError("recovered exp95 split metadata drift")
    meta_path = cache_dir / "meta.json"
    if sha256_file(meta_path) != authority["baseline_cache_meta_sha256"]:
        raise ExperimentError("B0 cache metadata drift")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("cache_key") != authority["baseline_cache_key"] or meta.get("bounded") != 30_000:
        raise ExperimentError("B0 cache contract drift")
    for origin in ORIGINS:
        path = cache_dir / f"{origin}.npy"
        if sha256_file(path) != authority["baseline_cache_logit_sha256"][origin]:
            raise ExperimentError(f"B0 cache logit hash drift: {origin}")
    return {
        "passed": True,
        "v93_archive_sha256": authority["v93_archive_sha256"],
        "catboost_prep_sha256": prep_hash,
        "lookup_sha256": authority["trackman_lookup_sha256"],
        "baseline_cache_key": authority["baseline_cache_key"],
        "recovered_feature_count": len(prep["features"]),
        "recovered_categorical_features": list(prep["cats"]),
        "recovered_params": recovered,
    }


def load_lookup(path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    lookup_path = Path(path)
    if sha256_file(lookup_path) != config["authority"]["trackman_lookup_sha256"]:
        raise ExperimentError("TrackMan lookup does not match reviewed v2 artifact")
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if lookup.get("contract_version") != tm_v2.CONTRACT_VERSION:
        raise ExperimentError("TrackMan lookup contract version drift")
    if not isinstance(lookup.get("seasons"), dict):
        raise ExperimentError("TrackMan lookup seasons missing")
    return lookup


def preprocess_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact v93 base-frame preprocessing; TrackMan columns are added afterward."""
    result = frame.copy()
    for column in result.select_dtypes("float64").columns:
        result[column] = result[column].astype("float32")
    for column in result.select_dtypes("int64").columns:
        result[column] = result[column].astype("int32")
    for column in ("top_bottom", "game_type", "base_state"):
        result[column] = result[column].astype("category")
    result["platoon"] = (result["pitcher_hand"] * 2 + result["batter_hand"]).astype("category")
    result["count_state"] = (result["balls_before"] * 3 + result["strikes_before"]).astype("category")
    return result


def apply_trackman_features(
    base_frame: pd.DataFrame,
    aggregate_lookup: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    required = ("season", "balls_before", "strikes_before", "outs_before")
    if any(column not in base_frame for column in required):
        raise ExperimentError("main context columns missing")
    records: list[dict[str, Any]] = []
    for row in base_frame.loc[:, list(required)].itertuples(index=False, name=None):
        mapping = dict(zip(required, row))
        season = tm_v2._season(mapping["season"])
        record = tm_v2.apply_context_prior_lookup_row(mapping, aggregate_lookup, season)
        records.append({column: record[column] for column in config["trackman_features"]})
    features = pd.DataFrame(records, index=base_frame.index,
                            columns=list(config["trackman_features"]))
    if "__backoff_level" in features:
        raise ExperimentError("audit-only backoff level leaked into feature frame")
    result = base_frame.copy()
    for column in config["trackman_features"]:
        result[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    expected = list(config["base_features"]) + list(config["trackman_features"])
    if list(result.loc[:, expected].columns) != expected:
        raise ExperimentError("C1 56-feature ordering failed")
    return result


def assert_feature_contracts(config: Mapping[str, Any]) -> dict[str, Any]:
    c0 = list(config["base_features"])
    c1 = c0 + list(config["trackman_features"])
    if c1 != c0 + list(tm_v2.FEATURE_COLUMNS) or len(c1) != 56:
        raise ExperimentError("C1 must equal C0 plus exactly seven reviewed features")
    if set(config["categorical_features"]).intersection(config["trackman_features"]):
        raise ExperimentError("TrackMan prior features must remain numeric")
    return {
        "C0_features": c0,
        "C1_features": c1,
        "categorical_features": list(config["categorical_features"]),
        "only_difference": list(config["trackman_features"]),
    }


def _raw_feature_columns(config: Mapping[str, Any]) -> list[str]:
    return [column for column in config["base_features"] if column not in DERIVED_BASE]


def read_feature_projection(train_csv: str | Path, config: Mapping[str, Any]) -> pd.DataFrame:
    projection = [ROW_ID, *_raw_feature_columns(config)]
    header = pd.read_csv(train_csv, nrows=0, encoding="utf-8-sig")
    missing = sorted(set(projection) - set(header.columns))
    if missing:
        raise ExperimentError(f"official feature projection missing columns: {missing}")
    if TARGET in projection:
        raise ExperimentError("target entered feature projection")
    frame = pd.read_csv(train_csv, usecols=projection, encoding="utf-8-sig")
    frame = frame.loc[:, projection]
    normalized = frame[ROW_ID].astype(str)
    if frame[ROW_ID].isna().any() or normalized.duplicated().any():
        raise ExperimentError("row_id must be present and unique")
    return preprocess_base(frame)


def origin_positions(frame: pd.DataFrame, origin: str) -> tuple[np.ndarray, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    season = pd.to_numeric(frame["season"], errors="coerce").to_numpy()
    if origin == "r2022":
        train = regular & (season <= 2021)
        validation = regular & (season == 2022)
    elif origin == "r2023":
        train = regular & (season <= 2022)
        validation = regular & (season == 2023)
    else:
        raise ExperimentError("only r2022/r2023 origins are permitted")
    train_positions = np.flatnonzero(train)
    validation_positions = np.flatnonzero(validation)
    if not len(train_positions) or not len(validation_positions):
        raise ExperimentError(f"empty origin geometry: {origin}")
    return train_positions, validation_positions


def bounded_positions(validation_positions: Sequence[int], budget: int = 30_000) -> np.ndarray:
    positions = np.asarray(validation_positions, dtype=np.int64)
    return positions[: min(int(budget), len(positions))].copy()


def split_positions(source_positions: Sequence[int], config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(source_positions, dtype=np.int64)
    split = config["catboost"]["es_split"]
    n_es = int(len(positions) * float(split["es_fraction"]))
    if n_es <= 0 or n_es >= len(positions):
        raise ExperimentError("invalid exp95 fit/ES row count")
    rng = np.random.RandomState(int(split["seed"]))
    es_relative = np.sort(rng.choice(len(positions), size=n_es, replace=False))
    fit_mask = np.ones(len(positions), dtype=bool)
    fit_mask[es_relative] = False
    fit = positions[np.flatnonzero(fit_mask)]
    es = positions[es_relative]
    if set(fit).intersection(es) or len(fit) + len(es) != len(positions):
        raise ExperimentError("fit/ES split parity failed")
    return fit, es


def _scoped_csv_rows(path: str | Path, positions: Sequence[int], usecols: Sequence[str]) -> pd.DataFrame:
    selected = frozenset(int(value) for value in positions)
    if not selected:
        raise ExperimentError("empty scoped CSV projection")
    return pd.read_csv(path, usecols=list(usecols), encoding="utf-8-sig",
                       skiprows=lambda line: line > 0 and (line - 1) not in selected)


def read_fit_labels(
    train_csv: str | Path,
    feature_frame: pd.DataFrame,
    positions: Sequence[int],
    origin: str,
) -> np.ndarray:
    positions_array = np.asarray(positions, dtype=np.int64)
    train_positions, _ = origin_positions(feature_frame, origin)
    if not set(positions_array).issubset(set(train_positions)):
        raise ExperimentError("fit labels requested outside the origin training scope")
    scoped = _scoped_csv_rows(train_csv, positions_array, (ROW_ID, "season", "game_type", TARGET))
    expected_ids = feature_frame.iloc[positions_array][ROW_ID].astype(str).tolist()
    if scoped[ROW_ID].astype(str).tolist() != expected_ids:
        raise ExperimentError("fit-label row_id alignment failure")
    if (pd.to_numeric(scoped["season"], errors="coerce") >= 2024).any():
        raise ExperimentError("2024 target access is forbidden")
    target = pd.to_numeric(scoped[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ExperimentError("fit target must be finite binary")
    return target.to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class SealedOriginLogits:
    origin: str
    validation_positions: tuple[int, ...]
    position_hash: str
    row_id_hash: str
    c0_logits: tuple[float, ...]
    c1_logits: tuple[float, ...]


def seal_outer_logits(
    origin: str,
    feature_frame: pd.DataFrame,
    validation_positions: Sequence[int],
    c0_logits: Sequence[float],
    c1_logits: Sequence[float],
) -> SealedOriginLogits:
    if origin not in ORIGINS:
        raise ExperimentError("sealed origin must be r2022/r2023")
    _, expected_positions = origin_positions(feature_frame, origin)
    positions = np.asarray(validation_positions, dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        raise ExperimentError("sealed positions differ from the exact full outer validation")
    c0 = np.asarray(c0_logits, dtype=np.float64)
    c1 = np.asarray(c1_logits, dtype=np.float64)
    if len(c0) != len(positions) or len(c1) != len(positions):
        raise ExperimentError("both-arm logits must match the outer positions")
    if not np.isfinite(c0).all() or not np.isfinite(c1).all():
        raise ExperimentError("nonfinite outer logits cannot be sealed")
    return SealedOriginLogits(
        origin=origin,
        validation_positions=tuple(int(value) for value in positions),
        position_hash=sequence_hash(positions.tolist()),
        row_id_hash=sequence_hash(feature_frame.iloc[positions][ROW_ID].astype(str).tolist()),
        c0_logits=tuple(float(value) for value in c0),
        c1_logits=tuple(float(value) for value in c1),
    )


def read_outer_labels(
    train_csv: str | Path,
    feature_frame: pd.DataFrame,
    sealed: SealedOriginLogits,
) -> np.ndarray:
    _, expected = origin_positions(feature_frame, sealed.origin)
    positions = np.asarray(sealed.validation_positions, dtype=np.int64)
    if not np.array_equal(positions, expected):
        raise ExperimentError("outer-label seal positions are stale")
    if sealed.position_hash != sequence_hash(positions.tolist()):
        raise ExperimentError("outer-label position hash drift")
    expected_ids = feature_frame.iloc[positions][ROW_ID].astype(str).tolist()
    if sealed.row_id_hash != sequence_hash(expected_ids):
        raise ExperimentError("outer-label row hash drift")
    scoped = _scoped_csv_rows(train_csv, positions, (ROW_ID, "season", "game_type", TARGET))
    if scoped[ROW_ID].astype(str).tolist() != expected_ids:
        raise ExperimentError("outer-label row_id alignment failure")
    expected_year = int(sealed.origin.removeprefix("r"))
    if not pd.to_numeric(scoped["season"], errors="coerce").eq(expected_year).all():
        raise ExperimentError("outer-label season scope mismatch")
    target = pd.to_numeric(scoped[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ExperimentError("outer target must be finite binary")
    return target.to_numpy(dtype=np.float64)


def catboost_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    params = dict(config["catboost"]["params"])
    params["random_seed"] = int(seed)
    return params


def candidate_prep_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen deployable C1 preprocessing/training metadata contract."""
    return {
        "features": list(config["base_features"]) + list(config["trackman_features"]),
        "cats": list(config["categorical_features"]),
        "seeds": list(config["catboost"]["seeds"]),
        "params": dict(config["catboost"]["params"]),
        "early_stopping_rounds": int(config["catboost"]["early_stopping_rounds"]),
        "use_best_model": bool(config["catboost"]["use_best_model"]),
        "es_seed": int(config["catboost"]["es_split"]["seed"]),
        "es_frac": float(config["catboost"]["es_split"]["es_fraction"]),
        "es_algorithm": str(config["catboost"]["es_split"]["algorithm"]),
        "no_post_es_refit": bool(config["catboost"]["es_split"]["no_post_es_refit"]),
        "lookup_sha256": str(config["authority"]["trackman_lookup_sha256"]),
        "protocol": CONTRACT_VERSION,
    }


def assert_matched_training_contract(
    config: Mapping[str, Any],
    source_positions: Sequence[int],
    target: Sequence[float],
    fit_positions: Sequence[int],
    es_positions: Sequence[int],
) -> dict[str, Any]:
    feature_contract = assert_feature_contracts(config)
    common_fields = {
        "source_position_hash": sequence_hash(source_positions),
        "target_hash": float_array_hash(target),
        "fit_position_hash": sequence_hash(fit_positions),
        "es_position_hash": sequence_hash(es_positions),
        "seeds": list(config["catboost"]["seeds"]),
        "parent_parameter_hash": canonical_hash(config["catboost"]),
        "base_feature_hash": canonical_hash(config["base_features"]),
        "categorical_feature_hash": canonical_hash(config["categorical_features"]),
        "base_preprocessing_hash": canonical_hash({
            "downcast": "float64->float32,int64->int32",
            "categories": list(config["categorical_features"]),
            "platoon": "pitcher_hand*2+batter_hand",
            "count_state": "balls_before*3+strikes_before",
        }),
    }
    return {
        "passed": True,
        "identical_C0_C1": common_fields,
        "allowed_difference": feature_contract["only_difference"],
    }


def train_catboost_ensemble(
    feature_frame: pd.DataFrame,
    train_positions: Sequence[int],
    validation_positions: Sequence[int],
    train_target: Sequence[float],
    config: Mapping[str, Any],
    arm: str,
    output_dir: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Future model path; never called by static/CHEAP verification."""
    try:
        from catboost import CatBoostClassifier, Pool, __version__ as catboost_version
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ExperimentError("CatBoost is unavailable") from exc
    if catboost_version != config["catboost"]["required_version"]:
        raise ExperimentError(f"CatBoost {config['catboost']['required_version']} required")
    if arm not in ("C0", "C1"):
        raise ExperimentError("only matched C0/C1 arms are trainable")
    features = list(config["base_features"])
    if arm == "C1":
        features += list(config["trackman_features"])
    categories = list(config["categorical_features"])
    train_positions = np.asarray(train_positions, dtype=np.int64)
    validation_positions = np.asarray(validation_positions, dtype=np.int64)
    target = np.asarray(train_target, dtype=np.float64)
    if len(target) != len(train_positions):
        raise ExperimentError("training target/row mismatch")
    fit_source, es_source = split_positions(train_positions, config)
    relative = {int(position): index for index, position in enumerate(train_positions)}
    fit_target = target[[relative[int(position)] for position in fit_source]]
    es_target = target[[relative[int(position)] for position in es_source]]
    cat_indices = [features.index(column) for column in categories]
    destination = Path(output_dir)
    if destination.exists():
        raise ExperimentError("model output directory exists; refusing overwrite")
    destination.mkdir(parents=True)
    logits: list[np.ndarray] = []
    per_seed: dict[str, Any] = {}
    for seed in config["catboost"]["seeds"]:
        model = CatBoostClassifier(**catboost_params(config, int(seed)))
        model.fit(
            Pool(feature_frame.iloc[fit_source][features], fit_target, cat_features=cat_indices),
            eval_set=Pool(feature_frame.iloc[es_source][features], es_target, cat_features=cat_indices),
            early_stopping_rounds=int(config["catboost"]["early_stopping_rounds"]),
            use_best_model=bool(config["catboost"]["use_best_model"]),
            verbose=False,
        )
        prediction = np.asarray(
            model.predict(feature_frame.iloc[validation_positions][features],
                          prediction_type="RawFormulaVal"), dtype=np.float64,
        ).ravel()
        if not np.isfinite(prediction).all():
            raise ExperimentError(f"nonfinite {arm} logits for seed {seed}")
        model_path = destination / f"catboost_s{seed}.cbm"
        model.save_model(str(model_path))
        logits.append(prediction)
        per_seed[str(seed)] = {
            "tree_count": int(model.tree_count_),
            "model_sha256": sha256_file(model_path),
        }
    aggregate = np.mean(np.stack(logits, axis=0), axis=0)
    return aggregate, {
        "arm": arm,
        "feature_count": len(features),
        "features": features,
        "categorical_features": categories,
        "per_seed": per_seed,
        "aggregate_logit_sha256": float_array_hash(aggregate),
        "no_post_es_refit": True,
    }


def original_v93_cat_logits(
    feature_frame: pd.DataFrame,
    validation_positions: Sequence[int],
    config: Mapping[str, Any],
    v93_dir: str | Path,
) -> np.ndarray:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover
        raise ExperimentError("CatBoost is unavailable") from exc
    model_dir = Path(v93_dir) / "model"
    features = list(config["base_features"])
    positions = np.asarray(validation_positions, dtype=np.int64)
    logits = []
    for seed in config["catboost"]["seeds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_dir / f"catboost_s{seed}.cbm"))
        prediction = np.asarray(
            model.predict(feature_frame.iloc[positions][features],
                          prediction_type="RawFormulaVal"), dtype=np.float64,
        ).ravel()
        logits.append(prediction)
    result = np.mean(np.stack(logits, axis=0), axis=0)
    if not np.isfinite(result).all():
        raise ExperimentError("original v93 CatBoost logits are nonfinite")
    return result


def load_b0_logits(origin: str, cache_dir: str | Path, config: Mapping[str, Any]) -> np.ndarray:
    if origin not in ORIGINS:
        raise ExperimentError("invalid B0 origin")
    path = Path(cache_dir) / f"{origin}.npy"
    if sha256_file(path) != config["authority"]["baseline_cache_logit_sha256"][origin]:
        raise ExperimentError("B0 logit cache hash drift")
    logits = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).ravel()
    if len(logits) != 30_000 or not np.isfinite(logits).all():
        raise ExperimentError("B0 cache row/finite contract failure")
    return logits


def control_reproduction_diagnostics(
    original_logits: Sequence[float],
    reconstructed_logits: Sequence[float],
    target: Sequence[float] | None = None,
) -> dict[str, Any]:
    original = np.asarray(original_logits, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_logits, dtype=np.float64)
    row_parity = bool(original.shape == reconstructed.shape and original.ndim == 1)
    finite = bool(row_parity and np.isfinite(original).all() and np.isfinite(reconstructed).all())
    if not finite:
        raise ExperimentError("control reproduction finite/row-parity failure")
    difference = reconstructed - original
    correlation = 1.0 if np.array_equal(original, reconstructed) else float(np.corrcoef(original, reconstructed)[0, 1])
    exact = bool(np.array_equal(original, reconstructed))
    original_probability = common.sigmoid(original)
    reconstructed_probability = common.sigmoid(reconstructed)
    result: dict[str, Any] = {
        "control_label": "exact v93 CatBoost control" if exact else "reconstructed v93-style matched control",
        "exact_prediction_reproduction": exact,
        "original_logit_sha256": float_array_hash(original),
        "reconstructed_c0_logit_sha256": float_array_hash(reconstructed),
        "original_raw_probability_sha256": float_array_hash(original_probability),
        "reconstructed_c0_raw_probability_sha256": float_array_hash(reconstructed_probability),
        "original_logit_mean": float(original.mean()),
        "reconstructed_c0_logit_mean": float(reconstructed.mean()),
        "original_raw_probability_mean": float(original_probability.mean()),
        "reconstructed_c0_raw_probability_mean": float(reconstructed_probability.mean()),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "correlation": correlation,
        "finite": finite,
        "row_parity": row_parity,
    }
    if target is not None:
        labels = np.asarray(target, dtype=np.float64)
        if len(labels) != len(original):
            raise ExperimentError("control reproduction target row mismatch")
        result["original_raw_brier"] = float(np.mean((original_probability - labels) ** 2))
        result["reconstructed_c0_raw_brier"] = float(np.mean((reconstructed_probability - labels) ** 2))
    return result


def deployed_probs(logits: Sequence[float], config: Mapping[str, Any]) -> np.ndarray:
    clip = config["composite"]["clip"]
    values = np.asarray(logits, dtype=np.float64)
    result = np.clip(common.sigmoid(values + float(config["composite"]["c_logit"])),
                     float(clip[0]), float(clip[1]))
    if not np.isfinite(result).all():
        raise ExperimentError("deployed probabilities are nonfinite")
    return result


def replace_catboost_leg(
    b0_logits: Sequence[float],
    new_cat_logits: Sequence[float],
    original_cat_logits: Sequence[float],
    config: Mapping[str, Any],
) -> np.ndarray:
    b0 = np.asarray(b0_logits, dtype=np.float64)
    new = np.asarray(new_cat_logits, dtype=np.float64)
    original = np.asarray(original_cat_logits, dtype=np.float64)
    if b0.shape != new.shape or b0.shape != original.shape:
        raise ExperimentError("CatBoost leg replacement row mismatch")
    result = b0 + float(config["composite"]["lambda_cat"]) * (new - original)
    if not np.isfinite(result).all():
        raise ExperimentError("replacement composite logits are nonfinite")
    return result


def _probability_metrics(probability: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all():
        raise ExperimentError("metric row/finite failure")
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "bss": float(common.score(p, y)),
        "mean": float(p.mean()),
    }


def evaluate_origin(
    origin: str,
    full_target: Sequence[float],
    b0_bounded_logits: Sequence[float],
    original_cat_full_logits: Sequence[float],
    c0_full_logits: Sequence[float],
    c1_full_logits: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if origin not in ORIGINS:
        raise ExperimentError("only r2022/r2023 may be scored")
    y = np.asarray(full_target, dtype=np.float64)
    original = np.asarray(original_cat_full_logits, dtype=np.float64)
    c0 = np.asarray(c0_full_logits, dtype=np.float64)
    c1 = np.asarray(c1_full_logits, dtype=np.float64)
    if not (y.shape == original.shape == c0.shape == c1.shape) or y.ndim != 1:
        raise ExperimentError("full-origin raw CatBoost row identity failure")
    if not all(np.isfinite(values).all() for values in (original, c0, c1, y)):
        raise ExperimentError("nonfinite full-origin values")
    raw_c0 = _probability_metrics(common.sigmoid(c0), y)
    raw_c1 = _probability_metrics(common.sigmoid(c1), y)
    n_bounded = len(np.asarray(b0_bounded_logits))
    if n_bounded != min(30_000, len(y)):
        raise ExperimentError("bounded B0 length differs from first-30k contract")
    yb = y[:n_bounded]
    original_b = original[:n_bounded]
    c0_b = c0[:n_bounded]
    c1_b = c1[:n_bounded]
    b0_z = np.asarray(b0_bounded_logits, dtype=np.float64)
    c0_z = replace_catboost_leg(b0_z, c0_b, original_b, config)
    c1_z = replace_catboost_leg(b0_z, c1_b, original_b, config)
    b0_metrics = _probability_metrics(deployed_probs(b0_z, config), yb)
    c0_metrics = _probability_metrics(deployed_probs(c0_z, config), yb)
    c1_metrics = _probability_metrics(deployed_probs(c1_z, config), yb)
    checks = {
        "raw_cat_c1_brier_lt_c0": bool(raw_c1["brier"] < raw_c0["brier"]),
        "composite_c1_brier_lt_c0": bool(c1_metrics["brier"] < c0_metrics["brier"]),
        "composite_c1_brier_lt_b0_original_v93": bool(c1_metrics["brier"] < b0_metrics["brier"]),
        "finite_outputs": True,
        "exact_row_identity": True,
    }
    return {
        "origin": origin,
        "raw_cat_panel": "full_outer_validation",
        "raw_cat": {"C0": raw_c0, "C1": raw_c1},
        "bounded_fixed_deployed_composite": {"B0_original_v93": b0_metrics, "C0": c0_metrics, "C1": c1_metrics},
        "deployed_mean_shift_C1_vs_B0_original_v93": c1_metrics["mean"] - b0_metrics["mean"],
        "deployed_abs_mean_shift_C1_vs_B0_original_v93": abs(c1_metrics["mean"] - b0_metrics["mean"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def final_screen_verdict(origin_results: Mapping[str, Mapping[str, Any]]) -> str:
    if set(origin_results) != set(ORIGINS):
        raise ExperimentError("screen requires exactly r2022 and r2023")
    return "PACKAGE_GO" if all(bool(origin_results[origin]["passed"]) for origin in ORIGINS) else "PACKAGE_STOP"


def _outside_repo(path: Path, repo_root: Path) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise ExperimentError("output path must be outside the repository")


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    return canonical_hash({key: value for key, value in report.items()
                           if key != "canonical_report_sha256"})


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    """Future MEDIUM/EXPENSIVE entry point; not invoked in this implementation turn."""
    config = load_config(args.config)
    root = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    _outside_repo(output, root)
    if output.exists():
        raise ExperimentError("output directory exists; refusing overwrite")
    verify_static_authorities(root, config)
    authority = verify_external_authorities(root, config, args.v93_archive,
                                             args.lookup_json, args.baseline_cache_dir)
    frame = read_feature_projection(args.train_csv, config)
    lookup = load_lookup(args.lookup_json, config)
    c1_frame = apply_trackman_features(frame, lookup, config)
    output.mkdir(parents=True)
    results: dict[str, Any] = {}
    training: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    label_ledger: list[dict[str, Any]] = []
    for origin in ORIGINS:
        train_positions, validation_positions = origin_positions(frame, origin)
        fit_positions, es_positions = split_positions(train_positions, config)
        train_target = read_fit_labels(args.train_csv, frame, train_positions, origin)
        label_ledger.append({"origin": origin, "role": "outer_train_for_fit_and_ES", "rows": len(train_target)})
        parity = assert_matched_training_contract(config, train_positions, train_target,
                                                  fit_positions, es_positions)
        c0_logits, c0_meta = train_catboost_ensemble(
            frame, train_positions, validation_positions, train_target, config,
            "C0", output / "models" / origin / "C0")
        c1_logits, c1_meta = train_catboost_ensemble(
            c1_frame, train_positions, validation_positions, train_target, config,
            "C1", output / "models" / origin / "C1")
        sealed = seal_outer_logits(origin, frame, validation_positions, c0_logits, c1_logits)
        original_cat = original_v93_cat_logits(frame, validation_positions, config, args.v93_dir)
        outer_target = read_outer_labels(args.train_csv, frame, sealed)
        label_ledger.append({"origin": origin, "role": "outer_validation_after_both_arms_sealed", "rows": len(outer_target)})
        b0 = load_b0_logits(origin, args.baseline_cache_dir, config)
        results[origin] = evaluate_origin(origin, outer_target, b0, original_cat,
                                          c0_logits, c1_logits, config)
        reproduction[origin] = control_reproduction_diagnostics(original_cat,
                                                                 c0_logits, outer_target)
        training[origin] = {"parity": parity, "C0": c0_meta, "C1": c1_meta,
                            "sealed": {key: value for key, value in asdict(sealed).items()
                                       if key not in ("c0_logits", "c1_logits")}}
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "protocol_role": config["protocol_role"],
        "recovery_promotion": False,
        "git_sha": _git_sha(root),
        "runner_sha256": sha256_file(Path(__file__)),
        "authority": authority,
        "arms": config["arms"],
        "feature_contract": assert_feature_contracts(config),
        "training": training,
        "control_reproduction": reproduction,
        "origin_results": results,
        "verdict": final_screen_verdict(results),
        "label_access_ledger": label_ledger,
        "scope": config["scope"],
        "notes": {
            "control_name_rule": "exact v93 CatBoost control only when prediction hashes reproduce exactly; otherwise reconstructed v93-style matched control",
            "no_origin_average_compensation": True,
            "no_raw_trackman_scan": True,
            "bounded_geometry_representativeness_limitation": True,
        },
    }
    report["canonical_report_sha256"] = canonical_report_hash(report)
    path = output / "v93_trackman_context_prior_catboost_v1_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return {"report": str(path), "verdict": report["verdict"],
            "canonical_report_sha256": report["canonical_report_sha256"]}


def static_contract(repo_root: str | Path, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    authority = verify_static_authorities(repo_root, config)
    features = assert_feature_contracts(config)
    return {
        "contract_version": CONTRACT_VERSION,
        "static_pass": True,
        "authority": authority,
        "feature_contract": features,
        "arms": config["arms"],
        "package_gate_origins": list(ORIGINS),
        "origin_average_compensation": False,
        "target_column_in_feature_projection": False,
        "target_2024_access": False,
        "primary_access": False,
        "r2024_access": False,
        "test_distribution_access": False,
        "public_leaderboard_access": False,
        "raw_trackman_scan_at_inference": False,
        "model_training_or_scoring": False,
        "gpu_used": False,
        "recovery_policy_modified": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static")
    static.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    static.add_argument("--config", type=Path, default=CONFIG_PATH)
    authority = sub.add_parser("verify-authority")
    authority.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    authority.add_argument("--config", type=Path, default=CONFIG_PATH)
    authority.add_argument("--v93-archive", type=Path, required=True)
    authority.add_argument("--lookup-json", type=Path, required=True)
    authority.add_argument("--baseline-cache-dir", type=Path, required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    screen.add_argument("--config", type=Path, default=CONFIG_PATH)
    screen.add_argument("--train-csv", type=Path, required=True)
    screen.add_argument("--lookup-json", type=Path, required=True)
    screen.add_argument("--v93-archive", type=Path, required=True)
    screen.add_argument("--v93-dir", type=Path, required=True)
    screen.add_argument("--baseline-cache-dir", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "static":
            result = static_contract(args.repo_root, args.config)
        elif args.command == "verify-authority":
            config = load_config(args.config)
            result = verify_external_authorities(args.repo_root, config, args.v93_archive,
                                                 args.lookup_json, args.baseline_cache_dir)
        else:
            result = run_screen(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ExperimentError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
