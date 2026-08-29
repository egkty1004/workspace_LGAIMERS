#!/usr/bin/env python3
"""Train and package exact Feature-Lab C3 plus 17 strict-past TrackMan physics features."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "c3_trackman_physics_dr_v1.json"
TARGET = "control_success"
ROW_ID = "row_id"
LOOKUP_NAME = "trackman_physics_dr_v1_lookup.json"
FIXED_TIME = (2020, 1, 1, 0, 0, 0)
BASE_TOKENS = frozenset(("___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"))
DERIVED = frozenset(("platoon", "count_state", "base_out_state_24"))
CONTEXT = ("balls_before", "strikes_before", "outs_before")
FAMILIES = ("fastball", "breaking", "offspeed")
PHYSICS_SOURCE = (
    "rel_speed", "spin_rate", "induced_vert_break", "extension", "rel_height", "horz_break"
)
TRACKMAN_PROJECTION = ("season", *CONTEXT, "pitch_type_group", *PHYSICS_SOURCE)
PACKAGE_MEMBERS = (
    "model/catboost.cbm", "model/feature_contract.json", f"model/{LOOKUP_NAME}",
    "script.py", "requirements.txt",
)


class ContractError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = config["base_c3_features"]
    physics = config["physics_features"]
    expected_cats = ["top_bottom", "game_type", "base_state", "platoon", "count_state", "base_out_state_24"]
    if len(base) != 50 or len(set(base)) != 50 or base[-1] != "base_out_state_24":
        raise ContractError("C3 50-feature contract drift")
    if len(physics) != 17 or len(set(physics)) != 17 or any(not name.startswith("tm_phys_") for name in physics):
        raise ContractError("physics 17-feature contract drift")
    if config["categorical_features"] != expected_cats:
        raise ContractError("categorical contract drift")
    if len(base + physics) != 67 or any(name.startswith("tm_cf_") for name in physics):
        raise ContractError("final 67-feature contract drift")
    if config["trackman"]["families"] != list(FAMILIES) or config["trackman"]["support_threshold"] != 100:
        raise ContractError("TrackMan family/support contract drift")
    return config


def _outside_repo(path: Path, repo_root: Path) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise ContractError("output must be outside repository")


def verify_authorities(eda_json: Path, trackman_csv: Path, c3_contract: Path) -> dict[str, Any]:
    config = load_config()
    contract_path = c3_contract.resolve()
    root = contract_path.parents[2]
    paths = {
        "eda": eda_json.resolve(),
        "trackman_source": trackman_csv.resolve(),
        "c3_report": root / "feature_lab_smoke_report.json",
        "c3_model": contract_path.parent / "catboost.cbm",
        "c3_contract": contract_path,
        "c3_parent_zip": root / "final-packages" / "catboost-feature-lab-b1-C3-base-out-submit.zip",
    }
    expected = config["authorities"]
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"authority missing/not regular: {name}")
        if sha256_file(path) != expected[f"{name}_sha256"]:
            raise ContractError(f"authority SHA-256 drift: {name}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("features") != config["base_c3_features"]:
        raise ContractError("C3 feature order differs from authority")
    if contract.get("categorical_features") != config["categorical_features"]:
        raise ContractError("C3 categorical order differs from authority")
    if contract.get("prediction") != "native CatBoost predict_proba[:,1]":
        raise ContractError("C3 prediction contract drift")
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(np.isfinite(values))


def _integer(value: Any, low: int, high: int) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    integer = int(numeric)
    return integer if low <= integer <= high else None


def _context_key(row: Mapping[str, Any]) -> tuple[int, int, int] | None:
    balls = _integer(row.get("balls_before"), 0, 3)
    strikes = _integer(row.get("strikes_before"), 0, 2)
    outs = _integer(row.get("outs_before"), 0, 2)
    if balls is None or strikes is None or outs is None:
        return None
    return balls, strikes, outs


def _value_key(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "<MISSING>" if pd.isna(value) else str(value)
    if not math.isfinite(number):
        return str(number)
    return str(int(number)) if number.is_integer() else format(number, ".17g")


def read_filter_trackman(path: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, usecols=list(TRACKMAN_PROJECTION), encoding="utf-8-sig")
    frame = frame.loc[:, list(TRACKMAN_PROJECTION)]
    expected = config["trackman"]
    if len(frame) != expected["raw_rows"]:
        raise ContractError("TrackMan raw row-count drift")
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if seasons.isna().any() or not seasons.eq(np.floor(seasons)).all() or not seasons.between(2019, 2024).all():
        raise ContractError("TrackMan season contract drift")
    taxonomy = frame["pitch_type_group"].astype("string").str.strip().str.lower()
    if taxonomy.isna().any() or not taxonomy.isin((*FAMILIES, "other")).all():
        raise ContractError("TrackMan taxonomy drift")
    frame["pitch_type_group"] = taxonomy
    valid = frame.apply(lambda row: _context_key(row) is not None, axis=1)
    counts: Counter[tuple[int, str, str, str]] = Counter()
    bounds = {"balls_before": (0, 3), "strikes_before": (0, 2), "outs_before": (0, 2)}
    for position in np.flatnonzero(~valid.to_numpy()):
        row = frame.iloc[int(position)]
        season = int(row["season"])
        for field, (low, high) in bounds.items():
            if _integer(row[field], low, high) is None:
                counts[(season, field, "out_of_domain", _value_key(row[field]))] += 1
    manifest = [
        {"season": season, "field": field, "reason": reason, "value": value, "count": int(count)}
        for (season, field, reason, value), count in sorted(counts.items())
    ]
    summary = {
        "raw_rows": int(len(frame)), "excluded_rows": int((~valid).sum()),
        "retained_rows": int(valid.sum()), "residue_manifest": manifest,
        "normalization": False, "rounding": False, "clipping": False,
    }
    if summary["excluded_rows"] != expected["excluded_rows"] or summary["retained_rows"] != expected["retained_rows"]:
        raise ContractError("TrackMan residue count drift")
    if manifest != expected["residue_manifest"]:
        raise ContractError("TrackMan residue manifest drift")
    filtered = frame.loc[valid & frame["pitch_type_group"].isin(FAMILIES)].copy().reset_index(drop=True)
    filtered.attrs["source_quality_filter_applied"] = True
    return filtered, summary


def _summarize(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    array.sort()
    n = int(array.size)
    if not n:
        return {"n": 0, "mean": None, "std": None, "var": None, "median": None, "iqr": None}
    mean = float(np.mean(array))
    variance = float(np.var(array, ddof=0))
    if variance < 0 and variance >= -1e-12 * max(1.0, mean * mean):
        variance = 0.0
    if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0:
        raise ContractError("material invalid aggregation variance")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {"n": n, "mean": mean, "std": math.sqrt(variance), "var": variance,
            "median": float(median), "iqr": float(q75 - q25)}


def _stat_table(history: pd.DataFrame, value_column: str, family: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    subset = history if family is None else history.loc[history["pitch_type_group"] == family]
    values = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.loc[np.isfinite(values)].copy()
    subset["__value"] = values.loc[subset.index].astype(np.float64)
    output: dict[str, dict[str, dict[str, Any]]] = {"L0": {}, "L1": {}, "global": {}}
    output["global"]["GLOBAL"] = _summarize(subset["__value"].to_numpy())
    for level, columns in (("L0", list(CONTEXT)), ("L1", list(CONTEXT[:2]))):
        for key, group in subset.groupby(columns, sort=True, observed=True):
            parts = key if isinstance(key, tuple) else (key,)
            rendered = ":".join(str(int(value)) for value in parts)
            output[level][rendered] = _summarize(group["__value"].to_numpy())
    return output


def _stats_for_history(history: pd.DataFrame) -> dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]]:
    working = history.loc[history["pitch_type_group"].isin(FAMILIES)].copy()
    working["abs_horz_break"] = np.abs(pd.to_numeric(working["horz_break"], errors="coerce"))
    variables = ("rel_speed", "spin_rate", "induced_vert_break", "extension", "rel_height", "abs_horz_break")
    result: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for variable in variables:
        result[(variable, "ALL")] = _stat_table(working, variable, None)
        for family in FAMILIES:
            result[(variable, family)] = _stat_table(working, variable, family)
    return result


def _level_key(level: str, context: tuple[int, int, int]) -> str:
    if level == "L0":
        return ":".join(map(str, context))
    if level == "L1":
        return f"{context[0]}:{context[1]}"
    return "GLOBAL"


def _select(table: Mapping[str, Mapping[str, Mapping[str, Any]]], context: tuple[int, int, int],
            threshold: int) -> tuple[Mapping[str, Any] | None, str]:
    for level in ("L0", "L1", "global"):
        stat = table[level].get(_level_key(level, context))
        if stat is not None and int(stat["n"]) >= threshold:
            return stat, level
    return None, "NO_HISTORY"


def _global(table: Mapping[str, Mapping[str, Mapping[str, Any]]], threshold: int) -> Mapping[str, Any] | None:
    stat = table["global"].get("GLOBAL")
    return stat if stat is not None and int(stat["n"]) >= threshold else None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _feature_vector(stats: Mapping[tuple[str, str], Mapping[str, Any]], context: tuple[int, int, int],
                    threshold: int) -> tuple[list[float | None], list[str]]:
    values: list[float | None] = []
    levels: list[str] = []

    rel_table = stats[("rel_speed", "ALL")]
    rel, rel_level = _select(rel_table, context, threshold)
    rel_global = _global(rel_table, threshold)
    values.extend([None if rel is None else float(rel["mean"]), None if rel is None else float(rel["std"])])
    levels.extend([rel_level, rel_level])
    values.append(None if rel is None or rel_global is None else _safe_ratio(
        float(rel["mean"]) - float(rel_global["mean"]), float(rel_global["std"])))
    levels.append(rel_level)

    for family_b in ("breaking", "offspeed"):
        table_a = stats[("rel_speed", "fastball")]
        table_b = stats[("rel_speed", family_b)]
        chosen_a = chosen_b = None
        chosen_level = "NO_HISTORY"
        for level in ("L0", "L1", "global"):
            a = table_a[level].get(_level_key(level, context))
            b = table_b[level].get(_level_key(level, context))
            if a is not None and b is not None and int(a["n"]) >= threshold and int(b["n"]) >= threshold:
                chosen_a, chosen_b, chosen_level = a, b, level
                break
        ga, gb = _global(table_a, threshold), _global(table_b, threshold)
        denominator = None if ga is None or gb is None else math.sqrt((float(ga["var"]) + float(gb["var"])) / 2.0)
        values.append(None if chosen_a is None or chosen_b is None or denominator is None else _safe_ratio(
            float(chosen_a["mean"]) - float(chosen_b["mean"]), denominator))
        levels.append(chosen_level)

    for variable in ("spin_rate", "induced_vert_break"):
        for family in FAMILIES:
            table = stats[(variable, family)]
            selected, level = _select(table, context, threshold)
            glob = _global(table, threshold)
            values.append(None if selected is None or glob is None else _safe_ratio(
                float(selected["mean"]) - float(glob["mean"]), float(glob["std"])))
            levels.append(level)

    ext_table = stats[("extension", "ALL")]
    extension, ext_level = _select(ext_table, context, threshold)
    ext_global = _global(ext_table, threshold)
    values.append(None if extension is None or ext_global is None else _safe_ratio(
        float(extension["mean"]) - float(ext_global["mean"]), float(ext_global["std"])))
    levels.append(ext_level)

    height_table = stats[("rel_height", "ALL")]
    height, height_level = _select(height_table, context, threshold)
    height_global = _global(height_table, threshold)
    values.append(None if height is None or height_global is None else _safe_ratio(
        float(height["median"]) - float(height_global["median"]), float(height_global["iqr"])))
    values.append(None if height is None or height_global is None else (
        None if _safe_ratio(float(height["iqr"]), float(height_global["iqr"])) is None
        else float(_safe_ratio(float(height["iqr"]), float(height_global["iqr"]))) - 1.0))
    levels.extend([height_level, height_level])

    for family in FAMILIES:
        table = stats[("abs_horz_break", family)]
        selected, level = _select(table, context, threshold)
        glob = _global(table, threshold)
        values.append(None if selected is None or glob is None else _safe_ratio(
            float(selected["mean"]) - float(glob["mean"]), float(glob["std"])))
        levels.append(level)
    if len(values) != 17 or len(levels) != 17:
        raise ContractError("internal 17-feature construction drift")
    for value in values:
        if value is not None and not math.isfinite(value):
            raise ContractError("nonfinite lookup feature")
    return values, levels


def build_physics_lookup(filtered: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    if filtered.attrs.get("source_quality_filter_applied") is not True:
        raise ContractError("unfiltered TrackMan frame cannot reach lookup construction")
    threshold = int(config["trackman"]["support_threshold"])
    states = [(b, s, o) for b in range(4) for s in range(3) for o in range(3)]
    payload: dict[str, Any] = {
        "contract_version": config["contract_version"], "features": config["physics_features"],
        "support_threshold": threshold, "temporal_rule": "main season S uses TrackMan season < S",
        "families": list(FAMILIES), "states": {},
    }
    for season in range(2019, 2026):
        history = filtered.loc[pd.to_numeric(filtered["season"], errors="coerce") < season]
        season_payload: dict[str, Any] = {}
        if history.empty:
            for state in states:
                season_payload[":".join(map(str, state))] = {"values": [None] * 17, "levels": ["NO_HISTORY"] * 17}
        else:
            stats = _stats_for_history(history)
            for state in states:
                values, levels = _feature_vector(stats, state, threshold)
                season_payload[":".join(map(str, state))] = {"values": values, "levels": levels}
        payload["states"][str(season)] = season_payload
    return payload


def base_out_state_24(frame: pd.DataFrame) -> pd.Series:
    base = frame["base_state"].astype("string")
    outs = _numeric(frame, "outs_before")
    valid = base.isin(BASE_TOKENS) & outs.notna() & outs.eq(np.floor(outs)) & outs.between(0, 2)
    rendered = base.fillna("__MISSING__") + "|" + outs.fillna(-1).astype("int64").astype("string")
    return rendered.where(valid, "__MISSING__")


def preprocess_c3(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in frame.columns:
        if frame[column].dtype == np.float64:
            frame[column] = frame[column].astype(np.float32)
        elif frame[column].dtype == np.int64:
            frame[column] = frame[column].astype(np.int32)
    frame["platoon"] = (frame["pitcher_hand"] * 2 + frame["batter_hand"]).astype("category")
    frame["count_state"] = (frame["balls_before"] * 3 + frame["strikes_before"]).astype("category")
    for column in ("top_bottom", "game_type", "base_state", "platoon", "count_state"):
        frame[column] = frame[column].astype("category")
    frame["base_out_state_24"] = base_out_state_24(frame).astype("category")
    return frame


def apply_lookup(frame: pd.DataFrame, lookup: Mapping[str, Any], config: Mapping[str, Any],
                 *, return_levels: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    required = ["season", *CONTEXT]
    unique = frame.loc[:, required].drop_duplicates(ignore_index=True)
    records: dict[tuple[Any, ...], tuple[list[float], list[str]]] = {}
    for row in unique.itertuples(index=False, name=None):
        season = _integer(row[0], 2019, 2025)
        context = _context_key(dict(zip(required[1:], row[1:])))
        if season is None or context is None:
            raise ContractError("invalid main season/context")
        source = lookup.get("states", {}).get(str(season), {}).get(":".join(map(str, context)))
        if not isinstance(source, dict) or len(source.get("values", [])) != 17 or len(source.get("levels", [])) != 17:
            raise ContractError("lookup state missing/drift")
        values = [np.nan if value is None else float(value) for value in source["values"]]
        if any(not math.isfinite(value) for value in values if not math.isnan(value)):
            raise ContractError("lookup contains nonfinite value")
        records[tuple(row)] = (values, list(source["levels"]))
    ordered = [records[tuple(row)] for row in frame.loc[:, required].itertuples(index=False, name=None)]
    matrix = np.asarray([item[0] for item in ordered], dtype=np.float64)
    levels = np.asarray([item[1] for item in ordered], dtype=object)
    result = frame.copy()
    for index, column in enumerate(config["physics_features"]):
        result[column] = matrix[:, index].astype(np.float32)
    expected = config["base_c3_features"] + config["physics_features"]
    if list(result.loc[:, expected].columns) != expected or len(expected) != 67:
        raise ContractError("67-feature ordering failure")
    return (result, levels) if return_levels else result


def _diagnostics(frame: pd.DataFrame, levels: np.ndarray, config: Mapping[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"by_main_season": {}, "lookup_ranges_2020_2024": {}, "lookup_ranges_2025": {}}
    seasons = pd.to_numeric(frame["season"], errors="coerce").to_numpy()
    features = config["physics_features"]
    for season in range(2019, 2025):
        mask = seasons == season
        n = int(mask.sum())
        feature_diag: dict[str, Any] = {}
        counts = Counter(levels[mask].ravel().tolist()) if n else Counter()
        for column in features:
            values = pd.to_numeric(frame.loc[mask, column], errors="coerce").to_numpy(np.float64)
            finite = int(np.isfinite(values).sum())
            feature_diag[column] = {"finite": finite, "missing": n - finite,
                                    "finite_rate": finite / n if n else None}
        denominator = n * len(features)
        output["by_main_season"][str(season)] = {
            "rows": n, "features": feature_diag,
            "level_counts": {level: int(counts.get(level, 0)) for level in ("L0", "L1", "global", "NO_HISTORY")},
            "level_fractions": {level: (counts.get(level, 0) / denominator if denominator else None)
                                for level in ("L0", "L1", "global", "NO_HISTORY")},
        }
    def ranges(season_values: Sequence[int]) -> dict[str, Any]:
        collected = {name: [] for name in features}
        for season in season_values:
            for state in lookup["states"][str(season)].values():
                for name, value in zip(features, state["values"]):
                    if value is not None:
                        collected[name].append(float(value))
        return {name: {"min": (min(values) if values else None), "max": (max(values) if values else None)}
                for name, values in collected.items()}
    output["lookup_ranges_2020_2024"] = ranges(range(2020, 2025))
    output["lookup_ranges_2025"] = ranges((2025,))
    output["2025_range_exceedances"] = {
        name: {
            "below_historical_min": bool(output["lookup_ranges_2025"][name]["min"] is not None and
                                         output["lookup_ranges_2020_2024"][name]["min"] is not None and
                                         output["lookup_ranges_2025"][name]["min"] < output["lookup_ranges_2020_2024"][name]["min"]),
            "above_historical_max": bool(output["lookup_ranges_2025"][name]["max"] is not None and
                                         output["lookup_ranges_2020_2024"][name]["max"] is not None and
                                         output["lookup_ranges_2025"][name]["max"] > output["lookup_ranges_2020_2024"][name]["max"]),
        } for name in features
    }
    return output


def split_positions(n_rows: int, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    split = config["split"]
    rng = np.random.RandomState(int(split["random_state"]))
    es = np.sort(rng.choice(n_rows, size=int(n_rows * float(split["es_fraction"])), replace=False))
    mask = np.ones(n_rows, dtype=bool); mask[es] = False
    fit = np.flatnonzero(mask)
    if len(fit) != split["fit_rows"] or len(es) != split["es_rows"] or np.intersect1d(fit, es).size:
        raise ContractError("frozen 95/5 split mismatch")
    return fit, es


def _raw_columns(config: Mapping[str, Any]) -> list[str]:
    return [column for column in config["base_c3_features"] if column not in DERIVED]


def read_training(train_csv: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    usecols = [ROW_ID, *_raw_columns(config), TARGET]
    header = pd.read_csv(train_csv, nrows=0, encoding="utf-8-sig")
    if missing := sorted(set(usecols) - set(header.columns)):
        raise ContractError(f"official train projection missing: {missing}")
    frame = pd.read_csv(train_csv, usecols=usecols, encoding="utf-8-sig")
    if len(frame) != config["official_train_rows"]:
        raise ContractError("official train row-count drift")
    row_ids = frame.pop(ROW_ID).astype(str).tolist()
    if len(set(row_ids)) != len(row_ids):
        raise ContractError("row_id missing/duplicate")
    target = pd.to_numeric(frame.pop(TARGET), errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ContractError("target is not finite binary")
    return frame, target.to_numpy(np.float64), row_ids


def _git_sha(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def train(train_csv: Path, trackman_csv: Path, eda_json: Path, c3_contract: Path,
          output_dir: Path, repo_root: Path) -> Path:
    from catboost import CatBoostClassifier, Pool, __version__ as catboost_version
    config = load_config()
    authorities = verify_authorities(eda_json, trackman_csv, c3_contract)
    if catboost_version != config["catboost"]["version"]:
        raise ContractError(f"CatBoost version mismatch: {catboost_version}")
    _outside_repo(output_dir, repo_root)
    if output_dir.exists():
        raise ContractError("training output exists; refusing overwrite")
    total_started = time.time(); lookup_started = time.time()
    filtered, residue = read_filter_trackman(trackman_csv, config)
    lookup = build_physics_lookup(filtered, config)
    reversed_lookup = build_physics_lookup(filtered.iloc[::-1].reset_index(drop=True).set_flags(allows_duplicate_labels=False), config)
    if canonical_hash(lookup) != canonical_hash(reversed_lookup):
        raise ContractError("lookup row-order determinism failure")
    lookup_elapsed = time.time() - lookup_started
    raw, target, row_ids = read_training(train_csv, config)
    prepared, levels = apply_lookup(preprocess_c3(raw), lookup, config, return_levels=True)
    if pd.to_numeric(prepared.loc[pd.to_numeric(prepared["season"]) == 2019, config["physics_features"]].stack(), errors="coerce").notna().any():
        raise ContractError("2019 strict-past all-missing contract failure")
    diagnostics = _diagnostics(prepared, levels, config, lookup)
    features = config["base_c3_features"] + config["physics_features"]
    cats = config["categorical_features"]
    fit, es = split_positions(len(prepared), config)
    params = dict(config["catboost"]); params.pop("version")
    model = CatBoostClassifier(**params)
    cat_indices = [features.index(column) for column in cats]
    training_started = time.time()
    model.fit(Pool(prepared.iloc[fit][features], target[fit], cat_features=cat_indices),
              eval_set=Pool(prepared.iloc[es][features], target[es], cat_features=cat_indices),
              use_best_model=True, verbose=False)
    training_elapsed = time.time() - training_started
    probability = np.asarray(model.predict_proba(prepared.iloc[es][features], thread_count=6), dtype=np.float64)[:, 1]
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ContractError("invalid ES probabilities")
    output_dir.mkdir(parents=True)
    model_path = output_dir / "catboost.cbm"; model.save_model(str(model_path))
    lookup_path = output_dir / LOOKUP_NAME
    lookup_path.write_text(json.dumps(lookup, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    contract = {
        "contract_version": config["contract_version"], "features": features,
        "base_c3_features": config["base_c3_features"], "physics_features": config["physics_features"],
        "categorical_features": cats, "inference_threads": 6,
        "prediction": "native CatBoost predict_proba[:,1]", "lookup_name": LOOKUP_NAME,
        "lookup_sha256": sha256_file(lookup_path),
        "preprocessing": "exact Feature-Lab C3 plus 17 numeric strict-past physics features",
    }
    contract_path = output_dir / "feature_contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "contract_version": config["contract_version"], "git_sha": _git_sha(repo_root),
        "runner_sha256": sha256_file(Path(__file__).resolve()), "authorities": authorities,
        "residue_filter": residue, "lookup_sha256": sha256_file(lookup_path),
        "lookup_canonical_sha256": canonical_hash(lookup), "lookup_reversed_canonical_sha256": canonical_hash(reversed_lookup),
        "lookup_deterministic": True, "lookup_elapsed_seconds": lookup_elapsed,
        "source_rows": len(prepared), "source_row_hash": canonical_hash(row_ids),
        "target_hash": canonical_hash(target.tolist()), "fit_rows": len(fit), "es_rows": len(es),
        "fit_position_hash": canonical_hash(fit.tolist()), "es_position_hash": canonical_hash(es.tolist()),
        "feature_count": len(features), "features": features, "categorical_features": cats,
        "catboost_version": catboost_version, "params": config["catboost"],
        "best_iteration": int(model.get_best_iteration()), "tree_count": int(model.tree_count_),
        "model_sha256": sha256_file(model_path), "model_bytes": model_path.stat().st_size,
        "es_probability_hash": canonical_hash(probability.tolist()),
        "diagnostic_only_es_brier": float(np.mean((probability - target[es]) ** 2)),
        "training_elapsed_seconds": training_elapsed, "total_elapsed_seconds": time.time() - total_started,
        "structural_diagnostics": diagnostics,
        "scope": {"test_distribution_access": False, "raw_trackman_inference_dependency": False,
                  "cross_row_test_aggregation": False, "entity_crosswalk": False,
                  "current_pitch_measurement": False, "calibration": False, "ensemble": False,
                  "gpu_used": False},
    }
    report["canonical_report_sha256"] = canonical_hash(report)
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path


PACKAGE_SCRIPT = r'''#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

MODEL_DIR = Path(__file__).resolve().parent / "model"
TEST_PATH = Path(os.environ.get("LGA_TEST_PATH", "data/test.csv"))
SAMPLE_PATH = Path(os.environ.get("LGA_SAMPLE_PATH", "data/sample_submission.csv"))
OUT_PATH = Path(os.environ.get("LGA_OUT_PATH", "output/submission.csv"))
LOOKUP_PATH = MODEL_DIR / "trackman_physics_dr_v1_lookup.json"
BASE_TOKENS = frozenset(("___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"))

def _number(value, low, high):
    try: numeric = float(value)
    except (TypeError, ValueError): raise RuntimeError("invalid main context")
    if not np.isfinite(numeric) or not numeric.is_integer() or not low <= int(numeric) <= high: raise RuntimeError("invalid main context")
    return int(numeric)

def load_assets():
    contract = json.loads((MODEL_DIR / "feature_contract.json").read_text(encoding="utf-8"))
    payload = LOOKUP_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != contract["lookup_sha256"]: raise RuntimeError("lookup SHA drift")
    lookup = json.loads(payload.decode("utf-8"))
    if lookup.get("contract_version") != "aimers9-c3-trackman-physics-dr-v1": raise RuntimeError("lookup version drift")
    model = CatBoostClassifier(); model.load_model(str(MODEL_DIR / "catboost.cbm"))
    return model, contract, lookup

def prepare_frame(raw, contract, lookup):
    frame = raw.copy()
    for column in frame.columns:
        if frame[column].dtype == np.float64: frame[column] = frame[column].astype(np.float32)
        elif frame[column].dtype == np.int64: frame[column] = frame[column].astype(np.int32)
    frame["platoon"] = (frame["pitcher_hand"]*2 + frame["batter_hand"]).astype("category")
    frame["count_state"] = (frame["balls_before"]*3 + frame["strikes_before"]).astype("category")
    for column in ("top_bottom", "game_type", "base_state", "platoon", "count_state"): frame[column] = frame[column].astype("category")
    base = frame["base_state"].astype("string"); outs = pd.to_numeric(frame["outs_before"], errors="coerce")
    outs = outs.where(np.isfinite(outs)); valid = base.isin(BASE_TOKENS) & outs.notna() & outs.eq(np.floor(outs)) & outs.between(0,2)
    frame["base_out_state_24"] = (base.fillna("__MISSING__")+"|"+outs.fillna(-1).astype("int64").astype("string")).where(valid, "__MISSING__").astype("category")
    vectors = []
    for row in frame[["season","balls_before","strikes_before","outs_before"]].itertuples(index=False, name=None):
        season = _number(row[0], 2019, 2025); balls = _number(row[1],0,3); strikes = _number(row[2],0,2); out = _number(row[3],0,2)
        payload = lookup.get("states",{}).get(str(season),{}).get(f"{balls}:{strikes}:{out}")
        if not isinstance(payload,dict) or len(payload.get("values",[])) != 17: raise RuntimeError("missing lookup state")
        vectors.append([np.nan if value is None else float(value) for value in payload["values"]])
    values = np.asarray(vectors,dtype=np.float64)
    for index,column in enumerate(contract["physics_features"]): frame[column] = values[:,index].astype(np.float32)
    if contract["features"] != contract["base_c3_features"] + contract["physics_features"] or len(contract["features"]) != 67: raise RuntimeError("feature contract drift")
    if contract["categorical_features"] != ["top_bottom","game_type","base_state","platoon","count_state","base_out_state_24"]: raise RuntimeError("categorical contract drift")
    return frame.loc[:,contract["features"]]

def predict_frame(raw, model=None, contract=None, lookup=None):
    if model is None or contract is None or lookup is None: model,contract,lookup = load_assets()
    probability = np.asarray(model.predict_proba(prepare_frame(raw,contract,lookup),thread_count=6),dtype=np.float64)[:,1]
    if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all(): raise RuntimeError("invalid probability")
    return probability

def main():
    test = pd.read_csv(TEST_PATH,encoding="utf-8-sig")
    if "row_id" not in test or test["row_id"].isna().any() or test["row_id"].astype(str).duplicated().any(): raise RuntimeError("invalid row_id")
    row_ids = test["row_id"].copy(); probability = predict_frame(test)
    sample = pd.read_csv(SAMPLE_PATH,encoding="utf-8-sig")
    if len(sample)!=len(test) or list(sample["row_id"].astype(str))!=list(row_ids.astype(str)): raise RuntimeError("sample row mismatch")
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":row_ids,"control_success":probability}).to_csv(OUT_PATH,index=False,encoding="utf-8-sig")
if __name__ == "__main__": main()
'''


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME); info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3; info.external_attr = (stat.S_IFREG | (0o755 if name == "script.py" else 0o644)) << 16
    return info


def _zip_bytes(model_path: Path, contract: Mapping[str, Any], lookup_path: Path) -> bytes:
    payloads = {
        "model/catboost.cbm": model_path.read_bytes(),
        "model/feature_contract.json": (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode(),
        f"model/{LOOKUP_NAME}": lookup_path.read_bytes(), "script.py": PACKAGE_SCRIPT.encode(),
        "requirements.txt": b"catboost==1.2.10\n",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in PACKAGE_MEMBERS:
            archive.writestr(_zip_info(name), payloads[name])
    return output.getvalue()


def _import_script(path: Path):
    spec = importlib.util.spec_from_file_location("c3_tm_phys_package", path)
    if spec is None or spec.loader is None: raise ContractError("cannot import package script")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def build_validate(trained_dir: Path, train_csv: Path, output_zip: Path,
                   validation_rows: int, repo_root: Path) -> Path:
    from catboost import __version__ as catboost_version
    config = load_config(); _outside_repo(output_zip, repo_root)
    if output_zip.exists(): raise ContractError("output ZIP exists; refusing overwrite")
    if catboost_version != config["catboost"]["version"]: raise ContractError("CatBoost version drift")
    model_path = trained_dir / "catboost.cbm"; contract_path = trained_dir / "feature_contract.json"
    lookup_path = trained_dir / LOOKUP_NAME; report_path = trained_dir / "training_report.json"
    if not all(path.is_file() for path in (model_path, contract_path, lookup_path, report_path)):
        raise ContractError("trained artifacts missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["features"] != config["base_c3_features"] + config["physics_features"] or contract["categorical_features"] != config["categorical_features"]:
        raise ContractError("trained contract drift")
    first = _zip_bytes(model_path, contract, lookup_path); second = _zip_bytes(model_path, contract, lookup_path)
    if first != second: raise ContractError("deterministic ZIP rebuild failed")
    output_zip.parent.mkdir(parents=True, exist_ok=True); output_zip.write_bytes(first)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="aimers9-c3-tm-physics-validate-") as directory:
        root = Path(directory); package = root / "package"
        with zipfile.ZipFile(output_zip) as archive:
            if tuple(sorted(archive.namelist())) != tuple(sorted(PACKAGE_MEMBERS)): raise ContractError("package member drift")
            archive.extractall(package)
        module = _import_script(package / "script.py"); model, packaged_contract, lookup = module.load_assets()
        raw_columns = [ROW_ID, *_raw_columns(config)]
        frame = pd.read_csv(train_csv, usecols=raw_columns, nrows=validation_rows, encoding="utf-8-sig")
        if len(frame) != validation_rows: raise ContractError("validation row count shortfall")
        prediction = module.predict_frame(frame, model, packaged_contract, lookup)
        chunks = np.array_split(np.arange(len(frame), dtype=np.int64), 7)
        chunked = np.concatenate([module.predict_frame(frame.iloc[position], model, packaged_contract, lookup) for position in chunks])
        if not np.array_equal(prediction, chunked): raise ContractError("chunk dependence")
        shuffled = frame.sample(frac=1.0, random_state=991)
        restored = pd.Series(module.predict_frame(shuffled, model, packaged_contract, lookup), index=shuffled.index).sort_index().to_numpy()
        if not np.array_equal(prediction, restored): raise ContractError("shuffle dependence")
        for index in (0, len(frame) // 2, len(frame) - 1):
            if not np.array_equal(module.predict_frame(frame.iloc[[index]], model, packaged_contract, lookup), prediction[[index]]):
                raise ContractError("single-row dependence")
        test_path = root / "test.csv"; sample_path = root / "sample_submission.csv"; out_path = root / "submission.csv"
        frame.to_csv(test_path, index=False, encoding="utf-8-sig")
        pd.DataFrame({ROW_ID: frame[ROW_ID], TARGET: 0.0}).to_csv(sample_path, index=False, encoding="utf-8-sig")
        env = os.environ.copy(); env.update({"LGA_TEST_PATH": str(test_path), "LGA_SAMPLE_PATH": str(sample_path),
                                             "LGA_OUT_PATH": str(out_path), "NO_PROXY": "*", "no_proxy": "*"})
        proc = subprocess.run([sys.executable, str(package / "script.py")], cwd=root, env=env,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0: raise ContractError(f"offline package failed: {proc.stderr[-1000:]}")
        output = pd.read_csv(out_path, encoding="utf-8-sig")
        if list(output[ROW_ID].astype(str)) != list(frame[ROW_ID].astype(str)): raise ContractError("output row/order drift")
        probability_output = pd.to_numeric(output[TARGET], errors="coerce").to_numpy()
        if not np.isfinite(probability_output).all() or not ((probability_output >= 0) & (probability_output <= 1)).all():
            raise ContractError("output probability invalid")
    validation = {
        "package_sha256": sha256_file(output_zip), "package_bytes": output_zip.stat().st_size,
        "model_sha256": sha256_file(model_path), "lookup_sha256": sha256_file(lookup_path),
        "deterministic_rebuild": True, "members": list(PACKAGE_MEMBERS), "feature_count": 67,
        "categorical_features": config["categorical_features"], "validation_rows": validation_rows,
        "finite_native_probability": True, "row_id_order_preserved": True,
        "shuffle_independence": True, "chunk_independence": True, "single_row_independence": True,
        "offline_inference": True, "inference_threads": 6, "raw_trackman_dependency": False,
        "cross_test_row_aggregation": False, "validation_elapsed_seconds": time.time() - started,
    }
    validation["canonical_report_sha256"] = canonical_hash(validation)
    (trained_dir / "package_validation_report.json").write_text(
        json.dumps(validation, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_zip


def static_contract() -> dict[str, Any]:
    config = load_config()
    forbidden = ("trackman_history.csv", "zone_speed", "rel_side", "tagged_pitch_type", "auto_pitch_type",
                 "tm_cf_", "requests.", "urllib.", "C_LOGIT", "clip(")
    if any(token in PACKAGE_SCRIPT for token in forbidden): raise ContractError("package firewall violation")
    return {"contract_version": config["contract_version"], "feature_count": 67,
            "categorical_count": 6, "raw_trackman_scan": False, "static_pass": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train-csv", type=Path, required=True)
    train_parser.add_argument("--trackman-csv", type=Path, required=True)
    train_parser.add_argument("--eda-json", type=Path, required=True)
    train_parser.add_argument("--c3-contract", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--repo-root", type=Path, default=ROOT)
    build = sub.add_parser("build-validate")
    build.add_argument("--trained-dir", type=Path, required=True)
    build.add_argument("--train-csv", type=Path, required=True)
    build.add_argument("--output-zip", type=Path, required=True)
    build.add_argument("--validation-rows", type=int, default=245789)
    build.add_argument("--repo-root", type=Path, default=ROOT)
    sub.add_parser("static")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train(args.train_csv.resolve(), args.trackman_csv.resolve(), args.eda_json.resolve(),
                           args.c3_contract.resolve(), args.output_dir.resolve(), args.repo_root.resolve())
        elif args.command == "build-validate":
            result = build_validate(args.trained_dir.resolve(), args.train_csv.resolve(), args.output_zip.resolve(),
                                    args.validation_rows, args.repo_root.resolve())
        else:
            result = static_contract()
        print(json.dumps(result if isinstance(result, dict) else {"path": str(result)}, sort_keys=True))
        return 0
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
