#!/usr/bin/env python3
"""Target-free TrackMan repertoire-fingerprint crosswalk feasibility audit.

This module intentionally stops at Level P and source-order prerequisite
diagnostics.  It never constructs an exact pitch-row matcher.  The runtime
configuration is authoritative; the JSON contract is checked before any data
reader is opened.  All external reports are aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CONTRACT_VERSION = "aimers9-trackman-repertoire-crosswalk-v2"
ORIGINS = (2022, 2023, 2024)
MAIN_SEASONS = tuple(range(2019, 2025))
GAME_TYPE = "R"
FAMILIES = ("fastball", "breaking", "offspeed")
VERIFIER_CHANNELS = ("game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "batter_hand")
TARGET = "control_success"
SOURCE_ORDER_STATUS = "NOT_PROVEN"
LEVEL_R_NOT_RUN = "NOT_RUN_SOURCE_ORDER_NOT_PROVEN"
MAIN_DISCOVERY_COLUMNS = ("row_id", "season", "game_type")
MAIN_PROJECTION = (
    "row_id", "season", "game_type", "game_month", "game_dayofweek", "inning",
    "top_bottom", "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "pitcher_hand", "pitcher_team_id", "batter_hand",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)
TRACKMAN_PROJECTION = (
    "trackman_id", "season", "game_date", "game_month", "game_dayofweek",
    "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
    "pitcher_trackman_id", "pitcher_hand", "batter_hand", "pitcher_team",
    "pitch_type_group",
)
FORBIDDEN_MAIN = frozenset({TARGET})
FORBIDDEN_TRACKMAN = frozenset({
    "tagged_pitch_type", "auto_pitch_type", "rel_speed", "spin_rate",
    "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side",
    "zone_speed", "batter_trackman_id",
})
NULL_RISK_CEILING = 0.01
WILSON_Z_95 = 1.6448536269514722
NULL_CAL_NS = "aimers9-trackman-repertoire-crosswalk-v2/null/calibration"
NULL_EVAL_NS = "aimers9-trackman-repertoire-crosswalk-v2/null/evaluation"
FRAME_HASH_ALGORITHM = "canonical-jsonl-v1"
PRECISION_PROBE_CHUNK_SIZE = 100_000


class CrosswalkAuditError(RuntimeError):
    """Fail-closed schema, scope, contract, or identity-audit error."""


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CrosswalkAuditError(f"duplicate JSON contract key: {key}")
        result[key] = value
    return result


def load_contract_config(repo_root: str | Path) -> dict[str, Any]:
    path = Path(repo_root) / "configs" / "trackman_repertoire_crosswalk_v2.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise CrosswalkAuditError(f"cannot load contract config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CrosswalkAuditError("contract config must be an object")
    validate_contract_config(value)
    return value


def _contract_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    scope = config.get("scope", {})
    taxonomy = config.get("taxonomy", {})
    materialization = config.get("materialization", {})
    selection = config.get("selection", {})
    verification = config.get("verification", {})
    self_id = config.get("self_identification", {})
    nulls = config.get("nulls", {})
    level_r = config.get("level_r", {})
    output = config.get("output", {})
    return {
        "contract_version": config.get("contract_version"),
        "scope": {
            "main_seasons": list(scope.get("main_seasons", [])),
            "origins": list(scope.get("origins", [])),
            "game_type": scope.get("game_type"),
            "target_free": scope.get("target_free"),
            "test_free": scope.get("test_free"),
            "leaderboard_free": scope.get("leaderboard_free"),
            "external_free": scope.get("external_free"),
            "future_safe": scope.get("future_safe"),
            "main_projection": list(scope.get("main_projection", [])),
            "trackman_projection": list(scope.get("trackman_projection", [])),
            "forbidden_main": sorted(scope.get("forbidden_main", [])),
            "forbidden_trackman": sorted(scope.get("forbidden_trackman", [])),
        },
        "taxonomy": dict(taxonomy),
        "materialization": dict(materialization),
        "selection": dict(selection),
        "verification": dict(verification),
        "self_identification": dict(self_id),
        "nulls": dict(nulls),
        "level_r": dict(level_r),
        "output": dict(output),
    }


def _runtime_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "scope": {
            "main_seasons": list(MAIN_SEASONS), "origins": list(ORIGINS), "game_type": GAME_TYPE,
            "target_free": True, "test_free": True, "leaderboard_free": True,
            "external_free": True, "future_safe": True,
            "main_projection": list(MAIN_PROJECTION), "trackman_projection": list(TRACKMAN_PROJECTION),
            "forbidden_main": sorted(FORBIDDEN_MAIN), "forbidden_trackman": sorted(FORBIDDEN_TRACKMAN),
        },
        "taxonomy": {
            "canonical_families": list(FAMILIES), "expected_trackman_groups": ["fastball", "breaking", "offspeed", "other"], "trackman_other_mode": "exclude_other_renormalize_known",
            "alternate_mapping_allowed": False, "mode_must_freeze_before_matching": True,
            "compatibility_requires_main_three_family_sum": True,
            "compatibility_requires_documented_family_names": True,
            "source_level_denominator_proof": {"status": "NOT_PROVEN", "known_only_denominator_proven": False, "basis": "repository documentation names rate fields but does not prove TrackMan known-only denominator compatibility", "observational_rule": {"enabled": True, "requires_main_complete_three_family_sum": True, "requires_all_canonical_families": True, "requires_unexpected_groups_absent": True, "requires_other_rows_zero": True, "requires_missing_rows_zero": True}},
            "documentation_evidence": ["docs/data_description.md:asof_pitcher_fastball_rate/breaking/offspeed_rate", "docs/data_description.md:pitch_type_group"],
            "share_sum_tolerance": 1e-6, "minimum_complete_main_rows": 1,
            "minimum_trackman_known_rows": 1, "unexpected_group_policy": "reject",
        },
        "materialization": {
            "representation": "share_based_annual_three_family_trajectory",
            "source_order_allowed": False, "source_order_status": SOURCE_ORDER_STATUS,
            "asof_mode": "infer_reset_or_cumulative_from_support_invariants",
            "mode_min_transition_fraction": 0.95, "storage_precision_probe": "decimal_tokens_target_free",
            "storage_tolerance_multiplier": 2.0, "storage_tolerance_minimum": 1e-6,
            "cumulative_negative_increment_tolerance_multiplier": 2.0,
            "max_support_snapshot_tie_policy": "conflicting_ties_fail_closed",
            "count_reconstruction_is_primary_requirement": False,
        },
        "selection": {
            "channels": ["annual_mix_tv", "trajectory_delta_half_l1"],
            "distance": "annual_mix_tv_and_trajectory_delta_half_l1",
            "hand": "hard_compatibility_only", "hand_temporal_basis": "hands_by_season_pre_origin_only", "unknown_or_ambiguous_hand": "incompatible", "trajectory_delta_pairs": "calendar_adjacent_only", "null_candidate_universe": "finite_pre_origin_hard_hand_fit_matrix", "support": "eligibility_and_quality_only",
            "team": "not_used", "mutual_top1": True, "second_best_margin": True,
            "one_to_one": True, "ambiguous_ties_unmatched": True,
            "minimum_common_seasons": 2, "minimum_common_delta_seasons": 1,
        },
        "verification": {
            "channel": "held_out_non_repertoire_context",
            "columns": ["game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "batter_hand"],
            "selection_columns_excluded": ["pitch_type_group", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"],
            "mapping_frozen_before_verification": True, "verifier_can_change_mapping": False,
            "verifier_confirmed_manifest": False, "null_risk_ceiling": 0.01,
            "null_candidate_universe": "pre_origin_hard_hand_compatible",
            "aggregation": "per_channel_median_total_variation",
            "calibration_quantile": 0.01,
            "all_required_channels_conjunctive": True,
        },
        "self_identification": {
            "reference": "seasons <= origin - 2", "query": "season == origin - 1",
            "diagnostic_name": "next_season_repertoire_persistence",
            "not_full_multi_year_validation": True,
            "distance": "annual_mix_tv_reference_latest_to_single_query_season",
            "separate_null_contract": True,
        },
        "nulls": {
            "risk_ceiling": NULL_RISK_CEILING, "wilson_confidence": 0.95,
            "requested_calibration_transformations": 1000, "requested_evaluation_transformations": 1000,
            "trial_unit": "unique_origin_transformation_familywise_any_accept",
            "calibration_namespace": NULL_CAL_NS, "evaluation_namespace": NULL_EVAL_NS,
            "calibration_evaluation_disjoint": True, "duplicate_transformations_do_not_increase_n": True,
            "small_finite_space_is_not_repeated": True,
        },
        "level_r": {"scope": "prerequisite_diagnostic_only", "source_order_prerequisite": "SOURCE_ORDER_PROVEN", "exact_matcher_implemented": False, "not_run_status": LEVEL_R_NOT_RUN},
        "output": {"aggregate_only": True, "external_only": True, "files": ["trackman_repertoire_crosswalk_v2_report.json", "trackman_repertoire_crosswalk_v2_report.md"], "include_raw_mapping_ids": False, "include_target_values": False, "include_physics": False},
    }


def validate_contract_config(config: Mapping[str, Any]) -> None:
    if config.get("documentary_snapshot_checked_against_runtime_contract") is not True:
        raise CrosswalkAuditError("config must be runtime-authority checked")
    if _contract_projection(config) != _runtime_contract():
        raise CrosswalkAuditError("contract config diverges from runtime contract")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int_value(value: Any) -> int | None:
    x = _num(value)
    return int(x) if x is not None and x.is_integer() else None


def _text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value).strip()


def _id(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _hand(value: Any) -> str | None:
    value = _text(value)
    return {"l": "L", "left": "L", "r": "R", "right": "R", "1": "L", "2": "R"}.get(value.lower()) if value else None


def _top(value: Any) -> str | None:
    value = _text(value)
    return {"t": "T", "top": "T", "b": "B", "bottom": "B"}.get(value.lower()) if value else None


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in sorted(value.items(), key=lambda x: repr(x[0]))}
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (TypeError, ValueError):
            pass
    if _is_missing(value):
        return None
    return value


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    body = {k: v for k, v in report.items() if k != "canonical_report_sha256"}
    return canonical_hash(_safe(body))


def selection_map_hash(selection_map: Mapping[str, str]) -> str:
    """Stable aggregate hash for a mapping; raw IDs are not report material."""
    return canonical_hash(sorted((str(left), str(right)) for left, right in selection_map.items()))


def canonical_frame_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Hash frame schema and rows with a deterministic streaming JSONL contract."""
    cols = list(columns) if columns is not None else [c for c in frame.columns if not c.startswith("__")]
    digest = hashlib.sha256()
    header = json.dumps({"columns": cols}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    digest.update(header + b"\n")
    for row in frame[cols].itertuples(index=False, name=None):
        values = [None if _is_missing(value) else value.item() if hasattr(value, "item") else value for value in row]
        line = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
        digest.update(line + b"\n")
    return digest.hexdigest()


def validate_projection(frame: pd.DataFrame, *, trackman: bool = False) -> None:
    expected = TRACKMAN_PROJECTION if trackman else MAIN_PROJECTION
    forbidden = FORBIDDEN_TRACKMAN if trackman else FORBIDDEN_MAIN
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise CrosswalkAuditError(f"projection missing columns: {missing}")
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise CrosswalkAuditError(f"forbidden columns materialized: {present}")
    unexpected = sorted(set(frame.columns) - set(expected) - {"__source_position"})
    if unexpected:
        raise CrosswalkAuditError(f"unexpected projection columns: {unexpected}")


def _header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # pragma: no cover - pandas parser details vary
        raise CrosswalkAuditError(f"cannot read CSV header {path}: {exc}") from exc


def _scoped_read(path: Path, columns: Sequence[str], positions: Sequence[int] | None = None) -> pd.DataFrame:
    available = _header(path)
    missing = [c for c in columns if c not in available]
    if missing:
        raise CrosswalkAuditError(f"{path.name} missing columns: {missing}")
    skiprows = None
    wanted = None
    if positions is not None:
        wanted = tuple(sorted(set(int(x) for x in positions)))
        wanted_set = set(wanted)
        skiprows = lambda line: line > 0 and line - 1 not in wanted_set
    frame = pd.read_csv(path, usecols=list(columns), skiprows=skiprows)
    frame["__source_position"] = list(range(len(frame))) if wanted is None else list(wanted)
    if wanted is not None and len(frame) != len(wanted):
        raise CrosswalkAuditError("scoped row count mismatch")
    return frame


def discover_main_source_positions(main_csv: str | Path, *, seasons: Iterable[int] = MAIN_SEASONS, game_type: str = GAME_TYPE) -> pd.DataFrame:
    frame = _scoped_read(Path(main_csv), MAIN_DISCOVERY_COLUMNS)
    season = pd.to_numeric(frame["season"], errors="coerce")
    mask = season.isin(list(seasons)) & frame["game_type"].astype(str).eq(str(game_type))
    selected = frame.loc[mask].copy()
    selected["season"] = season.loc[selected.index].astype(int)
    return selected.reset_index(drop=True)


def read_scoped_main_features(main_csv: str | Path, positions: Sequence[int] | None = None, *, seasons: Iterable[int] = MAIN_SEASONS, game_type: str = GAME_TYPE) -> pd.DataFrame:
    if positions is None:
        positions = discover_main_source_positions(main_csv, seasons=seasons, game_type=game_type)["__source_position"].tolist()
    frame = _scoped_read(Path(main_csv), MAIN_PROJECTION, positions)
    validate_projection(frame)
    if frame["row_id"].map(_id).duplicated().any():
        raise CrosswalkAuditError("main row_id is not unique")
    return frame


def discover_trackman_source_positions(trackman_csv: str | Path, *, seasons: Iterable[int] = MAIN_SEASONS) -> pd.DataFrame:
    frame = _scoped_read(Path(trackman_csv), ("trackman_id", "season"))
    season = pd.to_numeric(frame["season"], errors="coerce")
    selected = frame.loc[season.isin(list(seasons))].copy()
    selected["season"] = season.loc[selected.index].astype(int)
    return selected.reset_index(drop=True)


def read_scoped_trackman_features(trackman_csv: str | Path, positions: Sequence[int] | None = None, *, seasons: Iterable[int] = MAIN_SEASONS) -> pd.DataFrame:
    if positions is None:
        positions = discover_trackman_source_positions(trackman_csv, seasons=seasons)["__source_position"].tolist()
    frame = _scoped_read(Path(trackman_csv), TRACKMAN_PROJECTION, positions)
    validate_projection(frame, trackman=True)
    return frame


def analyze_rate_storage_precision_tokens(tokens: Iterable[Any], *, multiplier: float = 2.0, minimum: float = 1e-6) -> dict[str, Any]:
    decimals: list[int] = []
    for token in tokens:
        if token is None:
            continue
        text = str(token).strip().lower()
        if not text or text in {"nan", "na", "none"}:
            continue
        if "e" in text:
            try:
                decimals.append(max(0, len(f"{float(text):.12f}".rstrip("0").split(".")[-1])))
            except ValueError:
                continue
        elif "." in text:
            decimals.append(len(text.split(".", 1)[1].rstrip("0")))
        else:
            decimals.append(0)
    observed = max(decimals, default=0)
    tolerance = max(float(minimum), float(multiplier) * (10.0 ** (-observed)))
    return {"observed_max_decimal_places": observed, "tolerance": tolerance, "method": "target_free_decimal_token_probe"}


def probe_rate_storage_precision(main_csv: str | Path, *, multiplier: float = 2.0, minimum: float = 1e-6) -> dict[str, Any]:
    """Probe only raw non-target rate tokens; no target or TrackMan fields are read."""
    columns = [f"asof_pitcher_{family}_rate" for family in FAMILIES]
    observed = 0
    try:
        chunks = pd.read_csv(main_csv, usecols=columns, dtype=str, chunksize=PRECISION_PROBE_CHUNK_SIZE)
        for raw in chunks:
            chunk_result = analyze_rate_storage_precision_tokens((value for column in columns for value in raw[column]), multiplier=multiplier, minimum=minimum)
            observed = max(observed, int(chunk_result["observed_max_decimal_places"]))
    except Exception as exc:  # pragma: no cover - pandas parser details vary
        raise CrosswalkAuditError(f"cannot probe rate storage precision: {exc}") from exc
    result = {"observed_max_decimal_places": observed, "tolerance": max(float(minimum), float(multiplier) * (10.0 ** (-observed))), "method": "target_free_decimal_token_probe_streaming_json_csv", "chunk_size": PRECISION_PROBE_CHUNK_SIZE}
    result["columns"] = columns
    result["target_free"] = True
    return result


def _valid_mix(row: Mapping[str, Any], tolerance: float) -> tuple[float, tuple[float, float, float]] | None:
    n = _num(row.get("asof_pitcher_pitchmix_n"))
    values = tuple(_num(row.get(f"asof_pitcher_{family}_rate")) for family in FAMILIES)
    if n is None or n < 0 or any(v is None or v < -tolerance or v > 1.0 + tolerance for v in values):
        return None
    shares = tuple(max(0.0, min(1.0, float(v))) for v in values)
    if abs(sum(shares) - 1.0) > tolerance:
        return None
    return float(n), shares  # type: ignore[arg-type]


def _snapshot(group: pd.DataFrame, tolerance: float) -> dict[str, Any] | None:
    valid = []
    for row in group.to_dict("records"):
        item = _valid_mix(row, tolerance)
        if item is not None:
            valid.append((item[0], item[1]))
    if not valid:
        return None
    maximum = max(item[0] for item in valid)
    tied = [item[1] for item in valid if abs(item[0] - maximum) <= tolerance]
    reference = tied[0]
    if any(sum(abs(a - b) for a, b in zip(reference, other)) > tolerance for other in tied[1:]):
        raise CrosswalkAuditError("conflicting maximum-support main snapshot")
    return {"support": maximum, "share": tuple(float(x) for x in reference)}


def _support_range(group: pd.DataFrame, tolerance: float) -> tuple[float, float] | None:
    values = []
    for row in group.to_dict("records"):
        item = _valid_mix(row, tolerance)
        if item is not None:
            values.append(item[0])
    return (min(values), max(values)) if values else None


def infer_main_asof_mode(frame: pd.DataFrame, config: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    for pitcher, pitcher_frame in frame.groupby("pitcher_id", dropna=False, sort=True):
        by_season: dict[int, dict[str, Any]] = {}
        for season, group in pitcher_frame.groupby("season", sort=True):
            item = _snapshot(group, tolerance)
            support_range = _support_range(group, tolerance)
            if item is not None and support_range is not None:
                by_season[int(season)] = {**item, "support_min": support_range[0], "support_max": support_range[1]}
        seasons = sorted(by_season)
        for previous, current in zip(seasons, seasons[1:]):
            prev_max = by_season[previous]["support_max"]
            curr_min = by_season[current]["support_min"]
            if curr_min >= prev_max - tolerance:
                classification = "CUMULATIVE"
            elif curr_min < prev_max - tolerance:
                classification = "SEASON_RESET"
            else:
                classification = "UNCLASSIFIED"
            transitions.append({"pitcher": _id(pitcher), "previous": previous, "current": current, "previous_support_max": prev_max, "current_support_min": curr_min, "classification": classification})
    if not transitions:
        return {"status": "INSUFFICIENT_TRANSITIONS", "mode": None, "transition_count": 0}
    cumulative_count = sum(1 for item in transitions if item["classification"] == "CUMULATIVE")
    reset_count = sum(1 for item in transitions if item["classification"] == "SEASON_RESET")
    cumulative = cumulative_count / len(transitions)
    reset = reset_count / len(transitions)
    minimum = float(config["materialization"]["mode_min_transition_fraction"])
    if cumulative >= minimum and cumulative > reset:
        mode = "CUMULATIVE"
    elif reset >= minimum and reset > cumulative:
        mode = "SEASON_RESET"
    else:
        mode = None
    return {"status": "PASS" if mode else "MATERIALIZATION_NOT_PROVEN", "mode": mode, "transition_count": len(transitions), "cumulative_count": cumulative_count, "reset_count": reset_count, "unclassified_count": len(transitions) - cumulative_count - reset_count, "cumulative_fraction": cumulative, "reset_fraction": reset, "classification_basis": "current-season minimum versus prior-season maximum; unordered within-season support ranges"}


def _context_token(row: Mapping[str, Any], field: str) -> str:
    if field == "inning":
        value = _int_value(row.get(field))
        return "MISSING" if value is None else "1_3" if value <= 3 else "4_6" if value <= 6 else "7_9" if value <= 9 else "10_PLUS"
    if field == "top_bottom":
        return _top(row.get(field)) or "MISSING"
    if field == "batter_hand":
        return _hand(row.get(field)) or "MISSING"
    if field in {"balls_before", "strikes_before", "outs_before", "game_month", "game_dayofweek"}:
        value = _int_value(row.get(field))
        return "MISSING" if value is None else str(value)
    return _text(row.get(field)) or "MISSING"


def _profile_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    fields = ("game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "batter_hand")
    result = {field: Counter() for field in fields}
    for row in rows:
        for field in fields:
            result[field][_context_token(row, field)] += 1
    return result


def build_main_profiles(frame: pd.DataFrame, mode: str, tolerance: float) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for pitcher, pitcher_frame in frame.groupby("pitcher_id", dropna=False, sort=True):
        pid = _id(pitcher)
        if pid is None:
            continue
        snapshots: dict[int, dict[str, Any]] = {}
        contexts: dict[int, dict[str, Counter[str]]] = {}
        hands: set[str] = set()
        hands_by_season: dict[int, list[str]] = {}
        for season, group in pitcher_frame.groupby("season", sort=True):
            season_int = int(season)
            item = _snapshot(group, tolerance)
            if item is not None:
                snapshots[season_int] = item
            contexts[season_int] = _profile_context(group.to_dict("records"))
            season_hands = sorted({_hand(x) for x in group["pitcher_hand"].tolist() if _hand(x) is not None})
            hands_by_season[season_int] = season_hands
            hands.update(season_hands)
        annual: dict[int, dict[str, Any]] = {}
        for season in sorted(snapshots):
            current = snapshots[season]
            if mode == "SEASON_RESET":
                annual[season] = {"share": current["share"], "support": current["support"]}
            elif mode == "CUMULATIVE":
                prior = max((s for s in snapshots if s < season), default=None)
                if prior is None:
                    continue
                previous = snapshots[prior]
                denominator = current["support"] - previous["support"]
                if denominator <= tolerance:
                    continue
                differences = tuple(current["support"] * a - previous["support"] * b for a, b in zip(current["share"], previous["share"]))
                if any(value < -tolerance for value in differences):
                    continue
                total = sum(differences)
                if total <= tolerance:
                    continue
                annual[season] = {"share": tuple(max(0.0, value / total) for value in differences), "support": denominator}
        profiles[pid] = {"annual": annual, "contexts": contexts, "hands": sorted(hands), "hands_by_season": hands_by_season, "rows": int(len(pitcher_frame)), "rows_by_season": {int(season): int(len(group)) for season, group in pitcher_frame.groupby("season", sort=True)}}
    return profiles


def build_trackman_profiles(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for pitcher, pitcher_frame in frame.groupby("pitcher_trackman_id", dropna=False, sort=True):
        pid = _id(pitcher)
        if pid is None:
            continue
        annual: dict[int, dict[str, Any]] = {}
        contexts: dict[int, dict[str, Counter[str]]] = {}
        hands: set[str] = set()
        hands_by_season: dict[int, list[str]] = {}
        for season, group in pitcher_frame.groupby("season", sort=True):
            counts = Counter(_text(value) for value in group["pitch_type_group"].tolist())
            known = {family: int(counts.get(family, 0)) for family in FAMILIES}
            known_total = sum(known.values())
            if known_total > 0:
                annual[int(season)] = {"share": tuple(known[family] / known_total for family in FAMILIES), "support": known_total, "other_count": int(counts.get("other", 0) or 0)}
            contexts[int(season)] = _profile_context(group.to_dict("records"))
            season_hands = sorted({_hand(x) for x in group["pitcher_hand"].tolist() if _hand(x) is not None})
            hands_by_season[int(season)] = season_hands
            hands.update(season_hands)
        profiles[pid] = {"annual": annual, "contexts": contexts, "hands": sorted(hands), "hands_by_season": hands_by_season, "rows": int(len(pitcher_frame)), "rows_by_season": {int(season): int(len(group)) for season, group in pitcher_frame.groupby("season", sort=True)}}
    return profiles


def assess_taxonomy_compatibility(main: pd.DataFrame, trackman: pd.DataFrame, config: Mapping[str, Any], tolerance: float | None = None) -> dict[str, Any]:
    tolerance = float(tolerance if tolerance is not None else config["taxonomy"]["share_sum_tolerance"])
    rate_columns = [f"asof_pitcher_{family}_rate" for family in FAMILIES]
    rates = main[rate_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = rates.notna().all(axis=1) & rates.ge(0).all(axis=1) & rates.le(1).all(axis=1)
    valid_sums = rates.loc[valid_mask].sum(axis=1)
    valid = int(valid_mask.sum())
    max_sum_error = float((valid_sums - 1.0).abs().max()) if not valid_sums.empty else None
    raw_groups = trackman["pitch_type_group"].map(_text)
    groups = {str(value) for value in raw_groups.dropna().unique()}
    required = set(FAMILIES)
    known = bool(required.issubset(groups))
    docs_ok = bool(config["taxonomy"].get("documentation_evidence"))
    sum_ok = bool(valid >= int(config["taxonomy"]["minimum_complete_main_rows"]) and max_sum_error is not None and max_sum_error <= tolerance)
    expected = set(config["taxonomy"].get("expected_trackman_groups", [*FAMILIES, "other"]))
    unexpected = sorted(groups - expected)
    group_counts = raw_groups.value_counts(dropna=False)
    known_rows = int(sum(int(group_counts.get(family, 0)) for family in FAMILIES))
    other_rows = int(group_counts.get("other", 0))
    missing_rows = int(raw_groups.isna().sum())
    source_proof = config["taxonomy"].get("source_level_denominator_proof", {})
    official_proof_ok = source_proof.get("status") == "PROVEN" and source_proof.get("known_only_denominator_proven") is True
    observational_rule = source_proof.get("observational_rule", {})
    observational_reasons: list[str] = []
    if not observational_rule.get("enabled"):
        observational_reasons.append("observational_rule_disabled")
    if observational_rule.get("requires_main_complete_three_family_sum") and not sum_ok:
        observational_reasons.append("main_complete_three_family_denominator_not_proven")
    if observational_rule.get("requires_all_canonical_families") and not known:
        observational_reasons.append("canonical_family_support_missing")
    if observational_rule.get("requires_unexpected_groups_absent") and unexpected:
        observational_reasons.append("unexpected_trackman_groups_present")
    if observational_rule.get("requires_other_rows_zero") and other_rows != 0:
        observational_reasons.append("trackman_other_rows_nonzero")
    if observational_rule.get("requires_missing_rows_zero") and missing_rows != 0:
        observational_reasons.append("trackman_missing_rows_nonzero")
    observational_proof_ok = not observational_reasons and known_rows >= int(config["taxonomy"]["minimum_trackman_known_rows"])
    proof_ok = official_proof_ok or observational_proof_ok
    mode = "exclude_other_renormalize_known" if proof_ok and known and docs_ok and sum_ok and not unexpected and known_rows >= int(config["taxonomy"]["minimum_trackman_known_rows"]) else None
    proof_status = "PROVEN" if proof_ok else "NOT_PROVEN"
    proof_channel = "official_contract" if official_proof_ok else "observational_known_only" if observational_proof_ok else None
    return {"status": "PASS" if mode else "SEMANTIC_COMPATIBILITY_NOT_PROVEN", "mode": mode, "canonical_families": list(FAMILIES), "trackman_groups_observed": sorted(groups), "unexpected_trackman_groups": unexpected, "trackman_known_rows": known_rows, "trackman_other_rows": other_rows, "trackman_missing_rows": missing_rows, "main_complete_rows": valid, "main_sum_max_abs_error": max_sum_error, "tolerance_used": tolerance, "documentation_evidence": list(config["taxonomy"].get("documentation_evidence", [])), "source_level_denominator_proof": {"status": proof_status, "channel": proof_channel, "official_contract_status": source_proof.get("status"), "official_contract_known_only_denominator_proven": source_proof.get("known_only_denominator_proven"), "observational_known_only_status": "PROVEN" if observational_proof_ok else "NOT_PROVEN", "observational_reasons": observational_reasons, "basis": source_proof.get("basis"), "decision_before_matching": True}, "decision_before_matching": True, "reason": None if mode else "source-level known-only denominator proof, documented family contract, expected TrackMan taxonomy, known support, or main denominator not proven"}


def _subprofile(profile: Mapping[str, Any], before: int | None = None) -> dict[str, Any]:
    annual = {int(s): value for s, value in profile.get("annual", {}).items() if before is None or int(s) < before}
    contexts = {int(s): value for s, value in profile.get("contexts", {}).items() if before is None or int(s) < before}
    hands_by_season = {int(s): list(values) for s, values in profile.get("hands_by_season", {}).items() if before is None or int(s) < before}
    if "hands_by_season" in profile:
        hand_values = sorted({value for values in hands_by_season.values() for value in values})
        hand_status = "KNOWN" if all(len(values) == 1 for values in hands_by_season.values()) and len(hand_values) == 1 else "AMBIGUOUS" if any(len(values) > 1 for values in hands_by_season.values()) or len(hand_values) > 1 else "UNKNOWN"
    else:
        hand_values = sorted(set(map(str, profile.get("hands", []))))
        hand_status = "KNOWN" if len(hand_values) == 1 else "UNKNOWN"
    return {"annual": annual, "contexts": contexts, "hands": hand_values, "hand_status": hand_status, "hands_by_season": hands_by_season, "rows": int(profile.get("rows", 0)), "rows_by_season": {int(s): int(n) for s, n in profile.get("rows_by_season", {}).items() if before is None or int(s) < before}}


def _tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * math.fsum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def repertoire_distance(left: Mapping[str, Any], right: Mapping[str, Any], *, min_seasons: int = 2, min_deltas: int = 1) -> dict[str, Any] | None:
    common = sorted(set(left.get("annual", {})) & set(right.get("annual", {})))
    if len(common) < min_seasons:
        return None
    level = [_tv(left["annual"][s]["share"], right["annual"][s]["share"]) for s in common]
    left_delta = {s: tuple(a - b for a, b in zip(left["annual"][s]["share"], left["annual"][p]["share"])) for p, s in zip(common, common[1:])}
    right_delta = {s: tuple(a - b for a, b in zip(right["annual"][s]["share"], right["annual"][p]["share"])) for p, s in zip(common, common[1:])}
    adjacent_delta_seasons = [s for p, s in zip(common, common[1:]) if s == p + 1]
    delta = [_tv(left_delta[s], right_delta[s]) for s in adjacent_delta_seasons if s in left_delta and s in right_delta]
    if len(delta) < min_deltas:
        return None
    return {"annual_mix_tv": max(level), "trajectory_delta_half_l1": max(delta), "common_seasons": common, "distance": max(max(level), max(delta))}


def _hard_hand_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("hand_status") != "KNOWN" or right.get("hand_status") != "KNOWN":
        return False
    a, b = set(left.get("hands", [])), set(right.get("hands", []))
    return bool(a.intersection(b))


def _assignment_key(pairs: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(a), str(b)) for a, b in pairs))


def _hash_order(values: Sequence[str], namespace: str, counter: int) -> list[str]:
    return sorted((str(value) for value in values), key=lambda value: (hashlib.sha256(f"{namespace}\0{counter}\0{value}".encode()).hexdigest(), value))


def _is_derangement(assignment: Iterable[tuple[str, str]]) -> bool:
    """Return true only when no normalized left identity maps to itself."""
    return all(str(left) != str(right) for left, right in assignment)


def generate_unique_null_transformations(left_ids: Sequence[str], right_ids: Sequence[str], *, requested: int, namespace: str, forbidden: Iterable[tuple[tuple[str, str], ...]] = (), require_derangement: bool = False, forbidden_partners: Mapping[str, str] | None = None, allowed_pairs: Mapping[str, Iterable[str]] | None = None) -> dict[str, Any]:
    left, right = sorted(set(map(str, left_ids))), sorted(set(map(str, right_ids)))
    pair_count = min(len(left), len(right))
    canonical = _assignment_key(zip(left[:pair_count], right[:pair_count]))
    forbidden_keys = {_assignment_key(item) for item in forbidden}
    partner_map = {str(key): str(value) for key, value in (forbidden_partners or {}).items()}
    allowed = {str(key): {str(value) for value in values} for key, values in (allowed_pairs or {}).items()}

    def valid(item: tuple[tuple[str, str], ...]) -> bool:
        if require_derangement and not _is_derangement(item):
            return False
        if allowed_pairs is not None and any(right not in allowed.get(left, set()) for left, right in item):
            return False
        return all(partner_map.get(left) != right for left, right in item)

    assignments: list[tuple[tuple[str, str], ...]] = []
    if pair_count <= 8:
        candidates = []
        if len(left) <= len(right):
            candidates = [_assignment_key(zip(left, perm)) for perm in itertools.permutations(right, pair_count)]
        else:
            candidates = [_assignment_key(zip(chosen, right)) for chosen in itertools.permutations(left, pair_count)]
        valid_candidates = {item for item in candidates if valid(item)}
        canonical_valid = canonical in valid_candidates
        seen = set(forbidden_keys)
        if canonical_valid:
            seen.add(canonical)
        for item in sorted(valid_candidates, key=lambda x: hashlib.sha256((namespace + "\0" + repr(x)).encode()).hexdigest()):
            if item not in seen:
                seen.add(item); assignments.append(item)
                if len(assignments) >= requested:
                    break
        finite = len(valid_candidates)
    else:
        seen = set(forbidden_keys)
        if valid(canonical):
            seen.add(canonical)
        for counter in range(max(256, requested * 50)):
            item = _assignment_key(zip(_hash_order(left, namespace + "/left", counter)[:pair_count], _hash_order(right, namespace + "/right", counter)[:pair_count]))
            if not valid(item):
                continue
            if item not in seen:
                seen.add(item); assignments.append(item)
                if len(assignments) >= requested:
                    break
        finite = None
    return {"assignments": assignments, "requested": int(requested), "generated_unique": len(assignments), "pair_count": pair_count, "finite_space": finite, "exhausted_space": finite is not None and len(seen) >= finite, "trial_unit": "unique_null_transformation_derangement" if require_derangement else "unique_null_transformation", "canonical_rejected": True, "require_derangement": bool(require_derangement), "forbidden_partner_map_applied": bool(partner_map), "allowed_pair_universe_applied": allowed_pairs is not None, "namespace": namespace}


def wilson_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float | None:
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    z = WILSON_Z_95 if confidence == 0.95 else 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return float(min(1.0, center + half))


def null_false_accept_summary(outcomes: Sequence[bool], *, requested: int, generated_unique: int, exhausted_space: bool, pair_count: int, accepted_counts: Sequence[int] | None = None) -> dict[str, Any]:
    successes = sum(bool(value) for value in outcomes)
    trials = len(outcomes)
    counts = list(accepted_counts or [])
    p95 = float(pd.Series(counts).quantile(0.95, interpolation="linear")) if counts else None
    ucb = wilson_upper_bound(successes, trials)
    histogram = {str(key): int(value) for key, value in sorted(Counter(int(count) for count in counts).items())}
    return {"false_accept_count": successes, "unique_trials": trials, "requested_trials": requested, "false_accept_rate_proxy": successes / trials if trials else None, "wilson_95_upper_bound_proxy": ucb, "passes_1pct_ceiling": bool(ucb is not None and ucb <= NULL_RISK_CEILING), "accepted_count_p95": p95, "accepted_count_max": max(counts, default=None), "accepted_count_histogram": histogram, "accepted_count_distribution": {"sample_count": len(counts), "histogram": histogram}, "trial_unit": "unique_null_transformation_familywise_any_accept", "small_space_exhausted": exhausted_space, "pair_count": pair_count, "insufficient_unique_trials": trials == 0}


def _distance_for_pair(left_profiles: Mapping[str, Mapping[str, Any]], right_profiles: Mapping[str, Mapping[str, Any]], pair: tuple[str, str], minimum: Mapping[str, int]) -> dict[str, Any] | None:
    return repertoire_distance(left_profiles[pair[0]], right_profiles[pair[1]], min_seasons=int(minimum["seasons"]), min_deltas=int(minimum["deltas"]))


def _distance_scalar(value: Any) -> float:
    return float(value["distance"]) if isinstance(value, Mapping) else float(value)


def assigned_pair_evidence(left_id: str, assigned_right_id: str, distances: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the assigned pair, never a row-global best/second margin."""
    ordered = sorted(((str(right), _distance_scalar(value)) for right, value in distances.items()), key=lambda item: (item[1], item[0]))
    assigned = next((distance for right, distance in ordered if right == str(assigned_right_id)), None)
    if assigned is None or len(ordered) < 2:
        return {"unique_top1": False, "eligible": False, "assigned_distance": assigned, "margin": None}
    best_right, best_distance = ordered[0]
    unique_top1 = best_right == str(assigned_right_id) and abs(ordered[1][1] - best_distance) > 1e-12
    margin = float(ordered[1][1] - assigned) if unique_top1 else None
    return {"unique_top1": unique_top1, "eligible": bool(unique_top1 and margin is not None and margin > 0.0), "assigned_distance": assigned, "margin": margin}


def calibrate_selection_null(left_profiles: Mapping[str, Mapping[str, Any]], right_profiles: Mapping[str, Mapping[str, Any]], *, requested: int = 1000, namespace: str = NULL_CAL_NS, minimum: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Freeze separate channel thresholds and a null-derived margin threshold."""
    minimum = minimum or {"seasons": 2, "deltas": 1}
    left, right = sorted(left_profiles), sorted(right_profiles)
    distances: dict[str, dict[str, Any]] = {}
    for main_id in left:
        row: dict[str, Any] = {}
        for tm_id in right:
            if not _hard_hand_compatible(left_profiles[main_id], right_profiles[tm_id]):
                continue
            value = _distance_for_pair(left_profiles, right_profiles, (main_id, tm_id), minimum)
            if value is not None:
                row[tm_id] = value
        distances[main_id] = row
    # Nulls use the exact same pre-origin hard-hand finite candidate universe
    # as the observed fit matrix.  Impossible cross-hand pairs can never form
    # a calibration or evaluation trial.
    allowed_pairs = {main_id: tuple(sorted(row)) for main_id, row in distances.items()}
    cal = generate_unique_null_transformations(left, right, requested=requested, namespace=namespace + "/selection", allowed_pairs=allowed_pairs)
    annual_stats: list[float] = []
    delta_stats: list[float] = []
    margin_stats: list[float] = []
    for assignment in cal["assignments"]:
        values = [distances[a].get(b) for a, b in assignment if b in distances.get(a, {})]
        values = [value for value in values if value is not None]
        pair_margins = [
            float(evidence["margin"])
            for a, b in assignment
            for evidence in [assigned_pair_evidence(a, b, distances.get(a, {}))]
            if evidence["eligible"]
        ]
        margin_stats.append(max(pair_margins, default=0.0))
        if values:
            annual_stats.append(min(float(value["annual_mix_tv"]) for value in values))
            delta_stats.append(min(float(value["trajectory_delta_half_l1"]) for value in values))
    annual_threshold = float(pd.Series(annual_stats).quantile(NULL_RISK_CEILING, interpolation="linear")) if annual_stats else None
    delta_threshold = float(pd.Series(delta_stats).quantile(NULL_RISK_CEILING, interpolation="linear")) if delta_stats else None
    margin_threshold = float(pd.Series(margin_stats).quantile(1.0 - NULL_RISK_CEILING, interpolation="linear")) if margin_stats else None
    evaluation = generate_unique_null_transformations(left, right, requested=requested, namespace=NULL_EVAL_NS + "/selection", forbidden=cal["assignments"], allowed_pairs=allowed_pairs)
    outcomes: list[bool] = []
    accepted_counts: list[int] = []
    for assignment in evaluation["assignments"]:
        accepted = 0
        for main_id, tm_id in assignment:
            value = distances.get(main_id, {}).get(tm_id)
            if value is None or annual_threshold is None or delta_threshold is None:
                continue
            evidence = assigned_pair_evidence(main_id, tm_id, distances.get(main_id, {}))
            if float(value["annual_mix_tv"]) <= annual_threshold and float(value["trajectory_delta_half_l1"]) <= delta_threshold and evidence["eligible"] and margin_threshold is not None and float(evidence["margin"]) >= margin_threshold:
                accepted += 1
        accepted_counts.append(accepted)
        outcomes.append(accepted > 0)
    evaluation_summary = null_false_accept_summary(outcomes, requested=requested, generated_unique=evaluation["generated_unique"], exhausted_space=evaluation["exhausted_space"], pair_count=evaluation["pair_count"], accepted_counts=accepted_counts)
    return {"annual_mix_threshold": annual_threshold, "trajectory_delta_threshold": delta_threshold, "second_best_margin_threshold": margin_threshold, "threshold_quantile_risk": NULL_RISK_CEILING, "calibration": {k: v for k, v in cal.items() if k != "assignments"}, "evaluation": evaluation_summary, "calibration_statistics_count": len(annual_stats), "calibration_evaluation_disjoint": not bool(set(map(repr, cal["assignments"])) & set(map(repr, evaluation["assignments"]))), "hand_eligible_candidate_universe": True, "allowed_pair_count": sum(len(value) for value in allowed_pairs.values()), "familywise_statistic": "minimum false-pair channel statistic per unique transformation; maximum margin statistic", "margin_definition": "second-best distance minus assigned distance only for unique top-1 assigned pairs", "thresholds_null_derived": True}


def _sorted_distance_items(values: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    return sorted(values.items(), key=lambda item: (float(item[1]["distance"]), item[0]))


def fit_repertoire_selection_map(main_profiles: Mapping[str, Mapping[str, Any]], trackman_profiles: Mapping[str, Mapping[str, Any]], *, taxonomy: Mapping[str, Any], calibration: Mapping[str, Any], minimum: Mapping[str, int] | None = None) -> dict[str, Any]:
    if taxonomy.get("status") != "PASS":
        return {"selection_map": {}, "statuses": {"HIGH_CONFIDENCE": 0, "AMBIGUOUS": 0, "UNMATCHED": len(main_profiles)}, "mapping_hash": canonical_hash([]), "frozen_before_verification": True, "reason": "SEMANTIC_COMPATIBILITY_NOT_PROVEN"}
    minimum = minimum or {"seasons": 2, "deltas": 1}
    annual_threshold = calibration.get("annual_mix_threshold")
    delta_threshold = calibration.get("trajectory_delta_threshold")
    margin_threshold = calibration.get("second_best_margin_threshold")
    if annual_threshold is None or delta_threshold is None or margin_threshold is None:
        return {"selection_map": {}, "statuses": {"HIGH_CONFIDENCE": 0, "AMBIGUOUS": 0, "UNMATCHED": len(main_profiles)}, "mapping_hash": selection_map_hash({}), "frozen_before_verification": True, "reason": "NULL_THRESHOLD_UNAVAILABLE", "annual_mix_threshold": annual_threshold, "trajectory_delta_threshold": delta_threshold, "second_best_margin_threshold": margin_threshold}
    # Build the complete hand-compatible finite matrix first.  Thresholding
    # before ranking would remove just-outside candidates and inflate margins.
    scores: dict[str, dict[str, Any]] = {}
    for main_id, main_profile in sorted(main_profiles.items()):
        candidates: dict[str, Mapping[str, Any]] = {}
        for tm_id, tm_profile in sorted(trackman_profiles.items()):
            if not _hard_hand_compatible(main_profile, tm_profile):
                continue
            distance = repertoire_distance(main_profile, tm_profile, min_seasons=minimum["seasons"], min_deltas=minimum["deltas"])
            if distance is not None:
                candidates[tm_id] = distance
        scores[main_id] = dict(candidates)
    reverse: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for main_id, candidates in scores.items():
        for tm_id, value in candidates.items():
            reverse[tm_id].append((main_id, value))
    reverse_top: dict[str, tuple[str | None, bool]] = {}
    for tm_id, candidates in reverse.items():
        ordered = sorted(candidates, key=lambda item: (float(item[1]["distance"]), item[0]))
        tie = len(ordered) > 1 and abs(float(ordered[1][1]["distance"]) - float(ordered[0][1]["distance"])) <= 1e-12
        reverse_top[tm_id] = (ordered[0][0] if ordered and not tie else None, tie)
    mapping: dict[str, str] = {}
    statuses = Counter()
    for main_id in sorted(scores):
        ordered = _sorted_distance_items(scores[main_id])
        if not ordered:
            statuses["UNMATCHED"] += 1; continue
        best_id, best = ordered[0]
        tie = len(ordered) > 1 and abs(float(ordered[1][1]["distance"]) - float(best["distance"])) <= 1e-12
        evidence = assigned_pair_evidence(main_id, best_id, scores[main_id])
        reverse_main, reverse_tie = reverse_top.get(best_id, (None, False))
        mutual = reverse_main == main_id and not reverse_tie
        channels_pass = float(best["annual_mix_tv"]) <= annual_threshold and float(best["trajectory_delta_half_l1"]) <= delta_threshold
        if tie or reverse_tie or not channels_pass or not evidence["eligible"] or evidence["margin"] < margin_threshold or not mutual:
            statuses["AMBIGUOUS"] += 1
        else:
            mapping[main_id] = best_id; statuses["HIGH_CONFIDENCE"] += 1
    return {"selection_map": mapping, "statuses": {key: int(statuses.get(key, 0)) for key in ("HIGH_CONFIDENCE", "AMBIGUOUS", "UNMATCHED")}, "mapping_hash": selection_map_hash(mapping), "frozen_before_verification": True, "annual_mix_threshold": annual_threshold, "trajectory_delta_threshold": delta_threshold, "second_best_margin_threshold": margin_threshold, "candidate_count": sum(len(value) for value in scores.values())}


def _distribution_tv(left: Mapping[str, int], right: Mapping[str, int]) -> float | None:
    lt, rt = sum(left.values()), sum(right.values())
    if lt <= 0 or rt <= 0:
        return None
    keys = sorted(set(left) | set(right), key=repr)
    return float(0.5 * math.fsum(abs(left.get(key, 0) / lt - right.get(key, 0) / rt) for key in keys))


def _context_stat(selection_map: Mapping[str, str], main_profiles: Mapping[str, Mapping[str, Any]], trackman_profiles: Mapping[str, Mapping[str, Any]], origin: int, *, right_assignment: Mapping[str, str] | None = None) -> dict[str, Any]:
    scores: list[float] = []
    per_field: dict[str, list[float]] = {field: [] for field in VERIFIER_CHANNELS}
    for main_id, original_tm_id in sorted(selection_map.items()):
        tm_id = right_assignment.get(main_id, original_tm_id) if right_assignment is not None else original_tm_id
        left = main_profiles.get(main_id, {}).get("contexts", {}).get(origin, {})
        right = trackman_profiles.get(tm_id, {}).get("contexts", {}).get(origin, {})
        if not left or not right:
            continue
        values = {field: _distribution_tv(left.get(field, Counter()), right.get(field, Counter())) for field in VERIFIER_CHANNELS}
        finite = [float(value) for value in values.values() if value is not None]
        for field, value in values.items():
            if value is not None:
                per_field[field].append(float(value))
        if finite:
            scores.append(max(finite))
    channel_statistics = {field: float(pd.Series(values).median()) if values else None for field, values in per_field.items()}
    return {"verified_pairs": len(scores), "values": scores, "distributions": {field: {"count": len(values), "median_tv": channel_statistics[field]} for field, values in per_field.items()}, "channel_statistics": channel_statistics, "statistic": float(pd.Series(scores).median()) if scores else None}


def verify_selection_map_oop(selection_map: Mapping[str, str], main_profiles: Mapping[str, Mapping[str, Any]], trackman_profiles: Mapping[str, Mapping[str, Any]], origin: int, *, requested: int = 1000) -> dict[str, Any]:
    """Method-level held-out context audit; mapping is immutable and never returned."""
    map_hash = selection_map_hash(selection_map)
    observed = _context_stat(selection_map, main_profiles, trackman_profiles, origin)
    left_ids = sorted(selection_map)
    right_ids = sorted(selection_map.values())
    pre_main = {identity: _subprofile(profile, origin) for identity, profile in main_profiles.items()}
    pre_trackman = {identity: _subprofile(profile, origin) for identity, profile in trackman_profiles.items()}
    allowed_pairs = {left: tuple(sorted(right for right in right_ids if _hard_hand_compatible(pre_main.get(left, {}), pre_trackman.get(right, {})))) for left in left_ids}
    calibration = generate_unique_null_transformations(left_ids, right_ids, requested=requested, namespace=f"{CONTRACT_VERSION}/verification/{origin}/calibration", forbidden_partners=selection_map, allowed_pairs=allowed_pairs)
    calibration_channel_stats: dict[str, list[float]] = {field: [] for field in VERIFIER_CHANNELS}
    for assignment in calibration["assignments"]:
        assignment_map = {left: right for left, right in assignment}
        channel_values = _context_stat(selection_map, main_profiles, trackman_profiles, origin, right_assignment=assignment_map)["channel_statistics"]
        for field, value in channel_values.items():
            if value is not None:
                calibration_channel_stats[field].append(float(value))
    channel_thresholds = {field: float(pd.Series(values).quantile(NULL_RISK_CEILING, interpolation="linear")) if values else None for field, values in calibration_channel_stats.items()}
    evaluation = generate_unique_null_transformations(left_ids, right_ids, requested=requested, namespace=f"{CONTRACT_VERSION}/verification/{origin}/evaluation", forbidden=calibration["assignments"], forbidden_partners=selection_map, allowed_pairs=allowed_pairs)
    outcomes: list[bool] = []
    accepted_counts: list[int] = []
    for assignment in evaluation["assignments"]:
        assignment_map = {left: right for left, right in assignment}
        channel_values = _context_stat(selection_map, main_profiles, trackman_profiles, origin, right_assignment=assignment_map)["channel_statistics"]
        accepted = int(all(channel_values[field] is not None and channel_thresholds[field] is not None and float(channel_values[field]) <= float(channel_thresholds[field]) for field in VERIFIER_CHANNELS))
        accepted_counts.append(accepted)
        outcomes.append(bool(accepted))
    null = null_false_accept_summary(outcomes, requested=requested, generated_unique=evaluation["generated_unique"], exhausted_space=evaluation["exhausted_space"], pair_count=evaluation["pair_count"], accepted_counts=accepted_counts)
    observed_channels = observed["channel_statistics"]
    channel_pass = {field: bool(observed_channels[field] is not None and channel_thresholds[field] is not None and float(observed_channels[field]) <= float(channel_thresholds[field])) for field in VERIFIER_CHANNELS}
    observed_channels_pass = bool(all(channel_pass.values()))
    method_pass = bool(observed_channels_pass and all(value is not None for value in channel_thresholds.values()) and null["passes_1pct_ceiling"] and null["unique_trials"] > 0)
    return {"method_level_only": True, "selection_map_hash_before": map_hash, "selection_map_hash_after": map_hash, "verifier_confirmed_manifest": False, "required_channels": list(VERIFIER_CHANNELS), "observed": {key: value for key, value in observed.items() if key != "values"}, "observed_channel_pass": channel_pass, "all_required_channels_pass": observed_channels_pass, "null": {"channel_thresholds": channel_thresholds, "calibration_unique_trials": len(calibration["assignments"]), "calibration_channel_statistics_count": {field: len(values) for field, values in calibration_channel_stats.items()}, "evaluation": null, "calibration_evaluation_disjoint": not bool(set(map(repr, calibration["assignments"])) & set(map(repr, evaluation["assignments"]))), "forbidden_partner_map_enforced": calibration["forbidden_partner_map_applied"] and evaluation["forbidden_partner_map_applied"], "allowed_pair_universe_applied": calibration["allowed_pair_universe_applied"] and evaluation["allowed_pair_universe_applied"], "null_trials_preserve_selected_partners": all(all(selection_map.get(left) != right for left, right in assignment) for assignment in (*calibration["assignments"], *evaluation["assignments"]))}, "distributions": observed["distributions"], "method_status": "PASS" if method_pass else "FAIL", "mapping_immutable": True}


def self_id_acceptance_gate(correct_accepted: int, wrong_accepted: int, eligible: int, null_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Require observed correct acceptance to exceed the frozen null envelope."""
    evaluation = null_contract.get("evaluation", {})
    p95 = evaluation.get("accepted_count_p95")
    ucb = evaluation.get("wilson_95_upper_bound_proxy")
    envelope_available = bool(p95 is not None and ucb is not None and eligible > 0)
    count_envelope = max(float(p95), float(ucb) * int(eligible)) if envelope_available else None
    correct_exceeds = bool(count_envelope is not None and int(correct_accepted) > count_envelope)
    observed_coverage = float(correct_accepted) / int(eligible) if eligible else None
    null_coverage = float(count_envelope) / int(eligible) if count_envelope is not None and eligible else None
    coverage_exceeds = bool(observed_coverage is not None and null_coverage is not None and observed_coverage > null_coverage)
    passes = bool(envelope_available and null_contract.get("passes_1pct_ceiling") and evaluation.get("unique_trials", 0) > 0 and int(wrong_accepted) == 0 and int(correct_accepted) > 0 and correct_exceeds and coverage_exceeds)
    return {"null_accepted_count_p95": p95, "null_wilson_95_upper_bound": ucb, "null_count_envelope": count_envelope, "correct_accepted_exceeds_null_p95": bool(p95 is not None and int(correct_accepted) > float(p95)), "correct_accepted_exceeds_null_count_envelope": correct_exceeds, "observed_correct_coverage": observed_coverage, "null_coverage_envelope": null_coverage, "coverage_exceeds_null_envelope": coverage_exceeds, "envelope_available": envelope_available, "passes": passes}


def next_season_self_identification(profiles: Mapping[str, Mapping[str, Any]], origin: int) -> dict[str, Any]:
    """Known-ID diagnostic: reference <= Y-2, single query season Y-1."""
    query_season = origin - 1
    reference: dict[str, tuple[float, float, float]] = {}
    queries: dict[str, tuple[float, float, float]] = {}
    for identity, profile in profiles.items():
        query = profile.get("annual", {}).get(query_season)
        prior = [season for season in profile.get("annual", {}) if season <= origin - 2]
        if query is not None and prior:
            reference[identity] = profile["annual"][max(prior)]["share"]
            queries[identity] = query["share"]
    eligible_ids = sorted(set(reference).intersection(queries))
    distances: dict[str, dict[str, float]] = {}
    for identity in eligible_ids:
        query = queries[identity]
        distances[identity] = {other: _tv(query, reference[other]) for other in eligible_ids}
    null = calibrate_self_id_null(distances, eligible_ids, origin)
    records = []
    for identity, candidates_map in sorted(distances.items()):
        candidates = sorted(candidates_map.items(), key=lambda x: (x[1], x[0]))
        if not candidates:
            continue
        best = candidates[0]
        evidence = assigned_pair_evidence(identity, best[0], candidates_map)
        accepted = bool(null["threshold"] is not None and best[1] <= null["threshold"] and evidence["eligible"] and null["margin_threshold"] is not None and float(evidence["margin"]) >= null["margin_threshold"])
        records.append({"correct": accepted and best[0] == identity, "wrong": accepted and best[0] != identity, "ambiguous": not evidence["eligible"], "accepted": accepted, "distance": best[1], "candidate_count": len(candidates)})
    correct = sum(item["correct"] for item in records)
    wrong = sum(item["wrong"] for item in records)
    ambiguous = sum(item["ambiguous"] for item in records)
    gate = self_id_acceptance_gate(correct, wrong, len(records), null)
    return {"diagnostic": "next_season_repertoire_persistence", "reference_period": "seasons <= origin - 2", "query_period": f"season == {query_season}", "eligible": len(records), "accepted": sum(item["accepted"] for item in records), "correct_accepted": correct, "wrong_accepted": wrong, "ambiguous_accepted": ambiguous, "coverage": correct / len(records) if records else None, "distance": "TV between latest reference annual share and single query-season share", "not_full_multi_year_validation": True, "null_contract": null, "acceptance_gate": gate, "status": "PASS" if gate["passes"] else "FAIL"}


def calibrate_self_id_null(distances: Mapping[str, Mapping[str, float]], identities: Sequence[str], origin: int, *, requested: int = 1000) -> dict[str, Any]:
    namespace = f"{CONTRACT_VERSION}/self-id/{int(origin)}"
    eligible = sorted(set(map(str, identities)).intersection(distances))
    calibration = generate_unique_null_transformations(eligible, eligible, requested=requested, namespace=namespace + "/calibration", require_derangement=True)
    calibration_min: list[float] = []
    calibration_margin: list[float] = []
    for assignment in calibration["assignments"]:
        values = [distances.get(left, {}).get(right) for left, right in assignment if right in distances.get(left, {})]
        values = [float(value) for value in values if value is not None]
        if values:
            calibration_min.append(min(values))
        pair_margins = [
            float(evidence["margin"])
            for left, right in assignment
            for evidence in [assigned_pair_evidence(left, right, distances.get(left, {}))]
            if evidence["eligible"]
        ]
        calibration_margin.append(max(pair_margins, default=0.0))
    threshold = float(pd.Series(calibration_min).quantile(NULL_RISK_CEILING, interpolation="linear")) if calibration_min else None
    margin_threshold = float(pd.Series(calibration_margin).quantile(1.0 - NULL_RISK_CEILING, interpolation="linear")) if calibration_margin else None
    evaluation = generate_unique_null_transformations(eligible, eligible, requested=requested, namespace=namespace + "/evaluation", forbidden=calibration["assignments"], require_derangement=True)
    accepted_counts: list[int] = []
    outcomes: list[bool] = []
    for assignment in evaluation["assignments"]:
        accepted = 0
        for left, right in assignment:
            distance = distances.get(left, {}).get(right)
            evidence = assigned_pair_evidence(left, right, distances.get(left, {}))
            if distance is not None and threshold is not None and evidence["eligible"] and margin_threshold is not None and distance <= threshold and float(evidence["margin"]) >= margin_threshold:
                accepted += 1
        accepted_counts.append(accepted)
        outcomes.append(accepted > 0)
    summary = null_false_accept_summary(outcomes, requested=requested, generated_unique=evaluation["generated_unique"], exhausted_space=evaluation["exhausted_space"], pair_count=evaluation["pair_count"], accepted_counts=accepted_counts)
    return {"threshold": threshold, "margin_threshold": margin_threshold, "calibration_unique_trials": len(calibration_min), "evaluation_unique_trials": evaluation["generated_unique"], "evaluation": summary, "passes_1pct_ceiling": summary["passes_1pct_ceiling"], "calibration_evaluation_disjoint": not bool(set(map(repr, calibration["assignments"])) & set(map(repr, evaluation["assignments"]))), "calibration_require_derangement": calibration["require_derangement"], "evaluation_require_derangement": evaluation["require_derangement"], "true_derangement": all(_is_derangement(item) for item in (*calibration["assignments"], *evaluation["assignments"])), "thresholds_null_derived": True}


def self_identification_null_design(eligible_ids: Sequence[str], origin: int, *, requested: int = 1000) -> dict[str, Any]:
    """Separate self-ID null contract; it is not the cross-source null."""
    namespace = f"{CONTRACT_VERSION}/self-id/{int(origin)}"
    calibration = generate_unique_null_transformations(eligible_ids, eligible_ids, requested=requested, namespace=namespace + "/calibration", require_derangement=True)
    evaluation = generate_unique_null_transformations(eligible_ids, eligible_ids, requested=requested, namespace=namespace + "/evaluation", forbidden=calibration["assignments"], require_derangement=True)
    return {"eligibility": "reference <= origin-2 and query == origin-1", "trial_unit": "unique_identity_derangement", "calibration_namespace": namespace + "/calibration", "evaluation_namespace": namespace + "/evaluation", "calibration_unique_trials": calibration["generated_unique"], "evaluation_unique_trials": evaluation["generated_unique"], "calibration_evaluation_disjoint": not bool(set(map(repr, calibration["assignments"])) & set(map(repr, evaluation["assignments"]))), "true_derangement": all(_is_derangement(item) for item in (*calibration["assignments"], *evaluation["assignments"])), "null_false_accept_is_risk_proxy": True, "not_cross_source_contract": True}


def cross_origin_partner_stability(origin_results: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    maps = {int(origin): result.get("selection_map", {}) for origin, result in origin_results.items()}
    origins = sorted(maps)
    comparisons = []
    for left, right in itertools.combinations(origins, 2):
        shared = set(maps[left]) & set(maps[right])
        same = sum(maps[left][identity] == maps[right][identity] for identity in shared)
        comparisons.append({"left_origin": left, "right_origin": right, "shared_main_ids": len(shared), "same_partner": same, "different_partner": len(shared) - same, "stability": same / len(shared) if shared else None})
    nonempty = all(bool(maps[origin]) for origin in origins) if origins else False
    conflict_free = nonempty and bool(comparisons) and all(item["shared_main_ids"] > 0 and item["different_partner"] == 0 for item in comparisons)
    return {"comparisons": comparisons, "evidence_type": "independent_origin_mapping_partner_stability", "not_two_holdouts_of_one_frozen_mapping": True, "nonempty_origin_maps": nonempty, "conflict_free": conflict_free}


def origin_selection_coverage(selection_map: Mapping[str, str], main_profiles: Mapping[str, Mapping[str, Any]], origin: int, selection_null: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [identity for identity, profile in main_profiles.items() if int(origin) in profile.get("rows_by_season", {})]
    high_confidence = sorted(set(selection_map).intersection(eligible))
    total_rows = sum(int(main_profiles[identity].get("rows_by_season", {}).get(int(origin), 0)) for identity in eligible)
    matched_rows = sum(int(main_profiles[identity].get("rows_by_season", {}).get(int(origin), 0)) for identity in high_confidence)
    p95 = selection_null.get("evaluation", {}).get("accepted_count_p95")
    return {"eligible_main_pitchers": len(eligible), "high_confidence_pitchers": len(high_confidence), "selection_player_coverage": len(high_confidence) / len(eligible) if eligible else None, "origin_main_rows": total_rows, "high_confidence_main_rows": matched_rows, "selection_main_row_coverage": matched_rows / total_rows if total_rows else None, "null_accepted_count_p95": p95, "high_confidence_exceeds_null_count_p95": bool(p95 is not None and len(high_confidence) > p95), "selection_map_only": True, "verifier_filtered": False}


def assess_source_order_provenance(main: pd.DataFrame) -> dict[str, Any]:
    return {"status": SOURCE_ORDER_STATUS, "reason": "exact intra-month chronology is NOT_PROVEN", "source_order_used": False, "rows_checked": int(len(main))}


def level_r_prerequisite(main: pd.DataFrame) -> dict[str, Any]:
    status = assess_source_order_provenance(main)
    return {"verdict": LEVEL_R_NOT_RUN, "source_order": status, "exact_matcher_implemented": False, "reason": "Level R exact matcher is out of scope and source order is not proven"}


def _fit_rows(frame: pd.DataFrame, origin: int) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame["season"], errors="coerce").lt(origin) & frame["game_type"].astype(str).eq(GAME_TYPE)].copy()


def _origin_rows(frame: pd.DataFrame, origin: int) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(origin) & frame["game_type"].astype(str).eq(GAME_TYPE)].copy()


def run_level_p(main: pd.DataFrame, trackman: pd.DataFrame, config: Mapping[str, Any], *, precision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if precision is None:
        rate_tokens = (value for column in ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate") for value in main[column])
        precision = analyze_rate_storage_precision_tokens(rate_tokens, multiplier=float(config["materialization"]["storage_tolerance_multiplier"]), minimum=float(config["materialization"]["storage_tolerance_minimum"]))
    taxonomy = assess_taxonomy_compatibility(main, trackman, config, tolerance=float(precision["tolerance"]))
    mode = infer_main_asof_mode(main, config, float(precision["tolerance"]))
    if taxonomy["status"] != "PASS" or mode.get("mode") is None:
        return {"verdict": "P_KILL", "taxonomy": taxonomy, "materialization": {"mode": mode, "precision": precision}, "origins": {}, "cross_origin_partner_stability": {"comparisons": [], "not_two_holdouts_of_one_frozen_mapping": True}, "level_r": level_r_prerequisite(main), "reason": taxonomy.get("reason") or "MATERIALIZATION_NOT_PROVEN"}
    main_profiles_all = build_main_profiles(main, str(mode["mode"]), float(precision["tolerance"]))
    tm_profiles_all = build_trackman_profiles(trackman)
    origins: dict[int, Any] = {}
    for origin in ORIGINS:
        main_fit = {identity: _subprofile(profile, origin) for identity, profile in main_profiles_all.items()}
        tm_fit = {identity: _subprofile(profile, origin) for identity, profile in tm_profiles_all.items()}
        self_main = next_season_self_identification(main_fit, origin)
        self_tm = next_season_self_identification(tm_fit, origin)
        minimum = {"seasons": int(config["selection"]["minimum_common_seasons"]), "deltas": int(config["selection"]["minimum_common_delta_seasons"])}
        null = calibrate_selection_null(main_fit, tm_fit, requested=int(config["nulls"]["requested_calibration_transformations"]), minimum=minimum)
        selection = fit_repertoire_selection_map(main_fit, tm_fit, taxonomy=taxonomy, calibration=null, minimum=minimum)
        verifier = verify_selection_map_oop(selection["selection_map"], main_profiles_all, tm_profiles_all, origin)
        coverage = origin_selection_coverage(selection["selection_map"], main_profiles_all, origin, null)
        self_pass = self_main["status"] == "PASS" and self_tm["status"] == "PASS"
        selection_null_pass = bool(null["evaluation"].get("passes_1pct_ceiling") and null["evaluation"].get("unique_trials", 0) > 0)
        selection_gate = bool(selection["statuses"]["HIGH_CONFIDENCE"] > 0 and selection_null_pass and coverage["high_confidence_exceeds_null_count_p95"])
        verifier_gate = verifier["method_status"] == "PASS"
        origins[origin] = {"selection_map": selection["selection_map"], "selection": {key: value for key, value in selection.items() if key != "selection_map"}, "selection_null": null, "selection_gate": {"passes_1pct_ceiling": selection_null_pass, "nonempty_high_confidence": selection["statuses"]["HIGH_CONFIDENCE"] > 0, "passes": selection_gate}, "self_identification": {"main": self_main, "trackman": self_tm, "passes": self_pass}, "origin_selection_coverage": coverage, "oop_verification": verifier, "origin_verdict": "PASS" if self_pass and selection_gate and verifier_gate else "FAIL"}
    stability = cross_origin_partner_stability(origins)
    public_origins = {}
    for origin, value in origins.items():
        public_origins[str(origin)] = {key: _safe(item) for key, item in value.items() if key != "selection_map"}
    hc = sum(value["selection"]["statuses"].get("HIGH_CONFIDENCE", 0) for value in origins.values())
    self_failure = any(not value["self_identification"]["passes"] for value in origins.values())
    if self_failure:
        verdict = "P_KILL"
        reason = "within-source self-identification prerequisite failed"
    elif all(value["origin_verdict"] == "PASS" for value in origins.values()) and stability["nonempty_origin_maps"] and stability["conflict_free"]:
        verdict = "P_PASS"
        reason = None
    else:
        verdict = "P_FAIL"
        reason = "selection, method-level OOP verification, coverage, null-risk, or partner-stability gate failed"
    return {"verdict": verdict, "reason": reason, "taxonomy": taxonomy, "materialization": {"mode": mode, "precision": precision}, "origins": public_origins, "cross_origin_partner_stability": stability, "level_r": level_r_prerequisite(main), "selection_map_hashes": {str(origin): value["selection"]["mapping_hash"] for origin, value in origins.items()}, "aggregate_high_confidence": hc}


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outside_repo(output_dir: Path, repo_root: Path) -> None:
    output = output_dir.resolve()
    repo = repo_root.resolve()
    if output == repo or repo in output.parents:
        raise CrosswalkAuditError("output directory must be outside repository")


def _aggregate_report_value(value: Any, key: str | None = None) -> Any:
    """Remove row-level mappings and ID collections before report serialization."""
    if key in {"selection_map", "mapping", "manifest", "verifier_confirmed_mapping"}:
        return None
    if isinstance(value, Mapping):
        result = {}
        for child_key, child_value in value.items():
            child_name = str(child_key)
            if child_name in {"selection_map", "mapping", "manifest", "verifier_confirmed_mapping"}:
                continue
            if child_name.endswith("_ids") and isinstance(child_value, (list, tuple, set, dict)):
                continue
            result[child_name] = _aggregate_report_value(child_value, child_name)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_aggregate_report_value(item) for item in value]
    return value


def build_report(level_p: Mapping[str, Any], *, main: pd.DataFrame, trackman: pd.DataFrame, repo_root: Path, config_path: Path, script_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_match = _contract_projection(config) == _runtime_contract()
    report = {"contract_version": CONTRACT_VERSION, "git_sha": _git_sha(repo_root), "runner_sha256": _sha256(script_path), "config_sha256": _sha256(config_path), "source_frame_hash_algorithm": FRAME_HASH_ALGORITHM, "main_source_frame_sha256": canonical_frame_hash(main), "trackman_source_frame_sha256": canonical_frame_hash(trackman), "main_row_count": int(len(main)), "trackman_row_count": int(len(trackman)), "runtime_contract": {"config_runtime_match": bool(runtime_match), "runtime_contract_sha256": canonical_hash(_runtime_contract()), "config_documentary_snapshot_checked": bool(config.get("documentary_snapshot_checked_against_runtime_contract"))}, "scope": {"target_access": False, "target_column_in_projection": False, "target_2022_access": False, "target_2023_access": False, "target_2024_access": False, "test_access": False, "test_distribution_access": False, "public_access": False, "public_leaderboard_evidence": False, "leaderboard_access": False, "external_access": False, "external_information_access": False, "trackman_pitch_type_group_historical_access": True, "trackman_physics_access": False, "trackman_physics_current_pitch_access": False, "source_order_used": False, "model_training": False, "model_training_or_scoring": False, "label_access_ledger": [], "branch_inert": True}, "taxonomy": _aggregate_report_value(level_p.get("taxonomy")), "materialization": _aggregate_report_value(level_p.get("materialization")), "origins": _aggregate_report_value(level_p.get("origins", {})), "cross_origin_partner_stability": _aggregate_report_value(level_p.get("cross_origin_partner_stability")), "selection_map_hashes": _aggregate_report_value(level_p.get("selection_map_hashes", {})), "level_r": _aggregate_report_value(level_p.get("level_r")), "verdict": level_p.get("verdict"), "privacy": {"aggregate_only": True, "raw_mapping_ids": False, "target_values": False, "physics": False, "row_level_manifest": False}, "selection_evidence": "repertoire_only", "verification_evidence": "held_out_non_repertoire_context_method_only", "mapping_not_modified_by_verifier": True, "verifier_confirmed_manifest": False}
    report["canonical_report_sha256"] = canonical_report_hash(report)
    return report


def write_report(report: Mapping[str, Any], output_dir: str | Path, *, repo_root: Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    _outside_repo(output, repo_root)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "trackman_repertoire_crosswalk_v2_report.json"
    md_path = output / "trackman_repertoire_crosswalk_v2_report.md"
    safe = _safe(report)
    json_path.write_text(json.dumps(safe, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# TrackMan repertoire crosswalk v2", "", f"- verdict: `{safe.get('verdict')}`", f"- canonical_report_sha256: `{safe.get('canonical_report_sha256')}`", "- target_access: `false`", "- test_distribution_access: `false`", "- model_training_or_scoring: `false`", "- selection map is frozen before method-level verification.", "- Level R exact matching: `NOT_RUN_SOURCE_ORDER_NOT_PROVEN`."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_from_paths(main_csv: str | Path, trackman_csv: str | Path, output_dir: str | Path, repo_root: str | Path) -> tuple[dict[str, Any], tuple[Path, Path]]:
    repo = Path(repo_root).resolve()
    config_path = repo / "configs" / "trackman_repertoire_crosswalk_v2.json"
    config = load_contract_config(repo)
    discovery = discover_main_source_positions(main_csv, seasons=config["scope"]["main_seasons"], game_type=config["scope"]["game_type"])
    main = read_scoped_main_features(main_csv, discovery["__source_position"].tolist(), seasons=config["scope"]["main_seasons"], game_type=config["scope"]["game_type"])
    tm_discovery = discover_trackman_source_positions(trackman_csv, seasons=config["scope"]["main_seasons"])
    trackman = read_scoped_trackman_features(trackman_csv, tm_discovery["__source_position"].tolist(), seasons=config["scope"]["main_seasons"])
    precision = probe_rate_storage_precision(main_csv, multiplier=float(config["materialization"]["storage_tolerance_multiplier"]), minimum=float(config["materialization"]["storage_tolerance_minimum"]))
    level_p = run_level_p(main, trackman, config, precision=precision)
    report = build_report(level_p, main=main, trackman=trackman, repo_root=repo, config_path=config_path, script_path=Path(__file__).resolve(), config=config)
    paths = write_report(report, output_dir, repo_root=repo)
    return report, paths


def static_contract(repo_root: str | Path) -> dict[str, Any]:
    config = load_contract_config(Path(repo_root))
    projection = _contract_projection(config)
    if TARGET in MAIN_PROJECTION or TARGET in TRACKMAN_PROJECTION:
        raise CrosswalkAuditError("target appears in an approved projection")
    if FORBIDDEN_TRACKMAN.intersection(TRACKMAN_PROJECTION):
        raise CrosswalkAuditError("forbidden TrackMan physics/type appears in projection")
    selection_fields = set(["pitch_type_group", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"])
    verifier_fields = set(config["verification"]["columns"])
    if selection_fields.intersection(verifier_fields):
        raise CrosswalkAuditError("selection and verifier evidence columns overlap")
    if config["verification"]["verifier_can_change_mapping"] or config["verification"]["verifier_confirmed_manifest"]:
        raise CrosswalkAuditError("verifier is allowed to change mapping or emit manifest")
    if config["level_r"]["exact_matcher_implemented"]:
        raise CrosswalkAuditError("Level R matcher must remain disabled")
    return {"contract_version": CONTRACT_VERSION, "config_valid": True, "target_access": False, "target_column_in_projection": False, "target_2022_access": False, "target_2023_access": False, "target_2024_access": False, "test_access": False, "test_distribution_access": False, "public_access": False, "public_leaderboard_evidence": False, "leaderboard_access": False, "external_access": False, "external_information_access": False, "trackman_pitch_type_group_historical_access": True, "trackman_physics_access": False, "trackman_physics_current_pitch_access": False, "source_order_used": False, "model_training": False, "model_training_or_scoring": False, "branch_inert": True, "label_access_ledger": [], "selection_verifier_columns_disjoint": True, "mapping_immutable": True, "verifier_confirmed_manifest": False, "level_r_exact_matcher": False, "source_order_status": SOURCE_ORDER_STATUS, "projection_columns": projection["scope"]["main_projection"], "aggregate_only": True, "row_level_manifest": False, "static_pass": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static")
    static.add_argument("--repo-root", required=True)
    run = sub.add_parser("run")
    run.add_argument("--main-csv", "--train-csv", dest="main_csv", required=True)
    run.add_argument("--trackman-csv", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--repo-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "static":
            print(json.dumps(static_contract(args.repo_root), sort_keys=True, allow_nan=False))
        else:
            report, paths = run_from_paths(args.main_csv, args.trackman_csv, args.output_dir, args.repo_root)
            print(json.dumps({"verdict": report["verdict"], "canonical_report_sha256": report["canonical_report_sha256"], "json": str(paths[0]), "markdown": str(paths[1])}, sort_keys=True, allow_nan=False))
        return 0
    except (CrosswalkAuditError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
