#!/usr/bin/env python3
"""Model-free, entity-free historical TrackMan context-prior audit.

The runner is deliberately independent of the TrackMan player crosswalk
audits.  It aggregates historical ``pitch_type_group`` rows by legal game
context and uses only strict prior-season history for a main row.  It never
reads a target, an identifier, a test distribution, or a current-pitch
measurement.  The ``run`` command is an official-data audit entry point; this
module's tests and ``static`` command do not open official data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CONTRACT_VERSION = "aimers9-trackman-crosswalk-free-context-priors-v1"
TARGET = "control_success"
MAIN_SEASONS = tuple(range(2019, 2025))
INFERENCE_SEASONS = (2025,)
TRACKMAN_HISTORICAL_SEASONS = tuple(range(2019, 2025))
MAIN_PROJECTION = ("row_id", "season", "balls_before", "strikes_before", "outs_before")
TRACKMAN_PROJECTION = ("season", "balls_before", "strikes_before", "outs_before", "pitch_type_group")
CONTEXT_COLUMNS = ("balls_before", "strikes_before", "outs_before")
L0_COLUMNS = ("balls_before", "strikes_before", "outs_before")
L1_COLUMNS = ("balls_before", "strikes_before")
FAMILIES = ("fastball", "breaking", "offspeed", "other")
FEATURE_COLUMNS = (
    "tm_cf_p_fastball", "tm_cf_p_breaking", "tm_cf_p_offspeed",
    "tm_cf_p_other", "tm_cf_entropy_norm4", "tm_cf_support",
    "tm_cf_context_vs_global_tv",
)
SUPPORT_THRESHOLD = 100
SIMPLEX_TOLERANCE = 1e-9
FRAME_HASH_ALGORITHM = "canonical-jsonl-v1"
LEVELS = ("L0", "L1", "global", "NO_HISTORY")
FAMILY_FORBIDDEN_MAIN = frozenset({TARGET, "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"})
FAMILY_FORBIDDEN_TRACKMAN = frozenset({
    "pitcher_trackman_id", "batter_trackman_id", "trackman_id", "game_id",
    "trackman_game_id", "pitch_no", "tagged_pitch_type", "auto_pitch_type",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
    "rel_height", "rel_side", "zone_speed", "release", "movement",
})
EXPECTED_GROUPS = frozenset(FAMILIES)


class ContextPriorAuditError(RuntimeError):
    """Fail-closed contract, schema, domain, or privacy error."""


class ContextDomainNotProven(ContextPriorAuditError):
    """Observed required context is missing or outside its frozen domain."""


class TaxonomyNotProven(ContextPriorAuditError):
    """Observed TrackMan pitch-family taxonomy is not frozen-compatible."""


class TemporalCausalityNotProven(ContextPriorAuditError):
    """Observed season values cannot be admitted to the frozen history scope."""


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextPriorAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _runtime_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "scope": {
            "main_seasons": list(MAIN_SEASONS), "inference_seasons": list(INFERENCE_SEASONS),
            "main_projection": list(MAIN_PROJECTION), "trackman_projection": list(TRACKMAN_PROJECTION),
            "forbidden_main": sorted(FAMILY_FORBIDDEN_MAIN), "forbidden_trackman": sorted(FAMILY_FORBIDDEN_TRACKMAN),
            "target_free": True, "test_free": True, "leaderboard_free": True,
            "external_free": True, "future_safe": True,
        },
        "context": {
            "required_columns": list(CONTEXT_COLUMNS),
            "domains": {"balls_before": [0, 3], "strikes_before": [0, 2], "outs_before": [0, 2]},
            "integer_like": True, "invalid_domain_policy": "CONTEXT_DOMAIN_NOT_PROVEN",
            "levels": list(LEVELS[:3]), "L0": list(L0_COLUMNS), "L1": list(L1_COLUMNS),
            "support_threshold": SUPPORT_THRESHOLD,
            "sparse_context_policy": "legal_sparse_context_falls_back_L0_to_L1_to_global",
            "backoff_level_audit_only": True,
        },
        "taxonomy": {
            "families": list(FAMILIES), "other_policy": "retain_as_fourth_family",
            "unexpected_or_missing_policy": "TAXONOMY_NOT_PROVEN", "redistribution": False,
            "simplex_tolerance": SIMPLEX_TOLERANCE,
        },
        "temporal": {
            "rule": "main season S uses TrackMan season < S", "prior_season_strict": True,
            "same_season_forbidden": True, "future_season_forbidden": True,
            "no_history_policy": "missing_probabilities_support_zero_level_NO_HISTORY",
            "season_order_source": "season_column_only",
        },
        "features": {
            "columns": list(FEATURE_COLUMNS), "backoff_level_not_feature": True,
            "probability_formula": "selected_level_family_count / selected_level_total_count",
            "entropy_formula": "-sum(p*log(p))/log(4), with 0*log(0)=0",
            "context_vs_global_tv_formula": "0.5*sum(abs(selected_probability-global_probability))",
        },
        "acceptance": {
            "verdicts": ["P_PASS", "P_FAIL", "P_KILL"],
            "kill_reasons": ["CONTEXT_DOMAIN_NOT_PROVEN", "TAXONOMY_NOT_PROVEN", "TEMPORAL_CAUSALITY_NOT_PROVEN", "FIREWALL_VIOLATION"],
            "p_fail_reasons": ["NO_VALID_LEGAL_FALLBACK", "NO_CONTEXTUAL_USAGE", "CONTEXT_VARIATION_COLLAPSED", "FINITE_SIMPLEX_OR_DETERMINISM_FAILURE"],
            "l1_support_below_100_is_not_failure": True, "minimum_l1_support_gate": None,
            "synthetic_2025_count_state_support_gate": "none", "kill_outcome_reportable": True,
            "execution_error_distinct_from_kill": True,
        },
        "output": {
            "aggregate_only": True, "external_only": True,
            "report_json": "trackman_crosswalk_free_context_priors_v1_report.json",
            "report_markdown": "trackman_crosswalk_free_context_priors_v1_report.md",
            "lookup_json": "trackman_crosswalk_free_context_priors_v1_lookup.json",
            "allow_nan": False, "include_row_ids": False, "include_targets": False,
            "include_physics": False, "include_entity_ids": False,
        },
    }


def _contract_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("scope", "context", "taxonomy", "temporal", "features", "acceptance", "output")
    value = {"contract_version": config.get("contract_version")}
    for key in keys:
        section = config.get(key)
        if not isinstance(section, Mapping):
            raise ContextPriorAuditError(f"missing contract section: {key}")
        value[key] = dict(section)
    # Lists whose order is not semantic are canonicalized before the
    # documentary snapshot is compared with the executable contract.
    value["scope"]["forbidden_main"] = sorted(value["scope"]["forbidden_main"])
    value["scope"]["forbidden_trackman"] = sorted(value["scope"]["forbidden_trackman"])
    return value


def validate_contract_config(config: Mapping[str, Any]) -> None:
    if config.get("documentary_snapshot_checked_against_runtime_contract") is not True:
        raise ContextPriorAuditError("config is not marked as runtime-authoritative")
    if _contract_projection(config) != _runtime_contract():
        raise ContextPriorAuditError("contract config diverges from runtime contract")


def load_contract_config(repo_root: str | Path) -> dict[str, Any]:
    path = Path(repo_root) / "configs" / "trackman_crosswalk_free_context_priors_v1.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextPriorAuditError(f"cannot load contract config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ContextPriorAuditError("contract config must be an object")
    validate_contract_config(config)
    return config


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None and result.is_integer() else None


def _season(value: Any) -> int | None:
    result = _integer(value)
    return result


def _family(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    return text if text in EXPECTED_GROUPS else None


def _raw_family_status(value: Any) -> str:
    if _is_missing(value):
        return "missing"
    text = str(value).strip().lower()
    return text if text in EXPECTED_GROUPS else "unexpected"


def _context_key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[int, ...] | None:
    values: list[int] = []
    for column in columns:
        value = _integer(row.get(column))
        bounds = {"balls_before": (0, 3), "strikes_before": (0, 2), "outs_before": (0, 2)}[column]
        if value is None or not bounds[0] <= value <= bounds[1]:
            return None
        values.append(value)
    return tuple(values)


def _validate_context_frame(frame: pd.DataFrame, *, label: str) -> None:
    missing = [column for column in CONTEXT_COLUMNS if column not in frame.columns]
    if missing:
        raise ContextDomainNotProven(f"{label} missing required context columns: {missing}")
    for position, row in enumerate(frame[list(CONTEXT_COLUMNS)].to_dict("records")):
        if _context_key(row, CONTEXT_COLUMNS) is None:
            raise ContextDomainNotProven(f"{label} row {position} has missing/out-of-domain context")


def validate_projection(frame: pd.DataFrame, *, trackman: bool = False) -> None:
    expected = TRACKMAN_PROJECTION if trackman else MAIN_PROJECTION
    forbidden = FAMILY_FORBIDDEN_TRACKMAN if trackman else FAMILY_FORBIDDEN_MAIN
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ContextPriorAuditError(f"projection missing columns: {missing}")
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise ContextPriorAuditError(f"forbidden columns materialized: {present}")
    unexpected = sorted(set(frame.columns) - set(expected) - {"__source_position"})
    if unexpected:
        raise ContextPriorAuditError(f"unexpected projection columns: {unexpected}")


def _read_projected(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(Path(path), usecols=list(columns))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ContextPriorAuditError(f"scoped projection failed for {path}: {exc}") from exc
    return frame


def read_main_features(path: str | Path) -> pd.DataFrame:
    frame = _read_projected(path, MAIN_PROJECTION)
    validate_projection(frame)
    frame = frame.copy().reset_index(drop=True)
    if frame["row_id"].map(lambda value: None if _is_missing(value) else str(value)).duplicated().any():
        raise ContextPriorAuditError("main row_id is not unique")
    return frame


def read_trackman_features(path: str | Path) -> pd.DataFrame:
    frame = _read_projected(path, TRACKMAN_PROJECTION)
    validate_projection(frame, trackman=True)
    frame = frame.copy().reset_index(drop=True)
    # Keep raw context and taxonomy values visible to the run-level
    # prerequisite gate so missing/out-of-domain input produces an aggregate
    # P_KILL report.  No downstream lookup consumes them before validation.
    return frame


def _validate_season_scope(frame: pd.DataFrame, allowed: Sequence[int], *, label: str) -> None:
    seasons = frame["season"].map(_season)
    if seasons.isna().any() or not seasons.isin(tuple(int(value) for value in allowed)).all():
        raise TemporalCausalityNotProven(f"{label} contains nonintegral or out-of-scope season")


def validate_trackman_taxonomy(frame: pd.DataFrame) -> dict[str, Any]:
    statuses = frame["pitch_type_group"].map(_raw_family_status)
    counts = Counter(str(value) for value in statuses)
    missing_rows = int(counts.get("missing", 0))
    unexpected_rows = int(sum(count for key, count in counts.items() if key not in EXPECTED_GROUPS and key not in {"missing", "unexpected"}))
    unexpected_rows += int(counts.get("unexpected", 0))
    status = "PASS" if not missing_rows and not unexpected_rows else "KILL"
    return {
        "status": status, "families": list(FAMILIES), "observed_groups": dict(sorted(counts.items())),
        "known_rows": int(sum(counts.get(family, 0) for family in FAMILIES)),
        "other_rows": int(counts.get("other", 0)), "missing_rows": missing_rows,
        "unexpected_rows": unexpected_rows,
        "other_policy": "retain_as_fourth_family", "redistribution": False,
    }


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    body = {key: value for key, value in report.items() if key != "canonical_report_sha256"}
    return canonical_hash(_json_value(body))


def canonical_frame_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    cols = list(columns) if columns is not None else list(frame.columns)
    digest = hashlib.sha256()
    digest.update(json.dumps({"columns": cols}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    for row in frame[cols].itertuples(index=False, name=None):
        values = [_json_value(value) for value in row]
        digest.update(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8") + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Bucket:
    total: int
    counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class ContextPriorModel:
    """Frozen lookup built from TrackMan rows; no identities are retained."""

    lookups: Mapping[int, Mapping[str, Mapping[tuple[int, ...], Bucket]]]
    trackman_row_count: int
    support_threshold: int = SUPPORT_THRESHOLD


def _bucket_add(target: dict[tuple[int, ...], list[int]], key: tuple[int, ...], family_index: int) -> None:
    counts = target.setdefault(key, [0, 0, 0, 0])
    counts[family_index] += 1


def _freeze_buckets(raw: Mapping[tuple[int, ...], Sequence[int]]) -> dict[tuple[int, ...], Bucket]:
    return {tuple(key): Bucket(sum(int(v) for v in counts), tuple(int(v) for v in counts)) for key, counts in sorted(raw.items())}


def _build_lookup_for_cutoff(trackman: pd.DataFrame, cutoff: int, threshold: int) -> Mapping[str, Mapping[tuple[int, ...], Bucket]]:
    prior = trackman.loc[trackman["season"].astype(int) < int(cutoff)]
    l0: dict[tuple[int, ...], list[int]] = {}
    l1: dict[tuple[int, ...], list[int]] = {}
    global_counts = [0, 0, 0, 0]
    for row in prior.to_dict("records"):
        family = _family(row["pitch_type_group"])
        key0 = _context_key(row, L0_COLUMNS)
        key1 = _context_key(row, L1_COLUMNS)
        if family is None or key0 is None or key1 is None:
            raise ContextPriorAuditError("invalid TrackMan row reached lookup construction")
        family_index = FAMILIES.index(family)
        global_counts[family_index] += 1
        _bucket_add(l0, key0, family_index)
        _bucket_add(l1, key1, family_index)
    return {
        "L0": _freeze_buckets(l0), "L1": _freeze_buckets(l1),
        "global": {(): Bucket(sum(global_counts), tuple(global_counts))},
    }


def build_context_prior_model(trackman: pd.DataFrame, *, seasons: Iterable[int] = (*MAIN_SEASONS, *INFERENCE_SEASONS), support_threshold: int = SUPPORT_THRESHOLD) -> ContextPriorModel:
    validate_projection(trackman, trackman=True)
    _validate_context_frame(trackman, label="TrackMan")
    taxonomy = validate_trackman_taxonomy(trackman)
    if taxonomy["status"] != "PASS":
        raise TaxonomyNotProven("TrackMan taxonomy is not compatible")
    normalized = trackman.copy()
    normalized["season"] = normalized["season"].map(_season)
    normalized["pitch_type_group"] = normalized["pitch_type_group"].map(_family)
    if normalized["season"].isna().any() or normalized["pitch_type_group"].isna().any():
        raise ContextPriorAuditError("invalid TrackMan season or family")
    normalized["season"] = normalized["season"].astype(int)
    lookups = {int(season): _build_lookup_for_cutoff(normalized, int(season), support_threshold) for season in sorted(set(int(s) for s in seasons))}
    return ContextPriorModel(lookups=lookups, trackman_row_count=int(len(normalized)), support_threshold=int(support_threshold))


def _bucket_for(lookup: Mapping[str, Mapping[tuple[int, ...], Bucket]], level: str, key: tuple[int, ...]) -> Bucket | None:
    if level == "global":
        return lookup["global"].get(())
    return lookup[level].get(key)


def _probabilities(bucket: Bucket) -> tuple[float, float, float, float]:
    if bucket.total <= 0:
        raise ContextPriorAuditError("cannot calculate probabilities from empty bucket")
    probabilities = tuple(float(value) / float(bucket.total) for value in bucket.counts)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities) or abs(sum(probabilities) - 1.0) > SIMPLEX_TOLERANCE:
        raise ContextPriorAuditError("nonfinite or invalid probability simplex")
    return probabilities


def _entropy(probabilities: Sequence[float]) -> float:
    value = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0) / math.log(4.0)
    return float(value)


def _prior_record(lookup: Mapping[str, Mapping[tuple[int, ...], Bucket]], key0: tuple[int, ...], key1: tuple[int, ...], threshold: int) -> dict[str, Any]:
    global_bucket = _bucket_for(lookup, "global", ())
    if global_bucket is None or global_bucket.total == 0:
        return {"backoff_level": "NO_HISTORY", "support": 0, "probabilities": None, "entropy": None, "context_vs_global_tv": None}
    selected_level = "global"
    selected_bucket = global_bucket
    l0 = _bucket_for(lookup, "L0", key0)
    l1 = _bucket_for(lookup, "L1", key1)
    if l0 is not None and l0.total >= threshold:
        selected_level, selected_bucket = "L0", l0
    elif l1 is not None and l1.total >= threshold:
        selected_level, selected_bucket = "L1", l1
    probabilities = _probabilities(selected_bucket)
    global_probabilities = _probabilities(global_bucket)
    tv = 0.5 * sum(abs(left - right) for left, right in zip(probabilities, global_probabilities))
    return {"backoff_level": selected_level, "support": int(selected_bucket.total), "probabilities": probabilities, "entropy": _entropy(probabilities), "context_vs_global_tv": float(tv)}


def _feature_record(record: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = record["probabilities"]
    return {
        "tm_cf_p_fastball": None if probabilities is None else probabilities[0],
        "tm_cf_p_breaking": None if probabilities is None else probabilities[1],
        "tm_cf_p_offspeed": None if probabilities is None else probabilities[2],
        "tm_cf_p_other": None if probabilities is None else probabilities[3],
        "tm_cf_entropy_norm4": record["entropy"],
        "tm_cf_support": int(record["support"]),
        "tm_cf_context_vs_global_tv": record["context_vs_global_tv"],
        "__backoff_level": record["backoff_level"],
    }


def apply_context_prior_row(row: Mapping[str, Any], model: ContextPriorModel, season: int) -> dict[str, Any]:
    key0 = _context_key(row, L0_COLUMNS)
    key1 = _context_key(row, L1_COLUMNS)
    if key0 is None or key1 is None:
        raise ContextDomainNotProven("main row has missing/out-of-domain context")
    lookup = model.lookups.get(int(season))
    if lookup is None:
        raise ContextPriorAuditError(f"no frozen temporal lookup for season {season}")
    record = _prior_record(lookup, key0, key1, model.support_threshold)
    return _feature_record(record)


def apply_context_prior_features(main: pd.DataFrame, model: ContextPriorModel) -> pd.DataFrame:
    validate_projection(main)
    _validate_context_frame(main, label="main")
    records = [apply_context_prior_row(row, model, int(_season(row["season"]))) for row in main.to_dict("records")]
    return pd.DataFrame(records, columns=[*FEATURE_COLUMNS, "__backoff_level"])


def _season_diagnostics(main: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for season, group in main.groupby("season", sort=True):
        indices = list(group.index)
        levels = Counter(str(features.loc[index, "__backoff_level"]) for index in indices)
        supports = [int(features.loc[index, "tm_cf_support"]) for index in indices]
        finite_rows = 0
        simplex_failures = 0
        for index in indices:
            values = [features.loc[index, column] for column in FEATURE_COLUMNS[:4]]
            if all(value is not None and math.isfinite(float(value)) for value in values):
                finite_rows += 1
                if abs(sum(float(value) for value in values) - 1.0) > SIMPLEX_TOLERANCE:
                    simplex_failures += 1
        total = len(indices)
        diagnostics[str(int(season))] = {
            "main_row_count": total,
            "backoff_counts": {level: int(levels.get(level, 0)) for level in LEVELS},
            "backoff_fractions": {level: (float(levels.get(level, 0)) / total if total else None) for level in LEVELS},
            "support_min": min(supports) if supports else None, "support_max": max(supports) if supports else None,
            "finite_probability_rows": finite_rows, "missing_probability_rows": total - finite_rows,
            "simplex_failures": simplex_failures,
        }
    return diagnostics


def evaluate_acceptance(main: pd.DataFrame, features: pd.DataFrame, trackman: pd.DataFrame, *, determinism: Mapping[str, Any] | None = None) -> dict[str, Any]:
    diagnostics = _season_diagnostics(main, features)
    contextual = [index for index, level in enumerate(features["__backoff_level"]) if level in {"L0", "L1"}]
    historical = [index for index, level in enumerate(features["__backoff_level"]) if level != "NO_HISTORY"]
    invalid = [index for index in historical if any(features.loc[index, column] is None or not math.isfinite(float(features.loc[index, column])) for column in FEATURE_COLUMNS[:4])]
    tv_values = [float(features.loc[index, "tm_cf_context_vs_global_tv"]) for index in contextual if features.loc[index, "tm_cf_context_vs_global_tv"] is not None]
    simplex_failures = sum(int(value["simplex_failures"]) for value in diagnostics.values())
    reasons: list[str] = []
    if invalid:
        reasons.append("NO_VALID_LEGAL_FALLBACK")
    if historical and not contextual:
        reasons.append("NO_CONTEXTUAL_USAGE")
    if contextual and tv_values and all(abs(value) <= SIMPLEX_TOLERANCE for value in tv_values):
        reasons.append("CONTEXT_VARIATION_COLLAPSED")
    if simplex_failures:
        reasons.append("FINITE_SIMPLEX_OR_DETERMINISM_FAILURE")
    determinism_evidence = dict(determinism or {"tested": False, "complete_feature_output_equal": True})
    if determinism_evidence.get("tested") and not determinism_evidence.get("complete_feature_output_equal", False):
        reasons.append("FINITE_SIMPLEX_OR_DETERMINISM_FAILURE")
    verdict = "P_FAIL" if reasons else "P_PASS"
    return {"verdict": verdict, "p_fail_reasons": sorted(set(reasons)), "kill_reasons": [], "season_diagnostics": diagnostics, "contextual_rows": len(contextual), "historical_rows": len(historical), "l1_support_below_threshold_is_not_failure": True, "support_threshold": SUPPORT_THRESHOLD, "determinism": determinism_evidence}


def _encode_context_key(key: Sequence[int]) -> str:
    """Deterministic, identity-free key used by the external aggregate lookup."""
    return ":".join(str(int(value)) for value in key)


def _decode_context_key(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        return tuple(int(part) for part in value.split(":"))
    except ValueError as exc:
        raise ContextPriorAuditError(f"invalid aggregate lookup context key: {value}") from exc


def _bucket_payload(bucket: Bucket, global_bucket: Bucket) -> dict[str, Any]:
    probabilities = _probabilities(bucket) if bucket.total else None
    global_probabilities = _probabilities(global_bucket) if global_bucket.total else None
    tv = None
    if probabilities is not None and global_probabilities is not None:
        tv = float(0.5 * sum(abs(left - right) for left, right in zip(probabilities, global_probabilities)))
    return {
        "support": int(bucket.total), "family_counts": {family: int(count) for family, count in zip(FAMILIES, bucket.counts)},
        "probabilities": None if probabilities is None else list(probabilities),
        "entropy": None if probabilities is None else _entropy(probabilities),
        "context_vs_global_tv": tv,
    }


def build_aggregate_lookup(model: ContextPriorModel) -> dict[str, Any]:
    """Serialize all aggregate buckets needed to reproduce feature lookup.

    Keys contain only legal context values and seasons; no identity or raw row
    is retained.  The selected backoff level remains an audit decision and is
    not serialized as a model feature.
    """
    result: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION, "support_threshold": model.support_threshold,
        "context_key_encoding": "colon-separated-integer-tuple", "seasons": {},
    }
    for season, lookup in sorted(model.lookups.items()):
        global_bucket = lookup["global"].get((), Bucket(0, (0, 0, 0, 0)))
        result["seasons"][str(season)] = {
            "global": {"GLOBAL": _bucket_payload(global_bucket, global_bucket)},
            "L0": {_encode_context_key(key): _bucket_payload(bucket, global_bucket) for key, bucket in sorted(lookup["L0"].items())},
            "L1": {_encode_context_key(key): _bucket_payload(bucket, global_bucket) for key, bucket in sorted(lookup["L1"].items())},
        }
    return result


def _record_from_payload(payload: Mapping[str, Any], level: str) -> dict[str, Any]:
    probabilities = payload.get("probabilities")
    if probabilities is not None:
        probabilities = tuple(float(value) for value in probabilities)
        if len(probabilities) != 4 or not all(math.isfinite(value) for value in probabilities):
            raise ContextPriorAuditError("aggregate lookup contains invalid probabilities")
    return {"backoff_level": level, "support": int(payload.get("support", 0)), "probabilities": probabilities, "entropy": payload.get("entropy"), "context_vs_global_tv": payload.get("context_vs_global_tv")}


def apply_context_prior_lookup_row(row: Mapping[str, Any], aggregate_lookup: Mapping[str, Any], season: int) -> dict[str, Any]:
    """Reproduce one feature record from the external aggregate lookup."""
    key0 = _context_key(row, L0_COLUMNS)
    key1 = _context_key(row, L1_COLUMNS)
    if key0 is None or key1 is None:
        raise ContextDomainNotProven("main row has missing/out-of-domain context")
    season_lookup = aggregate_lookup.get("seasons", {}).get(str(int(season)))
    if not isinstance(season_lookup, Mapping):
        raise ContextPriorAuditError(f"aggregate lookup has no season {season}")
    threshold = int(aggregate_lookup.get("support_threshold", SUPPORT_THRESHOLD))
    l0 = season_lookup.get("L0", {}).get(_encode_context_key(key0))
    l1 = season_lookup.get("L1", {}).get(_encode_context_key(key1))
    global_payload = season_lookup.get("global", {}).get("GLOBAL")
    if not isinstance(global_payload, Mapping) or int(global_payload.get("support", 0)) == 0:
        return _feature_record({"backoff_level": "NO_HISTORY", "support": 0, "probabilities": None, "entropy": None, "context_vs_global_tv": None})
    if isinstance(l0, Mapping) and int(l0.get("support", 0)) >= threshold:
        return _feature_record(_record_from_payload(l0, "L0"))
    if isinstance(l1, Mapping) and int(l1.get("support", 0)) >= threshold:
        return _feature_record(_record_from_payload(l1, "L1"))
    return _feature_record(_record_from_payload(global_payload, "global"))


def assess_determinism(main: pd.DataFrame, trackman: pd.DataFrame, model: ContextPriorModel) -> dict[str, Any]:
    """Rebuild from a fixed TrackMan permutation and compare complete output."""
    original = apply_context_prior_features(main, model)
    permutation = list(reversed(range(len(trackman))))
    permuted_trackman = trackman.iloc[permutation].reset_index(drop=True)
    rebuilt = build_context_prior_model(permuted_trackman, seasons=tuple(sorted(model.lookups)), support_threshold=model.support_threshold)
    permuted = apply_context_prior_features(main, rebuilt)
    original_hash = canonical_frame_hash(original)
    permuted_hash = canonical_frame_hash(permuted)
    return {
        "tested": True, "fixed_trackman_row_permutation_length": len(permutation), "fixed_trackman_row_permutation_sha256": canonical_hash(permutation),
        "original_feature_output_sha256": original_hash, "permuted_feature_output_sha256": permuted_hash,
        "complete_feature_output_equal": bool(original.equals(permuted)),
    }


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


def _outside_repo(output_dir: str | Path, repo_root: str | Path) -> None:
    output = Path(output_dir).resolve()
    repo = Path(repo_root).resolve()
    if output == repo or repo in output.parents:
        raise ContextPriorAuditError("output directory must be outside repository")


def build_report(main: pd.DataFrame, trackman: pd.DataFrame, features: pd.DataFrame | None, acceptance: Mapping[str, Any], taxonomy: Mapping[str, Any], *, repo_root: Path, config_path: Path, script_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION, "git_sha": _git_sha(repo_root), "runner_sha256": _sha256(script_path), "config_sha256": _sha256(config_path),
        "source_frame_hash_algorithm": FRAME_HASH_ALGORITHM,
        "main_source_frame_sha256": canonical_frame_hash(main), "trackman_source_frame_sha256": canonical_frame_hash(trackman),
        "main_row_count": int(len(main)), "trackman_row_count": int(len(trackman)),
        "projection": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)},
        "temporal_contract": {"rule": "season S uses TrackMan season < S", "same_season_used": False, "future_season_used": False, "row_independent": True},
        "taxonomy": _json_value(taxonomy), "features": {"columns": list(FEATURE_COLUMNS), "backoff_level_not_feature": True, "simplex_tolerance": SIMPLEX_TOLERANCE},
        "acceptance": _json_value(acceptance), "feature_output_available": features is not None,
        "scope": {"target_access": False, "target_column_in_projection": False, "test_access": False, "test_distribution_access": False, "public_access": False, "public_leaderboard_evidence": False, "external_information_access": False, "trackman_entity_access": False, "trackman_physics_access": False, "current_pitch_measurement_access": False, "model_training_or_scoring": False, "gpu_used": False, "label_access_ledger": [], "branch_inert": True},
        "privacy": {"aggregate_only": True, "row_ids": False, "entity_ids": False, "target_values": False, "physics": False, "raw_rows": False},
        "level_r": {"verdict": "OUT_OF_SCOPE_NOT_IMPLEMENTED", "exact_matcher_implemented": False, "source_order_used": False},
        "config_runtime": {"config_runtime_match": _contract_projection(config) == _runtime_contract(), "runtime_contract_sha256": canonical_hash(_runtime_contract())},
    }
    report["canonical_report_sha256"] = canonical_report_hash(report)
    return report


def write_outputs(report: Mapping[str, Any], lookup: Mapping[str, Any], output_dir: str | Path, *, repo_root: Path) -> tuple[Path, Path, Path]:
    _outside_repo(output_dir, repo_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    safe_report = _json_value(report)
    json_path = output / "trackman_crosswalk_free_context_priors_v1_report.json"
    md_path = output / "trackman_crosswalk_free_context_priors_v1_report.md"
    lookup_path = output / "trackman_crosswalk_free_context_priors_v1_lookup.json"
    json_path.write_text(json.dumps(safe_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lookup_path.write_text(json.dumps(_json_value(lookup), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    acceptance = safe_report["acceptance"]
    lines = [
        "# TrackMan crosswalk-free context priors v1", "", f"- verdict: `{acceptance['verdict']}`",
        f"- canonical_report_sha256: `{safe_report['canonical_report_sha256']}`",
        "- entity/crosswalk access: `false`", "- target/test/Public/external access: `false`",
        "- temporal rule: only TrackMan seasons strictly before each main-row season",
        "- backoff: legal sparse contexts use frozen `L0 -> L1 -> global` hierarchy",
        "- `tm_cf_support` is the selected-level historical row count; backoff level is audit-only.",
        "- Level R exact pitch-row matching: `OUT_OF_SCOPE_NOT_IMPLEMENTED`.",
    ]
    for season, value in sorted(acceptance["season_diagnostics"].items()):
        lines.append(f"- season {season} backoff counts: `{value['backoff_counts']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, lookup_path


def run_from_paths(main_csv: str | Path, trackman_csv: str | Path, output_dir: str | Path, repo_root: str | Path) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    repo = Path(repo_root).resolve()
    config_path = repo / "configs" / "trackman_crosswalk_free_context_priors_v1.json"
    config = load_contract_config(repo)
    main = read_main_features(main_csv)
    trackman = read_trackman_features(trackman_csv)
    taxonomy = validate_trackman_taxonomy(trackman)
    try:
        _validate_season_scope(main, MAIN_SEASONS, label="main")
        _validate_season_scope(trackman, TRACKMAN_HISTORICAL_SEASONS, label="TrackMan")
        _validate_context_frame(main, label="main")
        _validate_context_frame(trackman, label="TrackMan")
        if taxonomy["status"] != "PASS":
            raise TaxonomyNotProven("TAXONOMY_NOT_PROVEN")
        model = build_context_prior_model(trackman, seasons=(*MAIN_SEASONS, *INFERENCE_SEASONS), support_threshold=int(config["context"]["support_threshold"]))
        features = apply_context_prior_features(main, model)
        determinism = assess_determinism(main, trackman, model)
        acceptance = evaluate_acceptance(main, features, trackman, determinism=determinism)
        lookup = build_aggregate_lookup(model)
    except ContextDomainNotProven:
        acceptance = {"verdict": "P_KILL", "kill_reasons": ["CONTEXT_DOMAIN_NOT_PROVEN"], "p_fail_reasons": [], "feature_output_available": False, "season_diagnostics": {}, "determinism": {"tested": False}}
        features = None
        lookup = {"contract_version": CONTRACT_VERSION, "support_threshold": SUPPORT_THRESHOLD, "context_key_encoding": "colon-separated-integer-tuple", "seasons": {}, "available": False}
    except TaxonomyNotProven:
        acceptance = {"verdict": "P_KILL", "kill_reasons": ["TAXONOMY_NOT_PROVEN"], "p_fail_reasons": [], "feature_output_available": False, "season_diagnostics": {}, "determinism": {"tested": False}}
        features = None
        lookup = {"contract_version": CONTRACT_VERSION, "support_threshold": SUPPORT_THRESHOLD, "context_key_encoding": "colon-separated-integer-tuple", "seasons": {}, "available": False}
    except TemporalCausalityNotProven:
        acceptance = {"verdict": "P_KILL", "kill_reasons": ["TEMPORAL_CAUSALITY_NOT_PROVEN"], "p_fail_reasons": [], "feature_output_available": False, "season_diagnostics": {}, "determinism": {"tested": False}}
        features = None
        lookup = {"contract_version": CONTRACT_VERSION, "support_threshold": SUPPORT_THRESHOLD, "context_key_encoding": "colon-separated-integer-tuple", "seasons": {}, "available": False}
    report = build_report(main, trackman, features, acceptance, taxonomy, repo_root=repo, config_path=config_path, script_path=Path(__file__).resolve(), config=config)
    return report, write_outputs(report, lookup, output_dir, repo_root=repo)


def static_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_contract_config(root)
    if TARGET in MAIN_PROJECTION or TARGET in TRACKMAN_PROJECTION:
        raise ContextPriorAuditError("target appears in projection")
    if FAMILY_FORBIDDEN_TRACKMAN.intersection(TRACKMAN_PROJECTION):
        raise ContextPriorAuditError("forbidden identity/physics field appears in TrackMan projection")
    if FAMILY_FORBIDDEN_MAIN.intersection(MAIN_PROJECTION):
        raise ContextPriorAuditError("forbidden target/entity field appears in main projection")
    if not config["features"]["backoff_level_not_feature"]:
        raise ContextPriorAuditError("backoff level cannot be a first model feature")
    if config["acceptance"]["minimum_l1_support_gate"] is not None:
        raise ContextPriorAuditError("L1 support gate must not be required")
    return {"contract_version": CONTRACT_VERSION, "config_valid": True, "projection_columns": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)}, "target_access": False, "target_column_in_projection": False, "test_access": False, "test_distribution_access": False, "public_access": False, "external_information_access": False, "trackman_entity_access": False, "trackman_physics_access": False, "current_pitch_measurement_access": False, "model_training_or_scoring": False, "gpu_used": False, "branch_inert": True, "label_access_ledger": [], "level_r_exact_matcher": False, "backoff_level_is_model_feature": False, "l1_below_threshold_is_failure": False, "kill_outcome_reportable": True, "execution_error_distinct_from_kill": True, "static_pass": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static", help="static contract/firewall check; no data access")
    static.add_argument("--repo-root", required=True)
    run = subparsers.add_parser("run", help="official model-free structural audit")
    run.add_argument("--train-csv", "--main-csv", dest="main_csv", required=True)
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
            print(json.dumps({"verdict": report["acceptance"]["verdict"], "canonical_report_sha256": report["canonical_report_sha256"], "json": str(paths[0]), "markdown": str(paths[1]), "lookup": str(paths[2])}, sort_keys=True, allow_nan=False))
        return 0
    except (ContextPriorAuditError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
