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
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    from scripts import audit_trackman_context_domain_characterization_v1 as _pr13
    from scripts import audit_trackman_crosswalk_free_context_priors_v1 as _v1
except ImportError:  # pragma: no cover - supports direct scripts/ imports
    import audit_trackman_context_domain_characterization_v1 as _pr13
    import audit_trackman_crosswalk_free_context_priors_v1 as _v1


CONTRACT_VERSION = "aimers9-trackman-crosswalk-free-context-priors-v2"
TARGET = "control_success"
MAIN_SEASONS = tuple(range(2019, 2025))
INFERENCE_SEASONS = (2025,)
TRACKMAN_HISTORICAL_SEASONS = tuple(range(2019, 2025))
SOURCE_PROJECTION = ("season", "balls_before", "strikes_before", "outs_before")
MAIN_PROJECTION = SOURCE_PROJECTION
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
FRAME_HASH_ALGORITHM = _pr13.FRAME_HASH_ALGORITHM
LEVELS = ("L0", "L1", "global", "NO_HISTORY")
FAMILY_FORBIDDEN_MAIN = frozenset({TARGET, "row_id", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"})
FAMILY_FORBIDDEN_TRACKMAN = frozenset({
    "pitcher_trackman_id", "batter_trackman_id", "trackman_id", "game_id",
    "trackman_game_id", "pitch_no", "tagged_pitch_type", "auto_pitch_type",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
    "rel_height", "rel_side", "zone_speed", "release", "movement",
})
EXPECTED_GROUPS = frozenset(FAMILIES)
EXPECTED_SOURCE_ROW_COUNTS = {"main": 1_475_092, "trackman": 1_793_078}
EXPECTED_FILTERED_TRACKMAN_ROWS = 1_792_981
EXPECTED_EXCLUDED_TRACKMAN_ROWS = 97
EXPECTED_RESIDUE_MANIFEST = (
    (2022, "balls_before", "out_of_domain", "4", 1),
    (2022, "outs_before", "out_of_domain", "3", 72),
    (2022, "outs_before", "out_of_domain", "4", 12),
    (2022, "strikes_before", "out_of_domain", "3", 1),
    (2023, "outs_before", "out_of_domain", "3", 11),
)
EXPECTED_PR13_SOURCE_HASHES = {
    "main": "50a1fdeae8c0e1ece5302837e3fb748674f852d436c1931e2001d8a6b7c894e8",
    "trackman": "934d5757241e78389f0736717c550be8e80c8948f156af0d5abc6e956acdd24c",
}
PR13_CHARACTERIZATION_CONFIG_SHA256 = "0334bbb8caa7f1915134a7f4dcbb66e5a26ceb08abbe8133a70f247c2643299f"
PR13_CHARACTERIZATION_RUNNER_SHA256 = "370b5a09836cf95b7c4499ddea4ef881ee2fd5deb6155aca99c9ec00eb7e5090"
PR13_RUNNER_RELATIVE_PATH = "scripts/audit_trackman_context_domain_characterization_v1.py"
PR13_CONFIG_RELATIVE_PATH = "configs/trackman_context_domain_characterization_v1.json"
V1_RUNNER_RELATIVE_PATH = "scripts/audit_trackman_crosswalk_free_context_priors_v1.py"
V1_CONFIG_RELATIVE_PATH = "configs/trackman_crosswalk_free_context_priors_v1.json"
V1_RUNNER_SHA256 = "2562bd3b21dbe010526eb748add0baccab6b8e11b4ddf2a0c8f73baae4a4a293"
V1_CONFIG_SHA256 = "7cb0fd4b26b68fa59d7c726489343c5295af9bb5d2d914a82f27d5958a68d8f7"


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


def validate_pr13_authority(repo_root: str | Path) -> dict[str, Any]:
    """Pin the exact merged PR #13 helper/config used for source semantics."""
    root = Path(repo_root)
    runner = root / PR13_RUNNER_RELATIVE_PATH
    config = root / PR13_CONFIG_RELATIVE_PATH
    try:
        runner_sha = _sha256(runner)
        config_sha = _sha256(config)
    except OSError as exc:
        raise ContextPriorAuditError(f"PR13 authority artifact missing: {exc}") from exc
    evidence = {
        "runner_path": PR13_RUNNER_RELATIVE_PATH, "config_path": PR13_CONFIG_RELATIVE_PATH,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "expected_runner_sha256": PR13_CHARACTERIZATION_RUNNER_SHA256,
        "expected_config_sha256": PR13_CHARACTERIZATION_CONFIG_SHA256,
        "hashes_match": runner_sha == PR13_CHARACTERIZATION_RUNNER_SHA256 and config_sha == PR13_CHARACTERIZATION_CONFIG_SHA256,
    }
    if not evidence["hashes_match"]:
        raise ContextPriorAuditError("PR13_SOURCE_AUTHORITY_DRIFT")
    return evidence


def validate_v1_authority(repo_root: str | Path) -> dict[str, Any]:
    """Pin the merged v1 implementation whose lookup semantics are reused."""
    root = Path(repo_root)
    runner = root / V1_RUNNER_RELATIVE_PATH
    config = root / V1_CONFIG_RELATIVE_PATH
    try:
        runner_sha = _sha256(runner)
        config_sha = _sha256(config)
    except OSError as exc:
        raise ContextPriorAuditError(f"v1 authority artifact missing: {exc}") from exc
    shared_contract = {
        "families": tuple(FAMILIES) == tuple(_v1.FAMILIES),
        "features": tuple(FEATURE_COLUMNS) == tuple(_v1.FEATURE_COLUMNS),
        "context_columns": tuple(CONTEXT_COLUMNS) == tuple(_v1.CONTEXT_COLUMNS),
        "l0": tuple(L0_COLUMNS) == tuple(_v1.L0_COLUMNS),
        "l1": tuple(L1_COLUMNS) == tuple(_v1.L1_COLUMNS),
        "support_threshold": int(SUPPORT_THRESHOLD) == int(_v1.SUPPORT_THRESHOLD),
        "simplex_tolerance": float(SIMPLEX_TOLERANCE) == float(_v1.SIMPLEX_TOLERANCE),
        "levels": tuple(LEVELS) == tuple(_v1.LEVELS),
    }
    evidence = {
        "runner_path": V1_RUNNER_RELATIVE_PATH, "config_path": V1_CONFIG_RELATIVE_PATH,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "expected_runner_sha256": V1_RUNNER_SHA256, "expected_config_sha256": V1_CONFIG_SHA256,
        "hashes_match": runner_sha == V1_RUNNER_SHA256 and config_sha == V1_CONFIG_SHA256,
        "shared_semantics": shared_contract,
        "shared_semantics_match": all(shared_contract.values()),
    }
    if not evidence["hashes_match"]:
        raise ContextPriorAuditError("V1_SOURCE_AUTHORITY_DRIFT")
    if not evidence["shared_semantics_match"]:
        raise ContextPriorAuditError("V1_SHARED_SEMANTICS_DRIFT")
    return evidence


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
        "source_quality": {
            "eligible_source": "trackman_only",
            "predicate": "finite integer-like balls 0..3, strikes 0..2, outs 0..2",
            "normalization": False, "rounding": False, "clipping": False,
            "expected_raw_rows": 1793078, "expected_excluded_rows": 97,
            "expected_retained_rows": 1792981,
            "residue_manifest": [
                {"season": 2022, "field": "balls_before", "reason": "out_of_domain", "value": "4", "count": 1},
                {"season": 2022, "field": "outs_before", "reason": "out_of_domain", "value": "3", "count": 72},
                {"season": 2022, "field": "outs_before", "reason": "out_of_domain", "value": "4", "count": 12},
                {"season": 2022, "field": "strikes_before", "reason": "out_of_domain", "value": "3", "count": 1},
                {"season": 2023, "field": "outs_before", "reason": "out_of_domain", "value": "3", "count": 11},
            ],
            "drift_policy": "P_KILL",
            "excluded_rows_reach_lookup": False,
        },
        "provenance": {
            "source_hash_authority": "PR13_characterization_v1",
            "source_projection": list(SOURCE_PROJECTION),
            "source_hash_algorithm": FRAME_HASH_ALGORITHM,
            "expected_main_source_frame_sha256": EXPECTED_PR13_SOURCE_HASHES["main"],
            "expected_trackman_source_frame_sha256": EXPECTED_PR13_SOURCE_HASHES["trackman"],
            "pr13_config_sha256": PR13_CHARACTERIZATION_CONFIG_SHA256,
            "pr13_runner_sha256": PR13_CHARACTERIZATION_RUNNER_SHA256,
            "v1_runner_path": V1_RUNNER_RELATIVE_PATH,
            "v1_config_path": V1_CONFIG_RELATIVE_PATH,
            "v1_runner_sha256": V1_RUNNER_SHA256,
            "v1_config_sha256": V1_CONFIG_SHA256,
            "v1_runtime_not_source_hash_authority": True,
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
            "kill_reasons": [
                "2025_LEGAL_STATE_PROOF_FAILED", "CONTEXT_DOMAIN_NOT_PROVEN",
                "FIREWALL_VIOLATION", "PR13_SOURCE_AUTHORITY_DRIFT",
                "PR13_SOURCE_HASH_DRIFT", "SOURCE_QUALITY_RESIDUE_DRIFT",
                "TAXONOMY_NOT_PROVEN", "TEMPORAL_CAUSALITY_NOT_PROVEN",
                "V1_SHARED_SEMANTICS_DRIFT", "V1_SOURCE_AUTHORITY_DRIFT",
            ],
            "p_fail_reasons": ["NO_VALID_LEGAL_FALLBACK", "NO_CONTEXTUAL_USAGE", "CONTEXT_VARIATION_COLLAPSED", "FINITE_SIMPLEX_OR_DETERMINISM_FAILURE"],
            "l1_support_below_100_is_not_failure": True, "minimum_l1_support_gate": None,
            "synthetic_2025_count_state_support_gate": "none", "kill_outcome_reportable": True,
            "execution_error_distinct_from_kill": True,
        },
        "output": {
            "aggregate_only": True, "external_only": True,
            "report_json": "trackman_crosswalk_free_context_priors_v2_report.json",
            "report_markdown": "trackman_crosswalk_free_context_priors_v2_report.md",
            "lookup_json": "trackman_crosswalk_free_context_priors_v2_lookup.json",
            "allow_nan": False, "include_row_ids": False, "include_targets": False,
            "include_physics": False, "include_entity_ids": False,
        },
    }


def _contract_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("scope", "source_quality", "provenance", "context", "taxonomy", "temporal", "features", "acceptance", "output")
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
    path = Path(repo_root) / "configs" / "trackman_crosswalk_free_context_priors_v2.json"
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
    if tuple(frame.columns) != tuple(expected):
        raise ContextPriorAuditError(f"projection must be exactly {expected}")
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ContextPriorAuditError(f"projection missing columns: {missing}")
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise ContextPriorAuditError(f"forbidden columns materialized: {present}")
    unexpected = sorted(set(frame.columns) - set(expected))
    if unexpected:
        raise ContextPriorAuditError(f"unexpected projection columns: {unexpected}")


def _read_projected(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(Path(path), usecols=list(columns))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ContextPriorAuditError(f"scoped projection failed for {path}: {exc}") from exc
    return frame.loc[:, list(columns)]


def read_main_features(path: str | Path) -> pd.DataFrame:
    frame = _read_projected(path, MAIN_PROJECTION)
    validate_projection(frame)
    return frame.copy().reset_index(drop=True)


def read_trackman_features(path: str | Path) -> pd.DataFrame:
    frame = _read_projected(path, TRACKMAN_PROJECTION)
    validate_projection(frame, trackman=True)
    frame = frame.copy().reset_index(drop=True)
    # Keep raw context and taxonomy values visible to the run-level
    # prerequisite gate so missing/out-of-domain input produces an aggregate
    # P_KILL report.  No downstream lookup consumes them before validation.
    return frame


def pr13_source_scan(frame: pd.DataFrame, *, source: str) -> dict[str, Any]:
    """Use the merged PR #13 characterization implementation as hash authority."""
    if source not in {"main", "trackman"}:
        raise ContextPriorAuditError(f"invalid source for PR13 hash: {source}")
    projection = frame.loc[:, list(SOURCE_PROJECTION)]
    return _pr13.scan_frame(projection, source=source)


def pr13_source_hash(frame: pd.DataFrame, *, source: str) -> str:
    return str(pr13_source_scan(frame, source=source)["source_frame_sha256"])


def _residue_manifest(frame: pd.DataFrame) -> tuple[tuple[int | str, str, str, str, int], ...]:
    counts: Counter[tuple[int | str, str, str, str]] = Counter()
    for row in frame.to_dict("records"):
        season_value = _pr13._classify_v1(row.get("season"), "season")
        season: int | str = int(season_value["value"]) if season_value["status"] == "legal" else "__INVALID_SEASON__"
        for field in CONTEXT_COLUMNS:
            info = _pr13._classify_v1(row.get(field), field)
            if info["status"] != "legal":
                counts[(season, field, str(info["reason"]), str(_pr13._value_key(row.get(field))))] += 1
    return tuple((season, field, reason, value, int(count)) for (season, field, reason, value), count in sorted(counts.items(), key=lambda item: repr(item[0])))


def _residue_manifest_payload(manifest: Sequence[tuple[int | str, str, str, str, int]]) -> list[dict[str, Any]]:
    return [
        {"season": season, "field": field, "reason": reason, "value": value, "count": int(count)}
        for season, field, reason, value, count in manifest
    ]


def filter_trackman_history(trackman: pd.DataFrame, *, enforce_official_residue: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen TrackMan-only legal-context predicate exactly once.

    Taxonomy is checked before the predicate; season scope is checked by the
    caller.  The returned frame is marked so the lookup builder cannot be
    called with the unfiltered raw frame.
    """
    validate_projection(trackman, trackman=True)
    if validate_trackman_taxonomy(trackman)["status"] != "PASS":
        raise TaxonomyNotProven("TAXONOMY_NOT_PROVEN")
    valid_mask = trackman.apply(lambda row: _context_key(row, CONTEXT_COLUMNS) is not None, axis=1)
    manifest = _residue_manifest(trackman)
    excluded = int((~valid_mask).sum())
    summary = {
        "raw_rows": int(len(trackman)), "excluded_rows": excluded,
        "retained_rows": int(valid_mask.sum()), "excluded_manifest": _residue_manifest_payload(manifest),
        "predicate": "finite integer-like balls 0..3, strikes 0..2, outs 0..2",
        "normalization": False, "rounding": False, "clipping": False,
        "expected_raw_rows": EXPECTED_SOURCE_ROW_COUNTS["trackman"],
        "expected_excluded_rows": EXPECTED_EXCLUDED_TRACKMAN_ROWS,
        "expected_retained_rows": EXPECTED_FILTERED_TRACKMAN_ROWS,
        "contract_checked": bool(enforce_official_residue),
    }
    if enforce_official_residue:
        validate_residue_contract(trackman, summary)
    filtered = trackman.loc[valid_mask].copy().reset_index(drop=True)
    filtered.attrs["source_quality_filter_applied"] = True
    filtered.attrs["excluded_rows"] = excluded
    filtered.attrs["residue_manifest"] = _residue_manifest_payload(manifest)
    return filtered, summary


def validate_residue_contract(trackman: pd.DataFrame, summary: Mapping[str, Any] | None = None) -> None:
    current = dict(summary or {})
    if not current:
        mask = trackman.apply(lambda row: _context_key(row, CONTEXT_COLUMNS) is not None, axis=1)
        current = {"raw_rows": int(len(trackman)), "excluded_rows": int((~mask).sum()), "retained_rows": int(mask.sum()), "excluded_manifest": _residue_manifest_payload(_residue_manifest(trackman))}
    expected_manifest = _residue_manifest_payload(EXPECTED_RESIDUE_MANIFEST)
    if (
        current.get("raw_rows") != EXPECTED_SOURCE_ROW_COUNTS["trackman"]
        or current.get("excluded_rows") != EXPECTED_EXCLUDED_TRACKMAN_ROWS
        or current.get("retained_rows") != EXPECTED_FILTERED_TRACKMAN_ROWS
        or current.get("excluded_manifest") != expected_manifest
    ):
        raise ContextPriorAuditError("SOURCE_QUALITY_RESIDUE_DRIFT")


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


Bucket = _v1.Bucket
ContextPriorModel = _v1.ContextPriorModel


def _wrap_v1_error(exc: Exception) -> ContextPriorAuditError:
    """Translate delegated v1 contract errors into the v2 error namespace."""
    return ContextPriorAuditError(str(exc))


def build_context_prior_model(trackman: pd.DataFrame, *, seasons: Iterable[int] = (*MAIN_SEASONS, *INFERENCE_SEASONS), support_threshold: int = SUPPORT_THRESHOLD) -> ContextPriorModel:
    """Guard v2 source-quality state, then delegate lookup construction to v1."""
    validate_projection(trackman, trackman=True)
    if trackman.attrs.get("source_quality_filter_applied") is not True:
        raise ContextPriorAuditError("unfiltered TrackMan frame cannot reach lookup construction")
    _validate_season_scope(trackman, TRACKMAN_HISTORICAL_SEASONS, label="TrackMan lookup")
    _validate_context_frame(trackman, label="TrackMan")
    taxonomy = validate_trackman_taxonomy(trackman)
    if taxonomy["status"] != "PASS":
        raise TaxonomyNotProven("TrackMan taxonomy is not compatible")
    if tuple(FAMILIES) != tuple(_v1.FAMILIES) or tuple(FEATURE_COLUMNS) != tuple(_v1.FEATURE_COLUMNS):
        raise ContextPriorAuditError("V1_SHARED_SEMANTICS_DRIFT")
    if tuple(L0_COLUMNS) != tuple(_v1.L0_COLUMNS) or tuple(L1_COLUMNS) != tuple(_v1.L1_COLUMNS):
        raise ContextPriorAuditError("V1_SHARED_SEMANTICS_DRIFT")
    if int(support_threshold) != int(_v1.SUPPORT_THRESHOLD):
        raise ContextPriorAuditError("V1_SHARED_SEMANTICS_DRIFT")
    try:
        return _v1.build_context_prior_model(trackman, seasons=seasons, support_threshold=support_threshold)
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def _bucket_for(lookup: Mapping[str, Mapping[tuple[int, ...], Bucket]], level: str, key: tuple[int, ...]) -> Bucket | None:
    try:
        return _v1._bucket_for(lookup, level, key)
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def _probabilities(bucket: Bucket) -> tuple[float, float, float, float]:
    try:
        return _v1._probabilities(bucket)
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def _entropy(probabilities: Sequence[float]) -> float:
    return _v1._entropy(probabilities)


def _prior_record(lookup: Mapping[str, Mapping[tuple[int, ...], Bucket]], key0: tuple[int, ...], key1: tuple[int, ...], threshold: int) -> dict[str, Any]:
    try:
        return _v1._prior_record(lookup, key0, key1, threshold)
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def _feature_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return _v1._feature_record(record)


def apply_context_prior_row(row: Mapping[str, Any], model: ContextPriorModel, season: int) -> dict[str, Any]:
    try:
        return _v1.apply_context_prior_row(row, model, season)
    except _v1.ContextDomainNotProven as exc:
        raise ContextDomainNotProven(str(exc)) from exc
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def apply_context_prior_features(main: pd.DataFrame, model: ContextPriorModel) -> pd.DataFrame:
    validate_projection(main)
    _validate_context_frame(main, label="main")
    records = []
    for values in main.itertuples(index=False, name=None):
        row = dict(zip(MAIN_PROJECTION, values))
        records.append(apply_context_prior_row(row, model, int(_season(row["season"]))))
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
    if determinism_evidence.get("tested") and not determinism_evidence.get("complete_rebuild_equal", determinism_evidence.get("complete_feature_output_equal", False)):
        reasons.append("FINITE_SIMPLEX_OR_DETERMINISM_FAILURE")
    if determinism_evidence.get("serialized_lookup_parity_tested") and not determinism_evidence.get("serialized_lookup_parity_equal", False):
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


def build_aggregate_lookup(model: ContextPriorModel) -> dict[str, Any]:
    """Serialize lookup data using the reviewed v1 implementation."""
    try:
        result = dict(_v1.build_aggregate_lookup(model))
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc
    result["contract_version"] = CONTRACT_VERSION
    return result


def apply_context_prior_lookup_row(row: Mapping[str, Any], aggregate_lookup: Mapping[str, Any], season: int) -> dict[str, Any]:
    """Reproduce one feature record through the reviewed v1 lookup helper."""
    try:
        return _v1.apply_context_prior_lookup_row(row, aggregate_lookup, season)
    except _v1.ContextDomainNotProven as exc:
        raise ContextDomainNotProven(str(exc)) from exc
    except _v1.ContextPriorAuditError as exc:
        raise _wrap_v1_error(exc) from exc


def assess_determinism(main: pd.DataFrame, trackman: pd.DataFrame, model: ContextPriorModel) -> dict[str, Any]:
    """Rebuild from a fixed TrackMan permutation and compare complete output."""
    original = apply_context_prior_features(main, model)
    permutation = list(reversed(range(len(trackman))))
    permuted_trackman = trackman.iloc[permutation].reset_index(drop=True)
    rebuilt = build_context_prior_model(permuted_trackman, seasons=tuple(sorted(model.lookups)), support_threshold=model.support_threshold)
    permuted = apply_context_prior_features(main, rebuilt)
    original_hash = canonical_frame_hash(original)
    permuted_hash = canonical_frame_hash(permuted)
    original_lookup_hash = canonical_hash(build_aggregate_lookup(model))
    permuted_lookup_hash = canonical_hash(build_aggregate_lookup(rebuilt))
    feature_equal = bool(original.equals(permuted))
    lookup_equal = original_lookup_hash == permuted_lookup_hash
    return {
        "tested": True, "fixed_trackman_row_permutation_length": len(permutation), "fixed_trackman_row_permutation_sha256": canonical_hash(permutation),
        "original_feature_output_sha256": original_hash, "permuted_feature_output_sha256": permuted_hash,
        "complete_feature_output_equal": feature_equal,
        "original_lookup_sha256": original_lookup_hash,
        "permuted_lookup_sha256": permuted_lookup_hash,
        "serialized_lookup_equal": lookup_equal,
        "complete_rebuild_equal": feature_equal and lookup_equal,
    }


def assess_serialized_lookup_parity(
    main: pd.DataFrame,
    features: pd.DataFrame,
    aggregate_lookup: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare every aggregate feature row with the serialized lookup path."""
    columns = [*FEATURE_COLUMNS, "__backoff_level"]
    if len(main) != len(features):
        raise ContextPriorAuditError("serialized lookup parity row-count mismatch")
    digest = hashlib.sha256()
    digest.update(json.dumps({"columns": columns}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    mismatch_count = 0
    for position, values_tuple in enumerate(main.itertuples(index=False, name=None)):
        row = dict(zip(MAIN_PROJECTION, values_tuple))
        season = _season(row.get("season"))
        if season is None:
            raise TemporalCausalityNotProven("main season is invalid during lookup parity")
        reconstructed = apply_context_prior_lookup_row(row, aggregate_lookup, season)
        values = [_json_value(reconstructed[column]) for column in columns]
        expected = [_json_value(features.iloc[position][column]) for column in columns]
        if values != expected:
            mismatch_count += 1
        digest.update(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8") + b"\n")
    reconstructed_hash = digest.hexdigest()
    feature_hash = canonical_frame_hash(features, columns)
    return {
        "tested": True,
        "row_count": int(len(main)),
        "mismatch_count": int(mismatch_count),
        "feature_output_sha256": feature_hash,
        "serialized_lookup_output_sha256": reconstructed_hash,
        "equal": mismatch_count == 0 and reconstructed_hash == feature_hash,
    }


def assess_temporal_cutoffs(trackman: pd.DataFrame, model: ContextPriorModel) -> dict[str, Any]:
    """Record aggregate proof that each lookup contains strict prior seasons only."""
    if trackman.attrs.get("source_quality_filter_applied") is not True:
        raise ContextPriorAuditError("temporal proof requires filtered TrackMan history")
    seasons = trackman["season"].map(_season)
    if seasons.isna().any():
        raise TemporalCausalityNotProven("TrackMan season is invalid during temporal proof")
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for cutoff, lookup in sorted(model.lookups.items()):
        global_bucket = lookup["global"].get(())
        lookup_rows = 0 if global_bucket is None else int(global_bucket.total)
        eligible = seasons < int(cutoff)
        expected_rows = int(eligible.sum())
        used_seasons = sorted(set(int(value) for value in seasons.loc[eligible]))
        passed = lookup_rows == expected_rows and all(value < int(cutoff) for value in used_seasons)
        if not passed:
            failures.append(str(int(cutoff)))
        evidence[str(int(cutoff))] = {
            "lookup_rows": lookup_rows,
            "expected_strict_prior_rows": expected_rows,
            "used_seasons": used_seasons,
            "max_used_season": max(used_seasons) if used_seasons else None,
            "same_season_used": False,
            "future_season_used": False,
            "passed": passed,
        }
    proof_2025 = evidence.get("2025", {})
    return {
        "tested": True,
        "filtered_before_cutoff": True,
        "cutoffs": evidence,
        "failed_cutoffs": failures,
        "all_cutoffs_strict_prior_only": not failures,
        "lookup_2025_uses_2019_2024_only": bool(
            proof_2025.get("passed")
            and proof_2025.get("max_used_season") == 2024
            and proof_2025.get("lookup_rows") == EXPECTED_FILTERED_TRACKMAN_ROWS
        ),
    }


def audit_2025_legal_states(model: ContextPriorModel, *, expected_global_support: int | None = None) -> dict[str, Any]:
    """Check every legal 2025 L0 state without reading a test frame."""
    lookup = model.lookups.get(2025)
    if lookup is None:
        raise ContextPriorAuditError("2025 lookup is missing")
    global_bucket = lookup["global"].get(())
    if global_bucket is None or global_bucket.total <= 0:
        raise ContextPriorAuditError("2025 global history is empty")
    states: list[dict[str, Any]] = []
    level_counts: Counter[str] = Counter()
    failures: list[str] = []
    for balls, strikes, outs in itertools.product(range(4), range(3), range(3)):
        key0 = (balls, strikes, outs)
        key1 = (balls, strikes)
        record = _prior_record(lookup, key0, key1, model.support_threshold)
        level = str(record["backoff_level"])
        level_counts[level] += 1
        selected = _bucket_for(lookup, level, () if level == "global" else (key0 if level == "L0" else key1))
        probabilities = record["probabilities"]
        finite_simplex = bool(
            probabilities is not None
            and all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in probabilities)
            and abs(sum(float(value) for value in probabilities) - 1.0) <= SIMPLEX_TOLERANCE
        )
        support_exact = bool(selected is not None and int(record["support"]) == int(selected.total))
        if not finite_simplex:
            failures.append(f"state={key0}:invalid_probability")
        if not support_exact:
            failures.append(f"state={key0}:support_mismatch")
        states.append({
            "balls_before": balls, "strikes_before": strikes, "outs_before": outs,
            "backoff_level": level, "support": int(record["support"]),
            "support_exact": support_exact, "finite_simplex": finite_simplex,
        })
    global_support_match = expected_global_support is None or int(global_bucket.total) == int(expected_global_support)
    if not global_support_match:
        failures.append("2025_global_support_mismatch")
    return {
        "tested_l0_states": len(states), "expected_l0_states": 36,
        "all_legal_l0_states_tested": len(states) == 36,
        "valid_output_states": sum(int(item["finite_simplex"]) for item in states),
        "support_exact_states": sum(int(item["support_exact"]) for item in states),
        "level_counts": {level: int(level_counts.get(level, 0)) for level in LEVELS[:3]},
        "global_support": int(global_bucket.total),
        "expected_global_support": expected_global_support,
        "global_support_match": global_support_match,
        "state_failures": failures,
        "passed": not failures and len(states) == 36,
        "states": states,
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


def _empty_lookup() -> dict[str, Any]:
    return {"contract_version": CONTRACT_VERSION, "support_threshold": SUPPORT_THRESHOLD, "context_key_encoding": "colon-separated-integer-tuple", "seasons": {}, "available": False}


def _kill_acceptance(reason: str) -> dict[str, Any]:
    return {"verdict": "P_KILL", "kill_reasons": [reason], "p_fail_reasons": [], "feature_output_available": False, "season_diagnostics": {}, "determinism": {"tested": False}}


def build_report(main: pd.DataFrame, trackman: pd.DataFrame, features: pd.DataFrame | None, acceptance: Mapping[str, Any], taxonomy: Mapping[str, Any], *, repo_root: Path, config_path: Path, script_path: Path, config: Mapping[str, Any], source_scans: Mapping[str, Any] | None = None, source_quality: Mapping[str, Any] | None = None, state_proof: Mapping[str, Any] | None = None, pr13_authority: Mapping[str, Any] | None = None, v1_authority: Mapping[str, Any] | None = None, temporal_evidence: Mapping[str, Any] | None = None, lookup_parity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source_scans = dict(source_scans or {})
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION, "git_sha": _git_sha(repo_root), "runner_sha256": _sha256(script_path), "config_sha256": _sha256(config_path),
        "source_frame_hash_authority": "PR13_characterization_v1", "source_frame_hash_algorithm": FRAME_HASH_ALGORITHM, "source_projection": list(SOURCE_PROJECTION),
        "main_source_frame_sha256": source_scans.get("main", {}).get("source_frame_sha256"), "trackman_source_frame_sha256": source_scans.get("trackman", {}).get("source_frame_sha256"),
        "expected_source_hashes": dict(EXPECTED_PR13_SOURCE_HASHES),
        "pr13_authority": _json_value(pr13_authority or {"hashes_match": False}),
        "v1_authority": _json_value(v1_authority or {"hashes_match": False}),
        "source_scan_summary": {source: {"row_count": value.get("row_count"), "context_invalid_row_count": value.get("context_invalid_row_count"), "context_invalid_observation_count": value.get("context_invalid_observation_count")} for source, value in sorted(source_scans.items())},
        "main_row_count": int(len(main)), "trackman_row_count": int(len(trackman)),
        "projection": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)},
        "temporal_contract": {"rule": "main season S uses filtered TrackMan season < S", "same_season_used": False, "future_season_used": False, "row_independent": True, "filtered_before_lookup": True},
        "temporal_cutoff_evidence": _json_value(temporal_evidence or {"tested": False}),
        "source_quality": _json_value(source_quality or {"available": False}),
        "taxonomy": _json_value(taxonomy), "features": {"columns": list(FEATURE_COLUMNS), "backoff_level_not_feature": True, "simplex_tolerance": SIMPLEX_TOLERANCE},
        "synthetic_2025_legal_state_proof": _json_value(state_proof or {"tested": False}),
        "serialized_lookup_parity": _json_value(lookup_parity or {"tested": False}),
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
    json_path = output / "trackman_crosswalk_free_context_priors_v2_report.json"
    md_path = output / "trackman_crosswalk_free_context_priors_v2_report.md"
    lookup_path = output / "trackman_crosswalk_free_context_priors_v2_lookup.json"
    json_path.write_text(json.dumps(safe_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lookup_path.write_text(json.dumps(_json_value(lookup), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    acceptance = safe_report["acceptance"]
    lines = [
        "# TrackMan crosswalk-free context priors v2", "", f"- verdict: `{acceptance['verdict']}`",
        f"- canonical_report_sha256: `{safe_report['canonical_report_sha256']}`",
        "- entity/crosswalk access: `false`", "- target/test/Public/external access: `false`",
        "- PR13 source-quality residue is filtered before lookup construction; excluded rows never enter a lookup.",
        "- temporal rule: only filtered TrackMan seasons strictly before each main-row season",
        "- backoff: legal sparse contexts use frozen `L0 -> L1 -> global` hierarchy",
        "- `tm_cf_support` is the selected-level historical row count; backoff level is audit-only.",
        "- Level R exact pitch-row matching: `OUT_OF_SCOPE_NOT_IMPLEMENTED`.",
    ]
    lines.append(f"- 2025 legal-state proof: `{safe_report.get('synthetic_2025_legal_state_proof', {}).get('passed')}` over `{safe_report.get('synthetic_2025_legal_state_proof', {}).get('tested_l0_states')}` states; global support `{safe_report.get('synthetic_2025_legal_state_proof', {}).get('global_support')}`.")
    for season, value in sorted(acceptance["season_diagnostics"].items()):
        lines.append(f"- season {season} backoff counts: `{value['backoff_counts']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, lookup_path


def run_from_paths(main_csv: str | Path, trackman_csv: str | Path, output_dir: str | Path, repo_root: str | Path) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    repo = Path(repo_root).resolve()
    config_path = repo / "configs" / "trackman_crosswalk_free_context_priors_v2.json"
    config = load_contract_config(repo)
    main = read_main_features(main_csv)
    trackman = read_trackman_features(trackman_csv)
    source_scans = {"main": pr13_source_scan(main, source="main"), "trackman": pr13_source_scan(trackman, source="trackman")}
    taxonomy = validate_trackman_taxonomy(trackman)
    source_quality: dict[str, Any] = {"available": False}
    state_proof: dict[str, Any] = {"tested": False}
    temporal_evidence: dict[str, Any] = {"tested": False}
    lookup_parity: dict[str, Any] = {"tested": False}
    pr13_authority: dict[str, Any] = {"hashes_match": False}
    v1_authority: dict[str, Any] = {"hashes_match": False}
    features: pd.DataFrame | None = None
    lookup: dict[str, Any] = _empty_lookup()
    try:
        pr13_authority = validate_pr13_authority(repo)
        v1_authority = validate_v1_authority(repo)
        if source_scans["main"]["source_frame_sha256"] != EXPECTED_PR13_SOURCE_HASHES["main"] or source_scans["trackman"]["source_frame_sha256"] != EXPECTED_PR13_SOURCE_HASHES["trackman"]:
            raise ContextPriorAuditError("PR13_SOURCE_HASH_DRIFT")
        _validate_season_scope(main, MAIN_SEASONS, label="main")
        _validate_season_scope(trackman, TRACKMAN_HISTORICAL_SEASONS, label="TrackMan")
        _validate_context_frame(main, label="main")
        if taxonomy["status"] != "PASS":
            raise TaxonomyNotProven("TAXONOMY_NOT_PROVEN")
        filtered, source_quality = filter_trackman_history(trackman, enforce_official_residue=True)
        model = build_context_prior_model(filtered, seasons=(*MAIN_SEASONS, *INFERENCE_SEASONS), support_threshold=int(config["context"]["support_threshold"]))
        features = apply_context_prior_features(main, model)
        determinism = assess_determinism(main, filtered, model)
        temporal_evidence = assess_temporal_cutoffs(filtered, model)
        if not temporal_evidence["all_cutoffs_strict_prior_only"] or not temporal_evidence["lookup_2025_uses_2019_2024_only"]:
            raise TemporalCausalityNotProven("TEMPORAL_CAUSALITY_NOT_PROVEN")
        state_proof = audit_2025_legal_states(model, expected_global_support=EXPECTED_FILTERED_TRACKMAN_ROWS)
        if not state_proof["passed"]:
            raise ContextPriorAuditError("2025_LEGAL_STATE_PROOF_FAILED")
        lookup = build_aggregate_lookup(model)
        lookup_parity = assess_serialized_lookup_parity(main, features, lookup)
        determinism["serialized_lookup_parity_tested"] = True
        determinism["serialized_lookup_parity_equal"] = bool(lookup_parity["equal"])
        acceptance = evaluate_acceptance(main, features, filtered, determinism=determinism)
    except ContextDomainNotProven:
        acceptance = _kill_acceptance("CONTEXT_DOMAIN_NOT_PROVEN")
    except TaxonomyNotProven:
        acceptance = _kill_acceptance("TAXONOMY_NOT_PROVEN")
    except TemporalCausalityNotProven:
        acceptance = _kill_acceptance("TEMPORAL_CAUSALITY_NOT_PROVEN")
    except ContextPriorAuditError as exc:
        reason = str(exc) if str(exc) in {"PR13_SOURCE_AUTHORITY_DRIFT", "PR13_SOURCE_HASH_DRIFT", "V1_SOURCE_AUTHORITY_DRIFT", "V1_SHARED_SEMANTICS_DRIFT", "SOURCE_QUALITY_RESIDUE_DRIFT", "2025_LEGAL_STATE_PROOF_FAILED"} else "SOURCE_QUALITY_RESIDUE_DRIFT"
        acceptance = _kill_acceptance(reason)
    if acceptance["verdict"] == "P_KILL":
        features = None
        lookup = _empty_lookup()
    report = build_report(main, trackman, features, acceptance, taxonomy, repo_root=repo, config_path=config_path, script_path=Path(__file__).resolve(), config=config, source_scans=source_scans, source_quality=source_quality, state_proof=state_proof, pr13_authority=pr13_authority, v1_authority=v1_authority, temporal_evidence=temporal_evidence, lookup_parity=lookup_parity)
    return report, write_outputs(report, lookup, output_dir, repo_root=repo)


def static_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_contract_config(root)
    pr13_authority = validate_pr13_authority(root)
    v1_authority = validate_v1_authority(root)
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
    if tuple(SOURCE_PROJECTION) != tuple(_pr13.PROJECTION) or FRAME_HASH_ALGORITHM != _pr13.FRAME_HASH_ALGORITHM:
        raise ContextPriorAuditError("PR13 source-hash authority is not exact")
    if config["provenance"]["expected_main_source_frame_sha256"] != EXPECTED_PR13_SOURCE_HASHES["main"] or config["provenance"]["expected_trackman_source_frame_sha256"] != EXPECTED_PR13_SOURCE_HASHES["trackman"]:
        raise ContextPriorAuditError("PR13 expected source hashes diverge")
    if config["provenance"]["v1_runner_sha256"] != V1_RUNNER_SHA256 or config["provenance"]["v1_config_sha256"] != V1_CONFIG_SHA256:
        raise ContextPriorAuditError("V1 expected authority hashes diverge")
    return {"contract_version": CONTRACT_VERSION, "config_valid": True, "projection_columns": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)}, "target_access": False, "target_column_in_projection": False, "test_access": False, "test_distribution_access": False, "public_access": False, "external_information_access": False, "trackman_entity_access": False, "trackman_physics_access": False, "current_pitch_measurement_access": False, "model_training_or_scoring": False, "gpu_used": False, "branch_inert": True, "label_access_ledger": [], "level_r_exact_matcher": False, "backoff_level_is_model_feature": False, "l1_below_threshold_is_failure": False, "kill_outcome_reportable": True, "execution_error_distinct_from_kill": True, "pr13_source_hash_authority": True, "pr13_authority_hashes": pr13_authority, "v1_lookup_authority": v1_authority, "official_residue_contract": {"raw": EXPECTED_SOURCE_ROW_COUNTS["trackman"], "excluded": EXPECTED_EXCLUDED_TRACKMAN_ROWS, "retained": EXPECTED_FILTERED_TRACKMAN_ROWS}, "synthetic_2025_legal_states": 36, "static_pass": True}


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
