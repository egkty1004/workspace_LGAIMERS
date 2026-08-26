#!/usr/bin/env python3
"""Aggregate characterization of the PR #12 TrackMan context-domain kill.

This is a target-free diagnostic.  It reads only the four approved context
columns from each source and reports the difference between the executable
PR #12 semantic classifier and a separate raw-token diagnostic.  It never
normalizes, filters, joins, or constructs model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CONTRACT_VERSION = "aimers9-trackman-context-domain-characterization-v1"
MAIN_SOURCE = "main"
TRACKMAN_SOURCE = "trackman"
PROJECTION = ("season", "balls_before", "strikes_before", "outs_before")
MAIN_PROJECTION = PROJECTION
TRACKMAN_PROJECTION = PROJECTION
CONTEXT_COLUMNS = ("balls_before", "strikes_before", "outs_before")
EXPECTED_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
DOMAINS = {
    "balls_before": frozenset((0, 1, 2, 3)),
    "strikes_before": frozenset((0, 1, 2)),
    "outs_before": frozenset((0, 1, 2)),
}
FRAME_HASH_ALGORITHM = "canonical-multiset-json-v1"
MISSING_SEASON = "__INVALID_SEASON__"
INVALID_REASONS = ("missing", "unparseable", "nonfinite", "non_integer", "out_of_domain")
MACHINE_VERDICTS = ("SCAN_COMPLETE", "NOT_PROVEN")
NOT_PROVEN_REASONS = (
    "PRIOR_KILL_NOT_REPRODUCED",
    "SEASON_STRATIFICATION_NOT_PROVEN",
    "PROJECTION_NOT_PROVEN",
    "SCAN_NOT_COMPLETE",
)
FORBIDDEN_MAIN = frozenset({"control_success", "row_id", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"})
FORBIDDEN_TRACKMAN = frozenset({
    "pitcher_trackman_id", "batter_trackman_id", "trackman_id", "game_id",
    "trackman_game_id", "pitch_no", "pitch_type_group", "tagged_pitch_type",
    "auto_pitch_type", "rel_speed", "spin_rate", "induced_vert_break",
    "horz_break", "extension", "rel_height", "rel_side", "zone_speed",
    "release", "movement",
})


class CharacterizationError(RuntimeError):
    """A fail-closed contract, schema, privacy, or output error."""


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CharacterizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _runtime_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "scope": {
            "main_projection": list(MAIN_PROJECTION),
            "trackman_projection": list(TRACKMAN_PROJECTION),
            "expected_seasons": list(EXPECTED_SEASONS),
            "target_free": True,
            "test_free": True,
            "leaderboard_free": True,
            "external_free": True,
            "row_id_free": True,
            "entity_free": True,
            "physics_free": True,
        },
        "semantic_classifier": {
            "authority": "PR12_integer_context_key_semantics",
            "number_conversion": "float(value), finite only",
            "integer_conversion": "finite number whose is_integer() is true",
            "missing_semantics": "pandas.isna before float conversion",
            "domains": {key: sorted(values) for key, values in DOMAINS.items()},
            "invalid_policy": "classify aggregate only; never normalize/filter",
            "raw_lexical_diagnostics_are_not_classifier": True,
        },
        "diagnostics": {
            "season_stratification": "integer season values 2019-2024; invalid season is NOT_PROVEN",
            "legal_domain_histograms": "overall and season-stratified sorted legal value counts for every projected field",
            "invalid_concentration": "context-invalid counts/fractions by source, source x field, and source x season",
            "row_signature": "source x season x invalid-fields x reason/value signature aggregate counts",
            "row_order_invariant": True,
        },
        "verdict": {
            "machine_verdicts": list(MACHINE_VERDICTS),
            "not_proven_reasons": list(NOT_PROVEN_REASONS),
            "automatic_filterable_or_incompatible_classification": False,
            "external_review_interpretation_required": True,
            "prior_expected_verdict": "P_KILL",
            "prior_expected_reason": "CONTEXT_DOMAIN_NOT_PROVEN",
        },
        "output": {
            "aggregate_only": True,
            "external_only": True,
            "report_json": "trackman_context_domain_characterization_v1_report.json",
            "report_markdown": "trackman_context_domain_characterization_v1_report.md",
            "allow_nan": False,
            "include_row_ids": False,
            "include_targets": False,
            "include_entities": False,
            "include_physics": False,
            "include_raw_rows": False,
        },
    }


def _contract_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    value = {"contract_version": config.get("contract_version")}
    for key in ("scope", "semantic_classifier", "diagnostics", "verdict", "output"):
        section = config.get(key)
        if not isinstance(section, Mapping):
            raise CharacterizationError(f"missing contract section: {key}")
        value[key] = dict(section)
    value["scope"]["main_projection"] = list(value["scope"]["main_projection"])
    value["scope"]["trackman_projection"] = list(value["scope"]["trackman_projection"])
    value["scope"]["expected_seasons"] = list(value["scope"]["expected_seasons"])
    value["semantic_classifier"]["domains"] = {
        str(key): sorted(int(item) for item in items)
        for key, items in value["semantic_classifier"]["domains"].items()
    }
    value["verdict"]["machine_verdicts"] = list(value["verdict"]["machine_verdicts"])
    value["verdict"]["not_proven_reasons"] = list(value["verdict"]["not_proven_reasons"])
    return value


def validate_contract_config(config: Mapping[str, Any]) -> None:
    if config.get("documentary_snapshot_checked_against_runtime_contract") is not True:
        raise CharacterizationError("config is not marked runtime-authoritative")
    if _contract_projection(config) != _runtime_contract():
        raise CharacterizationError("contract config diverges from runtime contract")


def load_contract_config(repo_root: str | Path) -> dict[str, Any]:
    path = Path(repo_root) / "configs" / "trackman_context_domain_characterization_v1.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise CharacterizationError(f"cannot load contract config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CharacterizationError("contract config must be an object")
    validate_contract_config(config)
    return config


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if hasattr(result, "__len__"):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    """Match PR #12's _number semantics exactly."""
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    """Match PR #12's _integer semantics exactly."""
    result = _number(value)
    return int(result) if result is not None and result.is_integer() else None


def _season(value: Any) -> int | None:
    return _integer(value)


def _context_key(row: Mapping[str, Any], columns: Sequence[str] = CONTEXT_COLUMNS) -> tuple[int, ...] | None:
    """Match PR #12's _context_key domain checks exactly."""
    values: list[int] = []
    for column in columns:
        value = _integer(row.get(column))
        bounds = DOMAINS[column]
        if value is None or value not in bounds:
            return None
        values.append(value)
    return tuple(values)


def _classify_v1(value: Any, field: str) -> dict[str, Any]:
    """Classify using executable PR #12 semantics; never use raw lexical form."""
    if field == "season":
        number = _number(value)
        integer = _integer(value)
        if _is_missing(value):
            return {"status": "invalid", "reason": "missing", "value": None}
        if number is None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return {"status": "invalid", "reason": "unparseable", "value": str(value)}
            if not math.isfinite(parsed):
                return {"status": "invalid", "reason": "nonfinite", "value": str(value)}
            return {"status": "invalid", "reason": "non_integer", "value": str(value)}
        if integer is None:
            return {"status": "invalid", "reason": "non_integer", "value": str(value)}
        if integer not in EXPECTED_SEASONS:
            return {"status": "invalid", "reason": "out_of_domain", "value": str(integer), "numeric": integer}
        return {"status": "legal", "value": integer, "numeric": integer}

    number = _number(value)
    integer = _integer(value)
    if _is_missing(value):
        return {"status": "invalid", "reason": "missing", "value": None}
    if number is None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "unparseable", "value": str(value)}
        if not math.isfinite(parsed):
            return {"status": "invalid", "reason": "nonfinite", "value": str(value)}
        return {"status": "invalid", "reason": "non_integer", "value": str(value)}
    if integer is None:
        return {"status": "invalid", "reason": "non_integer", "value": str(value), "numeric": number}
    if integer not in DOMAINS[field]:
        return {"status": "invalid", "reason": "out_of_domain", "value": str(integer), "numeric": integer}
    return {"status": "legal", "value": integer, "numeric": integer}


def _value_key(value: Any) -> str:
    if _is_missing(value):
        return "<MISSING>"
    number = _number(value)
    if number is not None:
        if number.is_integer():
            return str(int(number))
        return format(number, ".17g")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return str(value)
    return str(value)


def _canonical_cell(value: Any) -> Any:
    if _is_missing(value):
        return None
    number = _number(value)
    if number is not None:
        return float(number)
    return str(value)


def _raw_lexical_kind(value: Any) -> str:
    if _is_missing(value):
        return "pandas_missing"
    text = str(value)
    if text == "":
        return "empty"
    stripped = text.strip()
    if text != stripped:
        return "whitespace_variant"
    try:
        number = float(text)
    except (TypeError, ValueError):
        return "non_numeric_token"
    if not math.isfinite(number):
        return "nonfinite_token"
    if number.is_integer():
        if any(marker in text.lower() for marker in (".", "e")):
            return "integer_numeric_noninteger_lexeme"
        return "integer_lexeme"
    return "decimal_lexeme"


def validate_projection(frame: pd.DataFrame, *, source: str) -> None:
    expected = MAIN_PROJECTION if source == MAIN_SOURCE else TRACKMAN_PROJECTION
    if tuple(frame.columns) != tuple(expected):
        raise CharacterizationError(f"{source} projection must be exactly {expected}")
    forbidden = FORBIDDEN_MAIN if source == MAIN_SOURCE else FORBIDDEN_TRACKMAN
    if forbidden.intersection(frame.columns):
        raise CharacterizationError(f"{source} projection contains forbidden columns")


class _Accumulator:
    def __init__(self, source: str) -> None:
        self.source = source
        self.total_rows = 0
        self.row_counter: Counter[tuple[Any, ...]] = Counter()
        self.row_signatures: Counter[tuple[str, tuple[str, ...], tuple[tuple[str, str, str], ...]]] = Counter()
        self.season_counts: Counter[str] = Counter()
        self.season_reason_counts: dict[str, Counter[str]] = {str(season): Counter() for season in EXPECTED_SEASONS}
        self.season_reason_counts[MISSING_SEASON] = Counter()
        self.context_invalid_row_count = 0
        self.context_invalid_observation_count = 0
        self.context_invalid_by_season: Counter[str] = Counter()
        self.fields: dict[str, dict[str, Any]] = {field: self._new_field() for field in (*PROJECTION,)}

    @staticmethod
    def _new_field() -> dict[str, Any]:
        return {
            "total_rows": 0,
            "missing_rows": 0,
            "finite_rows": 0,
            "nonfinite_rows": 0,
            "integer_like_rows": 0,
            "non_integer_rows": 0,
            "legal_rows": 0,
            "invalid_rows": 0,
            "invalid_reason_counts": Counter(),
            "invalid_value_counts": {reason: Counter() for reason in INVALID_REASONS},
            "legal_value_counts": Counter(),
            "legal_value_counts_by_season": {
                str(season): Counter() for season in EXPECTED_SEASONS
            } | {MISSING_SEASON: Counter()},
            "observed_min": None,
            "observed_max": None,
            "season": {
                str(season): {"total_rows": 0, "invalid_rows": 0, "legal_rows": 0, "reason_counts": Counter()}
                for season in EXPECTED_SEASONS
            },
            "season_invalid": {"total_rows": 0, "invalid_rows": 0, "legal_rows": 0, "reason_counts": Counter()},
            "raw": {
                "total_rows": 0,
                "empty_tokens": 0,
                "whitespace_tokens": 0,
                "token_counts": Counter(),
                "lexical_kind_counts": Counter(),
                "semantic_v1_legal_with_lexical_variation": 0,
            },
        }

    def add_frame(self, frame: pd.DataFrame, raw_frame: pd.DataFrame | None = None) -> None:
        validate_projection(frame, source=self.source)
        if raw_frame is None:
            raw_frame = frame
        validate_projection(raw_frame, source=self.source)
        if len(frame) != len(raw_frame):
            raise CharacterizationError(f"{self.source} semantic/raw row count mismatch")
        for semantic_values, raw_values in zip(frame.itertuples(index=False, name=None), raw_frame.itertuples(index=False, name=None)):
            self._add_row(semantic_values, raw_values)

    def _add_row(self, semantic_values: Sequence[Any], raw_values: Sequence[Any]) -> None:
        self.total_rows += 1
        self.row_counter[tuple(_canonical_cell(value) for value in semantic_values)] += 1
        row = dict(zip(PROJECTION, semantic_values))
        raw_row = dict(zip(PROJECTION, raw_values))
        season_info = _classify_v1(row["season"], "season")
        if season_info["status"] == "legal":
            season_key = str(int(season_info["value"]))
        else:
            season_key = MISSING_SEASON
        self.season_counts[season_key] += 1
        if season_info["status"] != "legal":
            self.season_reason_counts[season_key][season_info["reason"]] += 1

        invalid_fields: list[str] = []
        signature: list[tuple[str, str, str]] = []
        context_invalid = 0
        for field in PROJECTION:
            info = _classify_v1(row[field], field)
            state = self.fields[field]
            state["total_rows"] += 1
            raw_state = state["raw"]
            raw_state["total_rows"] += 1
            raw_text = "" if _is_missing(raw_row[field]) else str(raw_row[field])
            raw_kind = _raw_lexical_kind(raw_row[field])
            raw_state["token_counts"][raw_text] += 1
            raw_state["lexical_kind_counts"][raw_kind] += 1
            if raw_kind == "empty":
                raw_state["empty_tokens"] += 1
            if raw_kind == "whitespace_variant":
                raw_state["whitespace_tokens"] += 1
            if info["status"] == "legal":
                state["legal_rows"] += 1
                state["legal_value_counts"][str(info["value"])] += 1
                state["legal_value_counts_by_season"][season_key][str(info["value"])] += 1
                if raw_kind in {"whitespace_variant", "integer_numeric_noninteger_lexeme", "decimal_lexeme"}:
                    raw_state["semantic_v1_legal_with_lexical_variation"] += 1
            else:
                state["invalid_rows"] += 1
                reason = str(info["reason"])
                state["invalid_reason_counts"][reason] += 1
                state["invalid_value_counts"][reason][_value_key(row[field])] += 1
                invalid_fields.append(field)
                signature.append((field, reason, _value_key(row[field])))
                if field in CONTEXT_COLUMNS:
                    context_invalid += 1
            number = _number(row[field])
            if number is not None:
                state["finite_rows"] += 1
                state["observed_min"] = number if state["observed_min"] is None else min(state["observed_min"], number)
                state["observed_max"] = number if state["observed_max"] is None else max(state["observed_max"], number)
            else:
                state["nonfinite_rows"] += 1
            if _integer(row[field]) is not None:
                state["integer_like_rows"] += 1
            else:
                state["non_integer_rows"] += 1
            bucket = state["season"].get(season_key, state["season_invalid"])
            bucket["total_rows"] += 1
            if info["status"] == "legal":
                bucket["legal_rows"] += 1
            else:
                bucket["invalid_rows"] += 1
                bucket["reason_counts"][str(info["reason"])] += 1
        if context_invalid:
            self.context_invalid_row_count += 1
            self.context_invalid_observation_count += context_invalid
            self.context_invalid_by_season[season_key] += 1
        if invalid_fields:
            self.row_signatures[(season_key, tuple(sorted(invalid_fields)), tuple(sorted(signature)))] += 1

    @staticmethod
    def _sorted_counter(value: Counter[str]) -> dict[str, int]:
        return {str(key): int(value[key]) for key in sorted(value, key=str)}

    def _field_output(self, state: Mapping[str, Any]) -> dict[str, Any]:
        season_output: dict[str, Any] = {}
        for season in EXPECTED_SEASONS:
            item = state["season"][str(season)]
            season_output[str(season)] = {
                "total_rows": int(item["total_rows"]),
                "invalid_rows": int(item["invalid_rows"]),
                "invalid_fraction": (float(item["invalid_rows"]) / item["total_rows"] if item["total_rows"] else 0.0),
                "legal_rows": int(item["legal_rows"]),
                "reason_counts": self._sorted_counter(item["reason_counts"]),
            }
        invalid_item = state["season_invalid"]
        season_output[MISSING_SEASON] = {
            "total_rows": int(invalid_item["total_rows"]),
            "invalid_rows": int(invalid_item["invalid_rows"]),
            "invalid_fraction": (float(invalid_item["invalid_rows"]) / invalid_item["total_rows"] if invalid_item["total_rows"] else 0.0),
            "legal_rows": int(invalid_item["legal_rows"]),
            "reason_counts": self._sorted_counter(invalid_item["reason_counts"]),
        }
        raw = state["raw"]
        v1 = {
            "total_rows": int(state["total_rows"]),
            "missing_rows": int(state["invalid_reason_counts"].get("missing", 0)),
            "finite_rows": int(state["finite_rows"]),
            "nonfinite_rows": int(state["invalid_reason_counts"].get("nonfinite", 0)),
            "integer_like_rows": int(state["integer_like_rows"]),
            "non_integer_rows": int(state["invalid_reason_counts"].get("non_integer", 0) + state["invalid_reason_counts"].get("unparseable", 0)),
            "legal_rows": int(state["legal_rows"]),
            "invalid_rows": int(state["invalid_rows"]),
            "invalid_fraction": (float(state["invalid_rows"]) / state["total_rows"] if state["total_rows"] else 0.0),
            "invalid_reason_counts": self._sorted_counter(state["invalid_reason_counts"]),
            "invalid_value_counts": {reason: self._sorted_counter(state["invalid_value_counts"][reason]) for reason in INVALID_REASONS},
            "out_of_domain_unique_values": self._sorted_counter(state["invalid_value_counts"]["out_of_domain"]),
            "legal_domain_value_histogram": self._sorted_counter(state["legal_value_counts"]),
            "legal_domain_value_histogram_by_season": {
                season: self._sorted_counter(state["legal_value_counts_by_season"][season])
                for season in (*[str(value) for value in EXPECTED_SEASONS], MISSING_SEASON)
            },
            "observed_min": None if state["observed_min"] is None else float(state["observed_min"]),
            "observed_max": None if state["observed_max"] is None else float(state["observed_max"]),
            "season_diagnostics": season_output,
            "v1_semantics_authority": "PR12_number_integer_context_key",
        }
        raw_output = {
            "total_rows": int(raw["total_rows"]),
            "empty_tokens": int(raw["empty_tokens"]),
            "whitespace_tokens": int(raw["whitespace_tokens"]),
            "token_counts": self._sorted_counter(raw["token_counts"]),
            "lexical_kind_counts": self._sorted_counter(raw["lexical_kind_counts"]),
            "semantic_v1_legal_with_lexical_variation": int(raw["semantic_v1_legal_with_lexical_variation"]),
            "authority": "diagnostic_only_not_used_for_v1_compatibility",
        }
        return {"v1_compatibility": v1, "raw_lexical_diagnostics": raw_output}

    def finalize(self) -> dict[str, Any]:
        rows = [{"row": list(key), "count": int(count)} for key, count in sorted(self.row_counter.items(), key=lambda item: repr(item[0]))]
        row_signatures = [
            {
                "source": self.source,
                "season": season,
                "invalid_fields": list(fields),
                "reason_value_signature": [
                    {"field": field, "reason": reason, "value": value}
                    for field, reason, value in values
                ],
                "row_count": int(count),
            }
            for (season, fields, values), count in sorted(self.row_signatures.items(), key=lambda item: repr(item[0]))
        ]
        return {
            "source": self.source,
            "scan_complete": True,
            "row_count": int(self.total_rows),
            "source_frame_sha256": canonical_hash({"columns": list(PROJECTION), "rows": rows}),
            "fields": {field: self._field_output(self.fields[field]) for field in PROJECTION},
            "season_diagnostics": {
                "expected_seasons": list(EXPECTED_SEASONS),
                "row_counts": {str(season): int(self.season_counts.get(str(season), 0)) for season in EXPECTED_SEASONS},
                "invalid_season_rows": int(self.season_counts.get(MISSING_SEASON, 0)),
                "invalid_season_reason_counts": {
                    season: self._sorted_counter(counts)
                    for season, counts in sorted(self.season_reason_counts.items())
                    if counts
                },
            },
            "context_invalid_row_count": int(self.context_invalid_row_count),
            "context_invalid_observation_count": int(self.context_invalid_observation_count),
            "context_invalid_by_season": {
                str(season): int(self.context_invalid_by_season.get(str(season), 0))
                for season in EXPECTED_SEASONS
            } | {MISSING_SEASON: int(self.context_invalid_by_season.get(MISSING_SEASON, 0))},
            "context_invalid_observation_counts_by_field": {
                field: int(self.fields[field]["invalid_rows"]) for field in CONTEXT_COLUMNS
            },
            "row_signature_counts": row_signatures,
        }


def scan_frame(frame: pd.DataFrame, *, source: str, raw_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    accumulator = _Accumulator(source)
    accumulator.add_frame(frame, raw_frame=raw_frame)
    return accumulator.finalize()


def scan_csv(path: str | Path, *, source: str, chunksize: int = 100_000) -> dict[str, Any]:
    path = Path(path)
    expected = MAIN_PROJECTION if source == MAIN_SOURCE else TRACKMAN_PROJECTION
    try:
        header = pd.read_csv(path, nrows=0)
        missing = [column for column in expected if column not in header.columns]
        if missing:
            raise CharacterizationError(f"{source} source missing approved columns: {missing}")
        semantic_reader = pd.read_csv(path, usecols=list(expected), chunksize=chunksize)
        raw_reader = pd.read_csv(path, usecols=list(expected), dtype=str, keep_default_na=False, na_filter=False, chunksize=chunksize)
        accumulator = _Accumulator(source)
        for semantic_chunk in semantic_reader:
            try:
                raw_chunk = next(raw_reader)
            except StopIteration as exc:
                raise CharacterizationError(f"{source} semantic/raw chunk count mismatch") from exc
            accumulator.add_frame(semantic_chunk, raw_frame=raw_chunk)
        try:
            next(raw_reader)
        except StopIteration:
            return accumulator.finalize()
        raise CharacterizationError(f"{source} raw projection has extra rows")
    except CharacterizationError:
        raise
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise CharacterizationError(f"{source} projection scan failed for {path}: {exc}") from exc


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
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    body = {key: value for key, value in report.items() if key != "canonical_report_sha256"}
    return canonical_hash(_json_value(body))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _outside_repo(output_dir: str | Path, repo_root: str | Path) -> None:
    output = Path(output_dir).resolve()
    repo = Path(repo_root).resolve()
    if output == repo or repo in output.parents:
        raise CharacterizationError("output directory must be outside repository")


def _combined_row_signatures(main: Mapping[str, Any], trackman: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted([*main.get("row_signature_counts", []), *trackman.get("row_signature_counts", [])], key=lambda item: repr(item))


def _context_invalid_count(main: Mapping[str, Any], trackman: Mapping[str, Any]) -> int:
    return int(main.get("context_invalid_row_count", 0)) + int(trackman.get("context_invalid_row_count", 0))


def _fraction(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _context_concentration(source_diagnostics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic aggregate invalid counts, without interpretation."""
    by_source: list[dict[str, Any]] = []
    by_source_field: list[dict[str, Any]] = []
    by_source_season: list[dict[str, Any]] = []
    for source in (MAIN_SOURCE, TRACKMAN_SOURCE):
        diagnostic = source_diagnostics[source]
        total = int(diagnostic.get("row_count", 0))
        source_count = int(diagnostic.get("context_invalid_row_count", 0))
        by_source.append({
            "source": source,
            "context_invalid_row_count": source_count,
            "total_row_count": total,
            "context_invalid_fraction": _fraction(source_count, total),
        })
        field_counts = diagnostic.get("context_invalid_observation_counts_by_field", {})
        for field in CONTEXT_COLUMNS:
            count = int(field_counts.get(field, 0))
            by_source_field.append({
                "source": source,
                "field": field,
                "context_invalid_observation_count": count,
                "total_row_count": total,
                "context_invalid_fraction": _fraction(count, total),
            })
        season_counts = diagnostic.get("context_invalid_by_season", {})
        row_counts = diagnostic.get("season_diagnostics", {}).get("row_counts", {})
        for season in (*[str(value) for value in EXPECTED_SEASONS], MISSING_SEASON):
            count = int(season_counts.get(season, 0))
            if season == MISSING_SEASON:
                season_total = int(diagnostic.get("season_diagnostics", {}).get("invalid_season_rows", 0))
            else:
                season_total = int(row_counts.get(season, 0))
            by_source_season.append({
                "source": source,
                "season": season,
                "context_invalid_row_count": count,
                "total_row_count": season_total,
                "context_invalid_fraction": _fraction(count, season_total),
            })
    return {
        "context_invalid_by_source": by_source,
        "context_invalid_by_source_field": by_source_field,
        "context_invalid_by_source_season": by_source_season,
        "row_signature_counts": _combined_row_signatures(source_diagnostics[MAIN_SOURCE], source_diagnostics[TRACKMAN_SOURCE]),
    }


def build_report(
    main: Mapping[str, Any],
    trackman: Mapping[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    script_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not main.get("scan_complete") or not trackman.get("scan_complete"):
        reasons.append("SCAN_NOT_COMPLETE")
    if main.get("season_diagnostics", {}).get("invalid_season_rows", 0) or trackman.get("season_diagnostics", {}).get("invalid_season_rows", 0):
        reasons.append("SEASON_STRATIFICATION_NOT_PROVEN")
    if _context_invalid_count(main, trackman) == 0:
        reasons.append("PRIOR_KILL_NOT_REPRODUCED")
    verdict = "NOT_PROVEN" if reasons else "SCAN_COMPLETE"
    concentration = _context_concentration({MAIN_SOURCE: main, TRACKMAN_SOURCE: trackman})
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "git_sha": _git_sha(repo_root),
        "runner_sha256": _sha256(script_path),
        "config_sha256": _sha256(config_path),
        "source_frame_hash_algorithm": FRAME_HASH_ALGORITHM,
        "projection": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)},
        "source_provenance": {
            "main": {"source_name": MAIN_SOURCE, "path_basename": "not_recorded_in_static_report"},
            "trackman": {"source_name": TRACKMAN_SOURCE, "path_basename": "not_recorded_in_static_report"},
        },
        "source_diagnostics": {"main": _json_value(main), "trackman": _json_value(trackman)},
        "row_signature_counts": concentration["row_signature_counts"],
        "concentration": concentration,
        "verdict": verdict,
        "reasons": sorted(set(reasons)),
        "prior_kill_reproduction": {
            "prior_verdict": "P_KILL",
            "prior_reason": "CONTEXT_DOMAIN_NOT_PROVEN",
            "v1_compatible_context_invalid_row_count": _context_invalid_count(main, trackman),
            "status": "PRIOR_KILL_NOT_REPRODUCED" if _context_invalid_count(main, trackman) == 0 else "V1_COMPATIBLE_INVALID_OBSERVED",
            "machine_verdict_does_not_interpret_filterability": True,
        },
        "source_distinction": {
            "trackman_invalid_future_candidate": "source_local_exclusion_contract_only_after_external_review",
            "main_invalid_future_candidate": "row_local_fallback_hypothesis_required; no automatic filtering",
        },
        "scope": {
            "target_access": False,
            "target_column_in_projection": False,
            "test_access": False,
            "test_distribution_access": False,
            "public_access": False,
            "public_leaderboard_evidence": False,
            "external_information_access": False,
            "trackman_entity_access": False,
            "trackman_physics_access": False,
            "current_pitch_measurement_access": False,
            "model_training_or_scoring": False,
            "gpu_used": False,
            "label_access_ledger": [],
            "branch_inert": True,
        },
        "privacy": {
            "aggregate_only": True,
            "row_ids": False,
            "target_values": False,
            "entity_ids": False,
            "physics": False,
            "raw_rows": False,
        },
        "semantic_contract": _runtime_contract()["semantic_classifier"],
        "config_runtime": {
            "config_runtime_match": _contract_projection(config) == _runtime_contract(),
            "runtime_contract_sha256": canonical_hash(_runtime_contract()),
        },
    }
    report["canonical_report_sha256"] = canonical_report_hash(report)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TrackMan context-domain characterization v1",
        "",
        f"- machine verdict: `{report['verdict']}`",
        f"- reasons: `{report['reasons']}`",
        f"- canonical report SHA-256: `{report['canonical_report_sha256']}`",
        "- PR #12 semantic classifier: `_number` / `_integer` / `_context_key` compatible",
        "- raw lexical diagnostics are separate and never classify v1 compatibility",
        "- no automatic FILTERABLE_RESIDUE or DOMAIN_INCOMPATIBLE conclusion is emitted",
        "- interpretation of raw prevalence remains external-review-only",
        "",
        "## Aggregate source census",
        "",
    ]
    for source in (MAIN_SOURCE, TRACKMAN_SOURCE):
        diagnostic = report["source_diagnostics"][source]
        lines.append(f"### {source}")
        lines.append(f"- rows: `{diagnostic['row_count']}`")
        lines.append(f"- projected frame SHA-256: `{diagnostic['source_frame_sha256']}`")
        lines.append(f"- v1-compatible context invalid rows: `{diagnostic['context_invalid_row_count']}`")
        for field in PROJECTION:
            v1 = diagnostic["fields"][field]["v1_compatibility"]
            lines.append(f"- `{field}` invalid: `{v1['invalid_rows']}/{v1['total_rows']}` ({v1['invalid_fraction']:.12g})")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "TrackMan invalid observations may be considered only in a future source-local exclusion brief.",
        "Main invalid observations are not declared filterable; a future row-local fallback hypothesis is required.",
        "No target, ID, test, Public, physics, or raw-row material is included.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], output_dir: str | Path, *, repo_root: Path) -> tuple[Path, Path]:
    _outside_repo(output_dir, repo_root)
    output = Path(output_dir)
    output.mkdir(parents=False, exist_ok=False)
    safe_report = _json_value(report)
    json_path = output / "trackman_context_domain_characterization_v1_report.json"
    md_path = output / "trackman_context_domain_characterization_v1_report.md"
    json_path.write_text(json.dumps(safe_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(safe_report), encoding="utf-8")
    return json_path, md_path


def run_from_paths(main_csv: str | Path, trackman_csv: str | Path, output_dir: str | Path, repo_root: str | Path) -> tuple[dict[str, Any], tuple[Path, Path]]:
    repo = Path(repo_root).resolve()
    config_path = repo / "configs" / "trackman_context_domain_characterization_v1.json"
    script_path = Path(__file__).resolve()
    config = load_contract_config(repo)
    main = scan_csv(main_csv, source=MAIN_SOURCE)
    trackman = scan_csv(trackman_csv, source=TRACKMAN_SOURCE)
    report = build_report(main, trackman, repo_root=repo, config_path=config_path, script_path=script_path, config=config)
    return report, write_outputs(report, output_dir, repo_root=repo)


def static_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_contract_config(root)
    if tuple(MAIN_PROJECTION) != tuple(TRACKMAN_PROJECTION):
        raise CharacterizationError("source projections diverge")
    if set(MAIN_PROJECTION).intersection(FORBIDDEN_MAIN | FORBIDDEN_TRACKMAN):
        raise CharacterizationError("forbidden field appears in approved projection")
    if "control_success" in MAIN_PROJECTION or "control_success" in TRACKMAN_PROJECTION:
        raise CharacterizationError("target appears in projection")
    if set(config["verdict"]["machine_verdicts"]) != set(MACHINE_VERDICTS):
        raise CharacterizationError("machine verdict set diverges")
    if config["verdict"]["automatic_filterable_or_incompatible_classification"]:
        raise CharacterizationError("automatic interpretation is forbidden")
    if not config["semantic_classifier"]["raw_lexical_diagnostics_are_not_classifier"]:
        raise CharacterizationError("raw lexical diagnostics cannot become classifier")
    return {
        "contract_version": CONTRACT_VERSION,
        "config_valid": True,
        "projection": {"main": list(MAIN_PROJECTION), "trackman": list(TRACKMAN_PROJECTION)},
        "target_access": False,
        "target_column_in_projection": False,
        "test_access": False,
        "test_distribution_access": False,
        "public_access": False,
        "public_leaderboard_evidence": False,
        "external_information_access": False,
        "trackman_entity_access": False,
        "trackman_physics_access": False,
        "current_pitch_measurement_access": False,
        "model_training_or_scoring": False,
        "gpu_used": False,
        "label_access_ledger": [],
        "aggregate_only": True,
        "raw_lexical_diagnostics_separate": True,
        "automatic_interpretation": False,
        "machine_verdicts": list(MACHINE_VERDICTS),
        "row_order_invariant": True,
        "static_pass": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static", help="static contract/firewall check; no data access")
    static.add_argument("--repo-root", required=True)
    run = subparsers.add_parser("run", help="official aggregate characterization scan")
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
            print(json.dumps({"verdict": report["verdict"], "reasons": report["reasons"], "canonical_report_sha256": report["canonical_report_sha256"], "json": str(paths[0]), "markdown": str(paths[1])}, sort_keys=True, allow_nan=False))
        return 0
    except (CharacterizationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
