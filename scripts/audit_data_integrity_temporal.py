#!/usr/bin/env python3
"""CHEAP controls for the Aimers 9 data-integrity and temporal audit.

This module intentionally does not train or score a model.  The implemented CLI
is limited to semantic source inspection and CSV header validation.  Full-data
scans are a separately authorized MEDIUM phase.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


TARGET = "control_success"
ROW_ID = "row_id"
ALLOWED_SEASONS = frozenset(range(2019, 2024))
EXPLORATORY_TARGET_SEASONS = frozenset(range(2019, 2022))
SELECTION_TARGET_SEASONS = frozenset((2022, 2023))
BOUNDED_ROWS = 30_000

TRAIN_FEATURE_COLUMNS = (
    "row_id", "season", "game_month", "game_dayofweek", "inning",
    "top_bottom", "game_type", "balls_before", "strikes_before",
    "outs_before", "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id",
    "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)
TRAIN_COLUMNS = (*TRAIN_FEATURE_COLUMNS, TARGET)

TRACKMAN_COLUMNS = (
    "trackman_id", "season", "game_date", "game_month", "game_dayofweek",
    "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
    "strikes_before", "outs_before", "pitch_of_pa", "pitcher_trackman_id",
    "batter_trackman_id", "pitcher_hand", "batter_hand", "pitcher_team",
    "batter_team", "tagged_pitch_type", "auto_pitch_type",
    "pitch_type_group", "rel_speed", "spin_rate", "induced_vert_break",
    "horz_break", "extension", "rel_height", "rel_side", "zone_speed",
)
TRACKMAN_DATE_COLUMNS = ("season", "game_date")

_TRACKMAN_SLASH_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
_TRACKMAN_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BASE_STATE_FROM_RUNNERS = {
    (0, 0, 0): "___", (1, 0, 0): "1__", (0, 1, 0): "_2_",
    (0, 0, 1): "__3", (1, 1, 0): "12_", (1, 0, 1): "1_3",
    (0, 1, 1): "_23", (1, 1, 1): "123",
}
ASOF_COUNT_COLUMNS = (
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
)
ASOF_RATE_COLUMNS = tuple(
    column for column in TRAIN_FEATURE_COLUMNS if column.startswith("asof_")
    and column not in ASOF_COUNT_COLUMNS
)

AUDIT_CATEGORICAL_COLUMNS = (
    "season", "game_month", "game_dayofweek", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
)
AUDIT_NUMERIC_COLUMNS = (
    "inning", "run_total_before", "score_diff_pitcher_team",
    "home_win_expectancy", "li", "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_success_rate", "asof_batter_success_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)
AUDIT_COVERAGE_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
)
TRACKMAN_MEASUREMENT_COLUMNS = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
REPORT_SCHEMA_VERSION = "aimers9-data-integrity-audit-v1"


class AuditError(RuntimeError):
    """Fail-closed scope, schema, or semantic audit error."""


class Severity(str, Enum):
    BLOCKER = "BLOCKER"
    LIKELY_ISSUE = "LIKELY_ISSUE"
    BENIGN = "BENIGN"
    UNKNOWN = "UNKNOWN"


class EvidenceStatus(str, Enum):
    DOCUMENTED_CONTRACT = "DOCUMENTED_CONTRACT"
    EMPIRICALLY_VERIFIED = "EMPIRICALLY_VERIFIED"
    INFERENCE = "INFERENCE"
    NOT_PROVEN = "NOT_PROVEN"


class PipelineVerdict(str, Enum):
    PROVEN = "PROVEN"
    VIOLATION = "VIOLATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_PROVEN = "NOT_PROVEN"


class TargetRole(str, Enum):
    INTEGRITY_2019_2023 = "INTEGRITY_2019_2023"
    EXPLORE_2019_2021 = "EXPLORE_2019_2021"
    BOUNDED_R2022_R2023 = "BOUNDED_R2022_R2023"
    FEATURE_ONLY_2024 = "FEATURE_ONLY_2024"


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    evidence_status: EvidenceStatus
    reason: str
    pipeline_verdict: PipelineVerdict = PipelineVerdict.NOT_APPLICABLE

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "evidence_status": self.evidence_status.value,
            "pipeline_verdict": self.pipeline_verdict.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AuthorizedTargets:
    role: TargetRole
    seasons: tuple[int, ...]
    values: tuple[int, ...]
    origin: str | None = None


@dataclass(frozen=True)
class MaskSelection:
    pipeline: str
    origin: str
    mask_role: str
    full_positions: Sequence[int]
    selected_positions: Sequence[int]
    truncation: str

    def report(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        row_ids = [records[position].get(ROW_ID) for position in self.selected_positions]
        return {
            "pipeline": self.pipeline,
            "origin": self.origin,
            "mask_role": self.mask_role,
            "n_full": len(self.full_positions),
            "n_selected": len(self.selected_positions),
            "truncation": self.truncation,
            "mask_hash": canonical_hash(self.selected_positions),
            "row_id_hash": canonical_hash(row_ids),
            "source_position_min": min(self.selected_positions, default=None),
            "source_position_max": max(self.selected_positions, default=None),
        }


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_missing(value: Any) -> bool:
    """Return whether a scalar is None/empty/NaN/pandas-NA-like.

    Real-data helpers use this single predicate so pandas' ambiguous NA truth
    value is never evaluated directly. Containers are outside this scalar
    contract.
    """
    if value is None or (isinstance(value, str) and value == ""):
        return True
    try:
        unequal_to_self = value != value
    except (TypeError, ValueError):
        return False
    try:
        return bool(unequal_to_self)
    except (TypeError, ValueError):
        # pandas.NA-like scalar comparisons return an NA-like value whose truth
        # value is intentionally ambiguous.
        return True


def _season(record: Mapping[str, Any]) -> int:
    try:
        return int(record["season"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError("record has no valid season") from exc


def assert_structural_scope(records: Sequence[Mapping[str, Any]]) -> None:
    bad = sorted({_season(record) for record in records} - ALLOWED_SEASONS)
    if bad:
        raise AuditError(f"structural scope contains forbidden seasons: {bad}")


def extract_targets(
    records: Sequence[Mapping[str, Any]], role: TargetRole, *, origin: str | None = None
) -> AuthorizedTargets:
    """Extract only explicitly authorized targets.

    For 2024, this function performs no target extraction, conversion,
    aggregation, return, target-dependent branching, or reporting.  A season
    check may reject the request before the target field is accessed.
    """
    if role is TargetRole.FEATURE_ONLY_2024:
        raise AuditError("feature-only 2024 scope cannot request targets")

    seasons = tuple(_season(record) for record in records)
    if any(season == 2024 for season in seasons):
        raise AuditError("2024 targets are firewalled")

    if role is TargetRole.INTEGRITY_2019_2023:
        allowed = ALLOWED_SEASONS
    elif role is TargetRole.EXPLORE_2019_2021:
        allowed = EXPLORATORY_TARGET_SEASONS
    elif role is TargetRole.BOUNDED_R2022_R2023:
        if origin not in {"r2022", "r2023"}:
            raise AuditError("bounded target role requires origin r2022 or r2023")
        required = int(origin[1:])
        allowed = frozenset((required,))
    else:  # pragma: no cover - Enum makes this unreachable
        raise AuditError(f"unknown target role: {role}")

    bad = sorted(set(seasons) - allowed)
    if bad:
        raise AuditError(f"target role {role.value} cannot access seasons {bad}")

    values: list[int] = []
    for record in records:
        try:
            value = int(record[TARGET])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError("authorized target is missing or non-integer") from exc
        if value not in (0, 1):
            raise AuditError(f"authorized target is outside binary domain: {value}")
        values.append(value)
    return AuthorizedTargets(role=role, seasons=seasons, values=tuple(values), origin=origin)


def target_rate(targets: AuthorizedTargets) -> float:
    if not targets.values:
        raise AuditError("cannot aggregate an empty authorized target set")
    return sum(targets.values) / len(targets.values)


def read_2024_features_only(train_csv: Path):
    """Read 2024 non-target features via explicit column projection.

    This makes no claim that raw CSV bytes are physically unread.  It guarantees
    that ``control_success`` is not selected into the DataFrame or returned.
    """
    import pandas as pd  # imported only when the optional MEDIUM reader is called

    frame = pd.read_csv(
        train_csv,
        encoding="utf-8-sig",
        usecols=list(TRAIN_FEATURE_COLUMNS),
    )
    if TARGET in frame.columns:
        raise AuditError("2024 feature projection unexpectedly contains target")
    return frame.loc[frame["season"] == 2024].copy()


def validate_header(path: Path, expected: Sequence[str]) -> Finding:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            actual = next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuditError(f"cannot read CSV header {path}: {exc}") from exc
    if actual != list(expected):
        raise AuditError(
            f"header mismatch for {path.name}: expected {len(expected)} columns, "
            f"got {len(actual)}"
        )
    return Finding(
        rule=f"header:{path.name}",
        severity=Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=f"exact {len(expected)}-column contract",
    )


def audit_row_semantics(records: Sequence[Mapping[str, Any]]) -> list[Finding]:
    """Small-fixture row integrity checks used by CHEAP tests and future scans."""
    assert_structural_scope(records)
    violation_counts: Counter[str] = Counter()
    row_ids = [record.get(ROW_ID) for record in records]
    violation_counts["row_id_missing"] = sum(is_missing(value) for value in row_ids)
    nonmissing_row_ids = [value for value in row_ids if not is_missing(value)]
    violation_counts["row_id_duplicate"] = (
        len(nonmissing_row_ids) - len(set(nonmissing_row_ids))
    )

    input_fingerprints: Counter[str] = Counter()
    for record in records:
        try:
            required_state = (
                "balls_before", "strikes_before", "outs_before",
                "runner_on_1b", "runner_on_2b", "runner_on_3b",
                "num_runners_on", "base_state", "run_total_before",
                "run_top_before", "run_bot_before",
            )
            if any(is_missing(record.get(column)) for column in required_state):
                raise ValueError("required state value is missing")
            balls = int(record["balls_before"])
            strikes = int(record["strikes_before"])
            outs = int(record["outs_before"])
            runners = tuple(int(record[c]) for c in
                            ("runner_on_1b", "runner_on_2b", "runner_on_3b"))
            if balls not in range(4) or strikes not in range(3) or outs not in range(3):
                violation_counts["count_range"] += 1
            if any(value not in (0, 1) for value in runners):
                violation_counts["runner_flag_domain"] += 1
            if int(record["num_runners_on"]) != sum(runners):
                violation_counts["runner_count_mismatch"] += 1
            if record["base_state"] != BASE_STATE_FROM_RUNNERS.get(runners):
                violation_counts["base_state_mismatch"] += 1
            if int(record["run_total_before"]) != (
                int(record["run_top_before"]) + int(record["run_bot_before"])
            ):
                violation_counts["run_total_mismatch"] += 1
        except (KeyError, TypeError, ValueError):
            violation_counts["state_missing_or_invalid"] += 1

        for column in ASOF_COUNT_COLUMNS:
            if column in record and not is_missing(record[column]):
                try:
                    value = float(record[column])
                    if value < 0 or not value.is_integer():
                        violation_counts[f"{column}:count_domain"] += 1
                except (TypeError, ValueError):
                    violation_counts[f"{column}:not_numeric"] += 1
        for column in ASOF_RATE_COLUMNS:
            if column in record and not is_missing(record[column]):
                try:
                    value = float(record[column])
                    if not 0.0 <= value <= 1.0:
                        violation_counts[f"{column}:rate_domain"] += 1
                except (TypeError, ValueError):
                    violation_counts[f"{column}:not_numeric"] += 1

        present_features = tuple(
            (column, record.get(column)) for column in TRAIN_FEATURE_COLUMNS
            if column != ROW_ID and column in record
        )
        input_fingerprints[canonical_hash(present_features)] += 1

    duplicates = sum(count - 1 for count in input_fingerprints.values() if count > 1)
    violations = {
        name: count for name, count in sorted(violation_counts.items()) if count
    }
    findings: list[Finding] = []
    if violations:
        findings.append(Finding(
            rule="raw_row_integrity",
            severity=Severity.BLOCKER,
            evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
            reason=f"aggregate violation counts by type: {violations}",
        ))
    else:
        findings.append(Finding(
            rule="raw_row_integrity",
            severity=Severity.BENIGN,
            evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
            reason="row identity and documented state invariants passed",
        ))
    findings.append(Finding(
        rule="duplicate_input_states",
        severity=Severity.UNKNOWN if duplicates else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"{duplicates} repeated full-input states; exact pitch duplication is "
                "not inferable without date/game/pitch keys"),
    ))
    return findings


def audit_asof_records(records: Sequence[Mapping[str, Any]]) -> list[Finding]:
    assert_structural_scope(records)
    findings: list[Finding] = []
    domain_problems: list[str] = []
    for record in records:
        for column in ASOF_COUNT_COLUMNS:
            value = record.get(column)
            if is_missing(value):
                continue
            try:
                numeric = float(value)
                if numeric < 0 or not numeric.is_integer():
                    domain_problems.append(column)
            except (TypeError, ValueError):
                domain_problems.append(column)
        for column in ASOF_RATE_COLUMNS:
            value = record.get(column)
            if is_missing(value):
                continue
            try:
                if not 0.0 <= float(value) <= 1.0:
                    domain_problems.append(column)
            except (TypeError, ValueError):
                domain_problems.append(column)
    findings.append(Finding(
        rule="asof_domains",
        severity=Severity.BLOCKER if domain_problems else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"invalid domain values in {sorted(set(domain_problems))}"
                if domain_problems else "count/rate domains passed"),
    ))

    last_count: dict[Any, float] = {}
    decreases = 0
    for record in records:
        pitcher = record.get("pitcher_id")
        value = record.get("asof_pitcher_n")
        if is_missing(pitcher) or is_missing(value):
            continue
        numeric = float(value)
        if pitcher in last_count and numeric < last_count[pitcher]:
            decreases += 1
        last_count[pitcher] = numeric
    findings.append(Finding(
        rule="asof_count_source_order_decreases",
        severity=Severity.UNKNOWN if decreases else Severity.BENIGN,
        evidence_status=(EvidenceStatus.NOT_PROVEN if decreases
                         else EvidenceStatus.EMPIRICALLY_VERIFIED),
        reason=(f"{decreases} decreases in CSV order; chronology is not proven, so "
                "this is not automatically a violation"),
    ))
    findings.append(Finding(
        rule="asof_exact_reconstruction",
        severity=Severity.UNKNOWN,
        evidence_status=EvidenceStatus.NOT_PROVEN,
        reason="repository lacks authoritative generation logic and join keys",
    ))
    return findings


def _positions(records: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> tuple[int, ...]:
    return tuple(index for index, record in enumerate(records) if predicate(record))


def _bounded(positions: Sequence[int]) -> tuple[int, ...]:
    return tuple(positions[:BOUNDED_ROWS])


def _selection(
    records: Sequence[Mapping[str, Any]], pipeline: str, origin: str,
    mask_role: str, predicate: Callable[[Mapping[str, Any]], bool], *,
    truncate: bool = True,
) -> MaskSelection:
    full = _positions(records, predicate)
    selected = _bounded(full) if truncate else full
    return MaskSelection(
        pipeline=pipeline,
        origin=origin,
        mask_role=mask_role,
        full_positions=full,
        selected_positions=selected,
        truncation=("first_30000_true_positions_in_dataframe_order"
                    if truncate else "not_truncated_pretrained_artifact_path"),
    )


def build_pipeline_geometries(records: Sequence[Mapping[str, Any]]) -> list[MaskSelection]:
    """Reproduce each pipeline's actual bounded-mask geometry independently."""
    assert_structural_scope(records)
    out: list[MaskSelection] = []
    is_r = lambda record: record.get("game_type") == "R"
    year = lambda record: _season(record)

    cat_specs = {
        "r2022": (2020, 2021, 2021, 2022),
        "r2023": (2021, 2022, 2022, 2023),
    }
    for origin, (inner_max, inner_year, outer_max, outer_year) in cat_specs.items():
        out.extend((
            _selection(records, "recovery_catboost", origin, "inner_train",
                       lambda r, y=inner_max: is_r(r) and year(r) <= y),
            _selection(records, "recovery_catboost", origin, "inner_validation",
                       lambda r, y=inner_year: is_r(r) and year(r) == y),
            _selection(records, "recovery_catboost", origin, "outer_train",
                       lambda r, y=outer_max: is_r(r) and year(r) <= y),
            _selection(records, "recovery_catboost", origin, "outer_validation",
                       lambda r, y=outer_year: is_r(r) and year(r) == y),
        ))

    # Baseline reproduction loads pre-trained artifacts and truncates validation only.
    for outer_year in (2022, 2023):
        origin = f"r{outer_year}"
        out.append(_selection(
            records, "baseline_reproduction", origin, "pretrained_outer_validation",
            lambda r, y=outer_year: is_r(r) and year(r) == y,
        ))

    # Residual base OOF truncates both train and validation for every forward year.
    for outer_year in (2020, 2021, 2022, 2023):
        origin = f"oof{outer_year}"
        out.extend((
            _selection(records, "recovery_residual", origin, "base_oof_train",
                       lambda r, y=outer_year: is_r(r) and year(r) <= y - 1),
            _selection(records, "recovery_residual", origin, "base_oof_validation",
                       lambda r, y=outer_year: is_r(r) and year(r) == y),
        ))
    residual_roles = {
        "r2022": ((2020, 2021), 2022),
        "r2023": ((2020, 2021, 2022), 2023),
    }
    for origin, (fit_years, apply_year) in residual_roles.items():
        for fit_year in fit_years:
            out.append(_selection(
                records, "recovery_residual", origin,
                f"correction_fit_oof_validation_{fit_year}",
                lambda r, y=fit_year: is_r(r) and year(r) == y,
            ))
        out.append(_selection(
            records, "recovery_residual", origin, "correction_outer_apply",
            lambda r, y=apply_year: is_r(r) and year(r) == y,
        ))
    # r2023 C selection fits 2020 OOF and evaluates the 2021 OOF.
    out.extend((
        _selection(records, "recovery_residual", "r2023", "c_selection_fit_2020",
                   lambda r: is_r(r) and year(r) == 2020),
        _selection(records, "recovery_residual", "r2023", "c_selection_score_2021",
                   lambda r: is_r(r) and year(r) == 2021),
    ))

    # Calibration cache/labels/apply use the same first-N outer rows, but are
    # recorded as distinct roles because alignment must be proven per consumer.
    for outer_year in (2022, 2023):
        origin = f"r{outer_year}"
        predicate = lambda r, y=outer_year: is_r(r) and year(r) == y
        for role in ("baseline_cache_logits", "bounded_labels", "transform_apply"):
            out.append(_selection(records, "recovery_calibration", origin, role, predicate))
    out.append(_selection(
        records, "recovery_calibration", "r2023", "transform_fit_panel_2022",
        lambda r: is_r(r) and year(r) == 2022,
    ))
    return out


def _distribution(records: Sequence[Mapping[str, Any]], positions: Sequence[int], column: str) -> dict[str, float]:
    values = [
        "__MISSING__" if is_missing(records[position].get(column))
        else str(records[position].get(column))
        for position in positions
    ]
    counts = Counter(values)
    total = len(values) or 1
    return {key: count / total for key, count in sorted(counts.items())}


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    return 0.5 * math.fsum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys
    )


def compare_bounded_to_full(
    records: Sequence[Mapping[str, Any]], selection: MaskSelection,
    *, categorical: Sequence[str], numeric: Sequence[str], coverage: Sequence[str],
) -> dict[str, Any]:
    full = selection.full_positions
    bounded = selection.selected_positions
    categorical_tv = {
        column: _total_variation(
            _distribution(records, bounded, column),
            _distribution(records, full, column),
        ) for column in categorical
    }
    smd: dict[str, float | None] = {}
    missing_pp: dict[str, float] = {}
    for column in numeric:
        bounded_raw = [records[p].get(column) for p in bounded]
        full_raw = [records[p].get(column) for p in full]
        bounded_values = [float(v) for v in bounded_raw if not is_missing(v)]
        full_values = [float(v) for v in full_raw if not is_missing(v)]
        bounded_mean = sum(bounded_values) / len(bounded_values) if bounded_values else None
        full_mean = sum(full_values) / len(full_values) if full_values else None
        if full_values:
            assert full_mean is not None
            variance = sum((value - full_mean) ** 2 for value in full_values) / len(full_values)
            std = math.sqrt(variance)
        else:
            std = 0.0
        smd[column] = (
            (bounded_mean - full_mean) / std
            if bounded_mean is not None and full_mean is not None and std else None
        )
        missing_pp[column] = 100.0 * (
            sum(is_missing(v) for v in bounded_raw) / (len(bounded_raw) or 1)
            - sum(is_missing(v) for v in full_raw) / (len(full_raw) or 1)
        )
    coverage_ratio: dict[str, float | None] = {}
    for column in coverage:
        full_unique = {
            "__MISSING__" if is_missing(records[p].get(column)) else records[p].get(column)
            for p in full
        }
        bounded_unique = {
            "__MISSING__" if is_missing(records[p].get(column)) else records[p].get(column)
            for p in bounded
        }
        coverage_ratio[column] = (
            len(bounded_unique) / len(full_unique) if full_unique else None
        )
    report = selection.report(records)
    report.update({
        "categorical_total_variation": categorical_tv,
        "numeric_smd": smd,
        "missing_rate_delta_pp": missing_pp,
        "coverage_ratio": coverage_ratio,
    })
    return report


FORBIDDEN_NETWORK_IMPORTS = frozenset(("requests", "urllib", "httpx", "socket"))
FORBIDDEN_CLI_DESTINATIONS = frozenset((
    "test_csv", "sample_submission", "public", "leaderboard", "url",
))
FORBIDDEN_DATA_BASENAMES = frozenset(("test.csv", "sample_submission.csv"))


def semantic_source_audit(source: str) -> list[str]:
    """Inspect executable semantics, ignoring comments and unrelated prose."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"source cannot be parsed: {exc}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_NETWORK_IMPORTS:
                    problems.append(f"forbidden external-information import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_NETWORK_IMPORTS:
                problems.append(f"forbidden external-information import: {node.module}")
        elif isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name == "add_argument" and node.args:
                flag = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
                destinations = set()
                if isinstance(flag, str) and flag.startswith("--"):
                    destinations.add(flag[2:].replace("-", "_"))
                for keyword in node.keywords:
                    if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                        destinations.add(str(keyword.value.value))
                bad = destinations & FORBIDDEN_CLI_DESTINATIONS
                if bad:
                    problems.append(f"forbidden CLI destination: {sorted(bad)}")
            if func_name in {"read_csv", "read_table", "open"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if Path(first.value).name in FORBIDDEN_DATA_BASENAMES:
                        problems.append(f"forbidden data reader path: {first.value}")
    return sorted(set(problems))


def audit_2024_reader_semantics(source: str) -> list[str]:
    """Fail closed if the isolated reader stops projecting non-target columns.

    This inspects executable syntax only.  It does not pretend that column
    projection makes target bytes physically unread in a row-oriented CSV.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"source cannot be parsed: {exc}"]
    readers = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "read_2024_features_only"
    ]
    if len(readers) != 1:
        return ["exactly one read_2024_features_only function is required"]

    problems: list[str] = []
    read_csv_calls = []
    def contains_target_token(node: ast.AST) -> bool:
        return any(
            (isinstance(part, ast.Name) and part.id == "TARGET")
            or (isinstance(part, ast.Constant) and part.value == TARGET)
            for part in ast.walk(node)
        )

    for node in ast.walk(readers[0]):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name == "read_csv":
                read_csv_calls.append(node)
                projection = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "usecols"),
                    None,
                )
                if projection is None:
                    problems.append("2024 feature reader read_csv requires usecols projection")
                elif ast.dump(projection) != ast.dump(ast.parse(
                    "list(TRAIN_FEATURE_COLUMNS)", mode="eval"
                ).body):
                    problems.append(
                        "2024 feature reader usecols must be list(TRAIN_FEATURE_COLUMNS)"
                    )
            elif any(contains_target_token(argument) for argument in node.args):
                problems.append("2024 feature reader contains target-value access")
        elif isinstance(node, ast.Subscript):
            if contains_target_token(node.slice):
                problems.append("2024 feature reader contains target-value access")
    if len(read_csv_calls) != 1:
        problems.append("2024 feature reader requires exactly one read_csv call")
    return sorted(set(problems))


def audit_medium_reader_semantics(source: str) -> list[str]:
    """Verify the main and optional MEDIUM readers retain their projections."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"source cannot be parsed: {exc}"]
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    problems: list[str] = []

    def calls(function_name: str) -> list[ast.Call]:
        function = functions.get(function_name)
        if function is None:
            problems.append(f"missing MEDIUM reader function: {function_name}")
            return []
        return [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
        ]

    def invokes(function_name: str, called_name: str) -> bool:
        function = functions.get(function_name)
        if function is None:
            return False
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == called_name
            for node in ast.walk(function)
        )

    season_projection = ast.dump(ast.parse("[\"season\"]", mode="eval").body)
    feature_projection = ast.dump(ast.parse(
        "list(TRAIN_FEATURE_COLUMNS)", mode="eval"
    ).body)
    target_projection = ast.dump(ast.parse("[ROW_ID, TARGET]", mode="eval").body)
    trackman_projection = ast.dump(ast.parse(
        "list(TRACKMAN_COLUMNS)", mode="eval"
    ).body)
    discovery_calls = calls("_discover_scoped_source_rows")
    if len(discovery_calls) != 1:
        problems.append("scope discovery requires exactly one season-only read_csv call")
    else:
        projection = next(
            (keyword.value for keyword in discovery_calls[0].keywords if keyword.arg == "usecols"),
            ast.Constant(None),
        )
        if ast.dump(projection) != season_projection:
            problems.append("scope discovery must project season only")
    main_calls = calls("read_scoped_train_2019_2023")
    if not invokes("read_scoped_train_2019_2023", "_discover_scoped_source_rows"):
        problems.append("main reader must invoke season/source-row discovery")
    if not invokes("read_scoped_train_2019_2023", "_scoped_skiprows"):
        problems.append("main reader must derive scoped skiprows from discovery")
    seen_scoped_feature = False
    seen_scoped_target = False
    for call in main_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        projection = keywords.get("usecols")
        dumped = ast.dump(projection) if projection is not None else ""
        if dumped == feature_projection:
            seen_scoped_feature = (
                "skiprows" in keywords
                and isinstance(keywords["skiprows"], ast.Name)
                and keywords["skiprows"].id == "skiprows"
            )
            if not seen_scoped_feature:
                problems.append("main feature projection requires skiprows firewall")
        if dumped == target_projection:
            seen_scoped_target = (
                "skiprows" in keywords
                and isinstance(keywords["skiprows"], ast.Name)
                and keywords["skiprows"].id == "skiprows"
            )
            if not seen_scoped_target:
                problems.append("main target projection requires skiprows firewall")
    if not seen_scoped_feature:
        problems.append("main reader lacks scoped TRAIN_FEATURE_COLUMNS projection")
    if not seen_scoped_target:
        problems.append("main reader lacks scoped target projection")

    trackman_calls = calls("read_scoped_trackman_2019_2023")
    if not invokes("read_scoped_trackman_2019_2023", "_discover_scoped_source_rows"):
        problems.append("main Trackman reader must invoke season/source-row discovery")
    if len(trackman_calls) != 1:
        problems.append("main Trackman reader requires exactly one scoped projection")
    else:
        keywords = {keyword.arg: keyword.value for keyword in trackman_calls[0].keywords}
        projection = keywords.get("usecols")
        if projection is None or ast.dump(projection) != trackman_projection:
            problems.append("main Trackman reader must project TRACKMAN_COLUMNS")
        skiprows_value = keywords.get("skiprows")
        scoped_call = (
            isinstance(skiprows_value, ast.Call)
            and isinstance(skiprows_value.func, ast.Name)
            and skiprows_value.func.id == "_scoped_skiprows"
        )
        if not scoped_call:
            problems.append("main Trackman projection requires skiprows firewall")

    date_trackman_calls = calls("read_scoped_trackman_game_date_columns_2019_2023")
    if not invokes(
        "read_scoped_trackman_game_date_columns_2019_2023",
        "_discover_scoped_source_rows",
    ):
        problems.append("targeted Trackman date reader must invoke season/source-row discovery")
    if len(date_trackman_calls) != 1:
        problems.append("targeted Trackman date reader requires exactly one scoped projection")
    else:
        keywords = {keyword.arg: keyword.value for keyword in date_trackman_calls[0].keywords}
        projection = keywords.get("usecols")
        date_projection = ast.dump(ast.parse(
            "list(TRACKMAN_DATE_COLUMNS)", mode="eval"
        ).body)
        if projection is None or ast.dump(projection) != date_projection:
            problems.append("targeted Trackman date reader must project TRACKMAN_DATE_COLUMNS")
        skiprows_value = keywords.get("skiprows")
        scoped_call = (
            isinstance(skiprows_value, ast.Call)
            and isinstance(skiprows_value.func, ast.Name)
            and skiprows_value.func.id == "_scoped_skiprows"
        )
        if not scoped_call:
            problems.append("targeted Trackman date projection requires skiprows firewall")

    abs_calls = calls("run_abs_2024_feature_audit")
    if len(abs_calls) != 1:
        problems.append("ABS feature-only runner requires exactly one read_csv call")
    elif ast.dump(next(
        (keyword.value for keyword in abs_calls[0].keywords if keyword.arg == "usecols"),
        ast.Constant(None),
    )) != feature_projection:
        problems.append("ABS feature-only runner must project TRAIN_FEATURE_COLUMNS")
    return sorted(set(problems))


PIPELINE_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "common_legacy_lgb": (
        "repro_979/common.py",
        ("load_train", "preprocess_for_submission", "train_model"),
    ),
    "mlp": (
        "repro_979/e6c_blend_folds.py",
        ("build_fold", "train_seed"),
    ),
    "recovery_catboost": (
        "repro_979/recovery_catboost_runner.py",
        ("_load_train", "_bounded_mask", "_inner_outer_masks", "_fit_inner", "_fit_outer"),
    ),
    "recovery_residual": (
        "repro_979/recovery_residual_runner.py",
        ("_bounded_mask", "build_base_oof_masks", "_compute_stats", "fit_correction"),
    ),
    "recovery_calibration": (
        "repro_979/recovery_calibration_runner.py",
        ("_bounded_labels", "build_causal_panel", "fit_beta", "fit_isotonic"),
    ),
    "baseline_reproduction": (
        "repro_979/recovery_evaluator.py",
        ("reproduce_baseline", "cmd_reproduce_baseline"),
    ),
}


def audit_pipeline_sources(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for pipeline, (relative, required_functions) in PIPELINE_SOURCES.items():
        path = repo_root / relative
        if not path.is_file():
            findings.append(Finding(
                rule=f"pipeline_source:{pipeline}", severity=Severity.BLOCKER,
                evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
                pipeline_verdict=PipelineVerdict.NOT_PROVEN,
                reason=f"missing source {relative}",
            ))
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        missing = sorted(set(required_functions) - functions)
        findings.append(Finding(
            rule=f"pipeline_source:{pipeline}",
            severity=Severity.BLOCKER if missing else Severity.BENIGN,
            evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
            pipeline_verdict=PipelineVerdict.NOT_PROVEN,
            reason=(f"missing expected functions {missing}" if missing
                    else ("required source path/functions present; fit/dataflow semantics "
                          "remain not proven by presence alone; "
                          f"sha256={hashlib.sha256(source.encode()).hexdigest()}")),
        ))

    notebook = repo_root / "baseline" / (
        "[Baseline_Train]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습).ipynb"
    )
    try:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        call_names: set[str] = set()
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            tree = ast.parse("".join(cell.get("source", [])))
            call_names.update(
                node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Name, ast.Attribute))
            )
        required = {"ColumnTransformer", "OrdinalEncoder", "SimpleImputer", "Pipeline", "fit"}
        missing = sorted(required - call_names)
    except (OSError, ValueError, SyntaxError) as exc:
        missing = [f"notebook parse error: {exc}"]
    findings.append(Finding(
        rule="pipeline_source:official_baseline",
        severity=Severity.BLOCKER if missing else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        pipeline_verdict=PipelineVerdict.NOT_PROVEN,
        reason=(f"missing expected calls {missing}" if missing else
                "expected calls are present; fit/dataflow semantics remain not proven by call presence"),
    ))
    return findings


def classify_project_status_sha(status_text: str, current_sha: str) -> Finding:
    match = re.search(r"Evidence base: `master` at `([0-9a-f]{7,40})`", status_text)
    recorded = match.group(1) if match else None
    if recorded and not current_sha.startswith(recorded) and recorded != current_sha:
        return Finding(
            rule="project_status_evidence_base",
            severity=Severity.BENIGN,
            evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
            reason=(f"status snapshot {recorded} differs from HEAD {current_sha}; documentary "
                    "drift is benign unless a separate policy/state contradiction is found"),
        )
    return Finding(
        rule="project_status_evidence_base", severity=Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason="status snapshot matches current HEAD",
    )


def abs_feature_only_contract() -> dict[str, Any]:
    result = {
        "branch_inert": True,
        "target_column_in_projection": False,
        "may_tune_thresholds": False,
        "may_select_features": False,
        "may_choose_transformations": False,
        "may_modify_recovery_policy": False,
        "requires_separate_experiment_brief_for_modeling": True,
        "pre_registered_metrics": (
            "numeric_smd", "quantile_change", "categorical_total_variation",
            "missing_rate_delta_pp", "entity_coverage_change",
        ),
    }
    return result


def assess_trackman_usage(
    *, documented_shared_pitch_key: bool, authoritative_player_crosswalk: bool,
    documented_main_game_date_key: bool, prior_season_coverage: bool,
    stable_league_measurements: bool, observed_identifier_overlap: bool = False,
) -> dict[str, dict[str, str]]:
    """Classify safe usage levels without treating raw ID overlap as a crosswalk."""
    del observed_identifier_overlap  # empirical overlap alone is never crosswalk proof
    usage = {
        "A_exact_row_join": {
            "verdict": "PROVEN" if documented_shared_pitch_key else "NOT_PROVEN",
            "requirement": "authoritative shared pitch key and one-to-one uniqueness",
        },
        "B_game_date_join": {
            "verdict": "PROVEN" if documented_main_game_date_key else "NOT_PROVEN",
            "requirement": "authoritative main-data game/date key",
        },
        "C_prior_season_player_team": {
            "verdict": ("PROVEN" if authoritative_player_crosswalk and prior_season_coverage
                        else "NOT_PROVEN"),
            "requirement": "authoritative player/team mapping and Trackman season < Y",
        },
        "D_prior_season_league": {
            "verdict": ("PROVEN" if prior_season_coverage and stable_league_measurements
                        else "NOT_PROVEN"),
            "requirement": "stable measurements and Trackman season < Y",
        },
    }
    usage["E_unusable"] = {
        "verdict": ("NOT_APPLICABLE" if any(
            item["verdict"] == "PROVEN" for item in usage.values()
        ) else "NOT_PROVEN"),
        "requirement": (
            "use only if no A-D level can establish required keys, coverage, and leakage guards"
        ),
    }
    return usage


def validate_trackman_prior_season(observation_seasons: Iterable[int], prediction_year: int) -> None:
    bad = sorted({int(year) for year in observation_seasons if int(year) >= prediction_year})
    if bad:
        raise AuditError(
            f"Trackman future-safe rule violated for prediction year {prediction_year}: {bad}"
        )


class _TargetFrameRow(Mapping[str, Any]):
    def __init__(self, frame, position: int):
        self._frame = frame
        self._position = position

    def __getitem__(self, key: str) -> Any:
        if key not in {"season", TARGET}:
            raise KeyError(key)
        return self._frame.iat[self._position, self._frame.columns.get_loc(key)]

    def __iter__(self):
        return iter(("season", TARGET))

    def __len__(self) -> int:
        return 2


class TargetFrameRecordView(Sequence[Mapping[str, Any]]):
    """Lazy season/target rows consumed only inside the role firewall."""

    def __init__(self, frame):
        self._frame = frame

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int | slice) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _TargetFrameRow(self._frame, index)


def _missing_mask(series):
    return series.map(is_missing)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_frame_hash(frame, columns: Sequence[str]) -> str:
    """Hash only the explicitly scoped/projected frame, in source-row order."""
    digest = hashlib.sha256()
    digest.update((json.dumps(list(columns), separators=(",", ":")) + "\n").encode())
    for start in range(0, len(frame), 50_000):
        payload = frame.iloc[start:start + 50_000][list(columns)].to_csv(
            index=False, header=False, lineterminator="\n", na_rep="<NA>"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def parse_trackman_game_dates(values):
    """Parse the two observed Trackman game_date formats strictly.

    Format routing is explicit so parsing does not depend on source order or
    pandas' single-format inference for mixed-format input. Values outside the
    declared formats, including blank, null, and impossible calendar dates,
    remain ``NaT``.
    """
    import pandas as pd

    raw = pd.Series(values)
    strings = raw.astype("string")
    slash = strings.str.fullmatch(_TRACKMAN_SLASH_DATE_RE, na=False)
    iso = strings.str.fullmatch(_TRACKMAN_ISO_DATE_RE, na=False)
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if slash.any():
        parsed.loc[slash] = pd.to_datetime(
            strings.loc[slash], format="%m/%d/%Y", errors="coerce"
        )
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(
            strings.loc[iso], format="%Y-%m-%d", errors="coerce"
        )
    return parsed


def _git_sha(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head[5:]
        loose_ref = repo_root / ".git" / ref_name
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip()
        packed = repo_root / ".git" / "packed-refs"
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                sha, name = line.split(" ", 1)
                if name == ref_name:
                    return sha
    except (OSError, ValueError):
        pass
    return "unknown"


def _discover_scoped_source_rows(csv_path: Path, *, source_name: str):
    """Materialize season only, returning allowed-line flags and source positions."""
    import numpy as np
    import pandas as pd

    seasons = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["season"])
    numeric = pd.to_numeric(seasons["season"], errors="coerce")
    if _missing_mask(numeric).any():
        raise AuditError(f"{source_name} has invalid season values")
    allowed = numeric.isin(ALLOWED_SEASONS).to_numpy(dtype=bool)
    if not allowed.any():
        raise AuditError(f"{source_name} has no 2019-2023 rows")
    return bytearray(int(value) for value in allowed), np.flatnonzero(allowed)


def _scoped_skiprows(allowed_lines: bytearray):
    def skip_disallowed_row(line_number: int) -> bool:
        if line_number == 0:
            return False
        position = line_number - 1
        return position >= len(allowed_lines) or not allowed_lines[position]

    return skip_disallowed_row


def _attach_aligned_targets(features, targets):
    if len(targets) != len(features):
        raise AuditError("scoped target/feature row counts differ")
    feature_ids = features[ROW_ID].reset_index(drop=True)
    target_ids = targets[ROW_ID].reset_index(drop=True)
    feature_hash = canonical_hash(feature_ids.tolist())
    target_hash = canonical_hash(target_ids.tolist())
    if feature_hash != target_hash or not feature_ids.equals(target_ids):
        raise AuditError("scoped target ROW_ID order/hash does not match feature projection")
    features[TARGET] = targets[TARGET].to_numpy()
    features.attrs["target_alignment_row_id_hash"] = feature_hash
    return features


def read_scoped_train_2019_2023(train_csv: Path):
    """Discover season rows, then project only scoped features and aligned targets.

    Both full projections skip non-2019-2023 rows. The target pass includes ROW_ID
    for an exact ordered alignment check before target attachment.
    As with the feature-only reader, this does not claim that bytes are physically
    unread by a CSV parser.
    """
    import pandas as pd

    validate_header(train_csv, TRAIN_COLUMNS)
    allowed_lines, source_positions = _discover_scoped_source_rows(
        train_csv, source_name="train"
    )
    skiprows = _scoped_skiprows(allowed_lines)
    features = pd.read_csv(
        train_csv,
        encoding="utf-8-sig",
        usecols=list(TRAIN_FEATURE_COLUMNS),
        skiprows=skiprows,
    )
    targets = pd.read_csv(
        train_csv,
        encoding="utf-8-sig",
        usecols=[ROW_ID, TARGET],
        skiprows=skiprows,
    )
    if len(features) != len(source_positions):
        raise AuditError("scoped train feature projection row count mismatch")
    features["__source_position"] = source_positions
    features.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)
    return _attach_aligned_targets(features, targets)


def read_scoped_trackman_2019_2023(trackman_csv: Path):
    import pandas as pd

    validate_header(trackman_csv, TRACKMAN_COLUMNS)
    allowed_lines, source_positions = _discover_scoped_source_rows(
        trackman_csv, source_name="Trackman"
    )
    frame = pd.read_csv(
        trackman_csv,
        encoding="utf-8-sig",
        usecols=list(TRACKMAN_COLUMNS),
        skiprows=_scoped_skiprows(allowed_lines),
    )
    if len(frame) != len(source_positions):
        raise AuditError("scoped Trackman projection row count mismatch")
    frame["__source_position"] = source_positions
    frame.reset_index(drop=True, inplace=True)
    return frame


def read_scoped_trackman_game_date_columns_2019_2023(trackman_csv: Path):
    """Read only scoped Trackman season and game_date columns.

    Season is discovered in a season-only pass first. The date projection then
    uses the same source-position skiprows firewall as the main Trackman reader,
    so out-of-scope 2024 rows are not materialized into the projected frame.
    """
    import pandas as pd

    validate_header(trackman_csv, TRACKMAN_COLUMNS)
    allowed_lines, source_positions = _discover_scoped_source_rows(
        trackman_csv, source_name="Trackman"
    )
    frame = pd.read_csv(
        trackman_csv,
        encoding="utf-8-sig",
        usecols=list(TRACKMAN_DATE_COLUMNS),
        skiprows=_scoped_skiprows(allowed_lines),
    )
    if len(frame) != len(source_positions):
        raise AuditError("scoped Trackman date projection row count mismatch")
    frame["__source_position"] = source_positions
    frame.reset_index(drop=True, inplace=True)
    return frame


def _frame_selection(frame, pipeline: str, origin: str, mask_role: str, mask) -> MaskSelection:
    import numpy as np

    full = np.flatnonzero(mask.to_numpy(dtype=bool))
    return MaskSelection(
        pipeline=pipeline,
        origin=origin,
        mask_role=mask_role,
        full_positions=full,
        selected_positions=full[:BOUNDED_ROWS],
        truncation="first_30000_true_positions_in_dataframe_order",
    )


def build_pipeline_geometries_frame(frame) -> list[MaskSelection]:
    """Vectorized counterpart of the reviewed per-pipeline geometry."""
    season = frame["season"].astype(int)
    is_r = frame["game_type"] == "R"
    out: list[MaskSelection] = []
    cat_specs = {
        "r2022": (2020, 2021, 2021, 2022),
        "r2023": (2021, 2022, 2022, 2023),
    }
    for origin, (inner_max, inner_year, outer_max, outer_year) in cat_specs.items():
        for role, mask in (
            ("inner_train", is_r & (season <= inner_max)),
            ("inner_validation", is_r & (season == inner_year)),
            ("outer_train", is_r & (season <= outer_max)),
            ("outer_validation", is_r & (season == outer_year)),
        ):
            out.append(_frame_selection(frame, "recovery_catboost", origin, role, mask))
    for year in (2022, 2023):
        out.append(_frame_selection(
            frame, "baseline_reproduction", f"r{year}",
            "pretrained_outer_validation", is_r & (season == year),
        ))
    for year in (2020, 2021, 2022, 2023):
        out.extend((
            _frame_selection(frame, "recovery_residual", f"oof{year}",
                             "base_oof_train", is_r & (season <= year - 1)),
            _frame_selection(frame, "recovery_residual", f"oof{year}",
                             "base_oof_validation", is_r & (season == year)),
        ))
    for origin, fit_years, apply_year in (
        ("r2022", (2020, 2021), 2022),
        ("r2023", (2020, 2021, 2022), 2023),
    ):
        for year in fit_years:
            out.append(_frame_selection(
                frame, "recovery_residual", origin,
                f"correction_fit_oof_validation_{year}", is_r & (season == year),
            ))
        out.append(_frame_selection(
            frame, "recovery_residual", origin, "correction_outer_apply",
            is_r & (season == apply_year),
        ))
    out.extend((
        _frame_selection(frame, "recovery_residual", "r2023",
                         "c_selection_fit_2020", is_r & (season == 2020)),
        _frame_selection(frame, "recovery_residual", "r2023",
                         "c_selection_score_2021", is_r & (season == 2021)),
    ))
    for year in (2022, 2023):
        for role in ("baseline_cache_logits", "bounded_labels", "transform_apply"):
            out.append(_frame_selection(
                frame, "recovery_calibration", f"r{year}", role,
                is_r & (season == year),
            ))
    out.append(_frame_selection(
        frame, "recovery_calibration", "r2023", "transform_fit_panel_2022",
        is_r & (season == 2022),
    ))
    return out


def compare_bounded_frame(frame, selection: MaskSelection) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    full = frame.iloc[selection.full_positions]
    bounded = frame.iloc[selection.selected_positions]
    categorical_tv: dict[str, float] = {}
    for column in AUDIT_CATEGORICAL_COLUMNS:
        left = bounded[column].map(
            lambda value: "__MISSING__" if is_missing(value) else str(value)
        ).value_counts(normalize=True).to_dict()
        right = full[column].map(
            lambda value: "__MISSING__" if is_missing(value) else str(value)
        ).value_counts(normalize=True).to_dict()
        categorical_tv[column] = _total_variation(left, right)
    smd: dict[str, float | None] = {}
    missing_delta: dict[str, float] = {}
    for column in AUDIT_NUMERIC_COLUMNS:
        bounded_missing = _missing_mask(bounded[column])
        full_missing = _missing_mask(full[column])
        bv = pd.to_numeric(bounded.loc[~bounded_missing, column], errors="coerce").dropna()
        fv = pd.to_numeric(full.loc[~full_missing, column], errors="coerce").dropna()
        std = float(fv.std(ddof=0)) if len(fv) else 0.0
        smd[column] = (
            float((bv.mean() - fv.mean()) / std) if len(bv) and std else None
        )
        missing_delta[column] = 100.0 * (
            float(bounded_missing.mean()) if len(bounded) else 0.0
        ) - 100.0 * (float(full_missing.mean()) if len(full) else 0.0)
    coverage: dict[str, float | None] = {}
    for column in AUDIT_COVERAGE_COLUMNS:
        full_values = set(full.loc[~_missing_mask(full[column]), column].astype(str))
        bounded_values = set(
            bounded.loc[~_missing_mask(bounded[column]), column].astype(str)
        )
        coverage[column] = (
            len(bounded_values) / len(full_values) if full_values else None
        )
    selected_sources = bounded["__source_position"].astype(int).tolist()
    report = {
        "pipeline": selection.pipeline,
        "origin": selection.origin,
        "mask_role": selection.mask_role,
        "n_full": len(full),
        "n_selected": len(bounded),
        "truncation": selection.truncation,
        "mask_hash": canonical_hash(selected_sources),
        "row_id_hash": canonical_hash(bounded[ROW_ID].astype(str).tolist()),
        "source_position_min": min(selected_sources, default=None),
        "source_position_max": max(selected_sources, default=None),
        "source_positions_contiguous": bool(
            not selected_sources
            or np.all(np.diff(np.asarray(selected_sources, dtype=int)) == 1)
        ),
        "categorical_total_variation": categorical_tv,
        "numeric_smd": smd,
        "missing_rate_delta_pp": missing_delta,
        "coverage_ratio": coverage,
    }
    return report


def compare_all_bounded_geometries(
    frame, selections: Sequence[MaskSelection],
) -> list[dict[str, Any]]:
    """Reuse aggregate profiles for consumers with identical row geometry."""
    import numpy as np

    cache: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    identity_fields = {"pipeline", "origin", "mask_role"}
    for selection in selections:
        digest = hashlib.sha256()
        for positions in (selection.full_positions, selection.selected_positions):
            array = np.asarray(positions, dtype=np.int64)
            digest.update(len(array).to_bytes(8, "big"))
            digest.update(array.tobytes())
        key = digest.hexdigest()
        if key not in cache:
            computed = compare_bounded_frame(frame, selection)
            cache[key] = {
                name: value for name, value in computed.items()
                if name not in identity_fields
            }
        reports.append({
            "pipeline": selection.pipeline,
            "origin": selection.origin,
            "mask_role": selection.mask_role,
            **cache[key],
            "shared_geometry_profile_hash": key,
        })
    return reports


def audit_main_structural_frame(frame) -> tuple[dict[str, Any], list[Finding]]:
    import pandas as pd

    violations: Counter[str] = Counter()
    row_missing = _missing_mask(frame[ROW_ID])
    violations["row_id_missing"] = int(row_missing.sum())
    violations["row_id_duplicate"] = int(
        frame.loc[~row_missing, ROW_ID].astype(str).duplicated().sum()
    )
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in (
            "balls_before", "strikes_before", "outs_before", "runner_on_1b",
            "runner_on_2b", "runner_on_3b", "num_runners_on",
            "run_top_before", "run_bot_before", "run_total_before",
        )
    }
    violations["count_range"] = int(
        (~numeric["balls_before"].isin(range(4))
         | ~numeric["strikes_before"].isin(range(3))
         | ~numeric["outs_before"].isin(range(3))).sum()
    )
    runner_columns = ("runner_on_1b", "runner_on_2b", "runner_on_3b")
    violations["runner_flag_domain"] = int(
        sum((~numeric[column].isin((0, 1))).sum() for column in runner_columns)
    )
    runner_sum = sum(numeric[column] for column in runner_columns)
    violations["runner_count_mismatch"] = int(
        (numeric["num_runners_on"] != runner_sum).sum()
    )
    expected_base = [
        BASE_STATE_FROM_RUNNERS.get(tuple(int(value) for value in values))
        if all(value in (0, 1) for value in values) else None
        for values in zip(*(numeric[column] for column in runner_columns))
    ]
    violations["base_state_mismatch"] = int(
        (frame["base_state"].astype(str) != pd.Series(expected_base, index=frame.index)).sum()
    )
    violations["run_total_mismatch"] = int(
        (numeric["run_total_before"]
         != numeric["run_top_before"] + numeric["run_bot_before"]).sum()
    )
    duplicate_inputs = int(frame.duplicated(
        subset=[column for column in TRAIN_FEATURE_COLUMNS if column != ROW_ID]
    ).sum())
    near_state_columns = (
        "season", "game_month", "game_type", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before", "run_top_before",
        "run_bot_before", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    )
    near_duplicate_states = int(frame.duplicated(subset=list(near_state_columns)).sum())
    violations = Counter({key: value for key, value in violations.items() if value})
    findings = [Finding(
        rule="raw_row_integrity",
        severity=Severity.BLOCKER if violations else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"aggregate violation counts by type: {dict(sorted(violations.items()))}"
                if violations else "row identity and state invariants passed"),
    ), Finding(
        rule="duplicate_input_states",
        severity=Severity.UNKNOWN if duplicate_inputs or near_duplicate_states else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"{duplicate_inputs} repeated full-input states; pitch duplication "
                f"and {near_duplicate_states} repeated near-state keys; pitch duplication "
                "is not inferable without date/game/pitch keys"),
    )]
    return {
        "n_rows": int(len(frame)),
        "seasons": sorted(int(value) for value in frame["season"].unique()),
        "row_id_hash": canonical_hash(frame[ROW_ID].astype(str).tolist()),
        "violations": dict(sorted(violations.items())),
        "duplicate_full_input_state_count": duplicate_inputs,
        "duplicate_near_state_key_count": near_duplicate_states,
        "near_state_key_columns": list(near_state_columns),
        "row_semantics": {
            "documented": "one pre-pitch game-state observation",
            "exact_pitch_identity": "NOT_PROVEN",
            "exact_date_game_pitch_order_reconstruction": "NOT_PROVEN",
        },
    }, findings


def audit_temporal_frame(frame) -> dict[str, Any]:
    season = frame["season"].astype(int)
    month = frame["game_month"]
    runs: list[int] = []
    previous = None
    for value in season:
        if value != previous:
            runs.append(int(value))
            previous = value
    run_counts = Counter(runs)
    composition = (
        frame.groupby(["season", "game_month"], dropna=False).size()
        .rename("n_rows").reset_index()
    )
    is_r = frame["game_type"] == "R"
    masks = {
        "r2022": ((season <= 2021) & is_r, (season == 2022) & is_r),
        "r2023": ((season <= 2022) & is_r, (season == 2023) & is_r),
    }
    origin_checks = {}
    for origin, (train_mask, val_mask) in masks.items():
        train_years = season.loc[train_mask]
        val_years = season.loc[val_mask]
        origin_checks[origin] = {
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "row_disjoint": True,
            "max_train_season": int(train_years.max()) if len(train_years) else None,
            "validation_season": int(val_years.min()) if len(val_years) else None,
            "chronological_season_order": bool(
                len(train_years) and len(val_years)
                and train_years.max() < val_years.min()
            ),
            "non_empty": bool(train_mask.any() and val_mask.any()),
        }
    return {
        "season_month_composition": composition.to_dict("records"),
        "season_source_order_run_count": len(runs),
        "season_source_order_run_hash": canonical_hash(runs),
        "season_run_counts": {str(key): value for key, value in sorted(run_counts.items())},
        "season_interleaving_observed": any(value > 1 for value in run_counts.values()),
        "source_order_hash": canonical_hash(list(zip(season.tolist(), month.astype(str).tolist()))),
        "origin_checks": origin_checks,
        "row_id_chronological": "NOT_PROVEN",
        "csv_order_chronological": "NOT_PROVEN",
        "exact_intra_month_chronology": "NOT_PROVEN",
    }


def audit_asof_frame(frame) -> tuple[dict[str, Any], Finding]:
    import pandas as pd

    invalid: Counter[str] = Counter()
    missing_by_season: dict[str, dict[str, float]] = {}
    for year, group in frame.groupby("season", sort=True):
        missing_by_season[str(int(year))] = {
            column: float(_missing_mask(group[column]).mean())
            for column in (*ASOF_COUNT_COLUMNS, *ASOF_RATE_COLUMNS)
        }
    for column in ASOF_COUNT_COLUMNS:
        missing = _missing_mask(frame[column])
        values = pd.to_numeric(frame.loc[~missing, column], errors="coerce")
        invalid[column] += int((values.isna() | (values < 0) | (values % 1 != 0)).sum())
    for column in ASOF_RATE_COLUMNS:
        missing = _missing_mask(frame[column])
        values = pd.to_numeric(frame.loc[~missing, column], errors="coerce")
        invalid[column] += int((values.isna() | (values < 0) | (values > 1)).sum())
    cold_start = {}
    for count_column, rate_columns in (
        ("asof_pitcher_n", tuple(c for c in ASOF_RATE_COLUMNS if "pitcher_" in c and "pitchmix" not in c)),
        ("asof_batter_n", tuple(c for c in ASOF_RATE_COLUMNS if "batter_" in c)),
        ("asof_pitcher_pitchmix_n", (
            "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        )),
    ):
        zero = pd.to_numeric(frame[count_column], errors="coerce") == 0
        cold_start[count_column] = {
            "zero_count_rows": int(zero.sum()),
            "missing_rate_fraction_at_zero": {
                column: float(_missing_mask(frame.loc[zero, column]).mean()) if zero.any() else None
                for column in rate_columns
            },
        }
    mix_columns = (
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    complete_mix = ~frame[list(mix_columns)].apply(_missing_mask).any(axis=1)
    mix_sum = frame.loc[complete_mix, list(mix_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).sum(axis=1)
    pitcher_count = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
    source_decreases = int(
        pitcher_count.groupby(frame["pitcher_id"], sort=False).diff().lt(0).sum()
    )
    count_frame = frame[["season", *ASOF_COUNT_COLUMNS]].copy()
    for column in ASOF_COUNT_COLUMNS:
        count_frame[column] = pd.to_numeric(count_frame[column], errors="coerce")
    yearly_means = count_frame.groupby("season")[list(ASOF_COUNT_COLUMNS)].mean()
    return {
        "invalid_domain_counts": {key: value for key, value in sorted(invalid.items()) if value},
        "missing_rates_by_season": missing_by_season,
        "cold_start_relationships": cold_start,
        "complete_pitchmix_rows": int(complete_mix.sum()),
        "pitchmix_sum_not_one_count": int((mix_sum.sub(1.0).abs() > 1e-6).sum()),
        "source_order_pitcher_count_decreases": source_decreases,
        "source_order_decrease_verdict": "NOT_PROVEN_AS_VIOLATION",
        "yearly_count_means": {
            str(int(year)): {column: float(value) for column, value in row.items()}
            for year, row in yearly_means.iterrows()
        },
        "exact_asof_reconstruction": "NOT_PROVEN",
    }, Finding(
        rule="asof_domains",
        severity=Severity.BLOCKER if any(invalid.values()) else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"aggregate invalid-domain counts: {dict(sorted(invalid.items()))}"
                if any(invalid.values()) else "count/rate domains passed; missing values excluded"),
    )


def _authorized_targets_for_frame(frame, role: TargetRole, *, origin: str | None = None) -> AuthorizedTargets:
    return extract_targets(
        TargetFrameRecordView(frame),
        role,
        origin=origin,
    )


def target_eda_2019_2021(frame) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd

    scoped = frame.loc[frame["season"].astype(int).isin(EXPLORATORY_TARGET_SEASONS)].copy()
    authorized = _authorized_targets_for_frame(scoped, TargetRole.EXPLORE_2019_2021)
    scoped["__authorized_target"] = authorized.values
    scoped["__pitcher_history_bin"] = pd.cut(
        pd.to_numeric(scoped["asof_pitcher_n"], errors="coerce"),
        bins=[-1, 0, 9, 49, 99, float("inf")],
        labels=["0", "1-9", "10-49", "50-99", "100+"],
    )

    def aggregate(columns: Sequence[str]) -> list[dict[str, Any]]:
        result = (
            scoped.groupby(list(columns), dropna=False, observed=False)["__authorized_target"]
            .agg(["size", "mean"]).reset_index()
            .rename(columns={"size": "n_rows", "mean": "target_rate"})
        )
        return result.to_dict("records")

    return {
        "scope": "hypothesis-generating target-aware EDA; 2019-2021 only",
        "by_count_state": aggregate(("balls_before", "strikes_before")),
        "by_base_state": aggregate(("base_state",)),
        "by_pitcher_history_bin": aggregate(("__pitcher_history_bin",)),
        "r2022_r2023_exploratory_target_access": False,
    }, {
        "role": TargetRole.EXPLORE_2019_2021.value,
        "seasons": list(authorized.seasons and sorted(set(authorized.seasons))),
        "purpose": "pre-registered aggregate exploratory tables",
        "n_labels_accessed": len(authorized.values),
    }


def bounded_target_rate_diagnostics(frame, selections: Sequence[MaskSelection]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for origin in ("r2022", "r2023"):
        selection = next(
            item for item in selections
            if item.pipeline == "baseline_reproduction" and item.origin == origin
        )
        full = frame.iloc[selection.full_positions]
        bounded = frame.iloc[selection.selected_positions]
        full_targets = _authorized_targets_for_frame(
            full, TargetRole.BOUNDED_R2022_R2023, origin=origin
        )
        bounded_targets = _authorized_targets_for_frame(
            bounded, TargetRole.BOUNDED_R2022_R2023, origin=origin
        )
        diagnostics.append({
            "origin": origin,
            "mask_role": "pretrained_outer_validation",
            "n_full": len(full_targets.values),
            "n_bounded": len(bounded_targets.values),
            "full_target_rate": target_rate(full_targets),
            "bounded_target_rate": target_rate(bounded_targets),
            "bounded_minus_full_target_rate": (
                target_rate(bounded_targets) - target_rate(full_targets)
            ),
            "usage_limit": "representativeness diagnostic only; no feature/model tuning",
        })
        ledger.append({
            "role": TargetRole.BOUNDED_R2022_R2023.value,
            "origin": origin,
            "seasons": [int(origin[1:])],
            "purpose": "pre-registered first-30000-vs-full target-rate diagnostic",
            "n_full_labels_accessed": len(full_targets.values),
            "n_bounded_labels_accessed": len(bounded_targets.values),
        })
    return diagnostics, ledger


def preprocessing_provenance_report(repo_root: Path, frame) -> dict[str, Any]:
    """Per-pipeline evidence; no learner is instantiated or fit."""
    source_findings = {
        finding.rule.removeprefix("pipeline_source:"): finding.to_dict()
        for finding in audit_pipeline_sources(repo_root)
    }
    category_exposure: dict[str, dict[str, int]] = {}
    is_r = frame["game_type"] == "R"
    for year in (2022, 2023):
        train_mask = (frame["season"].astype(int) <= year - 1) & is_r
        val_mask = (frame["season"].astype(int) == year) & is_r
        category_exposure[f"r{year}"] = {}
        for column in ("top_bottom", "game_type", "base_state"):
            train_values = set(frame.loc[train_mask, column].dropna().astype(str))
            val_values = set(frame.loc[val_mask, column].dropna().astype(str))
            category_exposure[f"r{year}"][column] = len(val_values - train_values)

    def field(verdict: PipelineVerdict, evidence: str) -> dict[str, str]:
        return {"verdict": verdict.value, "evidence": evidence}

    result = {
        "common_legacy_lgb": {
            "source_presence": source_findings["common_legacy_lgb"],
            "train_validation_scope": field(
                PipelineVerdict.PROVEN,
                "train_model receives caller-separated X_tr/y_tr and X_va/y_va",
            ),
            "imputation": field(PipelineVerdict.NOT_APPLICABLE, "LightGBM native missing path"),
            "scaling": field(PipelineVerdict.NOT_APPLICABLE, "no scaler in active common path"),
            "categorical_metadata": field(
                PipelineVerdict.NOT_PROVEN,
                "whole-frame pandas category dtype exists; downstream leakage effect not demonstrated",
            ),
            "feather_cache": field(
                PipelineVerdict.NOT_PROVEN,
                "cache-first load behavior exists; cache content parity requires separate evidence",
            ),
            "observed_validation_only_category_counts": category_exposure,
        },
        "mlp": {
            "source_presence": source_findings["mlp"],
            "historical_pipeline_behavior": field(
                PipelineVerdict.PROVEN,
                "historical workflow includes a 2024-primary validation/OOF role",
            ),
            "current_recovery_policy_admissibility": field(
                PipelineVerdict.VIOLATION,
                "2024-primary target access is not admissible for current recovery selection or this audit",
            ),
            "train_validation_scope": field(
                PipelineVerdict.PROVEN, "build_fold constructs tr=train[tr_m], va=train[va_m]"
            ),
            "imputation_scaling": field(
                PipelineVerdict.PROVEN, "numeric mean/std are computed from tr only"
            ),
            "categorical_vocabulary": field(
                PipelineVerdict.PROVEN,
                "vocabulary is built from tr unique values; unseen validation values map to 0",
            ),
            "missing_categorical_behavior": field(
                PipelineVerdict.NOT_PROVEN, "astype(str) behavior is visible but semantic impact untested"
            ),
        },
        "recovery_catboost": {
            "source_presence": source_findings["recovery_catboost"],
            "train_validation_scope": field(
                PipelineVerdict.PROVEN, "separate bounded inner/outer masks feed CatBoost Pools"
            ),
            "imputation_scaling": field(
                PipelineVerdict.NOT_APPLICABLE, "no explicit imputer or scaler"
            ),
            "categorical_metadata": field(
                PipelineVerdict.NOT_PROVEN,
                "whole-frame category dtype exists; material validation-category path not demonstrated",
            ),
            "observed_validation_only_category_counts": category_exposure,
        },
        "recovery_residual": {
            "source_presence": source_findings["recovery_residual"],
            "fit_scope": field(
                PipelineVerdict.PROVEN,
                "means/stds and category levels use only declared prior OOF fit_years",
            ),
            "unseen_categories": field(
                PipelineVerdict.PROVEN, "one-hot columns are prior-fit levels; unseen values are all-zero"
            ),
            "target_access": field(
                PipelineVerdict.PROVEN, "correction fit reads only declared prior OOF validation years"
            ),
        },
        "recovery_calibration": {
            "source_presence": source_findings["recovery_calibration"],
            "feature_preprocessing": field(
                PipelineVerdict.NOT_APPLICABLE, "operates on frozen baseline logits and bounded labels"
            ),
            "fit_scope": field(
                PipelineVerdict.PROVEN, "r2023 transformation fit panel is prior 2022 OOF only"
            ),
        },
        "baseline_reproduction": {
            "source_presence": source_findings["baseline_reproduction"],
            "fit_scope": field(
                PipelineVerdict.NOT_APPLICABLE, "loads pre-trained v93 assets; validation inference only"
            ),
            "preprocessing": field(
                PipelineVerdict.NOT_PROVEN,
                "common preprocessing is applied to whole train; fitted asset provenance is external",
            ),
        },
        "official_baseline": {
            "source_presence": source_findings["official_baseline"],
            "historical_pipeline_behavior": field(
                PipelineVerdict.PROVEN,
                "official historical notebook uses 2024 labels as validation and later refit input",
            ),
            "current_recovery_policy_admissibility": field(
                PipelineVerdict.VIOLATION,
                "historical 2024 target use is not admissible for current recovery selection or this audit",
            ),
            "pipeline_fit_scope": field(
                PipelineVerdict.PROVEN, "preprocessor is inside Pipeline fitted on X_train"
            ),
            "numeric_imputation": field(
                PipelineVerdict.PROVEN, "SimpleImputer(median) is inside the fitted pipeline"
            ),
            "unseen_categories": field(
                PipelineVerdict.PROVEN, "OrdinalEncoder uses unknown_value=-1"
            ),
            "target_access_for_this_audit": field(
                PipelineVerdict.VIOLATION,
                "historical notebook source accesses 2024 validation labels; it is not executed here",
            ),
        },
        "global_category_metadata_leakage_verdict": field(
            PipelineVerdict.NOT_PROVEN,
            "metadata exposure is observed, but no material downstream learner path is demonstrated",
        ),
    }
    contract_fields = (
        "train_mask_fit_scope", "validation_mask_scope", "imputation_fit_scope",
        "scaling_fit_scope", "categorical_vocabulary", "unseen_category_handling",
        "missing_categorical_behavior", "target_access",
        "validation_only_category_exposure", "whole_frame_category_metadata",
        "feather_cache_behavior", "material_learner_influence",
    )

    def contract(**overrides: dict[str, str]) -> dict[str, dict[str, str]]:
        output = {
            name: field(PipelineVerdict.NOT_PROVEN, "not demonstrated by this path audit")
            for name in contract_fields
        }
        output.update(overrides)
        return output

    result["common_legacy_lgb"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "caller supplies X_tr/y_tr"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "caller supplies X_va/y_va"),
        imputation_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "LightGBM native missing path"),
        scaling_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no scaler"),
        target_access=field(PipelineVerdict.PROVEN, "caller-separated y_tr/y_va"),
        validation_only_category_exposure=field(
            PipelineVerdict.NOT_PROVEN, "aggregate unseen-category counts observed; effect unproven"
        ),
        whole_frame_category_metadata=field(
            PipelineVerdict.NOT_PROVEN, "whole-frame dtype exists; material path unproven"
        ),
        feather_cache_behavior=field(
            PipelineVerdict.NOT_PROVEN, "cache-first load exists; content parity unproven"
        ),
        material_learner_influence=field(
            PipelineVerdict.NOT_PROVEN, "no category-metadata leakage path demonstrated"
        ),
    )
    result["mlp"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "tr=train[tr_m]"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "va=train[va_m]"),
        imputation_fit_scope=field(PipelineVerdict.PROVEN, "numeric means from tr only"),
        scaling_fit_scope=field(PipelineVerdict.PROVEN, "numeric stds from tr only"),
        categorical_vocabulary=field(PipelineVerdict.PROVEN, "unique values from tr only"),
        unseen_category_handling=field(PipelineVerdict.PROVEN, "unseen values map to index 0"),
        missing_categorical_behavior=field(
            PipelineVerdict.NOT_PROVEN, "astype(str) behavior visible; effect untested"
        ),
        target_access=field(
            PipelineVerdict.PROVEN,
            "historical ytr/yv use corresponding masks; admissibility is reported separately",
        ),
        whole_frame_category_metadata=field(
            PipelineVerdict.NOT_APPLICABLE, "fold vocabulary is rebuilt from tr strings"
        ),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "not used in build_fold"),
        material_learner_influence=field(PipelineVerdict.PROVEN, "fold transforms feed tensors"),
    )
    result["recovery_catboost"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "bounded inner/outer train masks"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "bounded inner/outer validation masks"),
        imputation_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no explicit imputer"),
        scaling_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no scaler"),
        target_access=field(PipelineVerdict.PROVEN, "labels follow each declared mask"),
        validation_only_category_exposure=field(
            PipelineVerdict.NOT_PROVEN, "aggregate exposure measured; CatBoost effect unproven"
        ),
        whole_frame_category_metadata=field(
            PipelineVerdict.NOT_PROVEN, "whole-frame dtype exists; material path unproven"
        ),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "runner reads CSV"),
        material_learner_influence=field(
            PipelineVerdict.NOT_PROVEN, "category metadata effect not demonstrated"
        ),
    )
    result["recovery_residual"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "declared prior OOF fit_years"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "declared OOF/apply year"),
        imputation_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no imputer in correction transform"),
        scaling_fit_scope=field(PipelineVerdict.PROVEN, "mean/std from prior OOF fit years"),
        categorical_vocabulary=field(PipelineVerdict.PROVEN, "levels from prior OOF fit years"),
        unseen_category_handling=field(PipelineVerdict.PROVEN, "unseen values produce all-zero one-hot"),
        target_access=field(PipelineVerdict.PROVEN, "correction labels from fit_years only"),
        whole_frame_category_metadata=field(PipelineVerdict.NOT_APPLICABLE, "explicit string levels used"),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "not used"),
        material_learner_influence=field(PipelineVerdict.PROVEN, "transforms feed correction learner"),
    )
    result["recovery_calibration"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "prior OOF calibration panel"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "bounded origin apply panel"),
        imputation_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "logits/labels only"),
        scaling_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no feature scaler"),
        categorical_vocabulary=field(PipelineVerdict.NOT_APPLICABLE, "no categorical inputs"),
        unseen_category_handling=field(PipelineVerdict.NOT_APPLICABLE, "no categorical inputs"),
        missing_categorical_behavior=field(PipelineVerdict.NOT_APPLICABLE, "no categorical inputs"),
        target_access=field(PipelineVerdict.PROVEN, "bounded panel labels only"),
        validation_only_category_exposure=field(PipelineVerdict.NOT_APPLICABLE, "no categorical inputs"),
        whole_frame_category_metadata=field(PipelineVerdict.NOT_APPLICABLE, "no categorical inputs"),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "frozen logits use npy cache"),
        material_learner_influence=field(PipelineVerdict.PROVEN, "prior panel fits calibration transform"),
    )
    result["baseline_reproduction"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "pre-trained assets"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "bounded outer validation only"),
        imputation_fit_scope=field(PipelineVerdict.NOT_PROVEN, "stored prep provenance external"),
        scaling_fit_scope=field(PipelineVerdict.NOT_PROVEN, "stored prep provenance external"),
        categorical_vocabulary=field(PipelineVerdict.NOT_PROVEN, "stored prep provenance external"),
        target_access=field(PipelineVerdict.PROVEN, "r2022/r2023 scoring labels in historical path"),
        whole_frame_category_metadata=field(
            PipelineVerdict.NOT_PROVEN, "common preprocessing precedes validation inference"
        ),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "reads CSV and asset/cache files"),
        material_learner_influence=field(PipelineVerdict.NOT_PROVEN, "stored fit provenance not reconstructed"),
    )
    result["official_baseline"]["contract"] = contract(
        train_mask_fit_scope=field(PipelineVerdict.PROVEN, "Pipeline.fit receives X_train"),
        validation_mask_scope=field(PipelineVerdict.PROVEN, "historical notebook defines 2024 validation"),
        imputation_fit_scope=field(PipelineVerdict.PROVEN, "median imputer inside fitted Pipeline"),
        scaling_fit_scope=field(PipelineVerdict.NOT_APPLICABLE, "no scaler"),
        categorical_vocabulary=field(PipelineVerdict.PROVEN, "OrdinalEncoder inside fitted Pipeline"),
        unseen_category_handling=field(PipelineVerdict.PROVEN, "unknown_value=-1"),
        missing_categorical_behavior=field(PipelineVerdict.NOT_PROVEN, "not separately demonstrated"),
        target_access=field(
            PipelineVerdict.PROVEN,
            "historical 2024 label access is demonstrated; current-policy admissibility is separate",
        ),
        whole_frame_category_metadata=field(PipelineVerdict.NOT_APPLICABLE, "sklearn object pipeline"),
        feather_cache_behavior=field(PipelineVerdict.NOT_APPLICABLE, "not used"),
        material_learner_influence=field(PipelineVerdict.PROVEN, "fitted Pipeline feeds RandomForest"),
    )
    return result


def audit_trackman_frame(trackman, main_frame) -> tuple[dict[str, Any], Finding, Finding]:
    import pandas as pd

    trackman_missing = _missing_mask(trackman["trackman_id"])
    trackman_id_duplicates = int(
        trackman.loc[~trackman_missing, "trackman_id"].astype(str).duplicated().sum()
    )
    composite_duplicates = int(
        trackman.duplicated(subset=["trackman_game_id", "pitch_no"]).sum()
    )
    dates = parse_trackman_game_dates(trackman["game_date"])
    invalid_dates = int(dates.isna().sum())
    date_season_mismatch = int(
        (dates.dt.year.loc[~dates.isna()].astype(int)
         != trackman.loc[~dates.isna(), "season"].astype(int)).sum()
    )
    missingness = {
        str(int(year)): {
            column: float(_missing_mask(group[column]).mean())
            for column in TRACKMAN_COLUMNS
        }
        for year, group in trackman.groupby("season", sort=True)
    }
    measurement = {}
    for column in TRACKMAN_MEASUREMENT_COLUMNS:
        values = pd.to_numeric(trackman[column], errors="coerce")
        measurement[column] = {
            str(int(year)): {
                "n_nonmissing": int(group.notna().sum()),
                "mean": float(group.mean()) if group.notna().any() else None,
                "std": float(group.std(ddof=0)) if group.notna().any() else None,
            }
            for year, group in values.groupby(trackman["season"], sort=True)
        }
    main_pitchers = set(main_frame["pitcher_id"].dropna().astype(str))
    main_batters = set(main_frame["batter_id"].dropna().astype(str))
    tm_pitchers = set(trackman["pitcher_trackman_id"].dropna().astype(str))
    tm_batters = set(trackman["batter_trackman_id"].dropna().astype(str))

    def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
        shared = len(left & right)
        return {
            "left_unique": len(left),
            "right_unique": len(right),
            "overlap_count": shared,
            "left_coverage_fraction": shared / len(left) if left else None,
            "compatibility_verdict": "NOT_PROVEN",
            "note": "raw identifier overlap is not an authoritative crosswalk",
        }

    seasons = sorted(int(value) for value in trackman["season"].unique())
    prior_coverage = all(
        any(source < year for source in seasons) for year in (2020, 2021, 2022, 2023)
    )
    usability = assess_trackman_usage(
        documented_shared_pitch_key=False,
        authoritative_player_crosswalk=False,
        documented_main_game_date_key=False,
        prior_season_coverage=prior_coverage,
        stable_league_measurements=False,
    )
    problems = (
        int(trackman_missing.sum()) + trackman_id_duplicates
        + invalid_dates + date_season_mismatch
    )
    report = {
        "n_rows_2019_2023": int(len(trackman)),
        "seasons": seasons,
        "trackman_id_missing": int(trackman_missing.sum()),
        "trackman_id_duplicates": trackman_id_duplicates,
        "game_pitch_composite_duplicates": composite_duplicates,
        "game_pitch_composite_key_verdict": "UNKNOWN",
        "game_pitch_composite_key_documented_unique": False,
        "game_pitch_composite_key_used_for_join": False,
        "invalid_game_dates": invalid_dates,
        "game_date_season_mismatches": date_season_mismatch,
        "key_uniqueness_status": "EMPIRICALLY_VERIFIED_NOT_DOCUMENTED",
        "missing_rates_by_season": missingness,
        "measurement_summary_by_season": measurement,
        "pitcher_identifier_overlap": overlap(main_pitchers, tm_pitchers),
        "batter_identifier_overlap": overlap(main_batters, tm_batters),
        "team_identifier_compatibility": "NOT_PROVEN",
        "park_coverage": "NOT_PROVEN_NO_PARK_FIELD",
        "provider_or_equipment_change": "NOT_PROVEN",
        "exact_main_row_join": "NOT_PROVEN",
        "fuzzy_join_attempted": False,
        "safe_usage_levels": usability,
    }
    structural_finding = Finding(
        rule="trackman_structural_contract",
        severity=Severity.LIKELY_ISSUE if problems else Severity.BENIGN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"aggregate key/date issue count={problems}" if problems else
                "schema/key/date aggregate checks passed; compatibility remains not proven"),
    )
    composite_finding = Finding(
        rule="trackman_game_pitch_composite_uniqueness",
        severity=Severity.UNKNOWN,
        evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
        reason=(f"observed duplicate count={composite_duplicates}; official uniqueness is not "
                "documented and no proposed join depends on this key"),
    )
    return report, structural_finding, composite_finding


def structural_pretrend_2019_2023(frame) -> dict[str, Any]:
    import pandas as pd

    annual: dict[str, Any] = {}
    for year, group in frame.groupby("season", sort=True):
        numeric = {}
        for column in AUDIT_NUMERIC_COLUMNS:
            missing = _missing_mask(group[column])
            values = pd.to_numeric(group.loc[~missing, column], errors="coerce").dropna()
            numeric[column] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=0)) if len(values) else None,
                "missing_rate": float(missing.mean()),
            }
        category = {}
        for column in (
            "game_month", "game_type", "balls_before", "strikes_before",
            "outs_before", "base_state", "pitcher_hand", "batter_hand",
        ):
            distribution = group[column].map(
                lambda value: "__MISSING__" if is_missing(value) else str(value)
            ).value_counts(normalize=True)
            category[column] = {
                str(key): float(value) for key, value in distribution.sort_index().items()
            }
        annual[str(int(year))] = {
            "n_rows": int(len(group)),
            "numeric": numeric,
            "categorical_distributions": category,
            "pitcher_coverage": int(group["pitcher_id"].nunique(dropna=True)),
            "batter_coverage": int(group["batter_id"].nunique(dropna=True)),
            "team_coverage": int(len(set(group["pitcher_team_id"].dropna().astype(str))
                                     | set(group["batter_team_id"].dropna().astype(str)))),
        }
    return {
        "annual": annual,
        "interpretation": {
            "abs_regime_change": "HYPOTHESIS_ONLY",
            "causal_attribution": "NOT_PROVEN",
            "possible_causes": (
                "baseball_process_change", "measurement_generation_change",
                "player_population_change", "ordinary_seasonal_drift", "UNKNOWN",
            ),
        },
    }


def abs_boundary_diagnostics(feature_frame) -> dict[str, Any]:
    """Report all pre-specified boundary deltas without selecting features."""
    import pandas as pd

    numeric: dict[str, Any] = {}
    for column in AUDIT_NUMERIC_COLUMNS:
        values = pd.to_numeric(feature_frame[column], errors="coerce")
        means = values.groupby(feature_frame["season"].astype(int)).mean()
        prior_deltas = {
            f"{year - 1}_to_{year}": float(means.get(year) - means.get(year - 1))
            for year in range(2020, 2024)
            if year in means.index and year - 1 in means.index
        }
        boundary = (
            float(means.get(2024) - means.get(2023))
            if 2023 in means.index and 2024 in means.index else None
        )
        prior_abs_max = max((abs(value) for value in prior_deltas.values()), default=None)
        numeric[column] = {
            "prior_year_deltas": prior_deltas,
            "boundary_2023_to_2024": boundary,
            "boundary_exceeds_prior_abs_max": (
                abs(boundary) > prior_abs_max
                if boundary is not None and prior_abs_max is not None else None
            ),
            "usage": "structural-break candidate only; not feature selection",
        }
    categorical: dict[str, Any] = {}
    for column in (
        "game_month", "game_type", "balls_before", "strikes_before",
        "outs_before", "base_state", "pitcher_hand", "batter_hand",
    ):
        distributions: dict[int, dict[str, float]] = {}
        for year, group in feature_frame.groupby("season", sort=True):
            distributions[int(year)] = group[column].map(
                lambda value: "__MISSING__" if is_missing(value) else str(value)
            ).value_counts(normalize=True).to_dict()
        prior_tv = {
            f"{year - 1}_to_{year}": _total_variation(
                distributions[year - 1], distributions[year]
            )
            for year in range(2020, 2024)
            if year in distributions and year - 1 in distributions
        }
        boundary_tv = (
            _total_variation(distributions[2023], distributions[2024])
            if 2023 in distributions and 2024 in distributions else None
        )
        prior_max = max(prior_tv.values(), default=None)
        categorical[column] = {
            "prior_year_total_variation": prior_tv,
            "boundary_2023_to_2024_total_variation": boundary_tv,
            "boundary_exceeds_prior_max": (
                boundary_tv > prior_max
                if boundary_tv is not None and prior_max is not None else None
            ),
            "usage": "structural-break candidate only; not feature selection",
        }
    return {
        "numeric": numeric,
        "categorical": categorical,
        "abrupt_vs_gradual_interpretation": (
            "Compare the complete boundary metrics with prior-year deltas; no automatic "
            "feature, threshold, transformation, or causal decision is made."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if is_missing(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in report.items()
        if key not in {"canonical_report_sha256", "runtime", "generated_at"}
    }
    return canonical_hash(_json_safe(payload))


def _assert_report_privacy(report: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "raw_rows", "row_ids", "individual_labels", "label_values",
        "predictions", "model_artifacts", "submission",
    }
    problems: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if key_text in forbidden_keys:
                    problems.append(f"forbidden report key {path}.{key_text}")
                visit(item, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(report, "report")
    if problems:
        raise AuditError("privacy contract failed: " + "; ".join(problems))


def _ensure_external_output_dir(output_dir: Path, repo_root: Path) -> Path:
    output = output_dir.expanduser().resolve()
    repo = repo_root.resolve()
    if output == repo or repo in output.parents:
        raise AuditError(f"audit output must be outside Git repository: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_report(report: Mapping[str, Any], output_dir: Path, *, stem: str) -> tuple[Path, Path]:
    safe = _json_safe(report)
    _assert_report_privacy(safe)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    findings = safe.get("findings", [])
    lines = [
        f"# {safe.get('audit_kind', 'Data audit')}",
        "",
        f"- Schema: `{safe.get('schema_version')}`",
        f"- Git SHA: `{safe.get('git_sha')}`",
        f"- Canonical report SHA-256: `{safe.get('canonical_report_sha256')}`",
        f"- Finding count: {len(findings)}",
        "",
        "## Findings",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- **{finding['severity']} / {finding['rule']}**: {finding['reason']}"
        )
    lines.extend((
        "",
        "## Boundaries",
        "",
        "Aggregate diagnostics only; no raw rows, individual labels, predictions, models, or submissions.",
        "No causal ABS claim and no feature/model/policy selection is authorized by this report.",
        "",
    ))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def run_medium_audit(
    train_csv: Path, trackman_csv: Path, output_dir: Path, repo_root: Path,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    """Run the aggregate 2019-2023 audit. No model is trained or scored."""
    repo_root = repo_root.resolve()
    output = _ensure_external_output_dir(output_dir, repo_root)
    train = read_scoped_train_2019_2023(train_csv.resolve())
    trackman = read_scoped_trackman_2019_2023(trackman_csv.resolve())
    integrity_targets = _authorized_targets_for_frame(
        train, TargetRole.INTEGRITY_2019_2023
    )
    structural, structural_findings = audit_main_structural_frame(train)
    temporal = audit_temporal_frame(train)
    asof, asof_finding = audit_asof_frame(train)
    selections = build_pipeline_geometries_frame(train)
    bounded = compare_all_bounded_geometries(train, selections)
    bounded_target, bounded_ledger = bounded_target_rate_diagnostics(train, selections)
    exploratory, exploratory_ledger = target_eda_2019_2021(train)
    trackman_report, trackman_finding, trackman_composite_finding = (
        audit_trackman_frame(trackman, train)
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "audit_kind": "Data Integrity, Temporal Structure, ABS-Regime, and Trackman Usability Audit",
        "git_sha": _git_sha(repo_root),
        "scope": {
            "main_structural_seasons": [2019, 2020, 2021, 2022, 2023],
            "exploratory_target_seasons": [2019, 2020, 2021],
            "bounded_target_origins": ["r2022", "r2023"],
            "target_2024_access": False,
            "test_distribution_access": False,
            "public_leaderboard_evidence": False,
            "external_information_access": False,
            "model_training_or_scoring": False,
        },
        "sources": {
            "train": {
                "name": train_csv.name,
                "hash_kind": "canonical_projected_2019_2023_features_only_sha256",
                "sha256": _canonical_frame_hash(train, TRAIN_FEATURE_COLUMNS),
                "rows": len(train),
                "target_row_id_alignment": {
                    "ordered_exact_match": True,
                    "row_id_hash": train.attrs.get("target_alignment_row_id_hash"),
                },
            },
            "trackman": {
                "name": trackman_csv.name,
                "hash_kind": "canonical_projected_2019_2023_sha256",
                "sha256": _canonical_frame_hash(trackman, TRACKMAN_COLUMNS),
                "rows": len(trackman),
            },
            "audit_script": {
                "path": "scripts/audit_data_integrity_temporal.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
        "label_access_ledger": [{
            "role": TargetRole.INTEGRITY_2019_2023.value,
            "seasons": sorted(set(integrity_targets.seasons)),
            "purpose": "target availability and binary-domain validation only",
            "n_labels_accessed": len(integrity_targets.values),
            "target_rate_reported": False,
        }, exploratory_ledger, *bounded_ledger],
        "findings": [
            *(finding.to_dict() for finding in structural_findings),
            Finding(
                rule="target_domain_2019_2023",
                severity=Severity.BENIGN,
                evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
                reason=(f"{len(integrity_targets.values)} authorized targets passed "
                        "availability and binary-domain checks; no aggregate target rate reported"),
            ).to_dict(),
            asof_finding.to_dict(),
            trackman_finding.to_dict(),
            trackman_composite_finding.to_dict(),
        ],
        "sections": {
            "main_structural": structural,
            "temporal_source_order": temporal,
            "asof_provenance": asof,
            "first_30000_by_pipeline_geometry": bounded,
            "bounded_target_rate_diagnostic": bounded_target,
            "target_eda_2019_2021": exploratory,
            "preprocessing_provenance": preprocessing_provenance_report(repo_root, train),
            "abs_pretrend_2019_2023": structural_pretrend_2019_2023(train),
            "trackman_usability": trackman_report,
        },
        "expensive_work_executed": False,
    }
    # Boolean privacy declarations use non-forbidden key names in the serialized report.
    report["privacy_contract"] = {
        "aggregate_only": True,
        "contains_raw_row_material": False,
        "contains_individual_target_material": False,
        "contains_row_identifier_lists": False,
        "row_identifier_hashes_allowed": True,
        "contains_predictions_models_or_submissions": False,
    }
    report["canonical_report_sha256"] = canonical_report_hash(report)
    return report, _write_report(report, output, stem="audit_report")


def run_abs_2024_feature_audit(
    train_csv: Path, output_dir: Path, repo_root: Path,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    """Separately invoked, branch-inert 2024 feature-only structural diagnostic."""
    import pandas as pd

    repo_root = repo_root.resolve()
    output = _ensure_external_output_dir(output_dir, repo_root)
    validate_header(train_csv.resolve(), TRAIN_COLUMNS)
    features = pd.read_csv(
        train_csv.resolve(), encoding="utf-8-sig", usecols=list(TRAIN_FEATURE_COLUMNS)
    )
    if TARGET in features.columns:
        raise AuditError("2024 feature-only projection unexpectedly contains target")
    season = pd.to_numeric(features["season"], errors="coerce")
    scoped = features.loc[season.isin((*ALLOWED_SEASONS, 2024))].copy()
    prior = scoped.loc[scoped["season"].astype(int).isin(ALLOWED_SEASONS)]
    current = scoped.loc[scoped["season"].astype(int) == 2024]
    if current.empty:
        raise AuditError("2024 feature-only diagnostic has no 2024 rows")
    pretrend = structural_pretrend_2019_2023(prior)
    comparison = structural_pretrend_2019_2023(current)["annual"]["2024"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "audit_kind": "Isolated 2024 feature-only ABS structural diagnostic",
        "git_sha": _git_sha(repo_root),
        "scope": {
            "target_column_in_projection": False,
            "target_access": False,
            "branch_inert": True,
            "may_tune_thresholds": False,
            "may_select_features": False,
            "may_choose_transformations": False,
            "may_modify_recovery_policy": False,
            "requires_separate_experiment_brief_for_modeling": True,
        },
        "sources": {
            "train_features": {
                "name": train_csv.name,
                "hash_kind": "canonical_projected_2019_2024_features_only_sha256",
                "sha256": _canonical_frame_hash(scoped, TRAIN_FEATURE_COLUMNS),
                "rows": len(scoped),
            },
            "audit_script": {
                "path": "scripts/audit_data_integrity_temporal.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
        "label_access_ledger": [],
        "findings": [Finding(
            rule="abs_2024_feature_only",
            severity=Severity.UNKNOWN,
            evidence_status=EvidenceStatus.EMPIRICALLY_VERIFIED,
            reason="structural comparison only; ABS causality and predictive value are not proven",
        ).to_dict()],
        "sections": {
            "pretrend_2019_2023": pretrend,
            "feature_only_2024": comparison,
            "boundary_diagnostics_all_prespecified_features": abs_boundary_diagnostics(scoped),
            "interpretation": {
                "structural_break_candidates_only": True,
                "causal_claim": "NOT_PROVEN",
                "model_or_policy_action": "FORBIDDEN_IN_THIS_AUDIT",
            },
        },
        "privacy_contract": {
            "aggregate_only": True,
            "contains_raw_row_material": False,
            "contains_individual_target_material": False,
            "contains_row_identifier_lists": False,
        },
        "expensive_work_executed": False,
    }
    report["canonical_report_sha256"] = canonical_report_hash(report)
    return report, _write_report(report, output, stem="abs_feature_report")


def _static_report(repo_root: Path) -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    semantic_problems = semantic_source_audit(source)
    if semantic_problems:
        raise AuditError("semantic source audit failed: " + "; ".join(semantic_problems))
    reader_problems = audit_2024_reader_semantics(source)
    if reader_problems:
        raise AuditError("2024 reader semantic audit failed: " + "; ".join(reader_problems))
    medium_reader_problems = audit_medium_reader_semantics(source)
    if medium_reader_problems:
        raise AuditError(
            "MEDIUM reader semantic audit failed: " + "; ".join(medium_reader_problems)
        )
    pipeline_findings = audit_pipeline_sources(repo_root)
    head = _git_sha(repo_root)
    status_finding = classify_project_status_sha(
        (repo_root / "PROJECT_STATUS.md").read_text(encoding="utf-8"), head
    )
    return {
        "mode": "CHEAP_STATIC_ONLY",
        "semantic_source_audit": "PASS",
        "target_role_semantic_audit": "PASS",
        "medium_reader_semantic_audit": "PASS",
        "pipeline_findings": [finding.to_dict() for finding in pipeline_findings],
        "project_status": status_finding.to_dict(),
        "abs_2024_feature_only_contract": abs_feature_only_contract(),
        "medium_full_data_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data Integrity, Temporal, ABS-Regime, and Trackman Audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static", help="AST/source inspection only")
    static.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    headers = subparsers.add_parser("headers", help="CSV header-only validation")
    headers.add_argument("--train-csv", type=Path, required=True)
    headers.add_argument("--trackman-csv", type=Path, required=True)
    medium = subparsers.add_parser(
        "medium", help="aggregate 2019-2023 train and Trackman audit"
    )
    medium.add_argument("--train-csv", type=Path, required=True)
    medium.add_argument("--trackman-csv", type=Path, required=True)
    medium.add_argument("--output-dir", type=Path, required=True)
    medium.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    abs_2024 = subparsers.add_parser(
        "abs-2024-features", help="separate branch-inert 2024 feature-only diagnostic"
    )
    abs_2024.add_argument("--train-csv", type=Path, required=True)
    abs_2024.add_argument("--output-dir", type=Path, required=True)
    abs_2024.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "static":
            report = _static_report(args.repo_root.resolve())
        elif args.command == "headers":
            findings = (
                validate_header(args.train_csv.resolve(), TRAIN_COLUMNS),
                validate_header(args.trackman_csv.resolve(), TRACKMAN_COLUMNS),
            )
            report = {
                "mode": "CHEAP_HEADER_ONLY",
                "findings": [finding.to_dict() for finding in findings],
                "rows_read": 0,
                "medium_full_data_executed": False,
            }
        elif args.command == "medium":
            completed, paths = run_medium_audit(
                args.train_csv, args.trackman_csv, args.output_dir, args.repo_root
            )
            report = {
                "mode": "MEDIUM_AGGREGATE_AUDIT",
                "canonical_report_sha256": completed["canonical_report_sha256"],
                "outputs": [str(path) for path in paths],
            }
        else:
            completed, paths = run_abs_2024_feature_audit(
                args.train_csv, args.output_dir, args.repo_root
            )
            report = {
                "mode": "MEDIUM_ABS_2024_FEATURE_ONLY",
                "canonical_report_sha256": completed["canonical_report_sha256"],
                "outputs": [str(path) for path in paths],
            }
    except AuditError as exc:
        print(f"[AUDIT ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
