from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_data_integrity_temporal.py"
SPEC = importlib.util.spec_from_file_location("audit_data_integrity_temporal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class PoisonTarget:
    def __int__(self):
        raise AssertionError("2024 target was converted")

    def __float__(self):
        raise AssertionError("2024 target was converted")

    def __bool__(self):
        raise AssertionError("2024 target was used for branching")

    def __str__(self):
        raise AssertionError("2024 target was reported")


class PoisonTargetMapping(dict):
    def __getitem__(self, key):
        if key == "control_success":
            raise AssertionError("2024 target was extracted")
        return super().__getitem__(key)


def good_record(row_id: str, season: int = 2021, game_type: str = "R") -> dict[str, object]:
    return {
        "row_id": row_id,
        "season": season,
        "game_month": 5,
        "game_dayofweek": 2,
        "inning": 3,
        "top_bottom": "T",
        "game_type": game_type,
        "balls_before": 2,
        "strikes_before": 1,
        "outs_before": 1,
        "run_top_before": 2,
        "run_bot_before": 1,
        "run_total_before": 3,
        "runner_on_1b": 1,
        "runner_on_2b": 0,
        "runner_on_3b": 1,
        "num_runners_on": 2,
        "base_state": "1_3",
        "pitcher_id": 10,
        "batter_id": 20,
        "pitcher_team_id": 1,
        "batter_team_id": 2,
        "asof_pitcher_n": 10,
        "asof_batter_n": 20,
        "asof_pitcher_pitchmix_n": 8,
        "asof_pitcher_success_rate": 0.5,
        "asof_batter_success_rate": 0.4,
        "asof_pitcher_fastball_rate": 0.6,
        "control_success": 1,
    }


class HeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="aimers9-audit-header-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _header(self, name: str, columns: tuple[str, ...]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(columns)
        return path

    def test_exact_train_and_trackman_headers_pass(self) -> None:
        train = self._header("train.csv", audit.TRAIN_COLUMNS)
        trackman = self._header("trackman_history.csv", audit.TRACKMAN_COLUMNS)
        self.assertEqual(audit.validate_header(train, audit.TRAIN_COLUMNS).severity.value, "BENIGN")
        self.assertEqual(
            audit.validate_header(trackman, audit.TRACKMAN_COLUMNS).evidence_status.value,
            "EMPIRICALLY_VERIFIED",
        )

    def test_header_mismatch_fails_closed(self) -> None:
        train = self._header("train.csv", audit.TRAIN_COLUMNS[:-1])
        with self.assertRaisesRegex(audit.AuditError, "header mismatch"):
            audit.validate_header(train, audit.TRAIN_COLUMNS)

    def test_header_cli_reads_no_rows(self) -> None:
        train = self._header("train.csv", audit.TRAIN_COLUMNS)
        trackman = self._header("trackman_history.csv", audit.TRACKMAN_COLUMNS)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = audit.main([
                "headers", "--train-csv", str(train),
                "--trackman-csv", str(trackman),
            ])
        report = json.loads(output.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(report["rows_read"], 0)
        self.assertFalse(report["medium_full_data_executed"])


class TargetFirewallTests(unittest.TestCase):
    def test_2024_poison_target_is_rejected_before_access(self) -> None:
        record = PoisonTargetMapping(season=2024, control_success=PoisonTarget())
        with self.assertRaisesRegex(audit.AuditError, "2024 targets are firewalled"):
            audit.extract_targets([record], audit.TargetRole.INTEGRITY_2019_2023)

    def test_feature_only_role_never_returns_targets(self) -> None:
        with self.assertRaisesRegex(audit.AuditError, "cannot request targets"):
            audit.extract_targets([], audit.TargetRole.FEATURE_ONLY_2024)

    def test_exploratory_target_scope_stops_at_2021(self) -> None:
        with self.assertRaisesRegex(audit.AuditError, "cannot access seasons"):
            audit.extract_targets(
                [good_record("x", season=2022)],
                audit.TargetRole.EXPLORE_2019_2021,
            )

    def test_bounded_target_role_requires_exact_origin_year(self) -> None:
        targets = audit.extract_targets(
            [good_record("x", season=2022)],
            audit.TargetRole.BOUNDED_R2022_R2023,
            origin="r2022",
        )
        self.assertEqual(audit.target_rate(targets), 1.0)
        with self.assertRaisesRegex(audit.AuditError, "cannot access seasons"):
            audit.extract_targets(
                [good_record("x", season=2023)],
                audit.TargetRole.BOUNDED_R2022_R2023,
                origin="r2022",
            )

    def test_2024_feature_reader_uses_column_projection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aimers9-audit-feature-only-") as temp:
            path = Path(temp) / "train.csv"
            row = {column: 0 for column in audit.TRAIN_COLUMNS}
            row.update({
                "row_id": "r2024", "season": 2024, "top_bottom": "T",
                "game_type": "R", "base_state": "___",
                "control_success": "DO_NOT_PARSE_OR_RETURN",
            })
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit.TRAIN_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            frame = audit.read_2024_features_only(path)
            self.assertEqual(len(frame), 1)
            self.assertNotIn(audit.TARGET, frame.columns)
            self.assertEqual(tuple(frame.columns), audit.TRAIN_FEATURE_COLUMNS)


class SemanticSourceTests(unittest.TestCase):
    def test_comments_and_prose_do_not_trigger_forbidden_source_audit(self) -> None:
        source = '''\n# 2024 test Public leaderboard are policy words only.\n"""test.csv is mentioned in documentation."""\ndef safe():\n    return 1\n'''
        self.assertEqual(audit.semantic_source_audit(source), [])

    def test_network_import_is_rejected(self) -> None:
        self.assertIn(
            "forbidden external-information import: requests",
            audit.semantic_source_audit("import requests\n"),
        )

    def test_forbidden_cli_role_is_rejected(self) -> None:
        source = "parser.add_argument('--test-csv')\n"
        self.assertTrue(any("forbidden CLI" in item for item in audit.semantic_source_audit(source)))

    def test_forbidden_reader_path_is_rejected(self) -> None:
        source = "import pandas as pd\npd.read_csv('test.csv')\n"
        self.assertTrue(any("forbidden data reader" in item for item in audit.semantic_source_audit(source)))

    def test_audit_script_itself_passes_semantic_scan(self) -> None:
        self.assertEqual(audit.semantic_source_audit(SCRIPT.read_text(encoding="utf-8")), [])

    def test_2024_reader_requires_explicit_non_target_projection(self) -> None:
        source = '''
def read_2024_features_only(path):
    frame = pd.read_csv(path)
    return frame.loc[frame["season"] == 2024]
'''
        problems = audit.audit_2024_reader_semantics(source)
        self.assertTrue(any("usecols" in item for item in problems))

    def test_2024_reader_cannot_extract_target_values(self) -> None:
        source = '''
def read_2024_features_only(path):
    frame = pd.read_csv(path, usecols=list(TRAIN_FEATURE_COLUMNS))
    return frame["control_success"]
'''
        problems = audit.audit_2024_reader_semantics(source)
        self.assertTrue(any("target-value access" in item for item in problems))

    def test_audit_2024_reader_passes_target_role_semantics(self) -> None:
        self.assertEqual(
            audit.audit_2024_reader_semantics(SCRIPT.read_text(encoding="utf-8")),
            [],
        )

    def test_medium_readers_pass_projection_semantics(self) -> None:
        self.assertEqual(
            audit.audit_medium_reader_semantics(SCRIPT.read_text(encoding="utf-8")),
            [],
        )

    def test_unscoped_medium_target_reader_fails_semantics(self) -> None:
        source = '''
def read_scoped_train_2019_2023(path):
    features = pd.read_csv(path, usecols=list(TRAIN_FEATURE_COLUMNS))
    targets = pd.read_csv(path, usecols=[TARGET])
def run_abs_2024_feature_audit(path):
    return pd.read_csv(path, usecols=list(TRAIN_FEATURE_COLUMNS))
'''
        problems = audit.audit_medium_reader_semantics(source)
        self.assertTrue(any("skiprows firewall" in item for item in problems))

    def test_lazy_frame_firewall_rejects_2024_before_target_conversion(self) -> None:
        import pandas as pd

        frame = pd.DataFrame([{"season": 2024, "control_success": PoisonTarget()}])
        with self.assertRaisesRegex(audit.AuditError, "2024 targets are firewalled"):
            audit._authorized_targets_for_frame(
                frame, audit.TargetRole.INTEGRITY_2019_2023
            )


class IntegrityAndAsofTests(unittest.TestCase):
    def test_common_missing_predicate_handles_numpy_and_pandas_na(self) -> None:
        import numpy as np
        import pandas as pd

        for value in (None, "", np.nan, pd.NA):
            with self.subTest(value=type(value).__name__):
                self.assertTrue(audit.is_missing(value))
        for value in (0, 0.0, False, " ", "value"):
            with self.subTest(value=value):
                self.assertFalse(audit.is_missing(value))

    def test_cold_start_missing_asof_rates_are_not_domain_violations(self) -> None:
        import numpy as np
        import pandas as pd

        for missing in (np.nan, pd.NA):
            with self.subTest(missing=type(missing).__name__):
                record = good_record("cold-start")
                record["asof_pitcher_n"] = 0
                record["asof_pitcher_success_rate"] = missing
                asof_finding = audit.audit_asof_records([record])[0]
                row_finding = audit.audit_row_semantics([record])[0]
                self.assertEqual(asof_finding.severity.value, "BENIGN")
                self.assertEqual(row_finding.severity.value, "BENIGN")

    def test_valid_row_semantics_pass(self) -> None:
        findings = audit.audit_row_semantics([good_record("a"), good_record("b")])
        self.assertEqual(findings[0].severity.value, "BENIGN")

    def test_state_invariant_failure_is_blocker(self) -> None:
        record = good_record("a")
        record["run_total_before"] = 99
        findings = audit.audit_row_semantics([record])
        self.assertEqual(findings[0].severity.value, "BLOCKER")
        self.assertIn("run_total_mismatch", findings[0].reason)

    def test_finding_serialization_never_discloses_raw_row_id(self) -> None:
        sentinel = "SENTINEL-PRIVATE-ROW-ID"
        record = good_record(sentinel)
        record["run_total_before"] = 99
        serialized = json.dumps(
            [finding.to_dict() for finding in audit.audit_row_semantics([record])],
            sort_keys=True,
        )
        self.assertNotIn(sentinel, serialized)
        self.assertIn("run_total_mismatch", serialized)

    def test_duplicate_states_are_unknown_not_automatic_corruption(self) -> None:
        findings = audit.audit_row_semantics([good_record("a"), good_record("b")])
        duplicate = next(finding for finding in findings if finding.rule == "duplicate_input_states")
        self.assertEqual(duplicate.severity.value, "UNKNOWN")
        self.assertIn("not inferable", duplicate.reason)

    def test_source_order_count_decrease_is_not_violation(self) -> None:
        first = good_record("a")
        second = good_record("b")
        first["asof_pitcher_n"] = 20
        second["asof_pitcher_n"] = 10
        findings = audit.audit_asof_records([first, second])
        decrease = next(f for f in findings if f.rule == "asof_count_source_order_decreases")
        self.assertEqual(decrease.severity.value, "UNKNOWN")
        self.assertEqual(decrease.pipeline_verdict.value, "NOT_APPLICABLE")

    def test_asof_out_of_range_is_blocker(self) -> None:
        record = good_record("a")
        record["asof_pitcher_success_rate"] = 1.1
        finding = audit.audit_asof_records([record])[0]
        self.assertEqual(finding.severity.value, "BLOCKER")


class TotalVariationTests(unittest.TestCase):
    def test_total_variation_matches_definition(self) -> None:
        left = {"a": 0.5, "b": 0.5}
        right = {"a": 0.25, "b": 0.75}
        self.assertEqual(audit._total_variation(left, right), 0.25)

    def test_total_variation_is_invariant_to_mapping_insertion_order(self) -> None:
        left = {"z": 0.5, "a": 0.25, "m": 0.25}
        right = {"m": 0.2, "z": 0.3, "b": 0.5}
        reordered_left = dict(reversed(tuple(left.items())))
        reordered_right = dict(reversed(tuple(right.items())))

        expected = 0.5
        self.assertEqual(audit._total_variation(left, right), expected)
        self.assertEqual(
            audit._total_variation(reordered_left, reordered_right), expected
        )

    def test_total_variation_uses_sorted_key_order_for_fsum(self) -> None:
        observed: list[list[float]] = []
        original_fsum = audit.math.fsum

        def spy(values: Iterable[float]) -> float:
            values = list(values)
            observed.append(values)
            return original_fsum(values)

        with mock.patch.object(audit.math, "fsum", side_effect=spy):
            result = audit._total_variation(
                {"z": 4.0, "a": 1.0, "m": 3.0},
                {"z": 0.0, "b": 2.0, "a": 0.0},
            )

        self.assertEqual(result, 5.0)
        self.assertEqual(observed, [[1.0, 2.0, 3.0, 4.0]])

    def test_total_variation_is_hash_seed_invariant_in_subprocesses(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            "import audit_data_integrity_temporal as audit\n"
            "weights = [1.0 / (i + 1) for i in range(2000)]\n"
            "total = sum(weights)\n"
            "left = {f'k{i}': value / total for i, value in enumerate(weights)}\n"
            "right = {f'k{i}': value / total for i, value in enumerate(reversed(weights))}\n"
            "print(repr(audit._total_variation(left, right)))\n"
        )
        outputs = []
        for seed in ("1", "2", "3", "4"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs.append(completed.stdout.strip())

        self.assertTrue(outputs[0])
        self.assertEqual(len(set(outputs)), 1)


class AbsMetricContractTests(unittest.TestCase):
    @staticmethod
    def frame():
        import pandas as pd

        rows = []
        for year in range(2019, 2025):
            row: dict[str, object] = {"season": year}
            for column in audit.AUDIT_NUMERIC_COLUMNS:
                row[column] = float(year)
            for column in audit.ABS_CATEGORICAL_COLUMNS:
                row[column] = "A" if year % 2 else "B"
            for column in audit.AUDIT_COVERAGE_COLUMNS:
                row[column] = f"{column}-{year}"
            rows.append(row)
        return pd.DataFrame(rows)

    def test_all_five_families_exact_columns_and_transitions_are_emitted(self) -> None:
        diagnostics = audit.abs_boundary_diagnostics(self.frame())
        expected_families = {
            "numeric_smd", "quantile_change", "categorical_total_variation",
            "missing_rate_delta_pp", "entity_coverage_change",
        }
        self.assertEqual(
            expected_families,
            set(diagnostics) & expected_families,
        )
        expected_transitions = {
            "2019_to_2020", "2020_to_2021", "2021_to_2022",
            "2022_to_2023", "2023_to_2024",
        }
        for family in expected_families:
            columns = (
                audit.AUDIT_NUMERIC_COLUMNS
                if family in {"numeric_smd", "quantile_change", "missing_rate_delta_pp"}
                else audit.ABS_CATEGORICAL_COLUMNS
                if family == "categorical_total_variation"
                else audit.AUDIT_COVERAGE_COLUMNS
            )
            for column in columns:
                self.assertEqual(
                    set(diagnostics[family][column]["transitions"]),
                    expected_transitions,
                )

    def test_numeric_smd_is_signed_pooled_population_sd_and_finite(self) -> None:
        import pandas as pd

        frame = self.frame()
        frame.loc[frame["season"] == 2019, "inning"] = [1.0]
        frame.loc[frame["season"] == 2020, "inning"] = [3.0]
        frame = pd.concat([frame, frame.iloc[[0]].assign(season=2019, inning=3.0)])
        frame = pd.concat([frame, frame.iloc[[1]].assign(season=2020, inning=5.0)])
        result = audit.abs_boundary_diagnostics(frame)["numeric_smd"]["inning"]
        self.assertEqual(result["transitions"]["2019_to_2020"]["value"], 2.0)
        self.assertIsNone(result["transitions"]["2019_to_2020"]["reason"])

    def test_zero_pooled_variance_and_empty_invalid_numeric_fail_closed(self) -> None:
        import pandas as pd

        frame = self.frame()
        frame["inning"] = frame["inning"].astype(object)
        frame.loc[frame["season"] == 2019, "inning"] = [2.0]
        frame.loc[frame["season"] == 2020, "inning"] = [2.0]
        frame = pd.concat([frame, frame.iloc[[0]].assign(season=2019, inning=2.0)])
        frame = pd.concat([frame, frame.iloc[[1]].assign(season=2020, inning=2.0)])
        result = audit.abs_boundary_diagnostics(frame)["numeric_smd"]["inning"]
        self.assertEqual(result["transitions"]["2019_to_2020"]["value"], 0.0)

        unequal = frame.copy()
        unequal.loc[unequal["season"] == 2020, "inning"] = 3.0
        unequal_result = audit.abs_boundary_diagnostics(unequal)["numeric_smd"]["inning"]
        self.assertIsNone(unequal_result["transitions"]["2019_to_2020"]["value"])
        self.assertEqual(
            unequal_result["transitions"]["2019_to_2020"]["reason"],
            "degenerate_pooled_variance_unequal_means",
        )

        frame.loc[frame["season"] == 2020, "inning"] = "bad"
        invalid = audit.abs_boundary_diagnostics(frame)["numeric_smd"]["inning"]
        self.assertIsNone(invalid["transitions"]["2019_to_2020"]["value"])
        self.assertEqual(
            invalid["transitions"]["2019_to_2020"]["reason"],
            "no_valid_numeric_values_in_later_year",
        )

    def test_quantile_grid_linear_interpolation_and_missing_rate_semantics(self) -> None:
        frame = self.frame()
        frame.loc[frame["season"] == 2019, "inning"] = 10.0
        frame.loc[frame["season"] == 2020, "inning"] = 12.0
        diagnostics = audit.abs_boundary_diagnostics(frame)
        quantiles = diagnostics["quantile_change"]["inning"]["transitions"]["2019_to_2020"]
        self.assertEqual(
            set(quantiles["quantiles"]), {"0.10", "0.50", "0.90"}
        )
        self.assertEqual(
            [quantiles["quantiles"][key]["value"] for key in ("0.10", "0.50", "0.90")],
            [2.0, 2.0, 2.0],
        )
        self.assertEqual(audit.ABS_QUANTILE_INTERPOLATION, "linear")

        frame.loc[frame["season"] == 2019, "inning"] = None
        frame.loc[frame["season"] == 2020, "inning"] = 12.0
        missing = audit.abs_boundary_diagnostics(frame)["missing_rate_delta_pp"]["inning"]
        self.assertEqual(
            missing["transitions"]["2019_to_2020"]["value"], -100.0
        )

        frame = self.frame()
        frame["inning"] = frame["inning"].astype(object)
        frame.loc[frame["season"] == 2021, "inning"] = "bad"
        invalid_quantiles = audit.abs_boundary_diagnostics(frame)[
            "quantile_change"
        ]["inning"]
        transition = invalid_quantiles["transitions"]["2021_to_2022"]
        for item in transition["quantiles"].values():
            self.assertIsNone(item["value"])
            self.assertEqual(item["reason"], "no_valid_numeric_values_in_earlier_year")
        self.assertIsNone(invalid_quantiles["historical_abs_max_by_quantile"]["0.10"])
        self.assertIsNone(
            invalid_quantiles["boundary_exceeds_historical_abs_max_by_quantile"]["0.10"]
        )
        self.assertEqual(
            invalid_quantiles["comparison_reason_by_quantile"]["0.10"],
            "historical_transition_invalid:2020_to_2021:no_valid_numeric_values_in_later_year",
        )

    def test_absent_categorical_level_and_entity_prior_zero_are_explicit(self) -> None:
        frame = self.frame()
        frame.loc[frame["season"] == 2019, "game_type"] = "A"
        frame.loc[frame["season"] == 2020, "game_type"] = "B"
        categorical = audit.abs_boundary_diagnostics(frame)[
            "categorical_total_variation"
        ]["game_type"]["transitions"]["2019_to_2020"]
        self.assertEqual(categorical["value"], 1.0)

        frame.loc[frame["season"] == 2019, "pitcher_id"] = None
        coverage = audit.abs_boundary_diagnostics(frame)["entity_coverage_change"][
            "pitcher_id"
        ]["transitions"]["2019_to_2020"]
        self.assertEqual(coverage["prior_unique_count"], 0)
        self.assertIsNone(coverage["relative_unique_count_change"]["value"])
        self.assertEqual(
            coverage["relative_unique_count_change"]["reason"],
            "prior_unique_count_zero",
        )
        coverage_family = audit.abs_boundary_diagnostics(frame)["entity_coverage_change"][
            "pitcher_id"
        ]
        self.assertIsNone(coverage_family["historical_abs_max_relative_change"])
        self.assertIsNone(
            coverage_family["boundary_exceeds_historical_abs_max_relative_change"]
        )
        self.assertEqual(
            coverage_family["comparison_reason"],
            "historical_transition_invalid:2019_to_2020:prior_unique_count_zero",
        )

    def test_partial_historical_metric_history_fails_closed(self) -> None:
        frame = self.frame()
        partial = frame.loc[frame["season"] != 2020].copy()
        numeric = audit.abs_boundary_diagnostics(partial)["numeric_smd"]["inning"]
        self.assertIsNone(numeric["historical_abs_max"])
        self.assertIsNone(numeric["boundary_exceeds_historical_abs_max"])
        self.assertEqual(
            numeric["comparison_reason"],
            "historical_transition_invalid:2019_to_2020:no_valid_numeric_values_in_later_year",
        )

    def test_contract_provenance_and_explicit_negative_scope_fields(self) -> None:
        contract = audit.abs_feature_only_contract()
        self.assertEqual(
            contract["metric_contract_version"],
            audit.ABS_METRIC_CONTRACT_VERSION,
        )
        self.assertEqual(tuple(contract["quantile_grid"]), audit.ABS_QUANTILE_GRID)
        self.assertEqual(
            tuple(contract["transitions"]),
            tuple(f"{a}_to_{b}" for a, b in audit.ABS_TRANSITIONS),
        )

    def test_annual_cache_prepares_each_numeric_column_once_and_preserves_contract(self) -> None:
        frame = self.frame()
        groups = audit._abs_year_groups(frame)
        with mock.patch.object(
            audit,
            "_abs_numeric_values",
            wraps=audit._abs_numeric_values,
        ) as numeric_values, mock.patch.object(
            audit,
            "_abs_category_distribution",
            wraps=audit._abs_category_distribution,
        ) as category_distribution, mock.patch.object(
            audit,
            "_abs_entity_values",
            wraps=audit._abs_entity_values,
        ) as entity_values:
            diagnostics = audit.abs_boundary_diagnostics(frame)
        self.assertEqual(
            numeric_values.call_count,
            len(groups) * len(audit.AUDIT_NUMERIC_COLUMNS),
        )
        self.assertEqual(
            category_distribution.call_count,
            len(groups) * len(audit.ABS_CATEGORICAL_COLUMNS),
        )
        self.assertEqual(
            entity_values.call_count,
            len(groups) * len(audit.AUDIT_COVERAGE_COLUMNS),
        )

        summaries = audit._abs_prepare_annual_numeric_summaries(groups)
        self.assertEqual(set(summaries), set(range(2019, 2025)))
        inning_2019 = summaries[2019]["inning"]
        self.assertEqual(inning_2019["count"], 1)
        self.assertEqual(inning_2019["mean"], 2019.0)
        self.assertEqual(inning_2019["variance"], 0.0)
        self.assertEqual(inning_2019["quantiles"]["0.50"], 2019.0)

        numeric_transitions = diagnostics["numeric_smd"]["inning"]["transitions"]
        self.assertTrue(all(
            result == {
                "value": None,
                "reason": "degenerate_pooled_variance_unequal_means",
            }
            for result in numeric_transitions.values()
        ))
        quantile_boundary = diagnostics["quantile_change"]["inning"][
            "transitions"
        ]["2023_to_2024"]
        self.assertEqual(
            [quantile_boundary["quantiles"][key]["value"]
             for key in ("0.10", "0.50", "0.90")],
            [1.0, 1.0, 1.0],
        )
        transition_keys = [f"{a}_to_{b}" for a, b in audit.ABS_TRANSITIONS]
        expected_tv = {
            key: {"value": 1.0, "reason": None} for key in transition_keys
        }
        self.assertEqual(
            diagnostics["categorical_total_variation"]["game_month"]["transitions"],
            expected_tv,
        )
        expected_missing = {
            key: {"value": 0.0, "reason": None} for key in transition_keys
        }
        self.assertEqual(
            diagnostics["missing_rate_delta_pp"]["inning"]["transitions"],
            expected_missing,
        )
        expected_entity = {
            key: {
                "prior_unique_count": 1,
                "later_unique_count": 1,
                "unique_count_delta": 0,
                "relative_unique_count_change": {"value": 0.0, "reason": None},
            }
            for key in transition_keys
        }
        self.assertEqual(
            diagnostics["entity_coverage_change"]["pitcher_id"]["transitions"],
            expected_entity,
        )


class PipelineGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = []
        for year in range(2019, 2024):
            for index in range(3):
                self.records.append(good_record(f"R-{year}-{index}", year, "R"))
            self.records.append(good_record(f"F-{year}", year, "F"))

    def test_geometry_is_recorded_per_actual_pipeline_role(self) -> None:
        with mock.patch.object(audit, "BOUNDED_ROWS", 2):
            selections = audit.build_pipeline_geometries(self.records)
        by_pipeline: dict[str, list[audit.MaskSelection]] = {}
        for selection in selections:
            by_pipeline.setdefault(selection.pipeline, []).append(selection)

        self.assertEqual(
            {(item.origin, item.mask_role) for item in by_pipeline["recovery_catboost"]},
            {
                (origin, role)
                for origin in ("r2022", "r2023")
                for role in ("inner_train", "inner_validation", "outer_train", "outer_validation")
            },
        )
        self.assertEqual(
            {(item.origin, item.mask_role) for item in by_pipeline["baseline_reproduction"]},
            {
                ("r2022", "pretrained_outer_validation"),
                ("r2023", "pretrained_outer_validation"),
            },
        )
        self.assertNotIn(
            "outer_train",
            {item.mask_role for item in by_pipeline["baseline_reproduction"]},
        )
        self.assertEqual(len(by_pipeline["recovery_catboost"]), 8)
        self.assertIn(
            "base_oof_train",
            {item.mask_role for item in by_pipeline["recovery_residual"]},
        )
        self.assertIn(
            "c_selection_score_2021",
            {item.mask_role for item in by_pipeline["recovery_residual"]},
        )
        self.assertIn(
            "baseline_cache_logits",
            {item.mask_role for item in by_pipeline["recovery_calibration"]},
        )
        self.assertIn(
            "transform_fit_panel_2022",
            {item.mask_role for item in by_pipeline["recovery_calibration"]},
        )
        self.assertEqual(
            {(item.origin, item.mask_role) for item in by_pipeline["recovery_calibration"]},
            {
                (origin, role)
                for origin in ("r2022", "r2023")
                for role in ("baseline_cache_logits", "bounded_labels", "transform_apply")
            } | {("r2023", "transform_fit_panel_2022")},
        )

    def test_first_n_is_current_dataframe_order(self) -> None:
        with mock.patch.object(audit, "BOUNDED_ROWS", 2):
            selections = audit.build_pipeline_geometries(self.records)
        chosen = next(
            item for item in selections
            if item.pipeline == "recovery_catboost"
            and item.origin == "r2022" and item.mask_role == "outer_validation"
        )
        self.assertEqual(chosen.selected_positions, chosen.full_positions[:2])
        self.assertEqual(len(chosen.full_positions), 3)

    def test_bounded_profile_reports_hashes_and_coverage(self) -> None:
        with mock.patch.object(audit, "BOUNDED_ROWS", 2):
            selection = next(
                item for item in audit.build_pipeline_geometries(self.records)
                if item.pipeline == "baseline_reproduction" and item.origin == "r2022"
            )
        report = audit.compare_bounded_to_full(
            self.records,
            selection,
            categorical=("game_month", "game_type"),
            numeric=("asof_pitcher_n",),
            coverage=("pitcher_id", "batter_id", "pitcher_team_id"),
        )
        self.assertEqual(report["n_full"], 3)
        self.assertEqual(report["n_selected"], 2)
        self.assertEqual(len(report["mask_hash"]), 64)
        self.assertEqual(len(report["row_id_hash"]), 64)

    def test_bounded_profile_excludes_na_from_numeric_stats(self) -> None:
        import numpy as np
        import pandas as pd

        records = [
            {"row_id": "a", "value": np.nan},
            {"row_id": "b", "value": 2.0},
            {"row_id": "c", "value": pd.NA},
            {"row_id": "d", "value": 4.0},
        ]
        selection = audit.MaskSelection(
            pipeline="fixture",
            origin="fixture",
            mask_role="fixture",
            full_positions=(0, 1, 2, 3),
            selected_positions=(0, 1),
            truncation="fixture",
        )
        report = audit.compare_bounded_to_full(
            records,
            selection,
            categorical=(),
            numeric=("value",),
            coverage=(),
        )
        self.assertEqual(report["numeric_smd"]["value"], -1.0)
        self.assertEqual(report["missing_rate_delta_pp"]["value"], 0.0)


class PipelineAndStatusStaticTests(unittest.TestCase):
    def test_expected_pipeline_sources_exist_and_parse(self) -> None:
        repo = SCRIPT.parents[1]
        findings = audit.audit_pipeline_sources(repo)
        self.assertTrue(findings)
        self.assertTrue(all(finding.severity.value == "BENIGN" for finding in findings))
        self.assertTrue(
            all(finding.pipeline_verdict.value == "NOT_PROVEN" for finding in findings)
        )
        self.assertIn(
            "pipeline_source:baseline_reproduction",
            {finding.rule for finding in findings},
        )

    def test_project_status_sha_difference_is_benign(self) -> None:
        finding = audit.classify_project_status_sha(
            "Evidence base: `master` at `1234567`.",
            "f" * 40,
        )
        self.assertEqual(finding.severity.value, "BENIGN")
        self.assertIn("unless a separate policy/state contradiction", finding.reason)


class AbsAndTrackmanGuardTests(unittest.TestCase):
    def test_abs_feature_only_contract_is_branch_inert(self) -> None:
        contract = audit.abs_feature_only_contract()
        self.assertTrue(contract["branch_inert"])
        self.assertFalse(contract["target_column_in_projection"])
        self.assertFalse(contract["may_tune_thresholds"])
        self.assertFalse(contract["may_select_features"])
        self.assertFalse(contract["may_choose_transformations"])
        self.assertFalse(contract["may_modify_recovery_policy"])
        self.assertTrue(contract["requires_separate_experiment_brief_for_modeling"])

    def test_identifier_overlap_does_not_create_trackman_crosswalk(self) -> None:
        usage = audit.assess_trackman_usage(
            documented_shared_pitch_key=False,
            authoritative_player_crosswalk=False,
            documented_main_game_date_key=False,
            prior_season_coverage=True,
            stable_league_measurements=True,
            observed_identifier_overlap=True,
        )
        self.assertEqual(usage["A_exact_row_join"]["verdict"], "NOT_PROVEN")
        self.assertEqual(usage["B_game_date_join"]["verdict"], "NOT_PROVEN")
        self.assertEqual(usage["C_prior_season_player_team"]["verdict"], "NOT_PROVEN")
        self.assertEqual(usage["D_prior_season_league"]["verdict"], "PROVEN")

    def test_trackman_prior_season_rule_is_strict(self) -> None:
        audit.validate_trackman_prior_season((2019, 2020, 2021), prediction_year=2022)
        with self.assertRaisesRegex(audit.AuditError, "future-safe rule violated"):
            audit.validate_trackman_prior_season((2021, 2022), prediction_year=2022)

    def test_trackman_report_includes_explicit_unusable_status(self) -> None:
        usage = audit.assess_trackman_usage(
            documented_shared_pitch_key=False,
            authoritative_player_crosswalk=False,
            documented_main_game_date_key=False,
            prior_season_coverage=False,
            stable_league_measurements=False,
        )
        self.assertIn("E_unusable", usage)
        self.assertEqual(usage["E_unusable"]["verdict"], "NOT_PROVEN")


def complete_train_row(row_id: str, season: int, index: int) -> dict[str, object]:
    row: dict[str, object] = {column: 0 for column in audit.TRAIN_COLUMNS}
    for column in audit.ASOF_RATE_COLUMNS:
        row[column] = 0.5
    row.update({
        "row_id": row_id,
        "season": season,
        "game_month": 4 + index,
        "game_dayofweek": index,
        "inning": 1 + index,
        "top_bottom": "T" if index % 2 == 0 else "B",
        "game_type": "R",
        "balls_before": index % 4,
        "strikes_before": index % 3,
        "outs_before": index % 3,
        "run_top_before": index,
        "run_bot_before": 1,
        "run_total_before": index + 1,
        "score_diff_home": index - 1,
        "score_diff_pitcher_team": index - 1,
        "runner_on_1b": 1,
        "runner_on_2b": 0,
        "runner_on_3b": 0,
        "num_runners_on": 1,
        "base_state": "1__",
        "home_win_expectancy": 0.5,
        "away_win_expectancy": 0.5,
        "li": 1.0,
        "pitcher_id": 1000 + season * 10 + index,
        "batter_id": 2000 + season * 10 + index,
        "pitcher_hand": index % 2,
        "batter_hand": (index + 1) % 2,
        "pitcher_team_id": 10 + index,
        "batter_team_id": 20 + index,
        "asof_pitcher_n": season - 2018 + index,
        "asof_batter_n": season - 2018 + index,
        "asof_pitcher_pitchmix_n": season - 2018 + index,
        "asof_pitcher_fastball_rate": 0.5,
        "asof_pitcher_breaking_rate": 0.3,
        "asof_pitcher_offspeed_rate": 0.2,
        "control_success": (season + index) % 2,
    })
    return row


def complete_trackman_row(season: int, index: int) -> dict[str, object]:
    row: dict[str, object] = {column: 0 for column in audit.TRACKMAN_COLUMNS}
    row.update({
        "trackman_id": f"tm-{season}-{index}",
        "season": season,
        "game_date": f"{season}-05-{index + 1:02d}",
        "game_month": 5,
        "game_dayofweek": index,
        "trackman_game_id": f"game-{season}-{index}",
        "pitch_no": index + 1,
        "inning": 1,
        "top_bottom": "T",
        "pitcher_trackman_id": 1000 + season * 10 + index,
        "batter_trackman_id": 2000 + season * 10 + index,
        "pitcher_hand": "R",
        "batter_hand": "L",
        "pitcher_team": f"P{index}",
        "batter_team": f"B{index}",
        "tagged_pitch_type": "Fastball",
        "auto_pitch_type": "Fastball",
        "pitch_type_group": "Fastball",
        "rel_speed": 140.0 + index,
        "spin_rate": 2200.0 + index,
        "induced_vert_break": 15.0,
        "horz_break": 5.0,
        "extension": 6.0,
        "rel_height": 5.5,
        "rel_side": 1.5,
        "zone_speed": 130.0,
    })
    return row


class TrackmanDateParserTests(unittest.TestCase):
    def _audit_frame(self, rows: list[dict[str, object]]):
        import pandas as pd

        trackman = pd.DataFrame(rows, columns=audit.TRACKMAN_COLUMNS)
        main = pd.DataFrame({"pitcher_id": [], "batter_id": []})
        return audit.audit_trackman_frame(trackman, main)

    def test_mixed_valid_formats_are_order_independent(self) -> None:
        import pandas as pd

        for values, expected in (
            (
                ["05/10/2019", "2022-05-11"],
                [pd.Timestamp("2019-05-10"), pd.Timestamp("2022-05-11")],
            ),
            (
                ["2022-05-11", "05/10/2019"],
                [pd.Timestamp("2022-05-11"), pd.Timestamp("2019-05-10")],
            ),
        ):
            with self.subTest(values=values):
                parsed = audit.parse_trackman_game_dates(pd.Series(values))
                self.assertEqual(parsed.tolist(), expected)

    def test_variable_width_slash_dates_parse(self) -> None:
        import pandas as pd

        parsed = audit.parse_trackman_game_dates(
            pd.Series(["5/1/2021", "05/1/2021", "5/01/2021", "05/01/2021"])
        )
        self.assertEqual(parsed.dt.strftime("%Y-%m-%d").tolist(), ["2021-05-01"] * 4)

    def test_impossible_garbage_blank_and_null_dates_remain_invalid(self) -> None:
        import pandas as pd

        values = pd.Series([
            "02/29/2019", "2022-02-30", "not-a-date", "", None,
            " 05/10/2021", "5/1/21",
        ])
        parsed = audit.parse_trackman_game_dates(values)
        self.assertTrue(parsed.isna().all())

    def test_valid_leap_date_parses(self) -> None:
        import pandas as pd

        parsed = audit.parse_trackman_game_dates(pd.Series(["02/29/2020"]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2020-02-29"))

    def test_wrong_year_is_mismatch_not_invalid(self) -> None:
        row = complete_trackman_row(2021, 0)
        row["game_date"] = "2022-05-01"
        report, structural, _ = self._audit_frame([row])
        self.assertEqual(report["invalid_game_dates"], 0)
        self.assertEqual(report["game_date_season_mismatches"], 1)
        self.assertEqual(structural.severity.value, "LIKELY_ISSUE")

    def test_audit_structural_finding_is_benign_for_mixed_valid_formats(self) -> None:
        first = complete_trackman_row(2019, 0)
        second = complete_trackman_row(2022, 1)
        first["game_date"] = "05/01/2019"
        second["game_date"] = "2022-05-02"
        report, structural, _ = self._audit_frame([first, second])
        self.assertEqual(report["invalid_game_dates"], 0)
        self.assertEqual(report["game_date_season_mismatches"], 0)
        self.assertEqual(structural.severity.value, "BENIGN")

    def test_audit_structural_finding_flags_impossible_date(self) -> None:
        row = complete_trackman_row(2021, 0)
        row["game_date"] = "02/29/2019"
        report, structural, _ = self._audit_frame([row])
        self.assertEqual(report["invalid_game_dates"], 1)
        self.assertEqual(report["game_date_season_mismatches"], 0)
        self.assertEqual(structural.severity.value, "LIKELY_ISSUE")


class MediumSyntheticOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="aimers9-medium-synthetic-")
        self.root = Path(self.temp.name)
        self.train = self.root / "train.csv"
        self.trackman = self.root / "trackman_history.csv"
        interleaved_years = (2019, 2024, 2020, 2021, 2022, 2023)
        train_rows = [
            complete_train_row(
                "SENTINEL-PRIVATE-ROW-ID" if year == 2019 and index == 0
                else f"row-{year}-{index}",
                year,
                index,
            )
            for year in interleaved_years
            for index in range(2)
        ]
        for row in train_rows:
            if row["season"] == 2024:
                row["control_success"] = "FORBIDDEN-2024-TARGET"
        with self.train.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit.TRAIN_COLUMNS)
            writer.writeheader()
            writer.writerows(train_rows)
        with self.trackman.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit.TRACKMAN_COLUMNS)
            writer.writeheader()
            writer.writerows(
                complete_trackman_row(year, index)
                for year in interleaved_years
                for index in range(2)
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_medium_orchestration_schema_privacy_and_firewall(self) -> None:
        output = self.root / "main-output"
        report, paths = audit.run_medium_audit(
            self.train, self.trackman, output, SCRIPT.parents[1]
        )
        self.assertEqual(report["schema_version"], audit.REPORT_SCHEMA_VERSION)
        self.assertEqual([path.name for path in paths], ["audit_report.json", "audit_report.md"])
        self.assertFalse(report["scope"]["target_2024_access"])
        self.assertEqual(
            report["label_access_ledger"][1]["seasons"], [2019, 2020, 2021]
        )
        self.assertEqual(
            {item["origin"] for item in report["label_access_ledger"] if "origin" in item},
            {"r2022", "r2023"},
        )
        serialized = paths[0].read_text(encoding="utf-8") + paths[1].read_text(encoding="utf-8")
        self.assertNotIn("SENTINEL-PRIVATE-ROW-ID", serialized)
        self.assertNotIn("FORBIDDEN-2024-TARGET", serialized)
        self.assertNotIn('"row_ids"', serialized)
        self.assertIn("first_30000_by_pipeline_geometry", report["sections"])
        self.assertIn("E_unusable", report["sections"]["trackman_usability"]["safe_usage_levels"])
        self.assertNotIn(
            "excluded_out_of_scope_rows", report["sections"]["trackman_usability"]
        )
        self.assertEqual(
            report["sources"]["train"]["hash_kind"],
            "canonical_projected_2019_2023_features_only_sha256",
        )
        self.assertTrue(
            report["sources"]["train"]["target_row_id_alignment"]["ordered_exact_match"]
        )
        self.assertEqual(
            len(report["sources"]["train"]["target_row_id_alignment"]["row_id_hash"]),
            64,
        )
        preprocessing = report["sections"]["preprocessing_provenance"]
        expected_contract_fields = {
            "train_mask_fit_scope", "validation_mask_scope", "imputation_fit_scope",
            "scaling_fit_scope", "categorical_vocabulary", "unseen_category_handling",
            "missing_categorical_behavior", "target_access",
            "validation_only_category_exposure", "whole_frame_category_metadata",
            "feather_cache_behavior", "material_learner_influence",
        }
        for pipeline in (
            "common_legacy_lgb", "mlp", "recovery_catboost", "recovery_residual",
            "recovery_calibration", "baseline_reproduction", "official_baseline",
        ):
            self.assertEqual(set(preprocessing[pipeline]["contract"]), expected_contract_fields)
            self.assertEqual(
                preprocessing[pipeline]["source_presence"]["pipeline_verdict"],
                "NOT_PROVEN",
            )

    def test_interleaved_2024_train_features_are_not_materialized(self) -> None:
        import pandas as pd

        real_read_csv = pd.read_csv
        projected_seasons: list[int] = []

        def spy(*args, **kwargs):
            result = real_read_csv(*args, **kwargs)
            if kwargs.get("usecols") == list(audit.TRAIN_FEATURE_COLUMNS):
                projected_seasons.extend(result["season"].astype(int).tolist())
            return result

        with mock.patch.object(pd, "read_csv", side_effect=spy):
            frame = audit.read_scoped_train_2019_2023(self.train)
        self.assertNotIn(2024, projected_seasons)
        self.assertNotIn(2024, frame["season"].astype(int).tolist())
        self.assertTrue(any(frame["__source_position"].diff().dropna() > 1))

    def test_interleaved_2024_trackman_rows_are_not_materialized(self) -> None:
        import pandas as pd

        real_read_csv = pd.read_csv
        projected_seasons: list[int] = []

        def spy(*args, **kwargs):
            result = real_read_csv(*args, **kwargs)
            if kwargs.get("usecols") == list(audit.TRACKMAN_COLUMNS):
                projected_seasons.extend(result["season"].astype(int).tolist())
            return result

        with mock.patch.object(pd, "read_csv", side_effect=spy):
            frame = audit.read_scoped_trackman_2019_2023(self.trackman)
        self.assertNotIn(2024, projected_seasons)
        self.assertNotIn(2024, frame["season"].astype(int).tolist())
        self.assertTrue(any(frame["__source_position"].diff().dropna() > 1))

    def test_scoped_trackman_date_reader_does_not_materialize_2024_dates(self) -> None:
        import pandas as pd

        real_read_csv = pd.read_csv
        projected_dates: list[str] = []
        projected_seasons: list[int] = []

        def spy(*args, **kwargs):
            result = real_read_csv(*args, **kwargs)
            if kwargs.get("usecols") == list(audit.TRACKMAN_DATE_COLUMNS):
                projected_dates.extend(result["game_date"].astype(str).tolist())
                projected_seasons.extend(result["season"].astype(int).tolist())
            return result

        with mock.patch.object(pd, "read_csv", side_effect=spy):
            frame = audit.read_scoped_trackman_game_date_columns_2019_2023(
                self.trackman
            )
        self.assertNotIn(2024, projected_seasons)
        self.assertNotIn(2024, frame["season"].astype(int).tolist())
        self.assertTrue(projected_dates)
        self.assertTrue(all(not value.startswith("2024-") for value in projected_dates))
        self.assertEqual(tuple(frame.columns), (*audit.TRACKMAN_DATE_COLUMNS, "__source_position"))
        self.assertTrue(any(frame["__source_position"].diff().dropna() > 1))

    def test_target_row_id_or_order_mismatch_fails_closed(self) -> None:
        import pandas as pd

        features = pd.DataFrame({"row_id": ["a", "b"], "season": [2019, 2020]})
        targets = pd.DataFrame({"row_id": ["b", "a"], "control_success": [1, 0]})
        with self.assertRaisesRegex(audit.AuditError, "ROW_ID order/hash"):
            audit._attach_aligned_targets(features, targets)

    def test_historical_behavior_is_distinct_from_current_policy_admissibility(self) -> None:
        frame = audit.read_scoped_train_2019_2023(self.train)
        provenance = audit.preprocessing_provenance_report(SCRIPT.parents[1], frame)
        for pipeline in ("mlp", "official_baseline"):
            self.assertEqual(
                provenance[pipeline]["historical_pipeline_behavior"]["verdict"],
                "PROVEN",
            )
            self.assertEqual(
                provenance[pipeline]["current_recovery_policy_admissibility"]["verdict"],
                "VIOLATION",
            )

    def test_trackman_composite_duplicate_remains_unknown(self) -> None:
        trackman = audit.read_scoped_trackman_2019_2023(self.trackman)
        train = audit.read_scoped_train_2019_2023(self.train)
        trackman.loc[1, "trackman_game_id"] = trackman.loc[0, "trackman_game_id"]
        trackman.loc[1, "pitch_no"] = trackman.loc[0, "pitch_no"]
        report, structural, composite = audit.audit_trackman_frame(trackman, train)
        self.assertEqual(report["game_pitch_composite_key_verdict"], "UNKNOWN")
        self.assertEqual(composite.severity.value, "UNKNOWN")
        self.assertEqual(structural.severity.value, "BENIGN")

    def test_vectorized_geometry_matches_reviewed_geometry(self) -> None:
        frame = audit.read_scoped_train_2019_2023(self.train)
        records = frame[["row_id", "season", "game_type"]].to_dict("records")
        reviewed = audit.build_pipeline_geometries(records)
        vectorized = audit.build_pipeline_geometries_frame(frame)
        left = {
            (item.pipeline, item.origin, item.mask_role): (
                item.full_positions, item.selected_positions
            )
            for item in reviewed
        }
        right = {
            (item.pipeline, item.origin, item.mask_role): (
                tuple(item.full_positions), tuple(item.selected_positions)
            )
            for item in vectorized
        }
        self.assertEqual(left, right)

    def test_synthetic_medium_report_hash_is_deterministic(self) -> None:
        first, _ = audit.run_medium_audit(
            self.train, self.trackman, self.root / "first", SCRIPT.parents[1]
        )
        second, _ = audit.run_medium_audit(
            self.train, self.trackman, self.root / "second", SCRIPT.parents[1]
        )
        self.assertEqual(
            first["canonical_report_sha256"], second["canonical_report_sha256"]
        )
        with_runtime = dict(first, runtime={"seconds": 999}, generated_at="never")
        self.assertEqual(
            audit.canonical_report_hash(first), audit.canonical_report_hash(with_runtime)
        )

    def test_synthetic_abs_report_hash_is_deterministic(self) -> None:
        first, _ = audit.run_abs_2024_feature_audit(
            self.train, self.root / "abs-first", SCRIPT.parents[1]
        )
        second, _ = audit.run_abs_2024_feature_audit(
            self.train, self.root / "abs-second", SCRIPT.parents[1]
        )
        self.assertEqual(
            first["canonical_report_sha256"], second["canonical_report_sha256"]
        )

    def test_separate_abs_feature_only_runner_is_branch_inert(self) -> None:
        report, paths = audit.run_abs_2024_feature_audit(
            self.train, self.root / "abs-output", SCRIPT.parents[1]
        )
        self.assertEqual(report["label_access_ledger"], [])
        self.assertEqual(
            report["abs_metric_contract_version"],
            audit.ABS_METRIC_CONTRACT_VERSION,
        )
        self.assertTrue(report["scope"]["branch_inert"])
        self.assertFalse(report["scope"]["target_access"])
        self.assertFalse(report["scope"]["test_distribution_access"])
        self.assertFalse(report["scope"]["public_leaderboard_evidence"])
        self.assertFalse(report["scope"]["external_information_access"])
        self.assertFalse(report["scope"]["model_training_or_scoring"])
        self.assertFalse(report["scope"]["may_select_features"])
        self.assertIn(
            "boundary_diagnostics_all_prespecified_features", report["sections"]
        )
        self.assertNotIn("selected_features", report)
        self.assertNotIn("adopted_features", report)
        self.assertNotIn("policy_action", report)
        forbidden_adoption_keys = {
            "selected_features", "adopted_features", "chosen_feature_set",
            "selected_model", "adopted_model", "model_selection",
            "recovery_policy_update", "threshold_update", "calibration_update",
        }

        def assert_no_adoption_keys(value: object) -> None:
            if isinstance(value, dict):
                self.assertTrue(forbidden_adoption_keys.isdisjoint(value))
                for item in value.values():
                    assert_no_adoption_keys(item)
            elif isinstance(value, list):
                for item in value:
                    assert_no_adoption_keys(item)

        assert_no_adoption_keys(report)
        serialized = paths[0].read_text(encoding="utf-8")
        self.assertNotIn("FORBIDDEN-2024-TARGET", serialized)
        self.assertNotIn("SENTINEL-PRIVATE-ROW-ID", serialized)

    def test_medium_cli_smoke_uses_only_synthetic_inputs(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = audit.main([
                "medium",
                "--train-csv", str(self.train),
                "--trackman-csv", str(self.trackman),
                "--output-dir", str(self.root / "cli-output"),
                "--repo-root", str(SCRIPT.parents[1]),
            ])
        payload = json.loads(output.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["mode"], "MEDIUM_AGGREGATE_AUDIT")

    def test_output_inside_repository_fails_before_data_access(self) -> None:
        with self.assertRaisesRegex(audit.AuditError, "outside Git repository"):
            audit.run_medium_audit(
                self.root / "does-not-need-to-exist.csv",
                self.root / "does-not-need-to-exist-trackman.csv",
                SCRIPT.parents[1] / "forbidden-audit-output",
                SCRIPT.parents[1],
            )


class AbsDeterminismTests(unittest.TestCase):
    def test_boundary_diagnostics_ignore_irrelevant_column_insertion_order(self) -> None:
        base = AbsMetricContractTests.frame()
        with_extra = base.copy()
        with_extra.insert(0, "irrelevant_extra", list(range(len(with_extra))))
        reordered = with_extra.loc[:, list(reversed(with_extra.columns))]
        self.assertEqual(
            audit.abs_boundary_diagnostics(base),
            audit.abs_boundary_diagnostics(reordered),
        )

    def test_abs_boundary_payload_is_hash_seed_invariant_in_subprocesses(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            "import pandas as pd\n"
            "import audit_data_integrity_temporal as audit\n"
            "frame = pd.DataFrame({\n"
            "  'season': [2019, 2020, 2021, 2022, 2023, 2024],\n"
            "  'inning': [1, 2, 3, 4, 5, 6],\n"
            "  'game_month': ['May', 'June', 'May', 'June', 'May', 'June'],\n"
            "  'pitcher_id': ['p19', 'p20', 'p21', 'p22', 'p23', 'p24'],\n"
            "  'batter_id': ['b19', 'b20', 'b21', 'b22', 'b23', 'b24'],\n"
            "  'pitcher_team_id': ['t19', 't20', 't21', 't22', 't23', 't24'],\n"
            "  'batter_team_id': ['u19', 'u20', 'u21', 'u22', 'u23', 'u24'],\n"
            "})\n"
            "print(audit.canonical_hash(audit.abs_boundary_diagnostics(frame)))\n"
        )
        outputs = []
        for seed in ("1", "2", "3", "4"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs.append(completed.stdout.strip())
        self.assertTrue(outputs[0])
        self.assertEqual(len(set(outputs)), 1)


class BoundedGeometryTests(unittest.TestCase):
    @staticmethod
    def frame(n_per_origin: int = 40):
        import pandas as pd

        rows = []
        for season in (2022, 2023):
            for index in range(n_per_origin):
                row = {
                    column: 0 for column in audit.BOUNDED_GEOMETRY_PROJECTION_COLUMNS
                }
                row.update({
                    "row_id": f"{season}-{index}",
                    "season": season,
                    "game_month": (index % 4) + 1,
                    "game_dayofweek": index % 7,
                    "top_bottom": "T" if index % 2 else "B",
                    "game_type": "R",
                    "balls_before": index % 4,
                    "strikes_before": index % 3,
                    "outs_before": index % 3,
                    "base_state": ("___", "1__", "_2_")[index % 3],
                    "pitcher_hand": "R" if index % 2 else "L",
                    "batter_hand": "L" if index % 2 else "R",
                    "pitcher_team_id": index % 5,
                    "batter_team_id": index % 7,
                    "pitcher_id": f"p-{index % 13}",
                    "batter_id": f"b-{index % 17}",
                    "inning": float(index % 9),
                    "run_total_before": float(index % 6),
                    "score_diff_pitcher_team": float((index % 5) - 2),
                    "home_win_expectancy": index / max(1, n_per_origin),
                    "li": float(index % 8) / 8.0,
                    "asof_pitcher_n": index,
                    "asof_batter_n": index + 1,
                    "asof_pitcher_success_rate": 0.2 + (index % 5) / 10.0,
                    "asof_batter_success_rate": 0.3 + (index % 4) / 10.0,
                    "asof_pitcher_pitchmix_n": index + 2,
                    "asof_pitcher_fastball_rate": 0.5,
                    "asof_pitcher_breaking_rate": 0.2,
                    "asof_pitcher_offspeed_rate": 0.3,
                })
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame["__source_position"] = list(range(len(frame)))
        return frame

    def test_contract_and_outer_scope_are_frozen(self) -> None:
        contract = audit._bounded_geometry_contract()
        self.assertEqual(
            contract["version"], "aimers9-bounded-validation-geometry-v1"
        )
        self.assertEqual(
            contract["candidate_a"]["stratum_columns"],
            ["game_month", "count_state"],
        )
        self.assertIn("integer 1..12", contract["candidate_a"]["game_month"])
        self.assertIn("divmod", contract["candidate_a"]["quota_rule"])
        self.assertTrue(contract["candidate_a"]["promotion_eligible"])
        self.assertFalse(contract["candidate_b"]["promotion_eligible"])
        self.assertFalse(contract["candidate_c"]["promotion_eligible"])
        self.assertNotIn(audit.TARGET, audit.BOUNDED_GEOMETRY_PROJECTION_COLUMNS)

    def test_largest_remainder_floor_and_deterministic_ties(self) -> None:
        strata = {
            ("a",): tuple(range(3)),
            ("b",): tuple(range(3, 6)),
            ("c",): tuple(range(6, 8)),
        }
        quotas = audit._geometry_allocate_quotas(strata, 4)
        self.assertEqual(quotas, {("a",): 2, ("b",): 1, ("c",): 1})
        reordered = dict(reversed(tuple(strata.items())))
        self.assertEqual(quotas, audit._geometry_allocate_quotas(reordered, 4))

    def test_quota_capacity_redistribution_and_empty_strata(self) -> None:
        strata = {
            ("small",): (0,),
            ("large",): tuple(range(1, 11)),
        }
        quotas = audit._geometry_allocate_quotas(strata, 8)
        self.assertEqual(quotas[("small",)], 1)
        self.assertEqual(quotas[("large",)], 7)
        self.assertEqual(audit._geometry_allocate_quotas({}, 0), {})

    def test_missing_and_invalid_candidate_a_categories_share_missing_token(self) -> None:
        self.assertEqual(audit._geometry_stratum_component("game_month", None), "__MISSING__")
        self.assertEqual(audit._geometry_stratum_component("game_month", 13), "__MISSING__")
        self.assertEqual(audit._geometry_stratum_component("balls_before", -1), "__MISSING__")
        self.assertEqual(audit._geometry_stratum_component("strikes_before", 3), "__MISSING__")
        self.assertEqual(audit._geometry_stratum_component("balls_before", 2.0), "2")

    def test_within_stratum_spread_edge_cases_are_exact_and_unique(self) -> None:
        positions = tuple(range(10))
        self.assertEqual(audit._geometry_spread_positions(positions, 0), ())
        self.assertEqual(audit._geometry_spread_positions(positions, 1), (5,))
        self.assertEqual(audit._geometry_spread_positions(positions, 10), positions)
        selected = audit._geometry_spread_positions(positions, 4)
        self.assertEqual(selected, (1, 3, 6, 8))
        self.assertEqual(len(selected), len(set(selected)))

    def test_all_candidates_use_exact_budget_and_subset(self) -> None:
        frame = self.frame()
        with mock.patch.object(audit, "BOUNDED_ROWS", 7):
            full = audit._geometry_origin_positions(frame, "r2022")
            for candidate in audit.BOUNDED_GEOMETRY_CANDIDATES:
                selection = audit.select_bounded_geometry(frame, "r2022", candidate)
                self.assertEqual(selection.full_positions, full)
                self.assertEqual(len(selection.selected_positions), 7)
                self.assertEqual(len(set(selection.selected_positions)), 7)
                self.assertTrue(set(selection.selected_positions) <= set(full))

    def test_candidate_a_same_input_is_deterministic_and_strata_are_cached(self) -> None:
        frame = self.frame(100)
        with mock.patch.object(audit, "BOUNDED_ROWS", 17), mock.patch.object(
            audit, "_geometry_column_values", wraps=audit._geometry_column_values
        ) as column_values:
            first = audit.select_bounded_geometry(frame, "r2022", "candidate_a")
            second = audit.select_bounded_geometry(frame, "r2022", "candidate_a")
        self.assertEqual(first, second)
        # One precomputed array per stratum component for each selection.
        self.assertEqual(column_values.call_count, 10)

    def test_target_variants_cannot_change_geometry_selection(self) -> None:
        first = self.frame(40)
        second = first.copy()
        first[audit.TARGET] = [0] * len(first)
        second[audit.TARGET] = [1] * len(second)
        with mock.patch.object(audit, "BOUNDED_ROWS", 17):
            selected_first = audit.select_bounded_geometry(first, "r2022", "candidate_a")
            selected_second = audit.select_bounded_geometry(second, "r2022", "candidate_a")
        self.assertEqual(selected_first, selected_second)

    def test_small_panel_uses_minimum_budget(self) -> None:
        frame = self.frame(3)
        with mock.patch.object(audit, "BOUNDED_ROWS", 30_000):
            selection = audit.select_bounded_geometry(frame, "r2023", "candidate_a")
        self.assertEqual(len(selection.selected_positions), 3)
        self.assertEqual(selection.parameters["budget"], 3)

    def test_candidate_c_fails_closed_on_missing_or_duplicate_row_id(self) -> None:
        frame = self.frame(5)
        frame.loc[0, "row_id"] = None
        with self.assertRaisesRegex(audit.AuditError, "nonmissing row_id"):
            audit.select_bounded_geometry(frame, "r2022", "candidate_c")
        frame = self.frame(5)
        frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
        with self.assertRaisesRegex(audit.AuditError, "unique row_id"):
            audit.select_bounded_geometry(frame, "r2022", "candidate_c")

    def test_candidate_c_hash_namespace_and_hash_seed_are_deterministic(self) -> None:
        frame = self.frame()
        first = audit.select_bounded_geometry(frame, "r2022", "candidate_c")
        reordered = frame.loc[list(reversed(frame.index))].reset_index(drop=True)
        reordered["__source_position"] = list(range(len(reordered)))
        second = audit.select_bounded_geometry(reordered, "r2022", "candidate_c")
        self.assertEqual(first.parameters["namespace"], audit.BOUNDED_GEOMETRY_HASH_NAMESPACE)
        self.assertEqual(
            first.parameters["hash_payload"],
            "UTF8(namespace) + b'\\x00' + UTF8(str(row_id))",
        )
        contract = audit.bounded_geometry_contract()["candidate_c"]
        self.assertEqual(contract["namespace"], audit.BOUNDED_GEOMETRY_HASH_NAMESPACE)
        self.assertEqual(contract["hash_payload"], first.parameters["hash_payload"])
        self.assertEqual(contract["row_id_normalization"], "UTF8(str(value)), no trimming")
        self.assertEqual(contract["sort_tie_break"], "digest, normalized_row_id, source_position")
        self.assertEqual(contract["seed_policy"], "no seeds or alternate namespaces")
        expected_digest = hashlib.sha256(
            audit.BOUNDED_GEOMETRY_HASH_NAMESPACE.encode("utf-8")
            + b"\x00" + b"row-1"
        ).hexdigest()
        self.assertEqual(audit._geometry_candidate_c_digest("row-1"), expected_digest)
        self.assertNotEqual(first.selected_positions, second.selected_positions)

    def test_candidate_c_is_hash_seed_invariant_in_subprocesses(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            "import pandas as pd\n"
            "import audit_data_integrity_temporal as audit\n"
            "audit.BOUNDED_ROWS = 7\n"
            "frame = pd.DataFrame({\n"
            " 'row_id': [f'row-{i}' for i in range(20)],\n"
            " 'season': [2022]*20, 'game_type': ['R']*20,\n"
            "})\n"
            "selected = audit.select_bounded_geometry(frame, 'r2022', 'candidate_c').selected_positions\n"
            "print(audit.canonical_hash(selected))\n"
        )
        outputs = []
        for seed in ("1", "2", "3"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True,
                text=True, env=environment,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(len(set(outputs)), 1)

    def test_source_order_changes_rank_selection_and_provenance_not_chronology(self) -> None:
        frame = self.frame()
        with mock.patch.object(audit, "BOUNDED_ROWS", 7):
            first = audit.select_bounded_geometry(frame, "r2022", "candidate_b")
            reversed_frame = frame.iloc[::-1].reset_index(drop=True)
            reversed_frame["__source_position"] = list(range(len(reversed_frame)))
            second = audit.select_bounded_geometry(reversed_frame, "r2022", "candidate_b")
        self.assertNotEqual(first.selected_positions, second.selected_positions)
        self.assertTrue(audit._bounded_geometry_contract()["source_rank_not_chronology"])

    def test_metrics_have_all_families_and_source_bins(self) -> None:
        frame = self.frame()
        with mock.patch.object(audit, "BOUNDED_ROWS", 7):
            selection = audit.select_bounded_geometry(frame, "r2022", "candidate_a")
        report = audit.compare_bounded_geometry(frame, selection)
        self.assertEqual(
            set(report), {
                "origin", "geometry", "n_full", "n_selected", "budget",
                "parameters", "selected_position_hash", "selected_row_id_hash",
                "source_position_min", "source_position_max",
                "categorical_total_variation", "numeric_smd", "quantile_discrepancy",
                "missing_rate_delta_pp", "entity_coverage", "source_rank_coverage",
                "status",
            },
        )
        self.assertEqual(
            set(report["categorical_total_variation"]),
            set(audit.BOUNDED_GEOMETRY_CATEGORICAL_COLUMNS),
        )
        self.assertEqual(set(report["numeric_smd"]), set(audit.AUDIT_NUMERIC_COLUMNS))
        self.assertEqual(set(report["entity_coverage"]), set(audit.AUDIT_COVERAGE_COLUMNS))
        self.assertEqual(report["source_rank_coverage"]["bin_count"], 20)

    def test_geometry_distribution_fetches_column_once(self) -> None:
        frame = self.frame()
        positions = (0, 1, 2, 3, 4)
        with mock.patch.object(
            audit, "_geometry_column_values", wraps=audit._geometry_column_values
        ) as column_values:
            observed = audit._geometry_distribution(frame, "game_month", positions)
        self.assertEqual(column_values.call_count, 1)
        self.assertEqual(
            observed,
            {"1": 0.4, "2": 0.2, "3": 0.2, "4": 0.2},
        )

    def test_numeric_missing_and_zero_scale_cases_fail_closed(self) -> None:
        frame = self.frame(8)
        frame["inning"] = frame["inning"].astype(object)
        frame["inning"] = 1.0
        frame["inning"] = frame["inning"].astype(object)
        frame.loc[0, "inning"] = "bad"
        with mock.patch.object(audit, "BOUNDED_ROWS", 3):
            selection = audit.select_bounded_geometry(frame, "r2022", "candidate_a")
        report = audit.compare_bounded_geometry(frame, selection)
        self.assertEqual(report["numeric_smd"]["inning"]["value"], 0.0)
        self.assertGreater(report["missing_rate_delta_pp"]["inning"]["value"], 0.0)
        frame["inning"] = "bad"
        report = audit.compare_bounded_geometry(frame, selection)
        self.assertIsNone(report["numeric_smd"]["inning"]["value"])
        self.assertEqual(report["missing_rate_delta_pp"]["inning"]["value"], 0.0)
        self.assertEqual(
            report["numeric_smd"]["inning"]["reason"],
            "no_valid_numeric_values_in_full_panel",
        )

    def test_selected_all_invalid_keeps_finite_missing_rate_delta(self) -> None:
        frame = self.frame(4)
        frame["inning"] = frame["inning"].astype(object)
        frame["inning"] = 1.0
        frame["inning"] = frame["inning"].astype(object)
        frame.loc[0, "inning"] = "bad"
        selection = audit.BoundedGeometrySelection(
            origin="r2022",
            geometry="synthetic_selected_invalid",
            full_positions=(0, 1, 2, 3),
            selected_positions=(0,),
            parameters={},
        )
        report = audit.compare_bounded_geometry(frame, selection)
        self.assertIsNone(report["numeric_smd"]["inning"]["value"])
        self.assertEqual(
            report["numeric_smd"]["inning"]["reason"],
            "no_valid_numeric_values_in_bounded_panel",
        )
        self.assertTrue(all(
            item["value"] is None
            for item in report["quantile_discrepancy"]["inning"]["quantiles"].values()
        ))
        self.assertEqual(report["missing_rate_delta_pp"]["inning"]["value"], 75.0)

    def test_gate_is_conjunctive_and_b_c_cannot_rescue_primary(self) -> None:
        base = {
            "categorical_total_variation": {
                "game_month": {"value": 0.4, "reason": None},
            },
            "numeric_smd": {"x": {"value": 0.2, "reason": None}},
            "quantile_discrepancy": {
                "x": {"quantiles": {
                    "0.10": {"standardized_abs_delta": 0.2},
                    "0.50": {"standardized_abs_delta": 0.2},
                    "0.90": {"standardized_abs_delta": 0.2},
                }},
            },
            "missing_rate_delta_pp": {"x": {"value": 0.2, "reason": None}},
            "entity_coverage": {"x": {"shortfall": {"value": 0.2}}},
            "source_rank_coverage": {
                "source_bin_tv": {"value": 0.4, "reason": None},
                "all_supported_bins_occupied": True,
            },
            "status": "OK",
        }
        primary = json.loads(json.dumps(base))
        primary["categorical_total_variation"]["game_month"]["value"] = 0.1
        primary["source_rank_coverage"]["source_bin_tv"]["value"] = 0.1
        for family in ("numeric_smd", "missing_rate_delta_pp"):
            primary[family]["x"]["value"] = 0.1
        for item in primary["quantile_discrepancy"]["x"]["quantiles"].values():
            item["standardized_abs_delta"] = 0.1
        primary["entity_coverage"]["x"]["shortfall"]["value"] = 0.1
        reports = {
            "r2022": {"current_first_30k": base, "candidate_a": primary,
                      "candidate_b": base, "candidate_c": base},
            "r2023": {"current_first_30k": base, "candidate_a": primary,
                      "candidate_b": base, "candidate_c": base},
        }
        gate = audit.evaluate_bounded_geometry_gate(reports)
        self.assertEqual(gate["result"], "PRIMARY_PASS")
        self.assertTrue(gate["candidate_a_promotion_eligible"])
        primary["categorical_total_variation"]["game_month"]["value"] = 0.4
        gate = audit.evaluate_bounded_geometry_gate(reports)
        self.assertEqual(gate["result"], "PRIMARY_FAIL")
        self.assertTrue(gate["b_c_cannot_rescue_primary"])

    def test_25_and_50_percent_gates_are_individually_frozen(self) -> None:
        self.assertEqual(
            audit._geometry_relative_reduction_gate(
                {"value": 0.4}, {"value": 0.3}, 0.25
            )["status"],
            "PASS",
        )
        self.assertEqual(
            audit._geometry_relative_reduction_gate(
                {"value": 0.4}, {"value": 0.2}, 0.50
            )["status"],
            "PASS",
        )
        self.assertEqual(
            audit._geometry_relative_reduction_gate(
                {"value": 0.4}, {"value": 0.300001}, 0.25
            )["status"],
            "FAIL",
        )
        self.assertEqual(
            audit._geometry_relative_reduction_gate(
                {"value": 0.4}, {"value": 0.200001}, 0.50
            )["status"],
            "FAIL",
        )

    def test_three_of_five_family_rule_is_required(self) -> None:
        base = {
            "categorical_total_variation": {"game_month": {"value": 0.4}},
            "numeric_smd": {"x": {"value": 0.2}},
            "quantile_discrepancy": {"x": {"quantiles": {
                "0.10": {"standardized_abs_delta": 0.2},
                "0.50": {"standardized_abs_delta": 0.2},
                "0.90": {"standardized_abs_delta": 0.2},
            }}},
            "missing_rate_delta_pp": {"x": {"value": 0.2}},
            "entity_coverage": {"x": {"shortfall": {"value": 0.2}}},
            "source_rank_coverage": {
                "source_bin_tv": {"value": 0.4},
                "all_supported_bins_occupied": True,
            },
            "status": "OK",
        }
        candidate = json.loads(json.dumps(base))
        candidate["categorical_total_variation"]["game_month"]["value"] = 0.1
        candidate["numeric_smd"]["x"]["value"] = 0.1
        candidate["source_rank_coverage"]["source_bin_tv"]["value"] = 0.1
        reports = {
            origin: {
                "current_first_30k": base,
                "candidate_a": candidate,
                "candidate_b": candidate,
                "candidate_c": candidate,
            }
            for origin in ("r2022", "r2023")
        }
        gate = audit.evaluate_bounded_geometry_gate(reports)
        self.assertEqual(gate["result"], "PRIMARY_FAIL")
        for origin in ("r2022", "r2023"):
            self.assertEqual(
                gate["origins"][origin]["checks"]["strict_family_improvement_count"]["count"],
                2,
            )

    def test_each_p95_guardrail_is_frozen(self) -> None:
        expected = {
            "categorical_total_variation": 0.005,
            "numeric_smd": 0.02,
            "quantile_discrepancy": 0.02,
            "missing_rate_delta_pp": 0.05,
            "entity_coverage_shortfall": 0.01,
        }
        base = {
            "categorical_total_variation": {"game_month": {"value": 0.2}},
            "numeric_smd": {"x": {"value": 0.2}},
            "quantile_discrepancy": {"x": {"quantiles": {
                "0.10": {"standardized_abs_delta": 0.2},
                "0.50": {"standardized_abs_delta": 0.2},
                "0.90": {"standardized_abs_delta": 0.2},
            }}},
            "missing_rate_delta_pp": {"x": {"value": 0.2}},
            "entity_coverage": {"x": {"shortfall": {"value": 0.2}}},
        }
        for family, guardrail in expected.items():
            candidate = json.loads(json.dumps(base))
            excessive = 0.2 + guardrail + 1e-6
            if family == "quantile_discrepancy":
                for item in candidate[family]["x"]["quantiles"].values():
                    item["standardized_abs_delta"] = excessive
            elif family == "entity_coverage_shortfall":
                candidate["entity_coverage"]["x"]["shortfall"]["value"] = excessive
            else:
                candidate[family]["x" if family != "categorical_total_variation" else "game_month"]["value"] = excessive
            result = audit._geometry_family_gate(base, candidate, family, guardrail)
            self.assertEqual(result["status"], "FAIL", family)

    def test_finite_current_to_null_family_comparison_is_fail_closed(self) -> None:
        current = {"categorical_total_variation": {"game_month": {"value": 0.2}}}
        candidate = {"categorical_total_variation": {"game_month": {"value": None}}}
        result = audit._geometry_family_gate(
            current, candidate, "categorical_total_variation", 0.005
        )
        self.assertEqual(result["status"], "FAIL_CLOSED")

    def test_finite_current_to_null_candidate_is_fail_closed(self) -> None:
        current = {"value": 0.4}
        candidate = {"value": None, "reason": "empty"}
        result = audit._geometry_relative_reduction_gate(current, candidate, 0.25)
        self.assertEqual(result["status"], "FAIL_CLOSED")

    def test_zero_baseline_relative_reduction_is_fail_closed(self) -> None:
        for baseline in (0.0, 1e-13, 1e-12):
            result = audit._geometry_relative_reduction_gate(
                {"value": baseline}, {"value": baseline}, 0.25
            )
            self.assertEqual(result["status"], "FAIL_CLOSED")
            self.assertEqual(result["reason"], "zero_baseline_relative_reduction_undefined")
        normal = audit._geometry_relative_reduction_gate(
            {"value": 2e-12}, {"value": 1e-12}, 0.25
        )
        self.assertEqual(normal["status"], "PASS")

    def test_geometry_reader_static_firewall_passes_and_target_is_not_projected(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(audit.audit_bounded_geometry_reader_semantics(source), [])
        self.assertNotIn(audit.TARGET, audit.BOUNDED_GEOMETRY_PROJECTION_COLUMNS)

    def test_source_rank_small_and_empty_panels_are_explicit(self) -> None:
        small = audit._geometry_source_rank_coverage((0, 1, 2), (1,))
        self.assertEqual(len(small["full_count_by_bin"]), 20)
        self.assertEqual(len(small["selected_count_by_bin"]), 20)
        self.assertEqual(small["supported_bins"], [0, 6, 13])
        self.assertEqual(small["occupied_supported_bins"], [6])
        self.assertFalse(small["all_supported_bins_occupied"])
        empty = audit._geometry_source_rank_coverage((), ())
        self.assertEqual(empty["source_bin_tv"]["reason"], "empty_full_panel")
        self.assertEqual(len(empty["selected_count_by_bin"]), 20)

    def test_feature_only_reader_excludes_2024_and_target_from_frame(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aimers9-bounded-geometry-reader-") as temp:
            path = Path(temp) / "train.csv"
            rows = []
            for season, row_id in ((2022, "r22"), (2023, "r23"), (2024, "r24")):
                row = {column: 0 for column in audit.TRAIN_COLUMNS}
                row.update({
                    "row_id": row_id, "season": season, "game_type": "R",
                    "game_month": 5, "base_state": "___",
                    "control_success": "POISON-TARGET",
                })
                rows.append(row)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit.TRAIN_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            frame = audit.read_bounded_validation_features(path)
            self.assertEqual(frame["season"].tolist(), [2022, 2023])
            self.assertNotIn(audit.TARGET, frame.columns)
            self.assertNotIn("POISON-TARGET", frame.to_string())

    def test_bounded_geometry_cli_writes_feature_only_contract_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aimers9-bounded-geometry-cli-") as temp:
            root = Path(temp)
            path = root / "train.csv"
            rows = []
            for season in (2022, 2023):
                for index in range(35):
                    row = {column: 0 for column in audit.TRAIN_COLUMNS}
                    row.update({
                        "row_id": f"{season}-{index}",
                        "season": season,
                        "game_type": "R",
                        "game_month": index % 5 + 1,
                        "balls_before": index % 4,
                        "strikes_before": index % 3,
                        "base_state": "___",
                        "control_success": "POISON_TARGET",
                    })
                    rows.append(row)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit.TRAIN_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = audit.main([
                    "bounded-validation-geometry",
                    "--train-csv", str(path),
                    "--output-dir", str(root / "output"),
                    "--repo-root", str(SCRIPT.parents[1]),
                ])
            self.assertEqual(rc, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["mode"], "MEDIUM_BOUNDED_VALIDATION_GEOMETRY_FEATURE_ONLY")
            report_path = root / "output" / "bounded_geometry_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["geometry_contract_version"],
                audit.BOUNDED_GEOMETRY_CONTRACT_VERSION,
            )
            self.assertEqual(report["label_access_ledger"], [])
            self.assertFalse(report["scope"]["target_access"])
            self.assertFalse(report["scope"]["target_column_in_projection"])
            self.assertTrue(report["scope"]["branch_inert"])
            for field in (
                "target_2024_access", "test_distribution_access",
                "public_leaderboard_evidence", "external_information_access",
                "trackman_access", "model_training_or_scoring",
                "active_policy_modified",
            ):
                self.assertFalse(report["scope"][field], field)
            panels = report["sections"]["outer_validation_panels"]
            self.assertEqual(set(panels), {"r2022", "r2023"})
            for origin in panels.values():
                self.assertEqual(
                    set(origin),
                    {"current_first_30k", "candidate_a", "candidate_b", "candidate_c"},
                )
            self.assertIn(
                report["sections"]["acceptance_gate"]["result"],
                {"PRIMARY_PASS", "PRIMARY_FAIL", "FAIL_CLOSED"},
            )
            self.assertNotIn("POISON_TARGET", report_path.read_text(encoding="utf-8"))

            output_two = io.StringIO()
            with contextlib.redirect_stdout(output_two):
                second_rc = audit.main([
                    "bounded-validation-geometry",
                    "--train-csv", str(path),
                    "--output-dir", str(root / "output-two"),
                    "--repo-root", str(SCRIPT.parents[1]),
                ])
            self.assertEqual(second_rc, 0)
            report_two = json.loads(
                (root / "output-two" / "bounded_geometry_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report, report_two)
            self.assertEqual(
                report["canonical_report_sha256"],
                audit.canonical_report_hash(report),
            )

            def assert_finite_or_null(value: object) -> None:
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))
                elif isinstance(value, dict):
                    for item in value.values():
                        assert_finite_or_null(item)
                elif isinstance(value, list):
                    for item in value:
                        assert_finite_or_null(item)

            assert_finite_or_null(report)

    def test_geometry_report_is_hash_seed_invariant_for_same_order(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            "import pandas as pd\n"
            "import audit_data_integrity_temporal as audit\n"
            "frame = pd.DataFrame({\n"
            " 'row_id': ['a','b','c','d','e','f'],\n"
            " 'season': [2022]*6, 'game_type': ['R']*6,\n"
            " 'game_month': [1,2,3,4,5,6], 'balls_before': [0,1,2,3,0,1],\n"
            " 'strikes_before': [0,1,2,0,1,2], '__source_position': list(range(6)),\n"
            "})\n"
            "print(audit.canonical_hash(audit.select_bounded_geometry(frame, 'r2022', 'candidate_a').selected_positions))\n"
        )
        outputs = []
        for seed in ("1", "2", "3"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True,
                text=True, env=environment,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(len(set(outputs)), 1)


if __name__ == "__main__":
    unittest.main()
