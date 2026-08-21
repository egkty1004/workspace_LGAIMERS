from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
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

    def test_separate_abs_feature_only_runner_is_branch_inert(self) -> None:
        report, paths = audit.run_abs_2024_feature_audit(
            self.train, self.root / "abs-output", SCRIPT.parents[1]
        )
        self.assertEqual(report["label_access_ledger"], [])
        self.assertTrue(report["scope"]["branch_inert"])
        self.assertFalse(report["scope"]["target_access"])
        self.assertFalse(report["scope"]["may_select_features"])
        self.assertIn(
            "boundary_diagnostics_all_prespecified_features", report["sections"]
        )
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


if __name__ == "__main__":
    unittest.main()
