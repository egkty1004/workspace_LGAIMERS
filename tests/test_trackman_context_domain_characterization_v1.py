import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import audit_trackman_context_domain_characterization_v1 as audit  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "trackman_context_domain_characterization_v1.json"
SCRIPT_PATH = ROOT / "scripts" / "audit_trackman_context_domain_characterization_v1.py"


def frame(rows=None):
    if rows is None:
        rows = [
            {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0},
            {"season": 2020, "balls_before": 1, "strikes_before": 1, "outs_before": 1},
            {"season": 2021, "balls_before": 2, "strikes_before": 2, "outs_before": 2},
            {"season": 2022, "balls_before": 3, "strikes_before": 0, "outs_before": 0},
        ]
    return pd.DataFrame(rows, columns=audit.PROJECTION)


def report_for(main=None, trackman=None):
    main_scan = audit.scan_frame(main if main is not None else frame(), source=audit.MAIN_SOURCE)
    tm_scan = audit.scan_frame(trackman if trackman is not None else frame(), source=audit.TRACKMAN_SOURCE)
    config = audit.load_contract_config(ROOT)
    return audit.build_report(
        main_scan,
        tm_scan,
        repo_root=ROOT,
        config_path=CONFIG_PATH,
        script_path=SCRIPT_PATH,
        config=config,
    )


class ContractAndFirewallTests(unittest.TestCase):
    def test_runtime_config_and_static_firewall(self):
        config = audit.load_contract_config(ROOT)
        self.assertEqual(config["contract_version"], audit.CONTRACT_VERSION)
        result = audit.static_contract(ROOT)
        self.assertTrue(result["static_pass"])
        self.assertEqual(result["machine_verdicts"], ["SCAN_COMPLETE", "NOT_PROVEN"])
        self.assertTrue(result["raw_lexical_diagnostics_separate"])
        self.assertFalse(result["target_access"])
        self.assertFalse(result["target_column_in_projection"])

    def test_config_divergence_fails_closed(self):
        config = audit.load_contract_config(ROOT)
        changed = copy.deepcopy(config)
        changed["semantic_classifier"]["domains"]["balls_before"] = [0, 1, 2]
        with self.assertRaises(audit.CharacterizationError):
            audit.validate_contract_config(changed)

    def test_exact_projection_and_forbidden_column_guard(self):
        valid = frame()
        audit.validate_projection(valid, source=audit.MAIN_SOURCE)
        poisoned = valid.copy()
        poisoned["control_success"] = 1
        with self.assertRaises(audit.CharacterizationError):
            audit.validate_projection(poisoned, source=audit.MAIN_SOURCE)
        tm_poison = valid.copy()
        tm_poison["pitch_type_group"] = "fastball"
        with self.assertRaises(audit.CharacterizationError):
            audit.validate_projection(tm_poison, source=audit.TRACKMAN_SOURCE)

    def test_csv_projection_ignores_target_and_forbidden_poison(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main = frame()
            main["control_success"] = [0, 1, 0, 1]
            main["row_id"] = ["a", "b", "c", "d"]
            poisoned = main.copy()
            poisoned["control_success"] = [1, 1, 1, 1]
            poisoned["row_id"] = ["poison"] * len(poisoned)
            first_path = directory / "main.csv"
            second_path = directory / "poisoned.csv"
            main.to_csv(first_path, index=False)
            poisoned.to_csv(second_path, index=False)
            first = audit.scan_csv(first_path, source=audit.MAIN_SOURCE)
            second = audit.scan_csv(second_path, source=audit.MAIN_SOURCE)
            self.assertEqual(first, second)
            self.assertEqual(first["row_count"], len(frame()))

    def test_csv_trackman_projection_ignores_entity_physics_and_family_poison(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean = frame().copy()
            clean["pitcher_trackman_id"] = ["p1", "p2", "p3", "p4"]
            clean["rel_speed"] = [90.0, 91.0, 92.0, 93.0]
            clean["pitch_type_group"] = ["fastball", "breaking", "offspeed", "other"]
            poisoned = clean.copy()
            poisoned["pitcher_trackman_id"] = ["poison"] * len(poisoned)
            poisoned["rel_speed"] = [-999.0] * len(poisoned)
            poisoned["pitch_type_group"] = ["unexpected"] * len(poisoned)
            first_path = directory / "trackman.csv"
            second_path = directory / "poisoned-trackman.csv"
            clean.to_csv(first_path, index=False)
            poisoned.to_csv(second_path, index=False)
            first = audit.scan_csv(first_path, source=audit.TRACKMAN_SOURCE)
            second = audit.scan_csv(second_path, source=audit.TRACKMAN_SOURCE)
            self.assertEqual(first, second)
            self.assertEqual(first["row_count"], len(frame()))

    def test_target_and_forbidden_columns_never_appear_in_runtime_projection(self):
        self.assertEqual(audit.MAIN_PROJECTION, audit.TRACKMAN_PROJECTION)
        self.assertNotIn("control_success", audit.MAIN_PROJECTION)
        self.assertNotIn("row_id", audit.MAIN_PROJECTION)
        self.assertFalse(audit.FORBIDDEN_MAIN.intersection(audit.MAIN_PROJECTION))
        self.assertFalse(audit.FORBIDDEN_TRACKMAN.intersection(audit.TRACKMAN_PROJECTION))


class SemanticClassifierTests(unittest.TestCase):
    def test_pr12_whitespace_and_numeric_lexemes_are_not_v1_invalid(self):
        values = frame([{"season": " 2019 ", "balls_before": " 1 ", "strikes_before": "1.0", "outs_before": "0"}])
        raw = values.copy()
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE, raw_frame=raw)
        for field in audit.PROJECTION:
            self.assertEqual(scan["fields"][field]["v1_compatibility"]["invalid_rows"], 0)
        self.assertGreater(scan["fields"]["balls_before"]["raw_lexical_diagnostics"]["whitespace_tokens"], 0)
        self.assertGreater(scan["fields"]["strikes_before"]["raw_lexical_diagnostics"]["semantic_v1_legal_with_lexical_variation"], 0)

    def test_missing_detection(self):
        values = frame([{"season": 2019, "balls_before": None, "strikes_before": pd.NA, "outs_before": float("nan")}])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        for field in audit.CONTEXT_COLUMNS:
            result = scan["fields"][field]["v1_compatibility"]
            self.assertEqual(result["missing_rows"], 1)
            self.assertEqual(result["invalid_reason_counts"]["missing"], 1)

    def test_nonfinite_detection(self):
        values = frame([{"season": 2019, "balls_before": math.inf, "strikes_before": -math.inf, "outs_before": float("nan")}])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        for field in ("balls_before", "strikes_before"):
            result = scan["fields"][field]["v1_compatibility"]
            self.assertEqual(result["nonfinite_rows"], 1)
            self.assertEqual(result["invalid_reason_counts"]["nonfinite"], 1)
        missing = scan["fields"]["outs_before"]["v1_compatibility"]
        self.assertEqual(missing["missing_rows"], 1)
        self.assertEqual(missing["invalid_reason_counts"]["missing"], 1)

    def test_noninteger_detection(self):
        values = frame([{"season": 2019, "balls_before": 1.5, "strikes_before": "2.2", "outs_before": 0.25}])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        for field in audit.CONTEXT_COLUMNS:
            result = scan["fields"][field]["v1_compatibility"]
            self.assertEqual(result["non_integer_rows"], 1)
            self.assertEqual(result["invalid_reason_counts"]["non_integer"], 1)

    def test_each_legal_boundary_value(self):
        for field, legal in audit.DOMAINS.items():
            for value in sorted(legal):
                row = {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0}
                row[field] = value
                scan = audit.scan_frame(frame([row]), source=audit.MAIN_SOURCE)
                self.assertEqual(scan["fields"][field]["v1_compatibility"]["legal_rows"], 1)

    def test_each_illegal_boundary_value(self):
        for field, legal in audit.DOMAINS.items():
            for value in (min(legal) - 1, max(legal) + 1):
                row = {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0}
                row[field] = value
                scan = audit.scan_frame(frame([row]), source=audit.MAIN_SOURCE)
                result = scan["fields"][field]["v1_compatibility"]
                self.assertEqual(result["invalid_reason_counts"]["out_of_domain"], 1)
                self.assertEqual(result["out_of_domain_unique_values"][str(value)], 1)

    def test_season_stratification_and_invalid_season(self):
        values = frame([
            {"season": season, "balls_before": 0, "strikes_before": 0, "outs_before": 0}
            for season in (*audit.EXPECTED_SEASONS, 2025)
        ])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        season = scan["season_diagnostics"]
        self.assertEqual(season["row_counts"]["2019"], 1)
        self.assertEqual(season["row_counts"]["2024"], 1)
        self.assertEqual(season["invalid_season_rows"], 1)
        self.assertIn(audit.MISSING_SEASON, scan["fields"]["season"]["v1_compatibility"]["season_diagnostics"])

    def test_raw_lexical_diagnostic_is_separate_from_v1_classifier(self):
        values = frame([{"season": 2019, "balls_before": "1.0", "strikes_before": " 2 ", "outs_before": "0"}])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE, raw_frame=values)
        field = scan["fields"]["balls_before"]
        self.assertEqual(field["v1_compatibility"]["invalid_rows"], 0)
        self.assertGreater(field["raw_lexical_diagnostics"]["lexical_kind_counts"]["integer_numeric_noninteger_lexeme"], 0)


class AggregateDiagnosticTests(unittest.TestCase):
    def test_overall_and_season_legal_domain_histograms_are_sorted(self):
        values = frame([
            {"season": 2019, "balls_before": 3, "strikes_before": 2, "outs_before": 1},
            {"season": 2019, "balls_before": 0, "strikes_before": 1, "outs_before": 0},
            {"season": 2020, "balls_before": 3, "strikes_before": 0, "outs_before": 2},
        ])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        balls = scan["fields"]["balls_before"]["v1_compatibility"]
        self.assertEqual(balls["legal_domain_value_histogram"], {"0": 1, "3": 2})
        self.assertEqual(balls["legal_domain_value_histogram_by_season"]["2019"], {"0": 1, "3": 1})
        self.assertEqual(balls["legal_domain_value_histogram_by_season"]["2020"], {"3": 1})
        self.assertEqual(list(balls["legal_domain_value_histogram"].keys()), ["0", "3"])
        for field in audit.CONTEXT_COLUMNS:
            compatibility = scan["fields"][field]["v1_compatibility"]
            self.assertIn("legal_domain_value_histogram", compatibility)
            self.assertIn("legal_domain_value_histogram_by_season", compatibility)
            self.assertEqual(
                set(compatibility["legal_domain_value_histogram_by_season"]),
                {str(season) for season in audit.EXPECTED_SEASONS} | {audit.MISSING_SEASON},
            )

    def test_context_invalid_concentration_by_source_field_and_season(self):
        main = frame([
            {"season": 2019, "balls_before": 4, "strikes_before": 0, "outs_before": 0},
            {"season": 2020, "balls_before": 0, "strikes_before": 3, "outs_before": 0},
        ])
        trackman = frame([
            {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 3},
        ])
        report = report_for(main=main, trackman=trackman)
        concentration = report["concentration"]
        by_source = {item["source"]: item for item in concentration["context_invalid_by_source"]}
        self.assertEqual(by_source["main"]["context_invalid_row_count"], 2)
        self.assertEqual(by_source["main"]["context_invalid_fraction"], 1.0)
        self.assertEqual(by_source["trackman"]["context_invalid_row_count"], 1)
        field_counts = {
            (item["source"], item["field"]): item["context_invalid_observation_count"]
            for item in concentration["context_invalid_by_source_field"]
        }
        self.assertEqual(field_counts[("main", "balls_before")], 1)
        self.assertEqual(field_counts[("main", "strikes_before")], 1)
        self.assertEqual(field_counts[("trackman", "outs_before")], 1)
        season_counts = {
            (item["source"], item["season"]): item["context_invalid_row_count"]
            for item in concentration["context_invalid_by_source_season"]
        }
        self.assertEqual(season_counts[("main", "2019")], 1)
        self.assertEqual(season_counts[("main", "2020")], 1)
        self.assertEqual(season_counts[("trackman", "2019")], 1)
        self.assertEqual(len(concentration["row_signature_counts"]), 3)

    def test_invalid_season_concentration_uses_invalid_season_denominator(self):
        main = frame([
            {"season": 2025, "balls_before": 4, "strikes_before": 0, "outs_before": 0},
        ])
        trackman = frame([
            {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0},
        ])
        report = report_for(main=main, trackman=trackman)
        entries = {
            (item["source"], item["season"]): item
            for item in report["concentration"]["context_invalid_by_source_season"]
        }
        invalid_season = entries[("main", audit.MISSING_SEASON)]
        self.assertEqual(invalid_season["context_invalid_row_count"], 1)
        self.assertEqual(invalid_season["total_row_count"], 1)
        self.assertEqual(invalid_season["context_invalid_fraction"], 1.0)

    def test_row_order_invariance_and_source_frame_hash(self):
        values = frame([
            {"season": 2020, "balls_before": 4, "strikes_before": 0, "outs_before": 0},
            {"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0},
        ])
        first = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        second = audit.scan_frame(values.iloc[::-1].reset_index(drop=True), source=audit.MAIN_SOURCE)
        self.assertEqual(first, second)

    def test_aggregate_row_signature_counts(self):
        values = frame([{"season": 2022, "balls_before": 4, "strikes_before": 1, "outs_before": 3}])
        scan = audit.scan_frame(values, source=audit.MAIN_SOURCE)
        self.assertEqual(len(scan["row_signature_counts"]), 1)
        signature = scan["row_signature_counts"][0]
        self.assertEqual(signature["source"], "main")
        self.assertEqual(signature["season"], "2022")
        self.assertEqual(signature["invalid_fields"], ["balls_before", "outs_before"])
        self.assertEqual(signature["row_count"], 1)
        self.assertFalse(any("row_id" in str(item) for item in scan["row_signature_counts"]))

    def test_prior_kill_not_reproduced_is_not_proven(self):
        report = report_for()
        self.assertEqual(report["verdict"], "NOT_PROVEN")
        self.assertEqual(report["reasons"], ["PRIOR_KILL_NOT_REPRODUCED"])
        self.assertEqual(report["prior_kill_reproduction"]["v1_compatible_context_invalid_row_count"], 0)

    def test_invalid_context_yields_scan_complete_without_policy_interpretation(self):
        bad = frame([{"season": 2019, "balls_before": 9, "strikes_before": 0, "outs_before": 0}])
        report = report_for(main=bad, trackman=bad)
        self.assertEqual(report["verdict"], "SCAN_COMPLETE")
        self.assertEqual(report["reasons"], [])
        self.assertFalse(report["config_runtime"]["config_runtime_match"] is False)
        self.assertNotIn("FILTERABLE_RESIDUE", report["verdict"])
        self.assertNotIn("DOMAIN_INCOMPATIBLE", report["verdict"])

    def test_source_distinction_is_documented_without_filtering(self):
        report = report_for()
        self.assertIn("source_distinction", report)
        self.assertIn("source_local_exclusion_contract", report["source_distinction"]["trackman_invalid_future_candidate"])
        self.assertIn("row_local_fallback_hypothesis", report["source_distinction"]["main_invalid_future_candidate"])

    def test_aggregate_only_privacy_and_firewall_flags(self):
        report = report_for()
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["row_ids"])
        self.assertFalse(report["privacy"]["target_values"])
        self.assertFalse(report["privacy"]["raw_rows"])
        self.assertEqual(report["scope"]["label_access_ledger"], [])
        self.assertFalse(report["scope"]["target_access"])
        self.assertFalse(report["scope"]["test_access"])
        self.assertFalse(report["scope"]["trackman_entity_access"])

    def test_deterministic_canonical_report_hash(self):
        first = report_for()
        second = report_for()
        self.assertEqual(first, second)
        self.assertEqual(first["canonical_report_sha256"], audit.canonical_report_hash(first))
        self.assertEqual(first["canonical_report_sha256"], second["canonical_report_sha256"])

    def test_output_directory_guard_and_no_overwrite(self):
        report = report_for()
        with self.assertRaises(audit.CharacterizationError):
            audit.write_outputs(report, ROOT / "forbidden-output", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            paths = audit.write_outputs(report, output, repo_root=ROOT)
            self.assertTrue(paths[0].exists())
            with self.assertRaises(FileExistsError):
                audit.write_outputs(report, output, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
