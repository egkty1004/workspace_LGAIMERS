import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import audit_trackman_crosswalk_free_context_priors_v1 as audit  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def main_frame(seasons=(2019, 2020, 2022, 2023, 2024, 2025)):
    rows = []
    for index, season in enumerate(seasons):
        rows.append({
            "row_id": f"main-{season}-{index}", "season": season,
            "balls_before": index % 4, "strikes_before": index % 3,
            "outs_before": index % 3,
        })
    return pd.DataFrame(rows, columns=audit.MAIN_PROJECTION)


def trackman_frame(include_2024=True):
    rows = []
    families = audit.FAMILIES
    # 2019 is the only history for 2020--2024.  The first context has a
    # supported L0 and a non-zero deviation from the global prior.
    for index in range(120):
        rows.append({"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 0, "pitch_type_group": families[index % 2]})
    for index in range(50):
        rows.append({"season": 2019, "balls_before": 0, "strikes_before": 0, "outs_before": 1, "pitch_type_group": families[2 + (index % 2)]})
    for index in range(10):
        rows.append({"season": 2019, "balls_before": 1, "strikes_before": 0, "outs_before": 0, "pitch_type_group": families[index % 4]})
    if include_2024:
        for index in range(40):
            rows.append({"season": 2024, "balls_before": 3, "strikes_before": 2, "outs_before": 2, "pitch_type_group": families[index % 4]})
    return pd.DataFrame(rows, columns=audit.TRACKMAN_PROJECTION)


class ContractAndFirewallTests(unittest.TestCase):
    def test_runtime_config_and_static_firewall(self):
        config = audit.load_contract_config(ROOT)
        self.assertEqual(config["contract_version"], audit.CONTRACT_VERSION)
        static = audit.static_contract(ROOT)
        self.assertTrue(static["static_pass"])
        self.assertFalse(static["target_access"])
        self.assertFalse(static["trackman_entity_access"])
        self.assertFalse(static["trackman_physics_access"])
        self.assertFalse(static["backoff_level_is_model_feature"])

    def test_config_divergence_fails_closed(self):
        config = audit.load_contract_config(ROOT)
        changed = copy.deepcopy(config)
        changed["context"]["support_threshold"] = 99
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_contract_config(changed)

    def test_projection_rejects_target_entity_and_physics(self):
        main = main_frame().copy()
        main["control_success"] = 0
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_projection(main)
        trackman = trackman_frame().copy()
        trackman["trackman_id"] = "forbidden"
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_projection(trackman, trackman=True)

    def test_csv_reader_uses_exact_feature_projections(self):
        with tempfile.TemporaryDirectory() as directory:
            main_path = Path(directory) / "train.csv"
            tm_path = Path(directory) / "trackman.csv"
            main = main_frame((2019, 2020)).copy()
            main["control_success"] = [0, 1]
            main.to_csv(main_path, index=False)
            trackman_frame(False).to_csv(tm_path, index=False)
            read_main = audit.read_main_features(main_path)
            read_tm = audit.read_trackman_features(tm_path)
            self.assertEqual(tuple(read_main.columns), audit.MAIN_PROJECTION)
            self.assertEqual(tuple(read_tm.columns), audit.TRACKMAN_PROJECTION)
            self.assertNotIn("control_success", read_main.columns)

    def test_scoped_readers_ignore_poisoned_target_entities_and_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main = main_frame((2020, 2022)).copy()
            main["control_success"] = [0, 1]
            main["pitcher_id"] = ["p1", "p2"]
            main["pitcher_team_id"] = ["t1", "t2"]
            poisoned_main = main.copy()
            poisoned_main["control_success"] = [1, 0]
            poisoned_main["pitcher_id"] = ["poison-a", "poison-b"]
            poisoned_main["pitcher_team_id"] = ["poison-team-a", "poison-team-b"]
            tm = trackman_frame(True).head(40).copy()
            tm["pitcher_trackman_id"] = [f"tm-{index}" for index in range(len(tm))]
            tm["rel_speed"] = 90.0
            poisoned_tm = tm.copy()
            poisoned_tm["pitcher_trackman_id"] = "poisoned"
            poisoned_tm["rel_speed"] = -999.0
            main_path = directory / "main.csv"
            poisoned_main_path = directory / "poisoned-main.csv"
            tm_path = directory / "tm.csv"
            poisoned_tm_path = directory / "poisoned-tm.csv"
            main.to_csv(main_path, index=False)
            poisoned_main.to_csv(poisoned_main_path, index=False)
            tm.to_csv(tm_path, index=False)
            poisoned_tm.to_csv(poisoned_tm_path, index=False)
            clean_main = audit.read_main_features(main_path)
            changed_main = audit.read_main_features(poisoned_main_path)
            clean_tm = audit.read_trackman_features(tm_path)
            changed_tm = audit.read_trackman_features(poisoned_tm_path)
            clean_features = audit.apply_context_prior_features(clean_main, audit.build_context_prior_model(clean_tm, seasons=(2020, 2022)))
            changed_features = audit.apply_context_prior_features(changed_main, audit.build_context_prior_model(changed_tm, seasons=(2020, 2022)))
            pd.testing.assert_frame_equal(clean_main, changed_main)
            pd.testing.assert_frame_equal(clean_tm, changed_tm)
            pd.testing.assert_frame_equal(clean_features, changed_features)

    def test_invalid_context_is_domain_kill_not_sparse_fallback(self):
        bad = main_frame((2022,)).copy()
        bad.loc[0, "balls_before"] = 4
        with self.assertRaises(audit.ContextDomainNotProven):
            bad_trackman = trackman_frame(False).copy()
            bad_trackman.loc[0, "outs_before"] = 3
            audit.build_context_prior_model(bad_trackman)
        with self.assertRaises(audit.ContextDomainNotProven):
            audit.apply_context_prior_features(bad, audit.build_context_prior_model(trackman_frame(False)))

    def test_missing_trackman_family_is_taxonomy_failure(self):
        trackman = trackman_frame(False).copy()
        trackman.loc[0, "pitch_type_group"] = None
        with self.assertRaises(audit.TaxonomyNotProven):
            audit.build_context_prior_model(trackman)


class PriorConstructionTests(unittest.TestCase):
    def test_strict_prior_season_and_future_poison(self):
        base = trackman_frame(True)
        poisoned = base.copy()
        poisoned.loc[poisoned["season"] == 2024, "pitch_type_group"] = "other"
        main = main_frame((2022, 2023, 2024, 2025))
        first = audit.apply_context_prior_features(main, audit.build_context_prior_model(base, seasons=(2022, 2023, 2024, 2025)))
        second = audit.apply_context_prior_features(main, audit.build_context_prior_model(poisoned, seasons=(2022, 2023, 2024, 2025)))
        pd.testing.assert_frame_equal(first.iloc[:3].reset_index(drop=True), second.iloc[:3].reset_index(drop=True))
        self.assertTrue((first.iloc[3][list(audit.FEATURE_COLUMNS[:4])] != second.iloc[3][list(audit.FEATURE_COLUMNS[:4])]).any())

    def test_trackman_row_order_invariance(self):
        trackman = trackman_frame(True)
        main = main_frame((2022, 2023, 2024, 2025))
        first = audit.apply_context_prior_features(main, audit.build_context_prior_model(trackman, seasons=(2022, 2023, 2024, 2025)))
        shuffled = trackman.sample(frac=1.0, random_state=91).reset_index(drop=True)
        second = audit.apply_context_prior_features(main, audit.build_context_prior_model(shuffled, seasons=(2022, 2023, 2024, 2025)))
        pd.testing.assert_frame_equal(first, second)

    def test_runtime_determinism_evidence_rebuilds_complete_output(self):
        trackman = trackman_frame(True)
        main = main_frame((2022, 2023, 2024, 2025))
        model = audit.build_context_prior_model(trackman, seasons=(2022, 2023, 2024, 2025))
        evidence = audit.assess_determinism(main, trackman, model)
        self.assertTrue(evidence["tested"])
        self.assertTrue(evidence["complete_feature_output_equal"])
        self.assertEqual(evidence["original_feature_output_sha256"], evidence["permuted_feature_output_sha256"])

    def test_other_is_retained_as_fourth_simplex_family(self):
        model = audit.build_context_prior_model(trackman_frame(False), seasons=(2020,))
        features = audit.apply_context_prior_features(main_frame((2020,)), model)
        values = [float(features.loc[0, column]) for column in audit.FEATURE_COLUMNS[:4]]
        self.assertEqual(len(values), 4)
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_l1_below_100_falls_back_to_global_and_support_is_selected_level(self):
        model = audit.build_context_prior_model(trackman_frame(False), seasons=(2020,))
        row = pd.DataFrame([{"row_id": "sparse", "season": 2020, "balls_before": 1, "strikes_before": 0, "outs_before": 0}], columns=audit.MAIN_PROJECTION)
        features = audit.apply_context_prior_features(row, model)
        self.assertEqual(features.loc[0, "__backoff_level"], "global")
        self.assertEqual(features.loc[0, "tm_cf_support"], len(trackman_frame(False)))
        row_l0 = pd.DataFrame([{"row_id": "dense", "season": 2020, "balls_before": 0, "strikes_before": 0, "outs_before": 0}], columns=audit.MAIN_PROJECTION)
        dense = audit.apply_context_prior_features(row_l0, model)
        self.assertEqual(dense.loc[0, "__backoff_level"], "L0")
        self.assertEqual(dense.loc[0, "tm_cf_support"], 120)

    def test_2019_no_history_is_explicit(self):
        model = audit.build_context_prior_model(trackman_frame(False), seasons=(2019,))
        features = audit.apply_context_prior_features(main_frame((2019,)), model)
        self.assertEqual(features.loc[0, "__backoff_level"], "NO_HISTORY")
        self.assertEqual(features.loc[0, "tm_cf_support"], 0)
        self.assertTrue(pd.isna(features.loc[0, "tm_cf_p_fastball"]))

    def test_twelve_2025_count_states_need_no_l1_threshold(self):
        rows = []
        for index in range(12):
            rows.append({"row_id": f"2025-{index}", "season": 2025, "balls_before": index % 4, "strikes_before": index % 3, "outs_before": index % 3})
        main = pd.DataFrame(rows, columns=audit.MAIN_PROJECTION)
        features = audit.apply_context_prior_features(main, audit.build_context_prior_model(trackman_frame(False), seasons=(2025,)))
        self.assertEqual(len(features), 12)
        self.assertTrue(features["tm_cf_p_fastball"].notna().all())
        self.assertTrue(set(features["__backoff_level"]).issubset({"L0", "L1", "global"}))

    def test_aggregate_lookup_reproduces_selected_feature_records(self):
        trackman = trackman_frame(True)
        main = main_frame((2022, 2023, 2024, 2025))
        model = audit.build_context_prior_model(trackman, seasons=(2022, 2023, 2024, 2025))
        features = audit.apply_context_prior_features(main, model)
        lookup = audit.build_aggregate_lookup(model)
        self.assertIn("L0", lookup["seasons"]["2022"])
        self.assertIn("L1", lookup["seasons"]["2022"])
        for index, row in enumerate(main.to_dict("records")):
            reproduced = audit.apply_context_prior_lookup_row(row, lookup, int(row["season"]))
            expected = {column: features.loc[index, column] for column in [*audit.FEATURE_COLUMNS, "__backoff_level"]}
            self.assertEqual(reproduced, expected)
        reversed_lookup = audit.build_aggregate_lookup(audit.build_context_prior_model(trackman.iloc[::-1].reset_index(drop=True), seasons=(2022, 2023, 2024, 2025)))
        self.assertEqual(audit.canonical_hash(lookup), audit.canonical_hash(reversed_lookup))

    def test_entity_and_physics_columns_cannot_affect_features(self):
        trackman = trackman_frame(False)
        model = audit.build_context_prior_model(trackman, seasons=(2020,))
        main = main_frame((2020,))
        expected = audit.apply_context_prior_features(main, model)
        self.assertEqual(tuple(trackman.columns), audit.TRACKMAN_PROJECTION)
        pd.testing.assert_frame_equal(expected, audit.apply_context_prior_features(main, model))


class IndependenceAndAcceptanceTests(unittest.TestCase):
    def test_main_row_add_delete_shuffle_independence(self):
        model = audit.build_context_prior_model(trackman_frame(True), seasons=(2022, 2023, 2024, 2025))
        main = main_frame((2022, 2023, 2024))
        expected = audit.apply_context_prior_features(main, model)
        augmented = pd.concat([main, main_frame((2025,))], ignore_index=True)
        augmented_result = audit.apply_context_prior_features(augmented, model)
        pd.testing.assert_frame_equal(expected, augmented_result.iloc[: len(main)].reset_index(drop=True))
        shuffled = main.sample(frac=1, random_state=7).reset_index(drop=True)
        shuffled_result = audit.apply_context_prior_features(shuffled, model)
        self.assertEqual(set(tuple(row) for row in expected[list(audit.FEATURE_COLUMNS)].to_numpy()), set(tuple(row) for row in shuffled_result[list(audit.FEATURE_COLUMNS)].to_numpy()))

    def test_acceptance_does_not_fail_for_sparse_l1_alone(self):
        trackman = trackman_frame(False)
        main = pd.DataFrame([{"row_id": "x", "season": 2020, "balls_before": 1, "strikes_before": 0, "outs_before": 0}], columns=audit.MAIN_PROJECTION)
        model = audit.build_context_prior_model(trackman, seasons=(2020,))
        features = audit.apply_context_prior_features(main, model)
        result = audit.evaluate_acceptance(main, features, trackman)
        self.assertNotIn("NO_VALID_LEGAL_FALLBACK", result["p_fail_reasons"])
        self.assertTrue(result["l1_support_below_threshold_is_not_failure"])

    def test_context_variation_collapse_is_p_fail(self):
        trackman = trackman_frame(False).copy()
        trackman["pitch_type_group"] = "fastball"
        model = audit.build_context_prior_model(trackman, seasons=(2020,))
        main = main_frame((2020,))
        features = audit.apply_context_prior_features(main, model)
        result = audit.evaluate_acceptance(main, features, trackman)
        self.assertIn("CONTEXT_VARIATION_COLLAPSED", result["p_fail_reasons"])

    def test_canonical_report_hash_and_external_output_guard(self):
        main = main_frame((2020,))
        trackman = trackman_frame(False)
        model = audit.build_context_prior_model(trackman, seasons=(2020,))
        features = audit.apply_context_prior_features(main, model)
        acceptance = audit.evaluate_acceptance(main, features, trackman)
        taxonomy = audit.validate_trackman_taxonomy(trackman)
        report = audit.build_report(main, trackman, features, acceptance, taxonomy, repo_root=ROOT, config_path=ROOT / "configs" / "trackman_crosswalk_free_context_priors_v1.json", script_path=ROOT / "scripts" / "audit_trackman_crosswalk_free_context_priors_v1.py", config=audit.load_contract_config(ROOT))
        self.assertEqual(report["canonical_report_sha256"], audit.canonical_report_hash(report))
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.write_outputs(report, audit.build_aggregate_lookup(model), ROOT / "forbidden-output", repo_root=ROOT)

    def test_deterministic_aggregate_report(self):
        trackman = trackman_frame(False)
        main = main_frame((2020, 2022))
        model = audit.build_context_prior_model(trackman, seasons=(2020, 2022))
        features = audit.apply_context_prior_features(main, model)
        acceptance = audit.evaluate_acceptance(main, features, trackman)
        taxonomy = audit.validate_trackman_taxonomy(trackman)
        config = audit.load_contract_config(ROOT)
        kwargs = {"repo_root": ROOT, "config_path": ROOT / "configs" / "trackman_crosswalk_free_context_priors_v1.json", "script_path": ROOT / "scripts" / "audit_trackman_crosswalk_free_context_priors_v1.py", "config": config}
        first = audit.build_report(main, trackman, features, acceptance, taxonomy, **kwargs)
        second = audit.build_report(main, trackman, features, acceptance, taxonomy, **kwargs)
        self.assertEqual(first["canonical_report_sha256"], second["canonical_report_sha256"])
        json.dumps(first, allow_nan=False)

    def test_invalid_domain_run_is_reportable_p_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main = main_frame((2022,)).copy()
            main.loc[0, "balls_before"] = 4
            main_path = directory / "main.csv"
            tm_path = directory / "tm.csv"
            output = directory / "output"
            main.to_csv(main_path, index=False)
            trackman_frame(False).to_csv(tm_path, index=False)
            report, paths = audit.run_from_paths(main_path, tm_path, output, ROOT)
            self.assertEqual(report["acceptance"]["verdict"], "P_KILL")
            self.assertEqual(report["acceptance"]["kill_reasons"], ["CONTEXT_DOMAIN_NOT_PROVEN"])
            self.assertFalse(report["feature_output_available"])
            self.assertTrue(paths[0].exists())

    def test_invalid_trackman_context_run_is_reportable_p_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main_path = directory / "main.csv"
            tm_path = directory / "tm.csv"
            main_frame((2022,)).to_csv(main_path, index=False)
            invalid_trackman = trackman_frame(False).copy()
            invalid_trackman.loc[0, "outs_before"] = 3
            invalid_trackman.to_csv(tm_path, index=False)
            report, _ = audit.run_from_paths(main_path, tm_path, directory / "output", ROOT)
            self.assertEqual(report["acceptance"]["verdict"], "P_KILL")
            self.assertEqual(report["acceptance"]["kill_reasons"], ["CONTEXT_DOMAIN_NOT_PROVEN"])

    def test_raw_missing_trackman_family_is_taxonomy_kill_not_unexpected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main_path = directory / "main.csv"
            tm_path = directory / "tm.csv"
            main_frame((2022,)).to_csv(main_path, index=False)
            invalid_trackman = trackman_frame(False).copy()
            invalid_trackman.loc[0, "pitch_type_group"] = None
            invalid_trackman.to_csv(tm_path, index=False)
            report, _ = audit.run_from_paths(main_path, tm_path, directory / "output", ROOT)
            self.assertEqual(report["acceptance"]["verdict"], "P_KILL")
            self.assertEqual(report["acceptance"]["kill_reasons"], ["TAXONOMY_NOT_PROVEN"])
            self.assertEqual(report["taxonomy"]["missing_rows"], 1)
            self.assertEqual(report["taxonomy"]["unexpected_rows"], 0)

    def test_out_of_scope_season_is_temporal_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main_path = directory / "main.csv"
            tm_path = directory / "tm.csv"
            main_frame((2022,)).to_csv(main_path, index=False)
            invalid_trackman = trackman_frame(False).copy()
            invalid_trackman.loc[0, "season"] = 2025
            invalid_trackman.to_csv(tm_path, index=False)
            report, _ = audit.run_from_paths(main_path, tm_path, directory / "output", ROOT)
            self.assertEqual(report["acceptance"]["verdict"], "P_KILL")
            self.assertEqual(report["acceptance"]["kill_reasons"], ["TEMPORAL_CAUSALITY_NOT_PROVEN"])

    def test_valid_run_report_records_determinism_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main_path = directory / "main.csv"
            tm_path = directory / "tm.csv"
            main_frame((2020, 2022, 2024)).to_csv(main_path, index=False)
            trackman_frame(True).to_csv(tm_path, index=False)
            report, _ = audit.run_from_paths(main_path, tm_path, directory / "output", ROOT)
            self.assertEqual(report["acceptance"]["verdict"], "P_PASS")
            self.assertTrue(report["acceptance"]["determinism"]["tested"])
            self.assertTrue(report["acceptance"]["determinism"]["complete_feature_output_equal"])


if __name__ == "__main__":
    unittest.main()
