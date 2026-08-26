from __future__ import annotations

import copy
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_v93_trackman_context_prior_catboost_v1 as package
from scripts import experiment_v93_trackman_context_prior_catboost_v1 as experiment


ROOT = Path(__file__).resolve().parents[1]


def _payload(probabilities=(0.5, 0.2, 0.2, 0.1), support=200):
    if support == 0:
        return {
            "probabilities": None, "support": 0, "entropy": None,
            "context_vs_global_tv": None,
            "family_counts": {name: 0 for name in ("fastball", "breaking", "offspeed", "other")},
        }
    return {
        "probabilities": list(probabilities), "support": support,
        "entropy": 0.8, "context_vs_global_tv": 0.1,
        "family_counts": {
            "fastball": int(support * probabilities[0]),
            "breaking": int(support * probabilities[1]),
            "offspeed": int(support * probabilities[2]),
            "other": support - sum(int(support * value) for value in probabilities[:3]),
        },
    }


def _lookup():
    seasons = {}
    for season in range(2019, 2026):
        if season == 2019:
            global_payload = _payload(support=0)
            l0 = {}
            l1 = {}
        else:
            global_payload = _payload((0.4, 0.3, 0.2, 0.1), 1000 + season)
            l0 = {"1:1:1": _payload((0.6, 0.2, 0.1, 0.1), 120)}
            l1 = {"2:1": _payload((0.2, 0.4, 0.3, 0.1), 150)}
        seasons[str(season)] = {"L0": l0, "L1": l1, "global": {"GLOBAL": global_payload}}
    return {
        "contract_version": "aimers9-trackman-crosswalk-free-context-priors-v2",
        "context_key_encoding": "colon-separated integers",
        "support_threshold": 100,
        "seasons": seasons,
    }


def _base_frame(seasons=(2021, 2022, 2023)):
    config = experiment.load_config()
    rows = []
    for index, season in enumerate(seasons):
        row = {}
        for column in config["base_features"]:
            if column in experiment.DERIVED_BASE:
                continue
            row[column] = 0.25
        row.update({
            experiment.ROW_ID: f"row-{index}", "season": season,
            "game_month": 4, "game_dayofweek": 2, "inning": 5,
            "top_bottom": "T", "game_type": "R", "balls_before": 1,
            "strikes_before": 1, "outs_before": 1, "base_state": "000",
            "pitcher_hand": 0, "batter_hand": 1, "pitcher_id": 100 + index,
            "batter_id": 200 + index, "pitcher_team_id": 1, "batter_team_id": 2,
        })
        rows.append(row)
    return experiment.preprocess_base(pd.DataFrame(rows))


class ConfigAndAuthorityTests(unittest.TestCase):
    def test_runtime_config_is_exact_49_plus_7_contract(self):
        config = experiment.load_config()
        contract = experiment.assert_feature_contracts(config)
        self.assertEqual(49, len(contract["C0_features"]))
        self.assertEqual(56, len(contract["C1_features"]))
        self.assertEqual(contract["C0_features"] + contract["only_difference"], contract["C1_features"])
        self.assertNotIn("__backoff_level", contract["C1_features"])

    def test_config_divergence_fails_closed(self):
        config = experiment.load_config()
        config["catboost"]["es_split"]["algorithm"] = "different"
        with self.assertRaises(experiment.ExperimentError):
            experiment.validate_config(config)

    def test_v93_blend_and_inference_thread_contracts_fail_closed(self):
        config = experiment.load_config()
        config["composite"]["lambda_ftt"] = 0.2
        with self.assertRaises(experiment.ExperimentError):
            experiment.validate_config(config)
        config = experiment.load_config()
        config["package"]["inference_threads_max"] = 7
        with self.assertRaises(experiment.ExperimentError):
            experiment.validate_config(config)

    def test_static_authorities_match_merged_repository(self):
        result = experiment.static_contract(ROOT)
        self.assertTrue(result["static_pass"])
        self.assertFalse(result["model_training_or_scoring"])
        self.assertFalse(result["target_2024_access"])

    def test_catboost_params_only_add_seed(self):
        config = experiment.load_config()
        first = experiment.catboost_params(config, 42)
        second = experiment.catboost_params(config, 43)
        self.assertEqual(42, first.pop("random_seed"))
        self.assertEqual(43, second.pop("random_seed"))
        self.assertEqual(first, second)

    def test_candidate_prep_freezes_model_split_and_lookup_contract(self):
        config = experiment.load_config()
        prep = experiment.candidate_prep_contract(config)
        self.assertEqual(56, len(prep["features"]))
        self.assertEqual(config["catboost"]["params"], prep["params"])
        self.assertEqual(12345, prep["es_seed"])
        self.assertTrue(prep["no_post_es_refit"])
        self.assertEqual(config["authority"]["trackman_lookup_sha256"], prep["lookup_sha256"])

    def test_split_is_deterministic_choice_and_complement(self):
        config = experiment.load_config()
        positions = np.arange(200, dtype=np.int64)
        fit1, es1 = experiment.split_positions(positions, config)
        fit2, es2 = experiment.split_positions(positions, config)
        np.testing.assert_array_equal(fit1, fit2)
        np.testing.assert_array_equal(es1, es2)
        self.assertEqual(10, len(es1))
        self.assertEqual(set(positions), set(fit1) | set(es1))
        self.assertFalse(set(fit1) & set(es1))


class FeatureFactoryTests(unittest.TestCase):
    def test_c1_adds_only_seven_numeric_trackman_features(self):
        config = experiment.load_config()
        base = _base_frame((2022,))
        result = experiment.apply_trackman_features(base, _lookup(), config)
        self.assertEqual(list(config["base_features"]) + list(config["trackman_features"]),
                         list(result[list(config["base_features"]) + list(config["trackman_features"])].columns))
        self.assertTrue(all(str(result[column].dtype) == "float32" for column in config["trackman_features"]))
        self.assertEqual(120.0, float(result.iloc[0]["tm_cf_support"]))

    def test_lookup_uses_row_season_and_future_poison_does_not_change_2022(self):
        config = experiment.load_config()
        base = _base_frame((2022, 2024))
        first = experiment.apply_trackman_features(base, _lookup(), config)
        poisoned = _lookup()
        poisoned["seasons"]["2024"]["L0"]["1:1:1"] = _payload((0.1, 0.1, 0.1, 0.7), 999)
        second = experiment.apply_trackman_features(base, poisoned, config)
        pd.testing.assert_series_equal(first.iloc[0][config["trackman_features"]],
                                       second.iloc[0][config["trackman_features"]])
        self.assertFalse(first.iloc[1][config["trackman_features"]].equals(
            second.iloc[1][config["trackman_features"]]))

    def test_2019_no_history_is_missing_with_zero_support(self):
        config = experiment.load_config()
        result = experiment.apply_trackman_features(_base_frame((2019,)), _lookup(), config).iloc[0]
        self.assertEqual(0.0, float(result["tm_cf_support"]))
        for column in config["trackman_features"]:
            if column != "tm_cf_support":
                self.assertTrue(pd.isna(result[column]))

    def test_frozen_l0_l1_global_backoff_and_selected_support(self):
        config = experiment.load_config()
        base = _base_frame((2022, 2022, 2022))
        base.loc[base.index[1], ["balls_before", "strikes_before", "outs_before"]] = [2, 1, 0]
        base.loc[base.index[2], ["balls_before", "strikes_before", "outs_before"]] = [3, 2, 2]
        result = experiment.apply_trackman_features(base, _lookup(), config)
        self.assertEqual([120.0, 150.0, 3022.0], result["tm_cf_support"].tolist())

    def test_invalid_main_context_fails_closed(self):
        config = experiment.load_config()
        base = _base_frame((2022,))
        base.loc[base.index[0], "balls_before"] = 4
        with self.assertRaises(Exception):
            experiment.apply_trackman_features(base, _lookup(), config)

    def test_feature_generation_is_row_independent(self):
        config = experiment.load_config()
        base = _base_frame((2022, 2023, 2024))
        together = experiment.apply_trackman_features(base, _lookup(), config)
        for index in base.index:
            one = experiment.apply_trackman_features(base.loc[[index]], _lookup(), config)
            pd.testing.assert_series_equal(together.loc[index, config["trackman_features"]],
                                           one.loc[index, config["trackman_features"]])

    def test_raw_trackman_is_not_an_input_to_model_feature_factory(self):
        parameters = experiment.apply_trackman_features.__annotations__
        self.assertIn("aggregate_lookup", parameters)
        self.assertNotIn("trackman_history", parameters)


class GeometryAndFirewallTests(unittest.TestCase):
    def test_origin_geometry_and_bounded_first_30000(self):
        frame = _base_frame((*([2021] * 2), *([2022] * 30005), 2023))
        train, validation = experiment.origin_positions(frame, "r2022")
        self.assertEqual(2, len(train))
        self.assertEqual(30005, len(validation))
        bounded = experiment.bounded_positions(validation)
        self.assertEqual(30000, len(bounded))
        np.testing.assert_array_equal(validation[:30000], bounded)

    def test_only_r2022_r2023_origins_are_allowed(self):
        with self.assertRaises(experiment.ExperimentError):
            experiment.origin_positions(_base_frame(), "r2024")

    def test_both_arm_logits_required_before_outer_seal(self):
        frame = _base_frame((2021, 2022, 2022))
        _, positions = experiment.origin_positions(frame, "r2022")
        with self.assertRaises(experiment.ExperimentError):
            experiment.seal_outer_logits("r2022", frame, positions, [0.0, 0.0], [0.0])
        sealed = experiment.seal_outer_logits("r2022", frame, positions, [0.0, 0.1], [0.2, 0.3])
        self.assertEqual(2, len(sealed.c0_logits))
        self.assertEqual(2, len(sealed.c1_logits))

    def test_target_never_enters_feature_projection(self):
        config = experiment.load_config()
        self.assertNotIn(experiment.TARGET, [experiment.ROW_ID, *experiment._raw_feature_columns(config)])
        source = (ROOT / "scripts" / "experiment_v93_trackman_context_prior_catboost_v1.py").read_text()
        self.assertIn("read_outer_labels(args.train_csv, frame, sealed)", source)
        self.assertLess(source.index("sealed = seal_outer_logits"), source.index("read_outer_labels(args.train_csv, frame, sealed)"))

    def test_matched_contract_hashes_same_rows_targets_and_parent(self):
        config = experiment.load_config()
        rows = np.arange(20)
        fit, es = experiment.split_positions(rows, config)
        result = experiment.assert_matched_training_contract(config, rows, np.zeros(20), fit, es)
        self.assertTrue(result["passed"])
        self.assertEqual(config["trackman_features"], result["allowed_difference"])
        self.assertIn("target_hash", result["identical_C0_C1"])


class ScoringAndGateTests(unittest.TestCase):
    def setUp(self):
        self.config = experiment.load_config()

    def test_replacement_algebra_preserves_b0_when_cat_leg_same(self):
        b0 = np.array([0.1, -0.2])
        original = np.array([0.4, 0.5])
        result = experiment.replace_catboost_leg(b0, original, original, self.config)
        np.testing.assert_array_equal(b0, result)

    def test_replacement_algebra_changes_only_fixed_cat_weight(self):
        b0 = np.array([0.0, 0.0])
        old = np.array([0.0, 0.0])
        new = np.array([1.0, -1.0])
        expected = self.config["composite"]["lambda_cat"] * new
        np.testing.assert_allclose(expected, experiment.replace_catboost_leg(b0, new, old, self.config))

    def test_control_reproduction_exact_and_nonexact_names(self):
        exact = experiment.control_reproduction_diagnostics([0.1, 0.2], [0.1, 0.2])
        self.assertEqual("exact v93 CatBoost control", exact["control_label"])
        self.assertEqual(exact["original_logit_sha256"], exact["reconstructed_c0_logit_sha256"])
        self.assertEqual(exact["original_raw_probability_sha256"],
                         exact["reconstructed_c0_raw_probability_sha256"])
        nonexact = experiment.control_reproduction_diagnostics([0.1, 0.2], [0.1, 0.21])
        self.assertEqual("reconstructed v93-style matched control", nonexact["control_label"])
        self.assertGreater(nonexact["rmse"], 0.0)

    def test_origin_gate_requires_raw_c1_and_both_composite_comparators(self):
        y = np.ones(5)
        result = experiment.evaluate_origin("r2022", y, np.zeros(5), np.zeros(5),
                                            np.zeros(5), np.ones(5), self.config)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertIn("deployed_mean_shift_C1_vs_B0_original_v93", result)
        self.assertGreater(result["deployed_mean_shift_C1_vs_B0_original_v93"], 0.0)
        self.assertEqual(result["deployed_mean_shift_C1_vs_B0_original_v93"],
                         result["deployed_abs_mean_shift_C1_vs_B0_original_v93"])

    def test_origin_gate_fails_when_c1_does_not_beat_original_b0(self):
        y = np.ones(5)
        result = experiment.evaluate_origin("r2022", y, np.full(5, 2.0), np.zeros(5),
                                            np.zeros(5), np.ones(5), self.config)
        self.assertFalse(result["checks"]["composite_c1_brier_lt_b0_original_v93"])
        self.assertFalse(result["passed"])

    def test_final_gate_is_conjunctive_without_origin_compensation(self):
        self.assertEqual("PACKAGE_GO", experiment.final_screen_verdict({
            "r2022": {"passed": True}, "r2023": {"passed": True}}))
        self.assertEqual("PACKAGE_STOP", experiment.final_screen_verdict({
            "r2022": {"passed": True}, "r2023": {"passed": False}}))
        with self.assertRaises(experiment.ExperimentError):
            experiment.final_screen_verdict({"r2022": {"passed": True}})

    def test_canonical_report_hash_is_key_order_independent(self):
        left = {"origin": "r2022", "values": {"C0": 1, "C1": 2}}
        right = {"values": {"C1": 2, "C0": 1}, "origin": "r2022"}
        self.assertEqual(experiment.canonical_report_hash(left),
                         experiment.canonical_report_hash(right))


class PackageBuilderTests(unittest.TestCase):
    def _original_script(self):
        return '''import os\nimport pickle\nimport time\n\nCAT_THREADS = 6\n\n\ndef main():\n    feats = common.get_feature_cols(test.columns)\n\n    common.preprocess_for_submission(test)\n    X = test[feats].copy()\n    print(f"features: {len(feats)}", flush=True)\n    z_lgb += common.logit(bst.predict(X))\n    z_cat = predict_cat_z(test, feats, cats, MODEL_DIR, SEEDS)\n'''

    def test_rendered_package_separates_fixed_and_catboost_frames(self):
        rendered = package.render_candidate_script(self._original_script())
        self.assertIn("X = test[base_feats].copy()", rendered)
        self.assertIn("z_cat = predict_cat_z(cat_frame, cat_feats", rendered)
        self.assertIn("z_lgb += common.logit(bst.predict(X))", rendered)
        self.assertNotIn("trackman_history.csv", rendered)
        self.assertNotIn("__backoff_level", rendered)
        self.assertIn("CAT_THREADS = 6", rendered)

    def test_rendered_package_uses_pinned_lookup_and_no_test_aggregation(self):
        rendered = package.render_candidate_script(self._original_script())
        self.assertIn("27747c22e0ff25e86040f5825667f8d9c0b8d7e840037ad9df87072781160515", rendered)
        self.assertNotIn("test.groupby(", rendered)
        self.assertNotIn("test.merge(", rendered)

    def test_deterministic_zip_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "script.py").write_text("print('ok')\n")
            (root / "model").mkdir()
            (root / "model" / "x.bin").write_bytes(b"asset")
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            package.deterministic_zip(root, first)
            package.deterministic_zip(root, second)
            self.assertEqual(experiment.sha256_file(first), experiment.sha256_file(second))

    def test_deterministic_zip_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            output = Path(directory) / "out.zip"
            output.write_bytes(b"existing")
            with self.assertRaises(package.PackageError):
                package.deterministic_zip(root, output)

    def test_fixed_asset_hash_guard_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_bytes(b"fixed")
            (root / "model").mkdir()
            (root / "model" / "lgb.txt").write_bytes(b"model")
            expected = {
                "source_files": {"requirements.txt": experiment.sha256_file(root / "requirements.txt")},
                "model_files": {"lgb.txt": experiment.sha256_file(root / "model" / "lgb.txt")},
            }
            self.assertEqual(2, len(package.fixed_asset_hashes(root, expected)))
            (root / "model" / "lgb.txt").write_bytes(b"drift")
            with self.assertRaises(package.PackageError):
                package.fixed_asset_hashes(root, expected)

    def test_candidate_artifacts_require_exact_models_and_56_feature_prep(self):
        config = copy.deepcopy(experiment.load_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            for name in package.CATBOOST_MODEL_NAMES:
                (models / name).write_bytes(name.encode())
            lookup = root / "lookup.json"
            lookup.write_text("{}\n")
            config["authority"]["trackman_lookup_sha256"] = experiment.sha256_file(lookup)
            prep = root / "prep.pkl"
            prep.write_bytes(pickle.dumps(experiment.candidate_prep_contract(config)))
            result = package.validate_candidate_artifacts(models, prep, lookup, config)
            self.assertEqual(56, result["feature_count"])
            (models / "extra.bin").write_bytes(b"x")
            with self.assertRaises(package.PackageError):
                package.validate_candidate_artifacts(models, prep, lookup, config)

    def test_candidate_prep_parameter_drift_fails_closed(self):
        config = copy.deepcopy(experiment.load_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            for name in package.CATBOOST_MODEL_NAMES:
                (models / name).write_bytes(name.encode())
            lookup = root / "lookup.json"
            lookup.write_text("{}\n")
            config["authority"]["trackman_lookup_sha256"] = experiment.sha256_file(lookup)
            payload = experiment.candidate_prep_contract(config)
            payload["params"]["depth"] = 7
            prep = root / "prep.pkl"
            prep.write_bytes(pickle.dumps(payload))
            with self.assertRaises(package.PackageError):
                package.validate_candidate_artifacts(models, prep, lookup, config)


if __name__ == "__main__":
    unittest.main()
