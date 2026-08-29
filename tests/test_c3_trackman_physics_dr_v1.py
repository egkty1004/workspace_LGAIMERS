import hashlib
import importlib.util
import json
import math
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("physics_dr", ROOT / "scripts" / "c3_trackman_physics_dr_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrackManPhysicsContractTests(unittest.TestCase):
    def setUp(self):
        self.config = MODULE.load_config()

    def _filtered_fixture(self):
        rows = []
        for family_index, family in enumerate(MODULE.FAMILIES):
            for index in range(120):
                rows.append({
                    "season": 2019, "balls_before": 0, "strikes_before": 0,
                    "outs_before": 0, "pitch_type_group": family,
                    "rel_speed": 130.0 + family_index * 10 + index / 100,
                    "spin_rate": 1900.0 + family_index * 200 + index,
                    "induced_vert_break": 10.0 + family_index * 5 + index / 10,
                    "extension": 1.7 + index / 1000,
                    "rel_height": 1.6 + index / 1000,
                    "horz_break": (-1 if family_index == 0 else 1) * (10 + index / 10),
                })
        frame = pd.DataFrame(rows)
        frame.attrs["source_quality_filter_applied"] = True
        return frame

    def test_exact_50_plus_17_contract(self):
        self.assertEqual(50, len(self.config["base_c3_features"]))
        self.assertEqual(17, len(self.config["physics_features"]))
        self.assertEqual(67, len(self.config["base_c3_features"] + self.config["physics_features"]))
        self.assertFalse(any(name.startswith("tm_cf_") for name in self.config["physics_features"]))

    def test_categorical_contract_unchanged(self):
        self.assertEqual(
            ["top_bottom", "game_type", "base_state", "platoon", "count_state", "base_out_state_24"],
            self.config["categorical_features"],
        )

    def test_context_domain(self):
        self.assertEqual((3, 2, 2), MODULE._context_key({"balls_before": 3, "strikes_before": 2, "outs_before": 2}))
        self.assertIsNone(MODULE._context_key({"balls_before": 4, "strikes_before": 2, "outs_before": 2}))
        self.assertIsNone(MODULE._context_key({"balls_before": 1.5, "strikes_before": 1, "outs_before": 1}))

    def test_population_variance_and_stability(self):
        forward = MODULE._summarize(np.array([1.0, 2.0, 3.0, 4.0]))
        reverse = MODULE._summarize(np.array([4.0, 3.0, 2.0, 1.0]))
        self.assertEqual(forward, reverse)
        self.assertEqual(1.25, forward["var"])
        self.assertAlmostEqual(math.sqrt(1.25), forward["std"])

    def test_linear_median_iqr(self):
        stats = MODULE._summarize(np.array([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual(1.5, stats["median"])
        self.assertEqual(1.5, stats["iqr"])

    def test_zero_denominator_missing(self):
        self.assertIsNone(MODULE._safe_ratio(1.0, 0.0))
        self.assertIsNone(MODULE._safe_ratio(1.0, float("nan")))

    def test_lookup_requires_filtered_frame(self):
        frame = self._filtered_fixture()
        frame.attrs.clear()
        with self.assertRaises(MODULE.ContractError):
            MODULE.build_physics_lookup(frame, self.config)

    def test_2019_all_missing(self):
        lookup = MODULE.build_physics_lookup(self._filtered_fixture(), self.config)
        for state in lookup["states"]["2019"].values():
            self.assertEqual([None] * 17, state["values"])
            self.assertEqual(["NO_HISTORY"] * 17, state["levels"])

    def test_strict_prior_season(self):
        frame = self._filtered_fixture()
        lookup = MODULE.build_physics_lookup(frame, self.config)
        self.assertEqual([None] * 17, lookup["states"]["2019"]["0:0:0"]["values"])
        self.assertTrue(any(value is not None for value in lookup["states"]["2020"]["0:0:0"]["values"]))

    def test_family_specific_l0_and_global_fallback(self):
        lookup = MODULE.build_physics_lookup(self._filtered_fixture(), self.config)
        supported = lookup["states"]["2020"]["0:0:0"]
        unseen = lookup["states"]["2020"]["3:2:2"]
        self.assertTrue(all(level == "L0" for level in supported["levels"]))
        self.assertTrue(all(level == "global" for level in unseen["levels"]))
        self.assertAlmostEqual(0.0, unseen["values"][2])

    def test_separation_requires_common_supported_level(self):
        frame = self._filtered_fixture()
        frame = frame.loc[~((frame.pitch_type_group == "offspeed") & (frame.index % 3 != 0))].copy()
        frame.attrs["source_quality_filter_applied"] = True
        lookup = MODULE.build_physics_lookup(frame, self.config)
        state = lookup["states"]["2020"]["0:0:0"]
        self.assertEqual("NO_HISTORY", state["levels"][4])
        self.assertIsNone(state["values"][4])

    def test_other_family_never_in_physics_fixture(self):
        frame = self._filtered_fixture()
        extra = frame.iloc[:120].copy(); extra["pitch_type_group"] = "other"; extra["rel_speed"] = 9999.0
        combined = pd.concat([frame, extra], ignore_index=True); combined.attrs["source_quality_filter_applied"] = True
        a = MODULE.build_physics_lookup(frame, self.config)
        b = MODULE.build_physics_lookup(combined, self.config)
        self.assertEqual(MODULE.canonical_hash(a), MODULE.canonical_hash(b))

    def test_lookup_row_order_deterministic(self):
        frame = self._filtered_fixture()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        shuffled.attrs["source_quality_filter_applied"] = True
        first = MODULE.build_physics_lookup(frame, self.config)
        second = MODULE.build_physics_lookup(shuffled, self.config)
        self.assertEqual(MODULE.canonical_hash(first), MODULE.canonical_hash(second))

    def test_lookup_serialization_replay_parity(self):
        lookup = MODULE.build_physics_lookup(self._filtered_fixture(), self.config)
        replay = json.loads(json.dumps(lookup, sort_keys=True, allow_nan=False))
        raw = pd.DataFrame({
            "season": [2019, 2020], "balls_before": [0, 0], "strikes_before": [0, 0],
            "outs_before": [0, 0], "base_state": ["___", "___"],
            "pitcher_hand": [1, 1], "batter_hand": [2, 2], "top_bottom": ["T", "T"],
            "game_type": ["R", "R"],
        })
        for column in self.config["base_c3_features"]:
            if column not in raw and column not in MODULE.DERIVED:
                raw[column] = 0
        prepared = MODULE.preprocess_c3(raw)
        a = MODULE.apply_lookup(prepared, lookup, self.config)
        b = MODULE.apply_lookup(prepared, replay, self.config)
        np.testing.assert_equal(a[self.config["physics_features"]].to_numpy(), b[self.config["physics_features"]].to_numpy())

    def test_shuffle_chunk_single_row_independence(self):
        lookup = MODULE.build_physics_lookup(self._filtered_fixture(), self.config)
        raw = pd.DataFrame({"season": [2020] * 4, "balls_before": [0, 1, 2, 3],
                            "strikes_before": [0, 1, 2, 0], "outs_before": [0, 1, 2, 0]})
        for column in self.config["base_c3_features"]:
            if column not in raw and column not in MODULE.DERIVED:
                raw[column] = "___" if column == "base_state" else ("R" if column == "game_type" else ("T" if column == "top_bottom" else 1))
        prepared = MODULE.preprocess_c3(raw)
        full = MODULE.apply_lookup(prepared, lookup, self.config)[self.config["physics_features"]]
        shuffled = MODULE.apply_lookup(prepared.iloc[[2, 0, 3, 1]], lookup, self.config)[self.config["physics_features"]]
        np.testing.assert_equal(full.to_numpy(), shuffled.sort_index().to_numpy())
        for index in range(4):
            one = MODULE.apply_lookup(prepared.iloc[[index]], lookup, self.config)[self.config["physics_features"]]
            np.testing.assert_equal(full.iloc[[index]].to_numpy(), one.to_numpy())

    def test_base_out_state_contract(self):
        frame = pd.DataFrame({"base_state": ["___", "123", "bad"], "outs_before": [0, 2, 1]})
        self.assertEqual(["___|0", "123|2", "__MISSING__"], MODULE.base_out_state_24(frame).tolist())

    def test_split_contract(self):
        fit, es = MODULE.split_positions(1_475_092, self.config)
        self.assertEqual(1_401_338, len(fit)); self.assertEqual(73_754, len(es))
        self.assertEqual(0, np.intersect1d(fit, es).size)

    def test_package_has_no_raw_trackman_or_forbidden_features(self):
        source = MODULE.PACKAGE_SCRIPT
        for token in ("trackman_history.csv", "zone_speed", "rel_side", "tagged_pitch_type", "auto_pitch_type", "tm_cf_"):
            self.assertNotIn(token, source)
        self.assertIn("thread_count=6", source)

    def test_package_script_compiles(self):
        compile(MODULE.PACKAGE_SCRIPT, "script.py", "exec")

    def test_source_quality_exclusion_is_row_local(self):
        rows = []
        for season, balls, strikes, outs in ((2022, 0, 0, 0), (2022, 4, 0, 0), (2023, 0, 0, 3)):
            row = {"season": season, "balls_before": balls, "strikes_before": strikes,
                   "outs_before": outs, "pitch_type_group": "fastball"}
            row.update({column: 1.0 for column in MODULE.PHYSICS_SOURCE})
            rows.append(row)
        config = copy.deepcopy(self.config)
        config["trackman"]["raw_rows"] = 3
        config["trackman"]["excluded_rows"] = 2
        config["trackman"]["retained_rows"] = 1
        config["trackman"]["residue_manifest"] = [
            {"season": 2022, "field": "balls_before", "reason": "out_of_domain", "value": "4", "count": 1},
            {"season": 2023, "field": "outs_before", "reason": "out_of_domain", "value": "3", "count": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trackman.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            filtered, summary = MODULE.read_filter_trackman(path, config)
        self.assertEqual(1, len(filtered))
        self.assertEqual(2, summary["excluded_rows"])
        self.assertTrue(filtered.attrs["source_quality_filter_applied"])

    def test_static_contract(self):
        result = MODULE.static_contract()
        self.assertTrue(result["static_pass"])
        self.assertEqual(67, result["feature_count"])

    def test_config_is_runtime_authority(self):
        config = json.loads((ROOT / "configs" / "c3_trackman_physics_dr_v1.json").read_text())
        self.assertEqual(config, MODULE.load_config())
        self.assertEqual(MODULE.FAMILIES, tuple(config["trackman"]["families"]))

    def test_authority_hash_literals(self):
        expected = self.config["authorities"]
        self.assertEqual("10f5ce12e1ed0e0e5f2f6bf7ef3cb5b87fca893d1339d8f64914488afda1402b", expected["eda_sha256"])
        self.assertEqual("f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9", expected["trackman_source_sha256"])


if __name__ == "__main__":
    unittest.main()
