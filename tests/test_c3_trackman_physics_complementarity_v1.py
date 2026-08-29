import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "complementarity", ROOT / "scripts" / "c3_trackman_physics_complementarity_v1.py")
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class ComplementarityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = M.load_config()

    def test_exact_contract(self):
        self.assertEqual(50, len(self.cfg["base_c3_features"]))
        self.assertEqual(17, len(self.cfg["physics_features"]))
        self.assertEqual(67, len(self.cfg["base_c3_features"] + self.cfg["physics_features"]))
        self.assertEqual(4, M.static_contract()["fit_count"])

    def test_split_is_deterministic_and_matched(self):
        source = np.arange(1000, dtype=np.int64)
        first = M.split_positions(source, self.cfg)
        second = M.split_positions(source, self.cfg)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(950, len(first[0]))
        self.assertEqual(50, len(first[1]))

    def test_only_historical_origins(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["origins"]["r2022"]["train_rows"] = 3
        cfg["origins"]["r2022"]["validation_rows"] = 1
        frame = pd.DataFrame({"season": [2019, 2020, 2021, 2022, 2023, 2024],
                              "game_type": ["R"] * 6, "row_id": list("abcdef")})
        train, outer = M.origin_positions(frame, "r2022", cfg)
        self.assertEqual([0, 1, 2], train.tolist())
        self.assertEqual([3], outer.tolist())
        with self.assertRaises(M.ContractError):
            M.origin_positions(frame, "r2024", cfg)

    def test_seal_required_before_outer_labels(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["origins"]["r2022"]["train_rows"] = 1
        cfg["origins"]["r2022"]["validation_rows"] = 1
        frame = pd.DataFrame({"row_id": ["a", "b"], "season": [2021, 2022],
                              "game_type": ["R", "R"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"row_id": ["a", "b"], "season": [2021, 2022],
                          "game_type": ["R", "R"], "control_success": [0, 1]}).to_csv(path, index=False)
            with self.assertRaises(M.ContractError):
                M.read_outer_labels(path, frame, object(), cfg)
            sealed = M.seal_predictions("r2022", frame, [1], [0.2], [0.3], cfg)
            np.testing.assert_array_equal([1.0], M.read_outer_labels(path, frame, sealed, cfg))

    def test_analytic_formula(self):
        p0 = np.array([0.2, 0.7, 0.3, 0.6])
        p1 = np.array([0.1, 0.8, 0.2, 0.7])
        y = np.array([0., 1., 0., 1.])
        result = M.analytic_origin(p0, p1, y)
        d, r = p1-p0, y-p0
        self.assertAlmostEqual(np.mean(r*d), result["A"])
        self.assertAlmostEqual(np.mean(d*d), result["D"])
        self.assertAlmostEqual(np.mean(r*d)/np.mean(d*d), result["w_star_unclipped"])

    def test_stop_when_any_origin_A_nonpositive(self):
        vectors = {name: (np.array([0.2, 0.8]), np.array([0.3, 0.7]), np.array([0., 1.]))
                   for name in M.ORIGINS}
        origins = {name: M.analytic_origin(*vectors[name]) for name in M.ORIGINS}
        self.assertEqual("COMPLEMENTARITY_STOP", M.finalize(origins, vectors)["result"])

    def test_shared_weight_improves_both(self):
        vectors = {
            "r2022": (np.array([0.4, 0.6]), np.array([0.2, 0.8]), np.array([0., 1.])),
            "r2023": (np.array([0.35, 0.65]), np.array([0.25, 0.75]), np.array([0., 1.])),
        }
        origins = {name: M.analytic_origin(*vectors[name]) for name in M.ORIGINS}
        final = M.finalize(origins, vectors)
        self.assertEqual("COMPLEMENTARITY_GO", final["result"])
        self.assertGreater(final["w_shared"], 0)
        self.assertTrue(all(item["passed"] for item in final["direct"].values()))

    def test_feature_projection_excludes_target(self):
        self.assertNotIn(M.TARGET, M.raw_feature_columns(self.cfg))

    def test_lookup_hash_is_pinned(self):
        self.assertEqual("333da608fea0c3bd9e314262427d07b1d1385c5314c402d810a07e821efb55e2",
                         self.cfg["authorities"]["physics_lookup_sha256"])

    def test_config_is_runtime_authority(self):
        raw = json.loads((ROOT / "configs" / "c3_trackman_physics_complementarity_v1.json").read_text())
        self.assertEqual(raw, M.load_config())


if __name__ == "__main__":
    unittest.main()
