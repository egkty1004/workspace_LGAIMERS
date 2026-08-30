from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_v93_bounded_trackman_residual_diagnostic_v0 as runner  # noqa: E402


class BoundedResidualDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = runner.load_config()

    @staticmethod
    def identity_frame(year: int = 2022, rows: int = 6) -> pd.DataFrame:
        return pd.DataFrame({
            "row_id": [f"row-{index}" for index in range(rows)],
            "season": [year] * rows,
            "game_type": ["R"] * rows,
        })

    @staticmethod
    def alpha_cfg(**changes):
        cfg = copy.deepcopy(runner.load_config())
        cfg["alpha_analysis"].update(changes)
        return cfg

    def test_logit_residual_and_alpha_zero_deployed_parity(self):
        p_c3 = np.asarray([0.25, 0.5, 0.75])
        p_phys = np.asarray([0.5, 0.75, 0.9])
        expected = np.log(p_phys) - np.log1p(-p_phys) - np.log(p_c3) + np.log1p(-p_c3)
        actual = runner.logit_probability(p_phys) - runner.logit_probability(p_c3)
        np.testing.assert_allclose(actual, expected)
        z = np.asarray([-2.0, 0.0, 2.0])
        baseline = runner.deployed_probs(z, actual, 0.0, self.cfg)
        expected_baseline = np.clip(runner.stable_sigmoid(z + self.cfg["v93"]["c_logit"]), 0.3, 0.7)
        np.testing.assert_array_equal(baseline, expected_baseline)

    def test_first_disconnected_component_is_used_and_later_component_ignored(self):
        def g(alpha: float) -> float:
            if alpha == 0.0:
                return 0.0
            if alpha < 0.25:
                return alpha * (0.25 - alpha)
            if alpha < 0.70:
                return -0.01
            return 0.02

        result = runner.characterize_positive_component(g, self.cfg)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["later_positive_components_ignored"])
        self.assertAlmostEqual(result["U"], 0.25, delta=1e-11)

    def test_narrow_first_component_uses_positive_derivative_anchor(self):
        result = runner.characterize_positive_component(
            lambda alpha: alpha * (0.0005 - alpha) if alpha < 0.0005 else -0.01,
            self.cfg,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["first_nonpositive_grid_index"], 1)
        self.assertAlmostEqual(result["U"], 0.0005, delta=1e-8)

    def test_right_derivative_failure_fails_closed(self):
        result = runner.characterize_positive_component(lambda alpha: -alpha, self.cfg)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["reason"], "right_derivative_not_improvement_directed")

    def test_unbracketable_boundary_fails_closed(self):
        cfg = self.alpha_cfg(boundary_max_iterations=1)
        result = runner.characterize_positive_component(lambda alpha: alpha * (0.7 - alpha), cfg)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["reason"], "boundary_not_refined")

    def test_boundary_is_refined_deterministically(self):
        result = runner.characterize_positive_component(lambda alpha: alpha * (0.31 - alpha), self.cfg)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["boundary_established"])
        self.assertAlmostEqual(result["U"], 0.31, delta=1e-11)
        self.assertLessEqual(result["boundary_width"], 1e-12)

    def test_component_positive_through_domain_uses_frozen_upper_endpoint(self):
        result = runner.characterize_positive_component(lambda alpha: alpha, self.cfg)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["boundary_established"])
        self.assertEqual(result["boundary_kind"], "domain_upper_endpoint")
        self.assertEqual(result["U"], 1.0)

    def test_two_origin_conjunction_and_shared_alpha(self):
        origin_results = {
            name: {"analysis": {"status": "PASS", "reason": None, "U": upper}}
            for name, upper in (("r2022", 0.4), ("r2023", 0.2))
        }
        vectors = {
            name: (np.zeros(20), np.ones(20), np.ones(20))
            for name in runner.ORIGINS
        }
        result = runner.finalize_origins(origin_results, vectors, self.cfg)
        self.assertEqual(result["result"], "BOUNDED_SIGNAL_PRESENT")
        self.assertEqual(result["shared_interval"]["upper"], 0.2)
        self.assertEqual(result["alpha_diagnostic"], 0.1)
        self.assertTrue(all(item["passed"] for item in result["direct"].values()))

        origin_results["r2023"]["analysis"] = {
            "status": "FAIL", "reason": "right_derivative_not_improvement_directed"
        }
        stopped = runner.finalize_origins(origin_results, vectors, self.cfg)
        self.assertEqual(stopped["result"], "BOUNDED_SIGNAL_ABSENT")
        self.assertIsNone(stopped["alpha_diagnostic"])

    def test_exact_origin_and_bounded_position_scope(self):
        frame = self.identity_frame(rows=30000)
        full = runner.origin_positions(frame, "r2022")
        bounded = runner.bounded_positions(full, self.cfg)
        self.assertEqual(len(full), 30000)
        self.assertEqual(full[:8].tolist(), list(range(8)))
        self.assertEqual(bounded.tolist(), list(range(30000)))
        with self.assertRaises(runner.ContractError):
            runner.origin_positions(frame, "r2024")

        cfg = copy.deepcopy(self.cfg)
        cfg["matched_predictions"]["r2022"].update({
            "full_origin_rows": len(full),
            "full_position_hash": runner.array_hash(full),
            "full_row_id_hash": runner.canonical_hash(frame["row_id"].tolist()),
        })
        checked = runner.validate_origin_scope(frame, "r2022", full, cfg)
        self.assertEqual(checked["full_rows"], 30000)

    def test_sealed_prediction_row_parity_and_label_firewall(self):
        frame = self.identity_frame(rows=4)
        positions = np.arange(4, dtype=np.int64)
        sealed = runner.seal_residual_inputs(
            "r2022", frame, positions, np.zeros(4),
            np.full(4, 0.4), np.full(4, 0.6),
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "synthetic.csv"
            frame.assign(control_success=[0, 1, 1, 0]).to_csv(csv_path, index=False)
            with self.assertRaises(runner.ContractError):
                runner.read_outer_labels(csv_path, frame, None)
            target = runner.read_outer_labels(csv_path, frame, sealed)
        np.testing.assert_array_equal(target, [0, 1, 1, 0])
        with self.assertRaises(runner.ContractError):
            runner.seal_residual_inputs("r2022", frame, positions[:-1], np.zeros(4),
                                        np.full(4, 0.4), np.full(4, 0.6))

    def test_identity_projection_is_target_free_and_scope_flags_are_closed(self):
        self.assertNotIn(runner.TARGET, runner.IDENTITY_COLUMNS)
        self.assertFalse(self.cfg["scope"]["primary_target_access"])
        self.assertFalse(self.cfg["scope"]["r2024_target_access"])
        self.assertFalse(self.cfg["scope"]["test_access"])
        self.assertFalse(self.cfg["scope"]["public_access"])
        self.assertFalse(self.cfg["scope"]["external_information_access"])
        self.assertEqual(self.cfg["scope"]["model_fits"], 0)

    def test_config_runtime_divergence_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            altered = copy.deepcopy(self.cfg)
            altered["alpha_analysis"]["grid_points"] = 1000
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(runner.ContractError):
                runner.load_config(path)

    def test_complementarity_report_authority_drift_fails_closed(self):
        report = {
            "canonical_report_sha256": "canonical",
            "runner_sha256": "runner",
            "origins": {
                origin: {
                    "predictions_literal_sha256": self.cfg["matched_predictions"][origin]["sha256"],
                    "validation_rows": self.cfg["matched_predictions"][origin]["full_origin_rows"],
                    "validation_position_hash": self.cfg["matched_predictions"][origin]["full_position_hash"],
                    "validation_row_id_hash": self.cfg["matched_predictions"][origin]["full_row_id_hash"],
                    "row_parity": True,
                    "outer_labels_read_after_both_predictions_sealed": True,
                }
                for origin in runner.ORIGINS
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complementarity.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            cfg = copy.deepcopy(self.cfg)
            cfg["complementarity_report"].update({
                "path": str(path),
                "sha256": runner.sha256_file(path),
                "canonical_report_sha256": "canonical",
                "runner_sha256": "runner",
            })
            self.assertEqual(runner.verify_complementarity_report(path, cfg)["sha256"],
                             runner.sha256_file(path))
            path.write_text(json.dumps({**report, "changed": True}), encoding="utf-8")
            with self.assertRaises(runner.ContractError):
                runner.verify_complementarity_report(path, cfg)

    def test_output_inside_repository_fails(self):
        with self.assertRaises(runner.ContractError):
            runner._outside_repo(ROOT / "tmp-output", ROOT)

    def test_report_hash_and_markdown_are_deterministic(self):
        source = {"columns": list(runner.IDENTITY_COLUMNS), "rows": 4,
                  "identity_projection_hash": "source", "target_materialized_before_seal": False}
        origins = {name: {"rows": 2, "analysis": {"status": "FAIL",
                                                      "reason": "right_derivative_not_improvement_directed"},
                          "v93_deployed_brier": 0.25}
                   for name in runner.ORIGINS}
        final = {"result": "BOUNDED_SIGNAL_ABSENT", "shared_interval": None,
                 "alpha_diagnostic": None, "origin_analysis_failures": {}, "direct": {}}
        first = runner.build_report(self.cfg, "git", "runner", "config", source,
                                    {"authority": {"sha256": "a"}}, origins, final)
        second = runner.build_report(self.cfg, "git", "runner", "config", source,
                                     {"authority": {"sha256": "a"}}, origins, final)
        self.assertEqual(first, second)
        self.assertEqual(first["canonical_report_sha256"], runner.canonical_hash({
            key: value for key, value in first.items() if key != "canonical_report_sha256"
        }))
        self.assertEqual(runner.render_markdown(first), runner.render_markdown(second))

    def test_static_contract_declares_zero_fits(self):
        result = runner.static_contract()
        self.assertTrue(result["static_pass"])
        self.assertEqual(result["model_fits"], 0)
        self.assertTrue(result["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
