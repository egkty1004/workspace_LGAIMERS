import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_trackman_physics_eda_v1.py"
EXPECTED_RUNNER_SHA256 = "c7009bf6de36ff28e981677f50e3ef4f96ae31d7d93e5637df8339566fb1b5ff"


def load_runner():
    spec = importlib.util.spec_from_file_location("trackman_physics_eda_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EDA = load_runner()


class TrackmanPhysicsEdaV1Tests(unittest.TestCase):
    def test_runner_is_the_pinned_byte_identical_evidence_script(self):
        self.assertEqual(EDA.sha(RUNNER), EXPECTED_RUNNER_SHA256)

    def test_projection_is_aggregate_physics_scope_without_target_or_ids(self):
        self.assertEqual(EDA.USECOLS, EDA.META + EDA.PHYS)
        forbidden = {
            "control_success",
            "row_id",
            "pitcher_id",
            "pitcher_trackman_id",
            "player_id",
            "team_id",
            "game_id",
            "pitch_id",
        }
        self.assertTrue(forbidden.isdisjoint(EDA.USECOLS))

    def test_clean_coerces_tokens_and_excludes_nonfinite_values(self):
        cleaned, finite = EDA.clean(pd.Series(["1", "  ", np.inf, -np.inf, None, "0.5"]))
        self.assertEqual(cleaned.dtype, np.dtype("float64"))
        self.assertEqual(finite.tolist(), [True, False, False, False, False, True])
        self.assertEqual(cleaned.iloc[0], 1.0)
        self.assertEqual(cleaned.iloc[5], 0.5)

    def test_core_reports_aggregate_counts_not_rows(self):
        result = EDA.core(pd.Series([1.0, np.nan, np.inf, -np.inf]), include_quantiles=False)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["finite"], 1)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["posinf"], 1)
        self.assertEqual(result["neginf"], 1)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 1.0)
        self.assertNotIn("rows", result)

    def test_core_is_row_order_invariant(self):
        values = pd.Series([0.1, 0.2, np.nan, 0.3, np.inf])
        self.assertEqual(EDA.core(values), EDA.core(values.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)))

    def test_season_core_has_only_season_aggregates(self):
        frame = pd.DataFrame({"season": [2020, 2020, 2021], "rel_speed": [90.0, 91.0, 88.0]})
        result = EDA.season_core(frame, "rel_speed")
        self.assertEqual(sorted(result), ["2020", "2021"])
        self.assertEqual(result["2020"]["total"], 2)
        self.assertEqual(result["2021"]["finite"], 1)
        self.assertNotIn("quantiles", result["2020"])

    def test_outliers_contains_counts_and_no_row_material(self):
        result = EDA.outliers(pd.Series([90.0, 91.0, 92.0, 200.0]))
        self.assertIn("iqr_extreme_count", result)
        self.assertIn("mad_extreme_count", result)
        self.assertIn("lowest_values", result)
        self.assertIn("highest_values", result)
        self.assertNotIn("row_id", json.dumps(result))

    def test_category_summary_is_aggregate_only(self):
        frame = pd.DataFrame({"pitch_type_group": ["fastball", "other", "fastball", None]})
        result = EDA.category_summary(frame, "pitch_type_group")
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["unique"], 2)
        self.assertEqual(result["counts"]["fastball"], 2)
        self.assertEqual(result["counts"]["other"], 1)
        self.assertNotIn("row_id", result)

    def test_family_stats_is_group_aggregate(self):
        frame = pd.DataFrame(
            {
                "season": [2020, 2020],
                "pitch_type_group": ["fastball", "other"],
                **{column: [1.0, 2.0] for column in EDA.PHYS},
            }
        )
        result = EDA.family_stats(frame)
        self.assertEqual(sorted(result), ["2020|fastball", "2020|other"])
        self.assertEqual(result["2020|fastball"]["rows"], 1)
        self.assertEqual(result["2020|other"]["physics"]["rel_speed"]["finite"], 1)
        self.assertNotIn("row_id", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
