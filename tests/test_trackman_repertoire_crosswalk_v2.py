import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import audit_trackman_repertoire_crosswalk_v2 as audit  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def main_rows(*, include_target=False):
    rows = []
    for pitcher, hand, offset in (("m1", "L", 0.0), ("m2", "R", 0.1)):
        for season, n in ((2019, 10), (2020, 20), (2021, 30), (2022, 40), (2023, 50), (2024, 60)):
            base = 0.60 + offset
            rows.append({
                "row_id": f"{pitcher}-{season}", "season": season, "game_type": "R",
                "game_month": 4, "game_dayofweek": 1, "inning": 2,
                "top_bottom": "T", "balls_before": 1, "strikes_before": 2,
                "outs_before": 1, "base_state": "___", "pitcher_id": pitcher,
                "pitcher_hand": hand, "pitcher_team_id": "tm", "batter_hand": "R",
                "asof_pitcher_pitchmix_n": n,
                "asof_pitcher_fastball_rate": base,
                "asof_pitcher_breaking_rate": 0.25,
                "asof_pitcher_offspeed_rate": 0.15 - offset,
            })
    frame = pd.DataFrame(rows)
    if include_target:
        frame["control_success"] = [0] * len(frame)
    return frame


def trackman_rows():
    rows = []
    for pitcher, hand, offset in (("t1", "L", 0.0), ("t2", "R", 0.1)):
        for season in (2019, 2020, 2021, 2022, 2023, 2024):
            shares = [0.60 + offset, 0.25, 0.15 - offset]
            for index, (family, count) in enumerate(zip(audit.FAMILIES, (60, 25, 15))):
                rows.extend({
                    "trackman_id": f"{pitcher}-{season}-{index}-{j}",
                    "season": season, "game_date": f"{season}-04-01", "game_month": 4,
                    "game_dayofweek": 1, "inning": 2, "top_bottom": "T",
                    "balls_before": 1, "strikes_before": 2, "outs_before": 1,
                    "pitcher_trackman_id": pitcher, "pitcher_hand": hand,
                    "batter_hand": "R", "pitcher_team": "tm", "pitch_type_group": family,
                } for j in range(count))
    return pd.DataFrame(rows)


class ContractTests(unittest.TestCase):
    def test_runtime_config_is_authoritative(self):
        config = audit.load_contract_config(ROOT)
        self.assertEqual(config["contract_version"], audit.CONTRACT_VERSION)
        self.assertTrue(audit.static_contract(ROOT)["static_pass"])

    def test_config_divergence_fails_closed(self):
        config = audit.load_contract_config(ROOT)
        config["selection"]["team"] = "used"
        with self.assertRaises(audit.CrosswalkAuditError):
            audit.validate_contract_config(config)

    def test_main_projection_rejects_target(self):
        frame = main_rows(include_target=True)
        with self.assertRaises(audit.CrosswalkAuditError):
            audit.validate_projection(frame)

    def test_trackman_projection_rejects_physics(self):
        frame = trackman_rows().head(1).copy()
        frame["rel_speed"] = 90.0
        with self.assertRaises(audit.CrosswalkAuditError):
            audit.validate_projection(frame, trackman=True)

    def test_source_order_status_is_fail_closed(self):
        result = audit.assess_source_order_provenance(main_rows())
        self.assertEqual(result["status"], "NOT_PROVEN")
        self.assertFalse(result["source_order_used"])


class TaxonomyAndMaterializationTests(unittest.TestCase):
    def test_other_mode_requires_independent_source_level_proof(self):
        main = main_rows()
        tm = trackman_rows()
        config = audit.load_contract_config(ROOT)
        result = audit.assess_taxonomy_compatibility(main, tm, config)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["mode"], "exclude_other_renormalize_known")
        self.assertEqual(result["source_level_denominator_proof"]["status"], "PROVEN")
        self.assertEqual(result["source_level_denominator_proof"]["channel"], "observational_known_only")
        self.assertEqual(result["source_level_denominator_proof"]["official_contract_status"], "NOT_PROVEN")
        self.assertTrue(result["decision_before_matching"])

    def test_tiny_known_families_and_dominant_other_do_not_pass(self):
        main = main_rows()
        tm = trackman_rows()
        known = pd.concat([tm[tm["pitch_type_group"] == family].head(1) for family in audit.FAMILIES], ignore_index=True)
        other = tm.head(100).copy()
        other["pitch_type_group"] = "other"
        result = audit.assess_taxonomy_compatibility(main, pd.concat([known, other], ignore_index=True), audit.load_contract_config(ROOT))
        self.assertEqual(result["status"], "SEMANTIC_COMPATIBILITY_NOT_PROVEN")
        self.assertGreater(result["trackman_other_rows"], result["trackman_known_rows"])

    def test_missing_known_family_fails_semantic_prerequisite(self):
        main = main_rows()
        tm = trackman_rows()
        tm.loc[tm["pitch_type_group"] == "offspeed", "pitch_type_group"] = "other"
        config = audit.load_contract_config(ROOT)
        self.assertEqual(audit.assess_taxonomy_compatibility(main, tm, config)["status"], "SEMANTIC_COMPATIBILITY_NOT_PROVEN")

    def test_unexpected_taxonomy_group_fails_closed(self):
        main = main_rows()
        tm = trackman_rows()
        tm.loc[0, "pitch_type_group"] = "new_unapproved_family"
        result = audit.assess_taxonomy_compatibility(main, tm, audit.load_contract_config(ROOT))
        self.assertEqual(result["status"], "SEMANTIC_COMPATIBILITY_NOT_PROVEN")
        self.assertIn("new_unapproved_family", result["unexpected_trackman_groups"])

    def test_rounded_rate_fixture_does_not_require_integer_reconstruction(self):
        row = main_rows().iloc[0].to_dict()
        row["asof_pitcher_fastball_rate"] = 0.603
        row["asof_pitcher_breaking_rate"] = 0.247
        row["asof_pitcher_offspeed_rate"] = 0.150
        self.assertIsNotNone(audit._valid_mix(row, 0.01))
        probe = audit.analyze_rate_storage_precision_tokens(["0.603", "0.247", "0.150"])
        self.assertEqual(probe["observed_max_decimal_places"], 3)
        self.assertGreaterEqual(probe["tolerance"], 0.001)

    def test_source_row_order_does_not_change_annual_profiles(self):
        frame = main_rows()
        config = audit.load_contract_config(ROOT)
        mode = audit.infer_main_asof_mode(frame, config, 1e-6)
        self.assertEqual(mode["mode"], "CUMULATIVE")
        left = audit.build_main_profiles(frame, mode["mode"], 1e-6)
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        right = audit.build_main_profiles(shuffled, mode["mode"], 1e-6)
        self.assertEqual(audit.canonical_hash(left), audit.canonical_hash(right))

    def test_support_range_reset_and_ambiguous_modes(self):
        config = audit.load_contract_config(ROOT)
        reset = main_rows()
        reset["asof_pitcher_pitchmix_n"] = reset["season"].map({2019: 10, 2020: 9, 2021: 8, 2022: 7, 2023: 6, 2024: 5})
        result = audit.infer_main_asof_mode(reset, config, 1e-6)
        self.assertEqual(result["mode"], "SEASON_RESET")
        ambiguous = reset.copy()
        ambiguous["asof_pitcher_pitchmix_n"] = ambiguous["season"].map({2019: 10, 2020: 9, 2021: 20, 2022: 19, 2023: 30, 2024: 29})
        result = audit.infer_main_asof_mode(ambiguous, config, 1e-6)
        self.assertIsNone(result["mode"])
        self.assertEqual(result["status"], "MATERIALIZATION_NOT_PROVEN")

    def test_conflicting_max_support_snapshot_fails_closed(self):
        frame = main_rows().query("pitcher_id == 'm1' and season == 2019").copy()
        extra = frame.iloc[0].copy()
        extra["asof_pitcher_fastball_rate"] = 0.61
        extra["asof_pitcher_breaking_rate"] = 0.24
        extra["asof_pitcher_offspeed_rate"] = 0.15
        frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
        with self.assertRaises(audit.CrosswalkAuditError):
            audit._snapshot(frame, 1e-6)

    def test_cumulative_negative_increment_is_not_emitted(self):
        frame = main_rows().query("pitcher_id == 'm1'").copy()
        frame.loc[frame["season"] == 2020, "asof_pitcher_fastball_rate"] = 0.90
        frame.loc[frame["season"] == 2020, "asof_pitcher_breaking_rate"] = 0.05
        frame.loc[frame["season"] == 2020, "asof_pitcher_offspeed_rate"] = 0.05
        frame.loc[frame["season"] == 2021, "asof_pitcher_fastball_rate"] = 0.50
        frame.loc[frame["season"] == 2021, "asof_pitcher_breaking_rate"] = 0.35
        frame.loc[frame["season"] == 2021, "asof_pitcher_offspeed_rate"] = 0.15
        profiles = audit.build_main_profiles(frame, "CUMULATIVE", 1e-6)
        self.assertNotIn(2021, profiles["m1"]["annual"])


class SelectionVerificationTests(unittest.TestCase):
    def setUp(self):
        self.main = audit.build_main_profiles(main_rows(), "CUMULATIVE", 1e-6)
        self.tm = audit.build_trackman_profiles(trackman_rows())
        self.taxonomy = {"status": "PASS", "mode": "exclude_other_renormalize_known"}

    def test_selection_uses_repertoire_and_hand_only(self):
        null = audit.calibrate_selection_null(self.main, self.tm, requested=10)
        self.assertIn("annual_mix_threshold", null)
        self.assertIn("trajectory_delta_threshold", null)
        self.assertIn("second_best_margin_threshold", null)
        self.assertTrue(null["thresholds_null_derived"])
        result = audit.fit_repertoire_selection_map(self.main, self.tm, taxonomy=self.taxonomy, calibration=null)
        self.assertTrue(result["frozen_before_verification"])
        self.assertNotIn("team", result)

    def test_full_matrix_second_best_outside_channel_threshold_constrains_margin(self):
        distances = {
            ("m1", "t1"): {"annual_mix_tv": 0.01, "trajectory_delta_half_l1": 0.01, "distance": 0.01},
            ("m1", "t2"): {"annual_mix_tv": 0.11, "trajectory_delta_half_l1": 0.01, "distance": 0.02},
        }

        main = {"m1": {}}
        tm = {"t1": {}, "t2": {}}

        def fake_distance_by_identity(left, right, *, min_seasons=2, min_deltas=1):
            left_id = next(key for key, value in main.items() if value is left)
            right_id = next(key for key, value in tm.items() if value is right)
            return distances[(left_id, right_id)]

        with mock.patch.object(audit, "repertoire_distance", side_effect=fake_distance_by_identity):
            result = audit.fit_repertoire_selection_map(main, tm, taxonomy=self.taxonomy, calibration={"annual_mix_threshold": 0.10, "trajectory_delta_threshold": 0.10, "second_best_margin_threshold": 0.005})
        self.assertEqual(result["selection_map"], {"m1": "t1"})

    def test_semantic_failure_cannot_match(self):
        result = audit.fit_repertoire_selection_map(self.main, self.tm, taxonomy={"status": "SEMANTIC_COMPATIBILITY_NOT_PROVEN"}, calibration={})
        self.assertEqual(result["selection_map"], {})
        self.assertEqual(result["reason"], "SEMANTIC_COMPATIBILITY_NOT_PROVEN")

    def test_oop_verifier_cannot_change_selection_map(self):
        selection = {"m1": "t1"}
        original = audit.selection_map_hash(selection)
        evidence_a = audit.verify_selection_map_oop(selection, self.main, self.tm, 2022)
        poisoned = copy.deepcopy(self.tm)
        poisoned["t1"]["contexts"][2022]["game_month"] = audit.Counter({"99": 1000})
        evidence_b = audit.verify_selection_map_oop(selection, self.main, poisoned, 2022)
        self.assertEqual(original, audit.selection_map_hash(selection))
        self.assertEqual(evidence_a["selection_map_hash_before"], evidence_a["selection_map_hash_after"])
        self.assertEqual(evidence_b["selection_map_hash_before"], original)
        self.assertNotEqual(evidence_a["distributions"], evidence_b["distributions"])
        self.assertFalse(evidence_b["verifier_confirmed_manifest"])

    def test_oop_null_trials_reject_actual_frozen_partners(self):
        selection = {"m1": "t1", "m2": "t2"}
        evidence = audit.verify_selection_map_oop(selection, self.main, self.tm, 2022, requested=1000)
        self.assertTrue(evidence["null"]["forbidden_partner_map_enforced"])
        self.assertTrue(evidence["null"]["null_trials_preserve_selected_partners"])

    def test_context_selection_conflict_cannot_make_high_confidence(self):
        selection = {"m1": "t1"}
        self.assertEqual(audit.verify_selection_map_oop(selection, self.main, self.tm, 2022)["selection_map_hash_before"], audit.selection_map_hash(selection))
        # Verifier evidence is reported separately; there is no API to promote it.
        self.assertNotIn("selection_map", audit.verify_selection_map_oop(selection, self.main, self.tm, 2022))

    def test_self_id_is_next_season_not_full_trajectory(self):
        result = audit.next_season_self_identification(self.main, 2022)
        self.assertEqual(result["query_period"], "season == 2021")
        self.assertTrue(result["not_full_multi_year_validation"])
        self.assertEqual(result["diagnostic"], "next_season_repertoire_persistence")

    def test_self_id_has_separate_evaluation_null(self):
        distances = {left: {right: float(abs(i - j)) for j, right in enumerate(("a", "b", "c"))} for i, left in enumerate(("a", "b", "c"))}
        result = audit.calibrate_self_id_null(distances, ["a", "b", "c"], 2022, requested=1)
        self.assertEqual(result["evaluation_unique_trials"], 1)
        self.assertTrue(result["calibration_evaluation_disjoint"])
        self.assertTrue(result["true_derangement"])
        self.assertTrue(result["calibration_require_derangement"])
        self.assertTrue(result["evaluation_require_derangement"])
        self.assertIn("passes_1pct_ceiling", result["evaluation"])

    def test_assigned_non_top_pair_cannot_use_unrelated_row_margin(self):
        distances = {"top": 0.1, "assigned": 0.2, "unrelated": 99.0}
        evidence = audit.assigned_pair_evidence("left", "assigned", distances)
        self.assertFalse(evidence["unique_top1"])
        self.assertFalse(evidence["eligible"])
        self.assertIsNone(evidence["margin"])
        top = audit.assigned_pair_evidence("left", "top", distances)
        self.assertTrue(top["unique_top1"])
        self.assertTrue(top["eligible"])

    def test_cross_origin_stability_is_separate_evidence(self):
        results = {2022: {"selection_map": {"m1": "t1"}}, 2023: {"selection_map": {"m1": "t2"}}}
        result = audit.cross_origin_partner_stability(results)
        self.assertEqual(result["comparisons"][0]["different_partner"], 1)
        self.assertTrue(result["not_two_holdouts_of_one_frozen_mapping"])

    def test_cross_origin_stability_checks_all_pairs_and_nonadjacent_conflict(self):
        results = {2022: {"selection_map": {"m1": "t1"}}, 2023: {"selection_map": {"m1": "t1"}}, 2024: {"selection_map": {"m1": "t2"}}}
        result = audit.cross_origin_partner_stability(results)
        self.assertEqual([(item["left_origin"], item["right_origin"]) for item in result["comparisons"]], [(2022, 2023), (2022, 2024), (2023, 2024)])
        self.assertFalse(result["conflict_free"])

    def test_cross_origin_zero_overlap_cannot_pass(self):
        results = {2022: {"selection_map": {"m1": "t1"}}, 2023: {"selection_map": {"m2": "t2"}}, 2024: {"selection_map": {"m3": "t3"}}}
        result = audit.cross_origin_partner_stability(results)
        self.assertFalse(result["conflict_free"])
        self.assertTrue(all(item["shared_main_ids"] == 0 for item in result["comparisons"]))


class NullAndPrivacyTests(unittest.TestCase):
    def test_self_id_negligible_identifiability_fails_null_envelope(self):
        null = {"passes_1pct_ceiling": True, "evaluation": {"unique_trials": 100, "accepted_count_p95": 0, "wilson_95_upper_bound_proxy": 0.005}}
        gate = audit.self_id_acceptance_gate(correct_accepted=1, wrong_accepted=0, eligible=500, null_contract=null)
        self.assertFalse(gate["correct_accepted_exceeds_null_count_envelope"])
        self.assertEqual(gate["null_count_envelope"], 2.5)
        self.assertFalse(gate["coverage_exceeds_null_envelope"])
        self.assertFalse(gate["passes"])

    def test_unique_null_transformations_do_not_inflate_small_space(self):
        result = audit.generate_unique_null_transformations(["a", "b"], ["x", "y"], requested=1000, namespace="test")
        self.assertEqual(result["generated_unique"], 1)
        self.assertTrue(result["exhausted_space"])
        self.assertEqual(result["trial_unit"], "unique_null_transformation")

    def test_forbidden_partner_map_deranges_actual_selection(self):
        result = audit.generate_unique_null_transformations(["m1", "m2"], ["t1", "t2"], requested=1000, namespace="oop", forbidden_partners={"m1": "t1", "m2": "t2"})
        self.assertTrue(result["forbidden_partner_map_applied"])
        self.assertTrue(all(right != {"m1": "t1", "m2": "t2"}[left] for assignment in result["assignments"] for left, right in assignment))

    def test_calibration_and_evaluation_are_disjoint(self):
        result = audit.generate_unique_null_transformations(["a", "b", "c"], ["x", "y", "z"], requested=2, namespace="cal")
        eval_result = audit.generate_unique_null_transformations(["a", "b", "c"], ["x", "y", "z"], requested=2, namespace="eval", forbidden=result["assignments"])
        self.assertTrue(set(map(repr, result["assignments"])).isdisjoint(set(map(repr, eval_result["assignments"]))))

    def test_self_id_null_transformations_are_true_derangements(self):
        result = audit.generate_unique_null_transformations(["a", "b", "c"], ["a", "b", "c"], requested=1000, namespace="self", require_derangement=True)
        self.assertTrue(result["require_derangement"])
        self.assertTrue(result["exhausted_space"])
        self.assertTrue(all(left != right for assignment in result["assignments"] for left, right in assignment))

    def test_partial_derangement_rejects_overlapping_fixed_pair(self):
        result = audit.generate_unique_null_transformations(["a", "b"], ["a", "x"], requested=1000, namespace="partial", require_derangement=True)
        self.assertTrue(all(left != right for assignment in result["assignments"] for left, right in assignment))

    def test_wilson_is_null_risk_proxy(self):
        report = audit.null_false_accept_summary([False, True], requested=100, generated_unique=2, exhausted_space=False, pair_count=2, accepted_counts=[0, 2])
        self.assertEqual(report["false_accept_rate_proxy"], 0.5)
        self.assertIn("proxy", report["trial_unit"] + "_proxy")
        self.assertGreater(report["wilson_95_upper_bound_proxy"], 0.5)
        self.assertEqual(report["accepted_count_histogram"], {"0": 1, "2": 1})
        self.assertEqual(report["accepted_count_distribution"]["sample_count"], 2)

    def test_level_r_fail_closed(self):
        result = audit.level_r_prerequisite(main_rows())
        self.assertEqual(result["verdict"], audit.LEVEL_R_NOT_RUN)
        self.assertFalse(result["exact_matcher_implemented"])

    def test_p_pass_requires_real_gates(self):
        config = audit.load_contract_config(ROOT)
        result = audit.run_level_p(main_rows(), trackman_rows(), config)
        self.assertIn(result["verdict"], {"P_FAIL", "P_KILL"})
        self.assertNotEqual(result["verdict"], "P_PASS")

    def test_static_assertions_include_immutable_mapping_and_disjointness(self):
        result = audit.static_contract(ROOT)
        self.assertTrue(result["selection_verifier_columns_disjoint"])
        self.assertTrue(result["mapping_immutable"])
        self.assertFalse(result["verifier_confirmed_manifest"])
        self.assertFalse(result["target_2022_access"])
        self.assertFalse(result["target_2023_access"])
        self.assertFalse(result["target_2024_access"])
        self.assertFalse(result["test_access"])
        self.assertFalse(result["public_access"])
        self.assertFalse(result["external_access"])
        self.assertTrue(result["trackman_pitch_type_group_historical_access"])
        self.assertFalse(result["trackman_physics_current_pitch_access"])
        self.assertFalse(result["source_order_used"])
        self.assertFalse(result["model_training"])
        self.assertTrue(result["branch_inert"])

    def test_output_inside_repo_rejected(self):
        with self.assertRaises(audit.CrosswalkAuditError):
            audit._outside_repo(ROOT / "reports", ROOT)

    def test_report_hash_is_deterministic_and_aggregate(self):
        report = {"b": 1, "a": {"selection_map": {"secret": "id"}}, "canonical_report_sha256": "bad"}
        first = audit.canonical_report_hash(report)
        second = audit.canonical_report_hash(dict(reversed(list(report.items()))))
        self.assertEqual(first, second)
        self.assertNotEqual(first, "bad")

    def test_actual_report_privacy_and_runtime_contract_evidence(self):
        config = audit.load_contract_config(ROOT)
        level = {"verdict": "P_FAIL", "taxonomy": {}, "materialization": {}, "origins": {2022: {"selection_map": {"m1": "t1"}, "selection": {"mapping_hash": "hash"}, "oop_verification": {"verifier_confirmed_manifest": False}}}, "selection_map_hashes": {"2022": "aggregate-hash"}}
        report = audit.build_report(level, main=main_rows(), trackman=trackman_rows(), repo_root=ROOT, config_path=ROOT / "configs" / "trackman_repertoire_crosswalk_v2.json", script_path=ROOT / "scripts" / "audit_trackman_repertoire_crosswalk_v2.py", config=config)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn('"selection_map":', encoded)
        self.assertNotIn('"m1"', encoded)
        self.assertNotIn('"t1"', encoded)
        self.assertNotIn('"manifest": {', encoded)
        self.assertEqual(report["selection_map_hashes"], {"2022": "aggregate-hash"})
        self.assertTrue(report["runtime_contract"]["config_runtime_match"])
        self.assertEqual(report["source_frame_hash_algorithm"], audit.FRAME_HASH_ALGORITHM)

    def test_streaming_frame_hash_preserves_order_contract(self):
        frame = main_rows().head(4)
        self.assertEqual(audit.canonical_frame_hash(frame), audit.canonical_frame_hash(frame.copy()))
        self.assertNotEqual(audit.canonical_frame_hash(frame), audit.canonical_frame_hash(frame.iloc[::-1].reset_index(drop=True)))

    def test_mapping_hash_is_insertion_and_hash_seed_invariant(self):
        mapping = {"m2": "t2", "m1": "t1"}
        self.assertEqual(audit.selection_map_hash(mapping), audit.selection_map_hash(dict(reversed(list(mapping.items())))))
        code = "import sys; sys.path.insert(0, 'scripts'); import audit_trackman_repertoire_crosswalk_v2 as a; print(a.selection_map_hash({'m2':'t2','m1':'t1'}))"
        values = []
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            values.append(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True).strip())
        self.assertEqual(len(set(values)), 1)


if __name__ == "__main__":
    unittest.main()
