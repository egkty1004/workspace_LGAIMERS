import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import audit_trackman_crosswalk_free_context_priors_v2 as audit
from scripts import audit_trackman_context_domain_characterization_v1 as pr13


ROOT = Path(__file__).resolve().parents[1]


def main_frame(seasons=(2019, 2022, 2025)):
    rows = []
    for index, season in enumerate(seasons):
        rows.append({
            "season": season,
            "balls_before": index % 4,
            "strikes_before": index % 3,
            "outs_before": index % 3,
        })
    return pd.DataFrame(rows, columns=audit.MAIN_PROJECTION)


def trackman_frame(repeated=120):
    rows = []
    families = audit.FAMILIES
    for index in range(repeated):
        rows.append({
            "season": 2019 + (index % 4),
            "balls_before": 0,
            "strikes_before": 0,
            "outs_before": 0,
            "pitch_type_group": families[index % len(families)],
        })
    rows.extend([
        {"season": 2019, "balls_before": 1, "strikes_before": 1, "outs_before": 1, "pitch_type_group": "other"},
        {"season": 2020, "balls_before": 2, "strikes_before": 1, "outs_before": 2, "pitch_type_group": "fastball"},
    ])
    return pd.DataFrame(rows, columns=audit.TRACKMAN_PROJECTION)


class ContractAndHashTests(unittest.TestCase):
    def test_config_matches_runtime_and_pr13_authority(self):
        config = audit.load_contract_config(ROOT)
        self.assertEqual(audit._contract_projection(config), audit._runtime_contract())
        self.assertEqual(tuple(audit.SOURCE_PROJECTION), tuple(pr13.PROJECTION))
        self.assertEqual(audit.FRAME_HASH_ALGORITHM, pr13.FRAME_HASH_ALGORITHM)
        self.assertNotEqual(audit.FRAME_HASH_ALGORITHM, "canonical-jsonl-v1")
        self.assertTrue(audit.validate_pr13_authority(ROOT)["hashes_match"])
        v1 = audit.validate_v1_authority(ROOT)
        self.assertTrue(v1["hashes_match"])
        self.assertTrue(v1["shared_semantics_match"])

    def test_config_runtime_divergence_fails_closed(self):
        config = audit.load_contract_config(ROOT)
        config["context"]["support_threshold"] = 101
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_contract_config(config)

    def test_pr13_hash_same_for_reordered_multiset(self):
        frame = main_frame((2019, 2022, 2023, 2024))
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        self.assertEqual(audit.pr13_source_hash(frame, source="main"), audit.pr13_source_hash(reversed_frame, source="main"))

    def test_pr13_hash_changes_for_context_mutation(self):
        frame = main_frame((2019, 2022, 2023, 2024))
        mutated = frame.copy()
        mutated.loc[0, "outs_before"] = (int(mutated.loc[0, "outs_before"]) + 1) % 3
        self.assertNotEqual(audit.pr13_source_hash(frame, source="main"), audit.pr13_source_hash(mutated, source="main"))

    def test_pr13_hash_changes_for_row_add_and_delete(self):
        frame = main_frame((2019, 2022, 2023, 2024))
        added = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        deleted = frame.iloc[:-1].reset_index(drop=True)
        original = audit.pr13_source_hash(frame, source="main")
        self.assertNotEqual(original, audit.pr13_source_hash(added, source="main"))
        self.assertNotEqual(original, audit.pr13_source_hash(deleted, source="main"))

    def test_projection_rejects_target_identity_and_physics(self):
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_projection(pd.DataFrame({"season": [2019], "balls_before": [0], "strikes_before": [0], "outs_before": [0], "control_success": [1]}))
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.validate_projection(pd.DataFrame({"season": [2019], "balls_before": [0], "strikes_before": [0], "outs_before": [0], "pitch_type_group": ["other"], "rel_speed": [90]}), trackman=True)

    def test_scoped_readers_are_invariant_to_forbidden_column_poison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_a = pd.DataFrame({
                "season": [2022], "balls_before": [1], "strikes_before": [2],
                "outs_before": [0], "control_success": [0], "pitcher_id": [11],
            })
            main_b = main_a.copy()
            main_b["control_success"] = 1
            main_b["pitcher_id"] = 999
            main_a.to_csv(root / "main_a.csv", index=False)
            main_b.to_csv(root / "main_b.csv", index=False)
            self.assertTrue(audit.read_main_features(root / "main_a.csv").equals(
                audit.read_main_features(root / "main_b.csv")
            ))

            tm_a = pd.DataFrame({
                "season": [2021], "balls_before": [1], "strikes_before": [2],
                "outs_before": [0], "pitch_type_group": ["other"],
                "pitcher_trackman_id": [77], "rel_speed": [90.0],
            })
            tm_b = tm_a.copy()
            tm_b["pitcher_trackman_id"] = 123456
            tm_b["rel_speed"] = 105.0
            tm_a.to_csv(root / "tm_a.csv", index=False)
            tm_b.to_csv(root / "tm_b.csv", index=False)
            self.assertTrue(audit.read_trackman_features(root / "tm_a.csv").equals(
                audit.read_trackman_features(root / "tm_b.csv")
            ))

    def test_static_contract_is_read_only_firewall(self):
        result = audit.static_contract(ROOT)
        self.assertTrue(result["static_pass"])
        self.assertFalse(result["target_access"])
        self.assertFalse(result["trackman_entity_access"])
        self.assertFalse(result["trackman_physics_access"])
        self.assertFalse(result["level_r_exact_matcher"])


class ResidueAndLookupTests(unittest.TestCase):
    def test_frozen_predicate_excludes_only_invalid_context_and_records_manifest(self):
        raw = trackman_frame(8).copy()
        raw.loc[0, "balls_before"] = 4
        raw.loc[1, "outs_before"] = 3
        filtered, summary = audit.filter_trackman_history(raw)
        self.assertEqual(summary["raw_rows"], len(raw))
        self.assertEqual(summary["excluded_rows"], 2)
        self.assertEqual(summary["retained_rows"], len(filtered))
        self.assertEqual(sum(item["count"] for item in summary["excluded_manifest"]), 2)
        self.assertEqual(len(filtered), len(raw) - 2)
        self.assertTrue(all(audit._context_key(row, audit.CONTEXT_COLUMNS) is not None for row in filtered.to_dict("records")))

    def test_predicate_excludes_missing_nonfinite_and_noninteger_without_normalization(self):
        raw = trackman_frame(8)
        invalid = pd.DataFrame([
            {"season": 2022, "balls_before": None, "strikes_before": 0, "outs_before": 0, "pitch_type_group": "fastball"},
            {"season": 2022, "balls_before": 0, "strikes_before": float("inf"), "outs_before": 0, "pitch_type_group": "breaking"},
            {"season": 2023, "balls_before": 0, "strikes_before": 0, "outs_before": 1.5, "pitch_type_group": "other"},
        ], columns=audit.TRACKMAN_PROJECTION)
        combined = pd.concat([raw, invalid], ignore_index=True)
        filtered, summary = audit.filter_trackman_history(combined)
        self.assertEqual(summary["excluded_rows"], 3)
        self.assertEqual(summary["retained_rows"], len(raw))
        self.assertFalse(summary["normalization"])
        self.assertFalse(summary["rounding"])
        self.assertFalse(summary["clipping"])

    def test_official_residue_contract_drift_fails_closed(self):
        raw = trackman_frame(8)
        with self.assertRaisesRegex(audit.ContextPriorAuditError, "SOURCE_QUALITY_RESIDUE_DRIFT"):
            audit.filter_trackman_history(raw, enforce_official_residue=True)

    def test_unfiltered_trackman_cannot_reach_lookup_builder(self):
        with self.assertRaises(audit.ContextPriorAuditError):
            audit.build_context_prior_model(trackman_frame(), seasons=(2025,))

    def test_raw_taxonomy_is_checked_before_source_quality_filter(self):
        raw = trackman_frame(8).copy()
        raw.loc[0, "outs_before"] = 4
        raw.loc[0, "pitch_type_group"] = "unclassified"
        with self.assertRaises(audit.TaxonomyNotProven):
            audit.filter_trackman_history(raw)

    def test_excluded_rows_never_reach_lookup(self):
        raw = trackman_frame(120).copy()
        raw.loc[0, "outs_before"] = 4
        filtered, _ = audit.filter_trackman_history(raw)
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        self.assertEqual(model.trackman_row_count, len(filtered))
        self.assertEqual(model.lookups[2025]["global"][()].total, len(filtered))

    def test_other_remains_fourth_family_without_redistribution(self):
        raw = trackman_frame(120)
        filtered, _ = audit.filter_trackman_history(raw)
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        bucket = model.lookups[2025]["global"][()]
        self.assertEqual(len(bucket.counts), 4)
        self.assertGreater(bucket.counts[audit.FAMILIES.index("other")], 0)
        self.assertEqual(sum(bucket.counts), bucket.total)

    def test_strict_season_cutoff_excludes_same_and_future(self):
        raw = trackman_frame(120)
        raw.loc[1, "season"] = 2024
        filtered, _ = audit.filter_trackman_history(raw)
        model = audit.build_context_prior_model(filtered, seasons=(2024, 2025))
        self.assertEqual(model.lookups[2024]["global"][()].total, len(filtered[filtered["season"] < 2024]))
        self.assertEqual(model.lookups[2025]["global"][()].total, len(filtered[filtered["season"] < 2025]))
        temporal = audit.assess_temporal_cutoffs(filtered, model)
        self.assertTrue(temporal["all_cutoffs_strict_prior_only"])
        self.assertFalse(temporal["cutoffs"]["2024"]["same_season_used"])
        self.assertFalse(temporal["cutoffs"]["2024"]["future_season_used"])

    def test_future_season_poison_cannot_change_earlier_features(self):
        raw = trackman_frame(160)
        filtered, _ = audit.filter_trackman_history(raw)
        poisoned = filtered.copy()
        future = poisoned["season"] >= 2022
        poisoned.loc[future, "pitch_type_group"] = "other"
        poisoned.attrs.update(filtered.attrs)
        base_model = audit.build_context_prior_model(filtered, seasons=(2022,))
        poison_model = audit.build_context_prior_model(poisoned, seasons=(2022,))
        probe = main_frame((2022,))
        self.assertTrue(audit.apply_context_prior_features(probe, base_model).equals(
            audit.apply_context_prior_features(probe, poison_model)
        ))

    def test_l0_l1_global_support_and_sparse_backoff(self):
        raw = trackman_frame(120)
        filtered, _ = audit.filter_trackman_history(raw)
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        row = {"season": 2025, "balls_before": 3, "strikes_before": 2, "outs_before": 2}
        result = audit.apply_context_prior_row(row, model, 2025)
        self.assertEqual(result["__backoff_level"], "global")
        self.assertEqual(result["tm_cf_support"], model.lookups[2025]["global"][()].total)

    def test_2025_proof_covers_all_36_legal_states_and_exact_support(self):
        filtered, _ = audit.filter_trackman_history(trackman_frame(120))
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        proof = audit.audit_2025_legal_states(model, expected_global_support=len(filtered))
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["tested_l0_states"], 36)
        self.assertEqual(proof["valid_output_states"], 36)
        self.assertEqual(proof["support_exact_states"], 36)
        self.assertEqual(proof["global_support"], len(filtered))
        self.assertEqual(sum(proof["level_counts"].values()), 36)

    def test_2025_proof_fails_on_wrong_frozen_global_support(self):
        filtered, _ = audit.filter_trackman_history(trackman_frame(120))
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        proof = audit.audit_2025_legal_states(model, expected_global_support=len(filtered) + 1)
        self.assertFalse(proof["passed"])
        self.assertIn("2025_global_support_mismatch", proof["state_failures"])

    def test_2025_proof_freezes_exact_official_global_support_without_large_fixture(self):
        total = audit.EXPECTED_FILTERED_TRACKMAN_ROWS
        counts = (448246, 448245, 448245, 448245)
        bucket = audit.Bucket(total=total, counts=counts)
        model = audit.ContextPriorModel(
            lookups={2025: {"L0": {}, "L1": {}, "global": {(): bucket}}},
            trackman_row_count=total,
            support_threshold=audit.SUPPORT_THRESHOLD,
        )
        proof = audit.audit_2025_legal_states(
            model, expected_global_support=audit.EXPECTED_FILTERED_TRACKMAN_ROWS
        )
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["global_support"], 1_792_981)
        self.assertEqual(proof["tested_l0_states"], 36)
        self.assertEqual(proof["support_exact_states"], 36)

    def test_2025_proof_allows_sparse_l0_to_use_l1(self):
        rows = []
        for index in range(100):
            rows.append({"season": 2019, "balls_before": 1, "strikes_before": 1, "outs_before": index % 3, "pitch_type_group": audit.FAMILIES[index % 4]})
        rows.append({"season": 2019, "balls_before": 3, "strikes_before": 2, "outs_before": 2, "pitch_type_group": "other"})
        raw = pd.DataFrame(rows, columns=audit.TRACKMAN_PROJECTION)
        filtered, _ = audit.filter_trackman_history(raw)
        model = audit.build_context_prior_model(filtered, seasons=(2025,))
        row = {"season": 2025, "balls_before": 1, "strikes_before": 1, "outs_before": 2}
        feature = audit.apply_context_prior_row(row, model, 2025)
        self.assertEqual(feature["__backoff_level"], "L1")
        self.assertEqual(feature["tm_cf_support"], 100)
        proof = audit.audit_2025_legal_states(model, expected_global_support=len(filtered))
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["tested_l0_states"], 36)


class DeterminismAndPrivacyTests(unittest.TestCase):
    def test_reverse_source_rebuild_is_identical(self):
        filtered, _ = audit.filter_trackman_history(trackman_frame(120))
        main = main_frame((2020, 2022, 2024))
        model = audit.build_context_prior_model(filtered, seasons=(2020, 2022, 2024, 2025))
        result = audit.assess_determinism(main, filtered, model)
        self.assertTrue(result["tested"])
        self.assertTrue(result["complete_feature_output_equal"])
        self.assertTrue(result["serialized_lookup_equal"])
        self.assertTrue(result["complete_rebuild_equal"])
        self.assertEqual(result["original_feature_output_sha256"], result["permuted_feature_output_sha256"])

    def test_serialized_lookup_reproduces_features(self):
        filtered, _ = audit.filter_trackman_history(trackman_frame(120))
        main = main_frame((2020, 2022, 2024))
        model = audit.build_context_prior_model(filtered, seasons=(2020, 2022, 2024, 2025))
        features = audit.apply_context_prior_features(main, model)
        lookup = audit.build_aggregate_lookup(model)
        for index, row in enumerate(main.to_dict("records")):
            reconstructed = audit.apply_context_prior_lookup_row(row, lookup, int(row["season"]))
            expected = {column: features.loc[index, column] for column in [*audit.FEATURE_COLUMNS, "__backoff_level"]}
            self.assertEqual(reconstructed, expected)
        parity = audit.assess_serialized_lookup_parity(main, features, lookup)
        self.assertTrue(parity["tested"])
        self.assertTrue(parity["equal"])
        self.assertEqual(parity["mismatch_count"], 0)
        self.assertEqual(parity["feature_output_sha256"], parity["serialized_lookup_output_sha256"])

    def test_report_hash_is_finite_and_external_output_is_guarded(self):
        filtered, _ = audit.filter_trackman_history(trackman_frame(120))
        main = main_frame((2020, 2022))
        model = audit.build_context_prior_model(filtered, seasons=(2020, 2022, 2025))
        features = audit.apply_context_prior_features(main, model)
        acceptance = audit.evaluate_acceptance(main, features, filtered)
        taxonomy = audit.validate_trackman_taxonomy(filtered)
        config = audit.load_contract_config(ROOT)
        report = audit.build_report(main, filtered, features, acceptance, taxonomy, repo_root=ROOT, config_path=ROOT / "configs" / "trackman_crosswalk_free_context_priors_v2.json", script_path=ROOT / "scripts" / "audit_trackman_crosswalk_free_context_priors_v2.py", config=config, source_scans={"main": audit.pr13_source_scan(main, source="main"), "trackman": audit.pr13_source_scan(filtered, source="trackman")}, source_quality={"raw_rows": 120, "excluded_rows": 0, "retained_rows": 120}, state_proof={"passed": True})
        self.assertEqual(report["canonical_report_sha256"], audit.canonical_report_hash(report))
        json.dumps(report, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(audit.ContextPriorAuditError):
                audit.write_outputs(report, audit.build_aggregate_lookup(model), ROOT / "forbidden-output", repo_root=ROOT)

    def test_main_invalid_context_fails_closed_and_main_is_never_filtered(self):
        main = main_frame((2022,)).copy()
        main.loc[0, "balls_before"] = 4
        with self.assertRaises(audit.ContextDomainNotProven):
            audit._validate_context_frame(main, label="main")

    def test_source_hash_comes_from_pr13_multiset_not_v1_order_hash(self):
        frame = main_frame((2019, 2022, 2023))
        expected = pr13.scan_frame(frame, source="main")["source_frame_sha256"]
        self.assertEqual(audit.pr13_source_hash(frame, source="main"), expected)


if __name__ == "__main__":
    unittest.main()
