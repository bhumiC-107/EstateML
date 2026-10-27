import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from estate_ml.unsupervised_pipeline import (
    FeatureEngineeringConfig,
    PropertyFeaturePreprocessor,
    DEFAULT_SELECTION_THRESHOLDS,
    _add_quality_gates,
    _stability_score,
    assign_new_property,
    default_feature_selection,
    engineer_property_features,
    load_pipeline_artifact,
    save_pipeline_artifact,
    train_unsupervised_pipeline,
    validate_dataset,
)


class UnsupervisedPipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.frame = pd.DataFrame({
            "property_id": [f"BLR{i:04}" for i in range(80)],
            "area_sqft": np.r_[rng.normal(750, 25, 40), rng.normal(1450, 35, 40)],
            "BHK": [1] * 40 + [3] * 40,
            "bathrooms": [1] * 40 + [2] * 40,
            "property_age_years": np.r_[rng.integers(0, 6, 40), rng.integers(5, 25, 40)],
            "floor_number": [1, 2, 3, 4] * 20,
            "total_floors": [5] * 80,
            "location": ["North"] * 40 + ["South"] * 40,
            "condition": ["Good"] * 80,
            "property_type": ["Apartment"] * 80,
            "furnishing_status": ["Furnished"] * 80,
            "lift": [0, 1] * 40,
            "security": [1] * 80,
            "gym": [0, 1, 0, 0] * 20,
            "swimming_pool": [0] * 80,
            "power_backup": [1] * 80,
            "gated_community": [0, 1] * 40,
            "listing_date": ["2025-01-01"] * 80,
        })

    def _quick_pipeline(self, configurations=None):
        if configurations is None:
            configurations = [
                {"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}},
                {"Algorithm": "Agglomerative", "Parameters": {"n_clusters": 2}},
                {"Algorithm": "Gaussian Mixture", "Parameters": {"n_clusters": 2}},
                {"Algorithm": "DBSCAN", "Parameters": {"eps": 0.2, "min_samples": 5}},
            ]
        return train_unsupervised_pipeline(
            self.frame,
            selected_features=["area_sqft", "BHK", "bathrooms", "property_age_years", "location", "lift"],
            feature_config=FeatureEngineeringConfig(include_location=True, location_min_frequency=2),
            configurations=configurations,
            max_rows=None,
            max_configurations=8,
            stability_repeats=2,
            stability_sample_rows=60,
            random_seed=22,
        )

    def test_validation_reports_malformed_invalid_suspicious_and_duplicates(self):
        dirty = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        dirty.loc[1, "area_sqft"] = 0
        dirty["BHK"] = dirty["BHK"].astype(object)
        dirty.loc[2, "BHK"] = "unknown"
        dirty.loc[3, "property_age_years"] = 140
        clean, report, issues = validate_dataset(dirty)
        self.assertEqual(report["rows_retained"], 80)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertGreaterEqual(report["corrected_values_to_missing"], 2)
        self.assertEqual(report["suspicious_values_flagged_and_retained"]["property_age_years"], 1)
        self.assertTrue(pd.isna(clean.loc[1, "area_sqft"]))
        self.assertIn("flagged unusual; retained", set(issues["classification"]))

    def test_validation_excludes_duplicate_property_ids_and_reports_them(self):
        duplicate = self.frame.iloc[[0]].copy()
        duplicate["area_sqft"] += 200
        dirty = pd.concat([self.frame, duplicate], ignore_index=True)
        clean, report, issues = validate_dataset(dirty)
        self.assertEqual(len(clean), len(self.frame))
        self.assertEqual(report["duplicate_property_ids"], 1)
        excluded = issues.loc[issues["classification"] == "excluded duplicate ID"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded.iloc[0]["property_id"], self.frame.iloc[0]["property_id"])

    def test_one_successful_stability_refit_cannot_pass_gate(self):
        row = {
            "Configuration ID": 0, "Algorithm": "K-Means", "Status": "Success",
            "Clusters Found": 2, "Silhouette Score": 0.5, "Davies-Bouldin Score": 0.8,
            "Calinski-Harabasz Score": 120.0, "Stability ARI": 1.0,
            "Stability Repeats": 1, "Total Runtime (s)": 1.0, "Noise (%)": 0.0,
        }
        labels = np.array([0] * 40 + [1] * 40)
        gated = _add_quality_gates(pd.DataFrame([row]), {0: labels}, dict(DEFAULT_SELECTION_THRESHOLDS))
        self.assertFalse(bool(gated.iloc[0]["Selection Eligible"]))
        self.assertIn("Fewer than 2 stability refits succeeded", gated.iloc[0]["Eligibility Reasons"])

    def test_missing_metric_is_penalized_in_comparable_selection_rank(self):
        base = {
            "Algorithm": "K-Means", "Status": "Success", "Clusters Found": 2,
            "Stability ARI": 0.8, "Stability Repeats": 2, "Noise (%)": 0.0,
        }
        rows = [
            {**base, "Configuration ID": 0, "Silhouette Score": 0.9, "Davies-Bouldin Score": 0.8, "Calinski-Harabasz Score": 100.0, "Total Runtime (s)": 1.0},
            {**base, "Configuration ID": 1, "Silhouette Score": 0.8, "Davies-Bouldin Score": 0.9, "Calinski-Harabasz Score": np.nan, "Total Runtime (s)": 1.1},
            {**base, "Configuration ID": 2, "Silhouette Score": 0.7, "Davies-Bouldin Score": 1.0, "Calinski-Harabasz Score": 50.0, "Total Runtime (s)": 1.2},
        ]
        labels = {index: np.array([0] * 40 + [1] * 40) for index in range(3)}
        gated = _add_quality_gates(pd.DataFrame(rows), labels, dict(DEFAULT_SELECTION_THRESHOLDS))
        partial = gated.set_index("Configuration ID").loc[1]
        self.assertTrue(bool(partial["Selection Eligible"]))
        self.assertAlmostEqual(partial["Metric Rank"], (2 / 3 + 2 / 3 + 1.0) / 3)

    def test_stability_uses_label_permutation_invariant_adjusted_rand_index(self):
        matrix = np.r_[np.zeros((30, 1)), np.ones((30, 1))]
        baseline = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]

        class ReversedLabels:
            def fit_predict(self, values):
                return 1 - (values[:, 0] > 0.5).astype(int)

        with patch("estate_ml.unsupervised_pipeline._make_cluster_estimator", return_value=ReversedLabels()):
            score, repeats, error = _stability_score(
                matrix, baseline, "Agglomerative", {"n_clusters": 2},
                random_seed=11, repeats=2, max_rows=60,
            )
        self.assertEqual(len(repeats), 2)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(error, "")

    def test_feature_engineering_config_rejects_invalid_location_weights(self):
        with self.assertRaisesRegex(ValueError, "location_weight"):
            FeatureEngineeringConfig(location_weight=1.5)
        with self.assertRaisesRegex(ValueError, "location_min_frequency"):
            FeatureEngineeringConfig(location_min_frequency=0)

    def test_feature_engineering_and_missing_optional_columns(self):
        features = engineer_property_features(
            self.frame.head(3), ["floor_number", "total_floors", "lift", "gym"],
            FeatureEngineeringConfig(include_amenity_count=True),
        )
        self.assertIn("floor_ratio", features)
        self.assertNotIn("floor_number", features)
        self.assertIn("amenity_count", features)
        reduced = self.frame.drop(columns=["gym", "swimming_pool", "power_backup"])
        selected = default_feature_selection(reduced)
        self.assertNotIn("property_id", selected)
        processor = PropertyFeaturePreprocessor(selected, FeatureEngineeringConfig())
        matrix = processor.fit_transform(reduced)
        self.assertTrue(np.isfinite(matrix).all())

    def test_pipeline_accepts_price_labels_without_selecting_them_as_features(self):
        labelled = self.frame.assign(
            price_lakhs=np.linspace(40, 200, len(self.frame)),
            price_per_sqft=np.linspace(0.04, 0.2, len(self.frame)),
        )
        selected = default_feature_selection(labelled)
        self.assertNotIn("price_lakhs", selected)
        self.assertNotIn("price_per_sqft", selected)
        processor = PropertyFeaturePreprocessor(selected, FeatureEngineeringConfig())
        matrix = processor.fit_transform(labelled)
        self.assertFalse(any("price" in name.lower() for name in processor.get_feature_names_out()))
        with self.assertRaisesRegex(ValueError, "price-derived"):
            engineer_property_features(
                labelled, ["area_sqft", "price_per_sqft"], FeatureEngineeringConfig(),
            )

    def test_pipeline_trains_all_four_candidates_and_preserves_id_alignment(self):
        result = self._quick_pipeline()
        self.assertEqual(set(result.candidate_results["Algorithm"]), {"K-Means", "Agglomerative", "Gaussian Mixture", "DBSCAN"})
        self.assertTrue((result.candidate_results["Rows Evaluated"] == len(self.frame)).all())
        self.assertIn("Stability ARI", result.candidate_results)
        self.assertEqual(len(result.selected_labels), len(self.frame))
        self.assertTrue(result.selected_labels["property_id"].equals(result.visualization_data["property_id"]))
        self.assertTrue(result.selected_labels["cluster_label"].equals(result.visualization_data[f"{result.selected_algorithm}_cluster"]))
        self.assertEqual(result.candidate_results["Selected"].sum(), 1)

    def test_no_winner_when_only_degenerate_configuration_exists(self):
        result = self._quick_pipeline([{"Algorithm": "K-Means", "Parameters": {"n_clusters": 1}}])
        self.assertTrue(result.no_satisfactory_model)
        self.assertIsNone(result.selected_algorithm)
        self.assertIsNone(result.selected_model)
        self.assertTrue(result.candidate_results.iloc[0]["Eligibility Reasons"])

    def test_new_property_assignment_uses_fitted_preprocessor(self):
        result = self._quick_pipeline([{"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}}])
        self.assertEqual(result.selected_algorithm, "K-Means")
        new_property = self.frame.iloc[0].drop(labels=["property_id"]).to_dict()
        for column in [
            "property_age_years", "floor_number", "total_floors", "parking_spaces",
            "lift", "security", "gym", "swimming_pool", "power_backup", "gated_community",
        ]:
            new_property[column] = None
        assignment = assign_new_property(result, new_property)
        self.assertIn(assignment["cluster_label"], set(result.selected_labels["cluster_label"]))
        self.assertEqual(assignment["assignment_method"], "K-Means predict")

    def test_save_and_load_preserves_selected_model_and_assignment(self):
        result = self._quick_pipeline([{"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}}])
        with tempfile.TemporaryDirectory() as directory:
            model_path = save_pipeline_artifact(result, directory)
            loaded = load_pipeline_artifact(directory)
            self.assertTrue(model_path.is_file())
            self.assertEqual(loaded.selected_algorithm, result.selected_algorithm)
            probe = self.frame.iloc[5].drop(labels=["property_id"]).to_dict()
            self.assertEqual(assign_new_property(loaded, probe)["cluster_label"], assign_new_property(result, probe)["cluster_label"])

    def test_artifact_integrity_is_checked_before_deserialization(self):
        result = self._quick_pipeline([{"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}}])
        with tempfile.TemporaryDirectory() as directory:
            model_path = save_pipeline_artifact(result, directory)
            model_path.write_bytes(model_path.read_bytes() + b"corruption")
            with patch("estate_ml.unsupervised_pipeline.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "integrity check failed"):
                    load_pipeline_artifact(directory)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
