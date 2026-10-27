import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from estate_ml.clustering import (
    available_features,
    build_cluster_profile,
    apply_experiment_configuration,
    build_experiment_configurations,
    filter_cluster_properties,
    find_similar_properties,
    interpret_clusters,
    load_dataset,
    prepare_features,
    rank_experiment_results,
    run_clustering_experiments,
    run_clustering,
    _scores,
)


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame({
            "property_id": [f"p{i}" for i in range(60)],
            "area_sqft": np.r_[rng.normal(800, 30, 30), rng.normal(1500, 30, 30)],
            "BHK": [1] * 30 + [3] * 30,
            "location": ["North"] * 30 + ["South"] * 30,
            "condition": ["Good"] * 60,
            "lift": [0, 1] * 30,
            "listing_date": ["2025-01-01"] * 60,
        })

    def test_loading_deduplicates_and_cleans_infinite_numbers(self):
        frame = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        frame.loc[1, "area_sqft"] = np.inf
        loaded = load_dataset(frame)
        self.assertEqual(len(loaded), len(self.frame))
        self.assertTrue(pd.isna(loaded.loc[1, "area_sqft"]))

    def test_preparation_encodes_categories_and_derives_selected_date(self):
        matrix, names = prepare_features(self.frame, ["area_sqft", "location", "listing_date"])
        self.assertEqual(matrix.shape[0], len(self.frame))
        self.assertTrue(any(name.startswith("location_") for name in names))
        self.assertIn("listing_year", names)
        self.assertNotIn("property_id", names)

    def test_price_and_price_derived_columns_are_not_selectable_or_clusterable(self):
        labelled = self.frame.assign(
            price_lakhs=np.arange(len(self.frame), dtype=float),
            price_per_sqft=np.arange(len(self.frame), dtype=float),
            listing_cost=np.arange(len(self.frame), dtype=float),
            market_value=np.arange(len(self.frame), dtype=float),
        )
        self.assertNotIn("price_lakhs", available_features(labelled))
        self.assertNotIn("price_per_sqft", available_features(labelled))
        self.assertNotIn("listing_cost", available_features(labelled))
        self.assertNotIn("market_value", available_features(labelled))
        for target_feature in ("price_lakhs", "price_per_sqft", "listing_cost", "market_value"):
            with self.assertRaisesRegex(ValueError, "price-derived"):
                prepare_features(labelled, ["area_sqft", target_feature])

    def test_all_algorithms_return_aligned_labels_and_metrics(self):
        labelled, results = run_clustering(
            self.frame, ["area_sqft", "BHK", "location", "lift"],
            n_clusters=2, eps=0.45, min_samples=4, random_seed=17,
        )
        self.assertEqual(len(labelled), len(self.frame))
        self.assertEqual(set(results["Algorithm"]), {"K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture"})
        for algorithm in ("K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture"):
            self.assertIn(f"{algorithm}_cluster", labelled)
        self.assertTrue(labelled["property_id"].equals(self.frame["property_id"]))
        self.assertTrue(labelled[["PCA_1", "PCA_2"]].notna().all().all())
        dbscan = results.loc[results["Algorithm"] == "DBSCAN"].iloc[0]
        self.assertEqual(dbscan["Metric Population"], "Non-noise points only")
        self.assertTrue((results["Rows Used"] == len(self.frame)).all())

    def test_sample_is_reproducible_and_shared_by_all_algorithms(self):
        labelled, results = run_clustering(
            self.frame, ["area_sqft", "BHK", "location"], n_clusters=2,
            eps=0.45, min_samples=4, random_seed=12, max_rows=30,
        )
        self.assertEqual(len(labelled), 30)
        self.assertEqual(set(results["Rows Used"]), {30})
        self.assertEqual(set(results["Evaluation Set"]), {"Sample"})
        second, _ = run_clustering(
            self.frame, ["area_sqft", "BHK", "location"], n_clusters=2,
            eps=0.45, min_samples=4, random_seed=12, max_rows=30,
        )
        self.assertTrue(labelled["property_id"].equals(second["property_id"]))
        prepared, _ = prepare_features(labelled, ["area_sqft", "BHK", "location"])
        expected = KMeans(n_clusters=2, random_state=12, n_init=10).fit_predict(prepared)
        self.assertTrue(np.array_equal(labelled["K-Means_cluster"].to_numpy(), expected))

    def test_profile_excludes_noise_and_reports_property_characteristics(self):
        labelled = self.frame.head(5).copy()
        labelled["property_type"] = ["Apartment"] * 5
        labelled["furnishing_status"] = ["Furnished"] * 5
        labelled["property_age_years"] = [1, 2, 3, 4, 5]
        labelled["lift"] = [1, 1, 0, 1, 0]
        labelled["DBSCAN_cluster"] = [0, 0, 1, 1, -1]
        profile = build_cluster_profile(labelled, "DBSCAN_cluster")
        self.assertEqual(profile["Properties"].sum(), 4)
        self.assertEqual(profile.attrs["noise_points"], 1)
        self.assertIn("Mean area (sq ft)", profile)
        self.assertIn("Median area (sq ft)", profile)
        self.assertIn("Most common BHK", profile)
        self.assertIn("Mean property age (years)", profile)
        self.assertIn("Most common property type", profile)
        self.assertIn("Most common furnishing", profile)
        self.assertIn("Most common locations", profile)
        self.assertIn("lift available (%)", profile)

    def test_cluster_interpretation_compares_measured_values_with_overall(self):
        labelled = self.frame.copy()
        labelled["test_cluster"] = [0] * 30 + [1] * 30
        profile, interpretations = interpret_clusters(labelled, "test_cluster")
        self.assertEqual(profile["Properties"].sum(), len(labelled))
        self.assertAlmostEqual(profile["Share of properties (%)"].sum(), 100.0)
        first = next(item for item in interpretations if item["Cluster"] == 0)
        second = next(item for item in interpretations if item["Cluster"] == 1)
        self.assertTrue(any("Mean area is" in detail and "below" in detail for detail in first["Measured differences"]))
        self.assertTrue(any("Mean area is" in detail and "above" in detail for detail in second["Measured differences"]))

    def test_property_filter_keeps_original_ids_aligned_with_labels(self):
        labelled = self.frame.copy()
        labelled["test_cluster"] = [0] * 30 + [1] * 30
        selected = filter_cluster_properties(
            labelled, "test_cluster", 0, location="north", bhk=[1], area_min=700,
        )
        self.assertGreater(len(selected), 0)
        self.assertTrue((selected["test_cluster"] == 0).all())
        self.assertTrue(selected["property_id"].isin(labelled.loc[labelled["test_cluster"] == 0, "property_id"]).all())
        self.assertTrue(selected["location"].str.contains("North", case=False).all())
        self.assertTrue((selected["area_sqft"] >= 700).all())

    def test_nearest_neighbor_results_exclude_selected_id_and_identifier_feature(self):
        similar, names = find_similar_properties(
            self.frame, "p0", ["area_sqft", "BHK", "location"], n_neighbors=4,
        )
        self.assertEqual(len(similar), 4)
        self.assertNotIn("p0", set(similar["property_id"]))
        self.assertNotIn("property_id", names)
        self.assertIn("Feature distance", similar)
        self.assertIn("Similarity rank", similar)

    def test_invalid_metrics_are_unavailable_and_noise_is_excluded(self):
        x = np.array([[0.0], [1.0], [2.0], [100.0]])
        one_cluster = _scores(x, np.array([0, 0, 0, 0]), ignore_noise=False)
        self.assertTrue(all(np.isnan(value) for value in one_cluster))
        with_noise = _scores(x, np.array([0, 0, 1, -1]), ignore_noise=True)
        self.assertTrue(all(np.isfinite(value) for value in with_noise))

    def test_experiment_reuses_one_prepared_matrix_and_shared_sample(self):
        configurations = [
            {"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}},
            {"Algorithm": "Agglomerative", "Parameters": {"n_clusters": 2}},
            {"Algorithm": "DBSCAN", "Parameters": {"eps": 0.001, "min_samples": 1000}},
        ]
        with patch("estate_ml.clustering.prepare_features", wraps=prepare_features) as prepare:
            results, labels, rows = run_clustering_experiments(
                self.frame, ["area_sqft", "BHK", "location"], configurations,
                max_rows=30, random_seed=31, max_configurations=4,
            )
        prepare.assert_called_once()
        self.assertEqual(len(rows), 30)
        self.assertEqual(set(results["Rows Evaluated"]), {30})
        self.assertEqual(set(results["Evaluation Set"]), {"Sample"})
        self.assertEqual(set(labels), {0, 1, 2})
        dbscan = results.loc[results["Algorithm"] == "DBSCAN"].iloc[0]
        self.assertEqual(dbscan["Noise Points"], 30)
        self.assertEqual(dbscan["Clusters Found"], 0)
        self.assertTrue(np.isnan(dbscan["Silhouette Score"]))

    def test_experiment_records_invalid_and_undefined_metric_configurations(self):
        configurations = [
            {"Algorithm": "K-Means", "Parameters": {"n_clusters": 0}},
            {"Algorithm": "K-Means", "Parameters": {"n_clusters": 1}},
        ]
        results, labels, _ = run_clustering_experiments(
            self.frame, ["area_sqft", "BHK"], configurations,
        )
        failed = results.iloc[0]
        self.assertEqual(failed["Status"], "Failed")
        self.assertIn("n_clusters", failed["Error"])
        undefined = results.iloc[1]
        self.assertEqual(undefined["Status"], "Success")
        self.assertTrue(np.isnan(undefined["Silhouette Score"]))
        self.assertTrue(np.isnan(undefined["Davies-Bouldin Score"]))
        self.assertIn(1, labels)

    def test_experiment_configuration_builder_and_runner_enforce_limits(self):
        configs = build_experiment_configurations(
            ["K-Means", "DBSCAN"], [2, 3], [0.4, 0.8], [4], max_configurations=4,
        )
        self.assertEqual(len(configs), 4)
        with self.assertRaisesRegex(ValueError, "above the selected limit"):
            build_experiment_configurations(
                ["K-Means", "Agglomerative"], [2, 3, 4], [], [], max_configurations=4,
            )
        with self.assertRaisesRegex(ValueError, "exceeding the limit"):
            run_clustering_experiments(
                self.frame, ["area_sqft"], configs, max_configurations=3,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and"):
            build_experiment_configurations(["K-Means"], [2], [], [], max_configurations=25)

    def test_experiment_ranking_uses_multiple_metrics_and_keeps_failures_unranked(self):
        results = pd.DataFrame([
            {"Configuration ID": 0, "Status": "Success", "Silhouette Score": 0.6, "Davies-Bouldin Score": 0.8, "Calinski-Harabasz Score": 300, "Execution Time (s)": 2.0, "Noise (%)": 0},
            {"Configuration ID": 1, "Status": "Success", "Silhouette Score": 0.5, "Davies-Bouldin Score": 0.5, "Calinski-Harabasz Score": 400, "Execution Time (s)": 1.0, "Noise (%)": 5},
            {"Configuration ID": 2, "Status": "Failed", "Silhouette Score": np.nan, "Davies-Bouldin Score": np.nan, "Calinski-Harabasz Score": np.nan, "Execution Time (s)": 0.1, "Noise (%)": 0},
        ])
        ranked = rank_experiment_results(results)
        self.assertEqual(set(ranked.loc[ranked["Recommendation Rank"].notna(), "Configuration ID"]), {0, 1})
        self.assertEqual(ranked.iloc[-1]["Configuration ID"], 2)
        self.assertIn("runtime", ranked.iloc[0]["Recommendation Basis"])

    def test_selected_experiment_configuration_maps_labels_to_original_ids(self):
        configs = [{"Algorithm": "K-Means", "Parameters": {"n_clusters": 2}}]
        results, labels, rows = run_clustering_experiments(
            self.frame, ["area_sqft", "BHK"], configs, max_rows=30, random_seed=9,
        )
        applied, selected_result = apply_experiment_configuration(rows, 0, labels, results)
        self.assertEqual(selected_result["Configuration ID"], 0)
        self.assertTrue(applied["property_id"].equals(rows["property_id"]))
        self.assertTrue(np.array_equal(applied["K-Means_cluster"], labels[0]))
        self.assertTrue(applied["PCA_1"].equals(rows["PCA_1"]))
        with self.assertRaisesRegex(ValueError, "not found"):
            apply_experiment_configuration(rows, 99, labels, results)


if __name__ == "__main__":
    unittest.main()
