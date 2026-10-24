import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from estate_ml.train_model import CANDIDATES, _tune_candidate

from estate_ml.property_model import (
    FEATURE_SCHEMA, ID_COLUMN, TARGET, build_property_input, make_pipeline,
    predict_property, prepare_property_data,
)


def make_property_listings(row_count=40):
    data = {
        "property_id": [f"BLR{i:05d}" for i in range(row_count)],
        "location": ["Whitefield" if i % 2 else "Unknown" for i in range(row_count)],
        "latitude": [12.9 + (i % 10) / 1000 for i in range(row_count)],
        "longitude": [77.5 + (i % 10) / 1000 for i in range(row_count)],
        "area_sqft": [800 + i * 10 for i in range(row_count)],
        "BHK": [1 + i % 4 for i in range(row_count)],
        "bathrooms": [1 + i % 3 for i in range(row_count)],
        "balconies": [i % 3 for i in range(row_count)],
        "listing_date": [f"2025-{1 + i % 12:02d}-15" for i in range(row_count)],
        "property_age_years": [i % 10 for i in range(row_count)],
        "condition": ["Good" if i % 2 else "New" for i in range(row_count)],
        "floor_number": [i % 5 for i in range(row_count)],
        "total_floors": [5] * row_count,
        "property_type": ["Apartment"] * row_count,
        "furnishing_status": ["Unfurnished" if i % 2 else "Furnished" for i in range(row_count)],
        "parking_spaces": [i % 2 for i in range(row_count)],
        "lift": [i % 2 for i in range(row_count)],
        "security": [i % 2 for i in range(row_count)],
        "gym": [i % 2 for i in range(row_count)],
        "swimming_pool": [i % 2 for i in range(row_count)],
        "power_backup": [i % 2 for i in range(row_count)],
        "gated_community": [i % 2 for i in range(row_count)],
        "price_lakhs": [50 + i * 3 for i in range(row_count)],
    }
    return pd.DataFrame(data)


class PropertyModelTests(unittest.TestCase):
    def test_preprocessing_excludes_bad_targets_area_bhk_and_impossible_floors(self):
        frame = make_property_listings(5)
        frame["price_lakhs"] = frame["price_lakhs"].astype(object)
        frame["BHK"] = frame["BHK"].astype(object)
        frame.loc[1, "price_lakhs"] = "bad"
        frame.loc[2, "area_sqft"] = 0
        frame.loc[3, "BHK"] = 2.5
        frame.loc[4, "floor_number"] = 8
        X, y, report = prepare_property_data(frame)
        self.assertEqual(len(X), 1)
        self.assertEqual(len(y), 1)
        self.assertEqual(report["excluded_rows"], 4)
        self.assertEqual(report["exclusion_reasons"]["invalid_or_nonpositive_target"], 1)
        self.assertEqual(report["exclusion_reasons"]["invalid_or_nonpositive_area"], 1)
        self.assertEqual(report["exclusion_reasons"]["invalid_bhk"], 1)
        self.assertEqual(report["exclusion_reasons"]["invalid_floor_combination"], 1)
        self.assertNotIn(ID_COLUMN, X.columns)
        self.assertTrue((y > 0).all())

    def test_optional_corruption_is_marked_missing_for_pipeline_imputation(self):
        frame = make_property_listings()
        frame["latitude"] = frame["latitude"].astype(object)
        frame.loc[0, "latitude"] = "not-a-number"
        frame.loc[1, "condition"] = "   "
        frame.loc[2, "listing_date"] = "not a date"
        X, y, report = prepare_property_data(frame)
        self.assertTrue(pd.isna(X.loc[0, "latitude"]))
        self.assertEqual(X.loc[1, "condition"], "Unknown")
        self.assertTrue(pd.isna(X.loc[2, "listing_date"]))
        self.assertEqual(report["numeric_values_marked_missing_for_imputation"]["latitude"], 1)
        self.assertEqual(len(X), len(y))

    def test_exact_schema_excludes_identifier_and_target(self):
        frame = make_property_listings()
        X, _, _ = prepare_property_data(frame)
        self.assertEqual(list(X.columns), FEATURE_SCHEMA)
        self.assertNotIn("property_id", X.columns)
        self.assertNotIn(TARGET, X.columns)

    def test_prediction_schema_and_pipeline_smoke(self):
        frame = make_property_listings()
        X, y, _ = prepare_property_data(frame)
        model = make_pipeline(RandomForestRegressor, {"n_estimators": 10, "random_state": 7, "n_jobs": 1}, scale_numeric=False)
        model.fit(X, y)
        row = build_property_input(X.iloc[0].to_dict())
        self.assertEqual(list(row.columns), FEATURE_SCHEMA)
        prediction = predict_property(model, row)
        self.assertTrue(np.isfinite(prediction))
        unseen = row.iloc[0].to_dict()
        unseen["location"] = "Never-seen locality"
        unseen["condition"] = "Unseen condition"
        self.assertTrue(np.isfinite(predict_property(model, unseen)))
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            build_property_input({"property_id": "ignored", TARGET: 100})

    def test_minimal_user_input_accepts_unknown_optional_details(self):
        frame = make_property_listings()
        X, y, _ = prepare_property_data(frame)
        model = make_pipeline(
            RandomForestRegressor,
            {"n_estimators": 10, "random_state": 7, "n_jobs": 1},
            scale_numeric=False,
        )
        model.fit(X, y)
        values = {feature: None for feature in FEATURE_SCHEMA}
        values.update({
            "location": "Whitefield",
            "area_sqft": 1200,
            "BHK": 2,
            "bathrooms": 2,
            "balconies": 1,
        })
        row = build_property_input(values)
        self.assertTrue(pd.isna(row.loc[0, "latitude"]))
        self.assertTrue(pd.isna(row.loc[0, "property_age_years"]))
        self.assertTrue(pd.isna(row.loc[0, "listing_date"]))
        self.assertEqual(row.loc[0, "condition"], "Unknown")
        self.assertTrue(np.isfinite(predict_property(model, values)))

    def test_only_four_requested_regressors_have_grids_with_default_baselines(self):
        self.assertEqual(set(CANDIDATES), {
            "LinearRegression", "DecisionTreeRegressor",
            "RandomForestRegressor", "KNeighborsRegressor",
        })
        forbidden = {"GradientBoostingRegressor", "XGBRegressor", "ExtraTreesRegressor", "HistGradientBoostingRegressor"}
        self.assertFalse(forbidden.intersection(CANDIDATES))
        for name, spec in CANDIDATES.items():
            for key, choices in spec["grid"].items():
                self.assertIn(spec["baseline"][key], choices)
        self.assertIn("model__fit_intercept", CANDIDATES["LinearRegression"]["grid"])
        self.assertFalse(CANDIDATES["DecisionTreeRegressor"]["scale_numeric"])
        self.assertFalse(CANDIDATES["RandomForestRegressor"]["scale_numeric"])
        self.assertTrue(CANDIDATES["LinearRegression"]["scale_numeric"])
        self.assertTrue(CANDIDATES["KNeighborsRegressor"]["scale_numeric"])

    def test_gridsearch_candidate_uses_only_supplied_training_rows_and_reports_baseline(self):
        X, y, _ = prepare_property_data(make_property_listings())
        result = _tune_candidate(
            "LinearRegression", X.iloc[:30], y.iloc[:30],
            KFold(n_splits=3, shuffle=True, random_state=17),
        )
        self.assertIn("model__fit_intercept", result["best_params"])
        self.assertEqual(len(result["best_fold_rmse_lakhs"]), 3)
        self.assertGreater(result["baseline_cv_rmse_lakhs"], 0)
        self.assertGreater(result["best_cv_rmse_lakhs"], 0)
        self.assertTrue(result["search"].refit)

    def test_invalid_prediction_values_fail_clearly(self):
        frame = make_property_listings()
        row = frame.iloc[0].drop(labels=[ID_COLUMN, TARGET]).to_dict()
        row["BHK"] = 2.5
        with self.assertRaisesRegex(ValueError, "BHK must be a whole number"):
            build_property_input(row)
        row["BHK"] = 2
        row["area_sqft"] = 0
        with self.assertRaisesRegex(ValueError, "area_sqft must be positive"):
            build_property_input(row)
        row["area_sqft"] = 1200
        row["floor_number"] = 7
        row["total_floors"] = 5
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_property_input(row)


if __name__ == "__main__":
    unittest.main()
