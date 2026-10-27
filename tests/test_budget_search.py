import unittest

from estate_ml.budget_search import MAX_CANDIDATES, SearchRequest, budget_to_lakhs, get_supported_categories, preference_summary, search_properties, tradeoff, tradeoff_display
from estate_ml.property_model import FEATURE_SCHEMA, predict_property_batch


META = {"known_categories_from_random_training_partition": {
    "location": ["Alpha", "Beta"], "condition": ["New", "Good"],
    "property_type": ["Apartment"], "furnishing_status": ["Unfurnished"]}}


def price(_model, values):
    # Deterministic fixture: a configuration at exactly ₹50 L is feasible.
    return 20 + values["area_sqft"] / 100 + values["BHK"] * 2 + (0 if values["location"] == "Alpha" else 3)


class BudgetSearchTests(unittest.TestCase):
    def req(self, **kwargs):
        values = {"budget_lakhs": 50, "localities": ("Alpha",), "min_area_sqft": 800,
                  "result_limit": 10}
        values.update(kwargs)
        return SearchRequest(**values)

    def test_inr_conversion_and_units(self):
        self.assertEqual(budget_to_lakhs(5_000_000, "INR"), 50)
        with self.assertRaises(ValueError): budget_to_lakhs(0, "INR")

    def test_candidates_budget_remaining_ranking_dedup_and_hard_constraints(self):
        request = self.req(bhk=2, min_bathrooms=2, balcony_required=True)
        result = search_properties(None, META, request, predictor=price)
        self.assertTrue(result.candidates)
        for candidate in result.candidates:
            self.assertLessEqual(candidate["predicted_price_lakhs"], 50)
            self.assertGreaterEqual(candidate["area_sqft"], 800)
            self.assertGreaterEqual(candidate["bathrooms"], 2)
            self.assertGreater(candidate["balconies"], 0)
            self.assertAlmostEqual(candidate["budget_remaining_lakhs"], 50-candidate["predicted_price_lakhs"])
            self.assertAlmostEqual(candidate["budget_utilization_pct"], candidate["predicted_price_lakhs"]/50*100)
        again = search_properties(None, META, request, predictor=price)
        self.assertEqual(result.candidates, again.candidates)
        self.assertEqual(len({(r['location'],r['BHK'],r['area_sqft'],r['bathrooms'],r['balconies']) for r in result.candidates}), len(result.candidates))

    def test_low_budget_and_exact_price_boundary(self):
        exact_request = self.req(bhk=2, min_area_sqft=800, min_bathrooms=2, balcony_required=True)
        # Fixture override makes all configurations equal to the budget exactly.
        exact = search_properties(None, META, exact_request, predictor=lambda *_: 50)
        self.assertTrue(exact.candidates)
        self.assertTrue(all(r['budget_remaining_lakhs'] == 0 for r in exact.candidates))
        just_over = search_properties(None, META, exact_request, predictor=lambda *_: 50.00001)
        self.assertEqual(just_over.candidates, [])
        low = search_properties(None, META, SearchRequest(1, ("Alpha",), min_area_sqft=800), predictor=price)
        self.assertEqual(low.candidates, [])

    def test_invalid_locality_and_optional_amenities(self):
        with self.assertRaisesRegex(ValueError, "supported localities"):
            search_properties(None, META, SearchRequest(50, ("Nowhere",)), predictor=price)
        result = search_properties(None, META, SearchRequest(50, ("Alpha",), amenities=("gym",), result_limit=50), predictor=price)
        self.assertTrue(result.candidates)
        self.assertIn(result.candidates[0]["gym"], (0, 1))
        self.assertEqual({row["gym"] for row in result.candidates}, {0, 1})

    def test_hard_filters_are_never_relaxed_by_ranking(self):
        # Even a high score cannot admit over-budget/undersized/underspecified rows.
        request = self.req(bhk=4, min_area_sqft=1200, min_bathrooms=4,
                           balcony_required=True, priority="maximize area")
        result = search_properties(None, META, request, predictor=lambda *_: 51)
        self.assertEqual(result.candidates, [])
        with self.assertRaisesRegex(ValueError, "Minimum bathrooms"):
            search_properties(None, META, self.req(min_bathrooms=7), predictor=price)
        with self.assertRaisesRegex(ValueError, "at most 3 bathrooms"):
            search_properties(None, META, self.req(bhk=2, min_bathrooms=4), predictor=price)

    def test_bhk_bathroom_cap_and_batched_model_inference(self):
        class BatchModel:
            def __init__(self):
                self.calls = []

            def predict(self, frame):
                self.calls.append(frame.copy())
                self.assert_schema(frame)
                return [30.0] * len(frame)

            @staticmethod
            def assert_schema(frame):
                if list(frame.columns) != FEATURE_SCHEMA:
                    raise AssertionError("Incomplete model input schema")

        model = BatchModel()
        result = search_properties(model, META, self.req(bhk=2, min_bathrooms=1), max_candidates=80)
        self.assertTrue(result.candidates)
        self.assertTrue(all(row["bathrooms"] <= 3 for row in result.candidates))
        self.assertEqual(len(model.calls), 1)

        rows = [{feature: None for feature in FEATURE_SCHEMA}]
        rows[0].update({"location": "Alpha", "area_sqft": 1000, "BHK": 2,
                        "bathrooms": 2, "balconies": 0})
        self.assertEqual(predict_property_batch(model, rows), [30.0])

    def test_category_choices_are_soft_ranking_preferences(self):
        request = SearchRequest(500, ("Alpha",), condition="New", result_limit=50)
        result = search_properties(None, META, request, predictor=lambda *_: 25)
        self.assertTrue(result.candidates)
        self.assertEqual(result.candidates[0]["condition"], "New")
        self.assertIn("Good", {row["condition"] for row in result.candidates})

    def test_missing_category_metadata_falls_back_to_saved_encoder_and_warns(self):
        class Encoder:
            feature_names_in_ = ["location", "condition", "property_type", "furnishing_status"]
            categories_ = [["Alpha", "Beta"], ["Good", "New"], ["Apartment"], ["Unfurnished"]]
        class Model:
            named_steps = {"preprocess": type("Preprocess", (), {
                "named_transformers_": {"categorical": type("Pipe", (), {
                    "named_steps": {"onehot": Encoder()}})()}})()}
        categories = get_supported_categories(Model(), {})
        self.assertEqual(categories["location"], ["Alpha", "Beta"])
        result = search_properties(Model(), {}, SearchRequest(50, ("Alpha",)), predictor=price)
        self.assertIn("cannot be verified", result.candidates[0]["support_note"])

    def test_recorded_ranges_and_category_deduplication(self):
        metadata = {**META, "training_feature_ranges": {"area_sqft": [900, 1000]}}
        request = self.req(localities=("Alpha", "Alpha"), min_area_sqft=800)
        result = search_properties(None, metadata, request, predictor=price)
        self.assertEqual(result.supported_localities, ("Alpha",))
        self.assertTrue(any(row["out_of_training_range"] for row in result.candidates))

    def test_bounded_search_and_tradeoff_math(self):
        req = SearchRequest(50, ("Alpha",), include_alternatives=True, result_limit=20)
        result = search_properties(None, META, req, predictor=price, max_candidates=80)
        self.assertLessEqual(result.evaluated, 80)
        self.assertIn("Beta", result.supported_localities)
        if len(result.candidates) > 1:
            delta = tradeoff(result.candidates[0], result.candidates[1], 50)
            self.assertAlmostEqual(delta["difference_lakhs"], result.candidates[1]["predicted_price_lakhs"]-result.candidates[0]["predicted_price_lakhs"])
            self.assertTrue(delta["dimensions"])
            self.assertAlmostEqual(delta["baseline_remaining_lakhs"], 50-result.candidates[0]["predicted_price_lakhs"])
            self.assertAlmostEqual(delta["alternative_remaining_lakhs"], 50-result.candidates[1]["predicted_price_lakhs"])

    def test_tradeoff_reports_category_and_amenity_dimensions(self):
        baseline = {"predicted_price_lakhs": 40, "condition": "New", "gym": 0}
        alternative = {"predicted_price_lakhs": 44, "condition": "Good", "gym": 1}
        delta = tradeoff(baseline, alternative, 50)
        self.assertEqual(delta["difference_lakhs"], 4)
        self.assertAlmostEqual(delta["difference_pct"], 10)
        self.assertIn("condition", delta["dimensions"])
        self.assertIn("gym", delta["dimensions"])
        self.assertEqual(delta["alternative_remaining_lakhs"], 6)

    def test_display_helpers_handle_zero_and_missing_values(self):
        baseline = {"predicted_price_lakhs": 40, "area_sqft": None, "BHK": 2,
                    "condition": "Unknown", "location": "Alpha", "gym": None}
        alternative = {"predicted_price_lakhs": 40, "area_sqft": None, "BHK": 2,
                       "condition": "Good", "location": "Alpha", "gym": None}
        display = tradeoff_display(baseline, alternative, 50)
        self.assertEqual(display["price_difference_display"], "No change")
        self.assertEqual(display["price_difference_pct_display"], "No change")
        self.assertEqual(display["area_difference_display"], "Unavailable")
        self.assertEqual(display["bhk_difference_display"], "No change")
        self.assertEqual(display["category_differences"]["Condition"], "Unspecified → Good")
        self.assertEqual(display["features_gained"], [])

    def test_preference_summary_identifies_met_and_missed_soft_preferences(self):
        request = SearchRequest(50, ("Alpha",), condition="New", amenities=("gym",))
        met, missed = preference_summary({"location": "Beta", "condition": "Good", "gym": 0}, request)
        self.assertEqual(met, [])
        self.assertEqual(set(missed), {"Condition: New", "Gym", "Preferred locality"})

    def test_large_search_is_capped_even_when_override_is_larger(self):
        request = SearchRequest(10000, ("Alpha",), include_alternatives=True,
                                amenities=("lift", "security", "gym", "swimming_pool", "power_backup", "gated_community"),
                                result_limit=50)
        result = search_properties(None, META, request, predictor=lambda *_: 10, max_candidates=MAX_CANDIDATES + 1000)
        self.assertLessEqual(result.evaluated, MAX_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
