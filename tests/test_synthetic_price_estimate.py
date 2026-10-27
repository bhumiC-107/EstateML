import unittest
from pathlib import Path

from estate_ml.property_model import FEATURE_SCHEMA, load_property_artifacts, predict_property


ROOT = Path(__file__).resolve().parents[1]
DEMO_MODEL_DIR = ROOT / "artifacts" / "model" / "synthetic_price"


class SyntheticPriceEstimateTests(unittest.TestCase):
    def test_separate_synthetic_model_artifact_predicts_without_changing_main_model(self):
        model, metadata = load_property_artifacts(DEMO_MODEL_DIR)
        self.assertEqual(metadata["target"], "price_lakhs")
        self.assertEqual(list(getattr(model, "feature_names_in_", [])), FEATURE_SCHEMA)
        _, existing_metadata = load_property_artifacts(ROOT / "artifacts" / "model")
        self.assertNotEqual(existing_metadata.get("dataset_kind"), "synthetic")
        values = {feature: None for feature in FEATURE_SCHEMA}
        values.update({
            "location": "Whitefield",
            "area_sqft": 1200,
            "BHK": 2,
            "bathrooms": 2,
            "balconies": 1,
        })
        estimate = predict_property(model, values)
        self.assertGreater(estimate, 0)


if __name__ == "__main__":
    unittest.main()
