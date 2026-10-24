"""Smoke check for the property model used by the active Streamlit app."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from estate_ml.property_model import FEATURE_SCHEMA, load_property_artifacts, predict_property


def main() -> None:
    model, metadata = load_property_artifacts(ROOT / "artifacts" / "model")
    assert metadata.get("label") == "curated listing evaluation"
    assert metadata.get("feature_schema") == FEATURE_SCHEMA
    assert list(getattr(model, "feature_names_in_", [])) == FEATURE_SCHEMA
    print(f"[PASS] property model loaded with the {len(FEATURE_SCHEMA)}-feature schema")

    values = {feature: None for feature in FEATURE_SCHEMA}
    values.update({
        "location": "Whitefield",
        "area_sqft": 1200,
        "BHK": 2,
        "bathrooms": 2,
        "balconies": 1,
    })
    known_estimate = predict_property(model, values)
    values["location"] = "Unseen test locality"
    unseen_estimate = predict_property(model, values)
    assert all(math.isfinite(value) and value > 0 for value in (known_estimate, unseen_estimate))
    print("[PASS] model predicts finite positive prices with unknown optional details")
    print(f"[INFO] 1200 sq ft, 2 BHK: known locality INR {known_estimate:.2f} L; unseen locality INR {unseen_estimate:.2f} L")


if __name__ == "__main__":
    main()
