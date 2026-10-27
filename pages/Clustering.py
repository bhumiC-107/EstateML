from pathlib import Path

import streamlit as st

from estate_ml.property_model import load_property_artifacts, predict_property
from estate_ml.property_ui import friendly_input_message, render_property_inputs


BASE_DIR = Path(__file__).resolve().parent.parent
PRICE_MODEL_DIR = BASE_DIR / "artifacts" / "model"

st.set_page_config(
    page_title="EstateML — Property Price Estimator",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_price_model(artifact_version: int):
    """Load the existing supervised price model once per artifact version."""
    model, metadata = load_property_artifacts(PRICE_MODEL_DIR)
    categories = metadata.get("known_categories_from_random_training_partition", {})
    return model, metadata, categories.get("location", []), categories


try:
    metadata_path = PRICE_MODEL_DIR / "model_metadata.json"
    model, model_metadata, localities, categories = load_price_model(metadata_path.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"The property price model could not be loaded: {exc}")
    st.stop()

st.title("EstateML — Bengaluru Property Price Estimate")
st.write(
    "Enter the property details below to get a price estimate from the trained model. "
    "Add optional details if you know them; otherwise, the trained model handles missing information."
)

input_column, result_column = st.columns([1, 1], gap="large")

with input_column:
    with st.container(border=True):
        property_inputs = render_property_inputs(
            title="Property details",
            key="clustering_page_estimate",
            localities=localities,
            categories=categories,
        )
        estimate_clicked = st.button(
            "🏷 Estimate Property Value",
            type="primary",
            width="stretch",
            key="clustering_page_estimate_button",
        )
        if estimate_clicked:
            try:
                estimated_price = predict_property(model, property_inputs)
                st.session_state["clustering_page_prediction"] = {
                    "price_lakhs": estimated_price,
                    "inputs": property_inputs,
                }
            except Exception as exc:
                st.error(friendly_input_message(exc))

with result_column:
    with st.container(border=True):
        prediction = st.session_state.get("clustering_page_prediction")
        if prediction:
            values = prediction["inputs"]
            price_lakhs = float(prediction["price_lakhs"])
            price_per_sqft = price_lakhs * 100_000 / max(float(values["area_sqft"]), 1)
            st.subheader("Estimated property value")
            st.metric("Listing-based estimate", f"₹{price_lakhs:,.2f} lakhs")
            st.caption(
                (
                    "The model was trained on synthetic sample listings for demonstration; "
                    "this estimate may not reflect market values."
                    if model_metadata.get("dataset_kind") == "synthetic"
                    else "This estimate uses the project's supervised model and user-provided listings. "
                    "It is not a formal valuation or a guarantee of a sale price."
                )
            )
            details = st.columns(2)
            details[0].metric("Approx. price per sq ft", f"₹{price_per_sqft:,.0f}")
            details[1].metric("Property configuration", f"{values['BHK']} BHK · {values['bathrooms']} bath")
            st.write(f"**Locality:** {values['location']}")
            st.write(f"**Area:** {float(values['area_sqft']):,.0f} sq ft")
            st.write(f"**Balconies:** {values['balconies']}")
        else:
            st.subheader("Your estimate will appear here")
            st.info("Fill in the property details and choose **Estimate Property Value**.")

st.caption("The clustering model is separate from price prediction.")
