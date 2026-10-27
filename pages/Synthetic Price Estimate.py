"""Optional price estimator trained on the bundled synthetic demonstration CSV."""
from pathlib import Path

import streamlit as st

from estate_ml.property_model import load_property_artifacts, predict_property
from estate_ml.property_ui import friendly_input_message, render_property_inputs


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "artifacts" / "model" / "synthetic_price"

st.set_page_config(
    page_title="EstateML — Synthetic Price Demo",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_demo_model(artifact_version: int):
    model, metadata = load_property_artifacts(MODEL_DIR)
    categories = metadata.get("known_categories_from_random_training_partition", {})
    return model, metadata, categories


try:
    metadata_path = MODEL_DIR / "model_metadata.json"
    model, metadata, categories = load_demo_model(metadata_path.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"The optional synthetic-data model could not be loaded: {exc}")
    st.stop()

st.title("🧪 Synthetic-data price estimate")
st.warning(
    "This separate demo model uses synthetic Bengaluru property listings. "
    "Its estimates and evaluation scores are not evidence of real-market accuracy. "
    "The existing EstateML estimator remains unchanged."
)
st.write("Enter property details to get an estimate from the model trained on the supplied dataset.")

form_column, result_column = st.columns([1, 1], gap="large")
with form_column:
    with st.container(border=True):
        values = render_property_inputs(
            title="Property details",
            key="synthetic_price_demo",
            localities=categories.get("location", []),
            categories=categories,
        )
        estimate = st.button(
            "Estimate using synthetic-data model",
            type="primary",
            width="stretch",
            key="synthetic_price_estimate_button",
        )
        if estimate:
            try:
                price = predict_property(model, values)
                st.session_state["synthetic_price_demo_result"] = {
                    "price_lakhs": price,
                    "values": values,
                }
            except Exception as exc:
                st.error(friendly_input_message(exc))

with result_column:
    with st.container(border=True):
        result = st.session_state.get("synthetic_price_demo_result")
        if result:
            price = float(result["price_lakhs"])
            area = max(float(result["values"]["area_sqft"]), 1.0)
            st.subheader("Demo estimate")
            st.metric("Estimated price", f"₹{price:,.2f} lakhs")
            st.metric("Approx. price per sq ft", f"₹{price * 100_000 / area:,.0f}")
            st.caption("Synthetic-data demonstration only; not a real property valuation.")
        else:
            st.subheader("Your estimate will appear here")
            st.info("Enter property details and click the estimate button.")

with st.expander("How this demo model was evaluated"):
    primary = metadata.get("primary_selected_holdout_metrics", {})
    chronological = metadata.get("chronological_supplementary_evaluation", {}).get(
        "selected_holdout_metrics", {}
    )
    st.write(
        f"Model: {metadata.get('model_name', 'Unknown')} · "
        f"Training rows: {metadata.get('training_rows', 'N/A'):,} · "
        f"dataset: {metadata.get('dataset_file', 'N/A')}"
    )
    st.dataframe(
        [
            {
                "Evaluation": "Random holdout",
                "R² (not accuracy)": primary.get("r2"),
                "MAE (₹ lakhs)": primary.get("mae_lakhs"),
                "RMSE (₹ lakhs)": primary.get("rmse_lakhs"),
            },
            {
                "Evaluation": "Later-date holdout",
                "R² (not accuracy)": chronological.get("r2"),
                "MAE (₹ lakhs)": chronological.get("mae_lakhs"),
                "RMSE (₹ lakhs)": chronological.get("rmse_lakhs"),
            },
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Scores summarize performance on synthetic records only. R² is not a percentage accuracy score.")
