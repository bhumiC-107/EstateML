import streamlit as st
import pandas as pd
import html
from pathlib import Path
from estate_ml.property_model import load_property_artifacts, predict_property
from estate_ml.property_ui import friendly_input_message, render_property_inputs

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Property Comparator", layout="wide")

st.markdown("""
<style>
/* ── Page Background & Base Text ── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #f5f5f0 !important;
    color: #202238 !important;
}

[data-testid="stAppViewContainer"] .main,
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] {
    color: #202238 !important;
}

/* ── Typography & Headings ── */
h1, h2, h3, h4,
[data-testid="stAppViewContainer"] .main h1,
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3,
[data-testid="stAppViewContainer"] .main h4 {
    font-family: Georgia, 'Times New Roman', serif !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* ── Captions & Subtitles ── */
[data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"],
.stCaption {
    color: #596174 !important;
}

/* ── Input Labels & Fields ── */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] div {
    color: #1a1a2e !important;
    font-weight: 600 !important;
}

[data-testid="stAppViewContainer"] .main [data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

[data-testid="stAppViewContainer"] .main [data-baseweb="input"] input {
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}

[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {
    color: #1a1a2e !important;
}

/* ── Buttons ── */
div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background-color: #1a1a2e !important;
    color: #e2b96f !important;
    border: 1px solid #1a1a2e !important;
    padding: 0.6rem 2.5rem !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
}
div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #0f3460 !important;
    color: #ffffff !important;
    border-color: #0f3460 !important;
}

/* ── Cards ── */
.prop-card {
    background: #1a1a2e !important;
    border-radius: 12px !important;
    padding: 1.5rem 2rem !important;
    margin-bottom: 1rem !important;
    text-align: center !important;
    border: 1px solid #2d3748 !important;
}
.prop-card * {
    color: #f7fafc !important;
}
.prop-label {
    color: #c0c8d8 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.prop-price {
    color: #e2b96f !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0.3rem 0 !important;
}
.prop-range {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
}
.winner-badge {
    background: #e2b96f !important;
    color: #1a1a2e !important;
    border-radius: 20px !important;
    padding: 0.25rem 0.9rem !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    display: inline-block !important;
    margin-top: 0.5rem !important;
}

/* ── Tables & Containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #dde2ec !important;
    box-shadow: 0 2px 10px rgba(26, 26, 46, 0.06) !important;
    padding: 1.5rem !important;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #d9deea !important;
    background: #ffffff !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
}
section[data-testid="stSidebar"] * {
    color: #e2b96f !important;
}
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavLink"] span {
    color: #e2b96f !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background-color: #2d3748 !important;
}
section[data-testid="stSidebar"] a {
    color: #e2b96f !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return load_property_artifacts(BASE_DIR / "artifacts" / "model")

model, model_metadata = load_assets()
model_categories = model_metadata.get("known_categories_from_random_training_partition", {})
locality_options = model_categories.get("location", [])


# ── Page Header ───────────────────────────────────────────────────────────
st.title("⚖️ Property Comparator")
st.markdown("Compare two properties using the same model trained on curated Bengaluru listings.")
st.caption("Listing prices can differ from completed sale prices; treat these as estimates, not formal valuations.")

st.divider()

# ── Input Panels ──────────────────────────────────────────────────────────
col1, div, col2 = st.columns([5, 1, 5])

with col1:
    with st.container(border=True):
        property_a = render_property_inputs(
            title="Property A", key="compare_a", localities=locality_options,
            categories=model_categories, area_default=1000,
        )

with div:
    st.markdown("<div style='text-align:center;font-size:1.6rem;color:#596174;font-weight:800;margin-top:7rem;'>VS</div>", unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        property_b = render_property_inputs(
            title="Property B", key="compare_b", localities=locality_options,
            categories=model_categories, area_default=1500,
        )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    compare_clicked = st.button("⚖️ Compare Properties", width="stretch")

# ── Results ───────────────────────────────────────────────────────────────
if compare_clicked:
    try:
        property_a_price = predict_property(model, property_a)
        property_b_price = predict_property(model, property_b)
    except (TypeError, ValueError) as exc:
        st.info(friendly_input_message(exc))
        st.stop()

    property_a_area_sqft = property_a["area_sqft"]
    property_b_area_sqft = property_b["area_sqft"]
    property_a_bhk = property_a["BHK"]
    property_b_bhk = property_b["BHK"]
    property_a_bathrooms = property_a["bathrooms"]
    property_b_bathrooms = property_b["bathrooms"]
    property_a_balconies = property_a["balconies"]
    property_b_balconies = property_b["balconies"]
    property_a_location = property_a["location"]
    property_b_location = property_b["location"]
    property_a_price_per_sqft = property_a_price * 100000 / property_a_area_sqft
    property_b_price_per_sqft = property_b_price * 100000 / property_b_area_sqft

    lower_estimate = (
        "A" if property_a_price < property_b_price
        else "B" if property_b_price < property_a_price
        else None
    )

    st.markdown("###  Comparison Results")

    r1, r2 = st.columns(2)

    with r1:
        badge = '<div class="winner-badge">Lower estimate</div>' if lower_estimate == "A" else ''
        st.markdown(f"""
        <div class="prop-card">
            <div class="prop-label">Property A · {html.escape(str(property_a_location))} · {int(property_a_bhk)} BHK · {int(property_a_area_sqft)} sqft</div>
            <div class="prop-price">₹{property_a_price:.2f} L</div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        badge = '<div class="winner-badge">Lower estimate</div>' if lower_estimate == "B" else ''
        st.markdown(f"""
        <div class="prop-card">
            <div class="prop-label">Property B · {html.escape(str(property_b_location))} · {int(property_b_bhk)} BHK · {int(property_b_area_sqft)} sqft</div>
            <div class="prop-price">₹{property_b_price:.2f} L</div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    # Metrics comparison table
    st.markdown("###  Feature-by-Feature Breakdown")
    metrics_df = pd.DataFrame({
        "Metric":          ["Predicted Price (L)", "Price/Sqft (₹)", "Total Area (sqft)", "BHK", "Bathrooms", "Balconies"],
        "Property A": [f"₹{property_a_price:.2f}", f"₹{property_a_price_per_sqft:,.0f}", f"{property_a_area_sqft:.0f}", str(int(property_a_bhk)), str(int(property_a_bathrooms)), str(int(property_a_balconies))],
        "Property B": [f"₹{property_b_price:.2f}", f"₹{property_b_price_per_sqft:,.0f}", f"{property_b_area_sqft:.0f}", str(int(property_b_bhk)), str(int(property_b_bathrooms)), str(int(property_b_balconies))],
    })
    st.dataframe(metrics_df, width="stretch", hide_index=True)
    st.caption("The lower estimate is not necessarily the better value. Compare properties with similar condition, age, floor, and amenities.")

    # Give buyers a simple summary based on the priorities they can compare.
    st.markdown("### Quick summary")
    if lower_estimate:
        price_gap = abs(property_a_price - property_b_price)
        st.markdown(
            f"- **Lower upfront estimate:** Property {lower_estimate} is about ₹{price_gap:,.2f} lakh less."
        )
    else:
        st.markdown("- **Estimated price:** Both properties have the same estimate.")

    lower_cost_sqft = "A" if property_a_price_per_sqft < property_b_price_per_sqft else "B" if property_b_price_per_sqft < property_a_price_per_sqft else None
    if lower_cost_sqft:
        st.markdown(
            f"- **Lower estimated cost per square foot:** Property {lower_cost_sqft} "
            f"(₹{min(property_a_price_per_sqft, property_b_price_per_sqft):,.0f}/sq ft). "
            "This can help compare space for the price when the properties are otherwise similar."
        )
    else:
        st.markdown("- **Cost per square foot:** Both properties are about the same.")

    if property_a_area_sqft != property_b_area_sqft:
        roomier = "A" if property_a_area_sqft > property_b_area_sqft else "B"
        larger_area = abs(property_a_area_sqft - property_b_area_sqft)
        st.markdown(
            f"- **More floor space:** Property {roomier} has {larger_area:,.0f} more sq ft."
        )
    if property_a_bhk != property_b_bhk:
        more_bedrooms = "A" if property_a_bhk > property_b_bhk else "B"
        st.markdown(f"- **More bedrooms:** Property {more_bedrooms} has {abs(property_a_bhk - property_b_bhk)} more bedroom(s).")
    st.info("There isn’t one best choice for everyone: a lower estimate may suit a tighter budget, while more space or a lower cost per square foot may matter more to you. Estimates are based on curated listings and may differ from sale prices.")

    st.markdown("### Explore the finances")
    finance_property = st.selectbox(
        "Choose which property estimate to use",
        ["Property A", "Property B"],
        key="comparison_finance_property",
        help="We’ll carry this estimate into the EMI calculator so you can explore a loan payment.",
    )
    selected_estimate = property_a_price if finance_property == "Property A" else property_b_price
    st.session_state["comparison_property_price_lakhs"] = float(selected_estimate)
    st.page_link(
        "pages/finance.py",
        label="Continue to Finance",
        icon=":material/account_balance:",
        width="stretch",
    )

    # Download comparison
    st.download_button(
        "⬇ Download Comparison as CSV",
        data=metrics_df.to_csv(index=False),
        file_name="property_comparison.csv",
        mime="text/csv"
    )

st.caption("EstateML Property Intelligence · Side-by-Side Property Comparison")
