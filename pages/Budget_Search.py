from pathlib import Path

import pandas as pd
import streamlit as st

from estate_ml.budget_search import (
    AMENITIES,
    SearchRequest,
    budget_to_lakhs,
    get_supported_categories,
    preference_summary,
    search_properties,
    tradeoff_display,
)
from estate_ml.property_model import load_property_artifacts

BASE_DIR = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="What Can I Get for My Budget?", page_icon="🏠", layout="wide")
st.markdown("""
<style>
:root { --navy:#1a1a2e; --navy2:#2d3748; --gold:#c9922a; --gold-lt:#f0d080;
        --page-bg:#f5f5f0; --ink:#1a1a2e; --muted:#596174; --border:#dde2ec; }
.stApp { background:var(--page-bg); color:var(--ink); }
h1,h2,h3,h4 { color:var(--ink); }
[data-testid="stVerticalBlockBorderWrapper"] {
  background:#fff; border:1px solid var(--border); border-radius:12px;
  box-shadow:0 2px 10px rgba(26,26,46,.06); padding:1.1rem;
}
section[data-testid="stSidebar"] { background:var(--navy); }
section[data-testid="stSidebar"] * { color:var(--gold-lt); }
.budget-hero { background:var(--navy); color:#f7f6f2; border-radius:12px;
  padding:1.6rem 2rem; margin-bottom:1rem; }
.budget-hero h1 { color:#f7f6f2; margin:0 0 .4rem; font-size:1.9rem; }
.budget-hero p { color:#c0c8d8; margin:0; line-height:1.55; }
.result-kicker { color:var(--muted); text-transform:uppercase; letter-spacing:.06em;
  font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    return load_property_artifacts(BASE_DIR / "artifacts" / "model")


try:
    model, metadata = load_assets()
except Exception as exc:
    st.error(f"The EstateML model could not be loaded: {exc}")
    st.stop()

categories = get_supported_categories(model, metadata)
localities = categories.get("location", [])

st.markdown("""
<div class="budget-hero">
  <h1>🏠 What Can I Get for My Budget?</h1>
  <p>Set a budget and priorities to explore hypothetical property configurations that the existing EstateML model estimates could fit.</p>
</div>
""", unsafe_allow_html=True)
st.caption("Model-generated configurations are not verified property listings or statements of availability.")
if not metadata.get("training_feature_ranges"):
    st.warning("The curated training rows and feature-range metadata are not bundled. The saved model supplies valid categories, but support for any particular combination cannot be verified.")
if not localities:
    st.error("The saved model contains no recoverable locality categories, so a safe locality search cannot run.")
    st.stop()

# Inputs are intentionally unwrapped Streamlit widgets so a changed value can
# invalidate a previous result immediately on rerun.
with st.container(border=True):
    st.subheader("1 · Set your budget")
    budget = st.number_input(
        "Maximum budget (₹ lakhs)", min_value=1.0, max_value=100_000.0,
        value=100.0, step=5.0, format="%.1f",
        help="The model predicts price in lakhs of INR. ₹100 lakh = ₹1 crore.",
    )

with st.container(border=True):
    st.subheader("2 · Choose locality")
    any_locality = "Any supported locality"
    locality_choice = st.multiselect(
        "Preferred locality or localities",
        [any_locality, *localities], default=[any_locality],
        help="Choosing a locality ranks it as preferred. Enable alternatives to include other model-supported localities.",
    )
    include_alternatives = st.checkbox(
        "Also consider other supported localities", value=False,
        help="Adds all supported locality categories as alternatives. No geographic distance is implied.",
    )

with st.container(border=True):
    st.subheader("3 · Property configuration")
    config_a, config_b = st.columns(2)
    with config_a:
        bhk_option = st.selectbox("Bedrooms", ["Flexible", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"], index=0)
        min_area = st.slider("Minimum area (sq ft)", 400, 4719, 400, step=100)
        bathroom_cap = 6 if bhk_option == "Flexible" else min(6, int(bhk_option.split()[0]) + 1)
    with config_b:
        min_bath = st.selectbox(
            f"Minimum bathrooms (maximum {bathroom_cap} for this BHK)",
            [1, 2, 3, 4, 5, 6], index=0,
            help="Generated configurations are capped at BHK + 1 bathrooms, up to six. Flexible BHK applies the cap separately to each candidate.",
        )
        balcony_required = st.checkbox("Require at least one balcony", value=False)
    bathroom_invalid = bhk_option != "Flexible" and min_bath > bathroom_cap
    if bathroom_invalid:
        st.error(f"A {bhk_option} configuration can have at most {bathroom_cap} bathrooms in this search. Lower the minimum or choose a larger BHK.")

with st.expander("Optional preferences", expanded=False):
    pref_a, pref_b, pref_c = st.columns(3)
    with pref_a:
        property_type = st.selectbox("Preferred property type", ["Any", *categories.get("property_type", [])])
    with pref_b:
        furnishing = st.selectbox("Preferred furnishing", ["Any", *categories.get("furnishing_status", [])])
    with pref_c:
        condition = st.selectbox("Preferred condition", ["Any", *categories.get("condition", [])])
    amenity_labels = {
        "lift": "Lift", "security": "Security", "gym": "Gym",
        "swimming_pool": "Swimming pool", "power_backup": "Power backup",
        "gated_community": "Gated community",
    }
    amenity_columns = st.columns(3)
    selected_amenities = tuple(
        amenity for index, amenity in enumerate(AMENITIES)
        if amenity_columns[index % len(amenity_columns)].checkbox(amenity_labels[amenity], key=f"budget_{amenity}")
    )
    priority = st.selectbox(
        "Ranking priority", ["overall fit", "maximize area", "bedroom count", "amenities"],
        help="Changes ranking among configurations that meet every hard constraint.",
    )

st.caption("Hard filters: budget, selected BHK (unless flexible), minimum area, minimum bathrooms, and a required balcony. Locality is a search scope. Amenity, property type, furnishing, and condition are soft preferences.")

if not locality_choice:
    chosen_localities: tuple[str, ...] = ()
elif any_locality in locality_choice:
    chosen_localities = tuple(localities)
else:
    chosen_localities = tuple(locality_choice)

request = SearchRequest(
    budget_lakhs=budget_to_lakhs(budget, "lakhs"),
    localities=chosen_localities,
    bhk=None if bhk_option == "Flexible" else int(bhk_option.split()[0]),
    min_area_sqft=min_area,
    min_bathrooms=min_bath,
    balcony_required=balcony_required,
    property_type="Unknown" if property_type == "Any" else property_type,
    furnishing_status="Unknown" if furnishing == "Any" else furnishing,
    condition="Unknown" if condition == "Any" else condition,
    amenities=selected_amenities,
    priority=priority,
    include_alternatives=include_alternatives,
)

search_clicked = st.button("🔎 Find configurations", type="primary", use_container_width=True, disabled=bathroom_invalid)
if search_clicked:
    st.session_state.pop("budget_search_result", None)
    if not chosen_localities:
        st.warning("Choose at least one locality, or select Any supported locality.")
    else:
        try:
            found = search_properties(model, metadata, request)
            st.session_state["budget_search_result"] = (found, request)
        except ValueError as exc:
            st.error(str(exc))

stored = st.session_state.get("budget_search_result")
if stored and stored[1] != request:
    st.session_state.pop("budget_search_result", None)
    stored = None
    st.info("Your search preferences changed. Run the search again to refresh recommendations.")

st.markdown("## Search results")
if not stored and not bathroom_invalid and not search_clicked:
    st.info("Set your preferences, then click **Find configurations**. Recommendations will appear here.")

if stored:
    result, saved_request = stored
    st.caption(f"Evaluated {result.evaluated:,} candidates. Prices are model estimates in ₹ lakhs.")
    if not result.candidates:
        st.warning("No configuration met your budget and hard requirements. You can widen the locality scope, lower the minimum area, or make BHK flexible, then run the search again. No requirement was relaxed automatically.")
    else:
        preference_names = {
            "lift": "Lift", "security": "Security", "gym": "Gym",
            "swimming_pool": "Swimming pool", "power_backup": "Power backup",
            "gated_community": "Gated community",
        }
        for start in range(0, len(result.candidates), 2):
            card_columns = st.columns(2)
            for column, index in zip(card_columns, range(start, min(start + 2, len(result.candidates)))):
                candidate = result.candidates[index]
                with column:
                    with st.container(border=True):
                        st.markdown(f'<div class="result-kicker">Hypothetical configuration · #{index + 1}</div>', unsafe_allow_html=True)
                        st.markdown(f"### {candidate['location']} · {candidate['BHK']} BHK")
                        st.caption("Meets all hard filters for this search")
                        price_col, remaining_col, utilization_col = st.columns(3)
                        price = candidate["predicted_price_lakhs"]
                        price_col.metric("Estimated price", f"₹{price:,.2f} L")
                        price_col.caption(f"₹{price / 100:,.2f} crore")
                        remaining_col.metric("Budget remaining", f"₹{candidate['budget_remaining_lakhs']:,.2f} L")
                        utilization_col.metric("Budget used", f"{candidate['budget_utilization_pct']:.1f}%")
                        st.write(f"**Configuration:** {candidate['area_sqft']:,.0f} sq ft · {candidate['bathrooms']} bathrooms · {candidate['balconies']} balconies")
                        details = [candidate.get("property_type"), candidate.get("furnishing_status"), candidate.get("condition")]
                        details = [value for value in details if value not in (None, "", "Unknown")]
                        st.write("**Property details:** " + (" · ".join(details) if details else "Not specified"))
                        amenity_status = [
                            f"{preference_names[name]}: {'Yes' if candidate.get(name) == 1 else 'No'}"
                            for name in saved_request.amenities
                        ]
                        if amenity_status:
                            st.write("**Selected amenity preferences:** " + " · ".join(amenity_status))
                        met, missed = preference_summary(candidate, saved_request)
                        if met:
                            st.success("Soft preferences met: " + ", ".join(met))
                        if missed:
                            st.info("Soft preferences not met: " + ", ".join(missed))
                        st.caption("This is a model-generated hypothetical configuration, not a verified available property.")
                        st.caption(candidate["support_note"])

        st.subheader("Compare recommendations")
        selected_indexes = st.multiselect(
            "Choose up to three configurations for a side-by-side view",
            range(len(result.candidates)),
            format_func=lambda i: f"{i + 1}. {result.candidates[i]['location']} · {result.candidates[i]['BHK']} BHK · {result.candidates[i]['area_sqft']} sq ft",
            max_selections=3,
        )
        if len(selected_indexes) >= 2:
            comparison_rows = []
            for index in selected_indexes:
                candidate = result.candidates[index]
                comparison_rows.append({
                    "Configuration": f"{candidate['location']} · {candidate['BHK']} BHK",
                    "Estimated price (₹ L)": f"₹{candidate['predicted_price_lakhs']:,.2f}",
                    "Budget remaining (₹ L)": f"₹{candidate['budget_remaining_lakhs']:,.2f}",
                    "Budget used": f"{candidate['budget_utilization_pct']:.1f}%",
                    "Area (sq ft)": candidate["area_sqft"], "Bathrooms": candidate["bathrooms"],
                    "Balconies": candidate["balconies"], "Property type": candidate["property_type"],
                    "Furnishing": candidate["furnishing_status"], "Condition": candidate["condition"],
                })
            st.dataframe(pd.DataFrame(comparison_rows).set_index("Configuration").T, use_container_width=True)

        st.subheader("What Could I Get If I Compromise?")
        if len(result.candidates) >= 2:
            baseline_index = st.selectbox(
                "Baseline configuration", range(len(result.candidates)),
                format_func=lambda i: f"{i + 1}. {result.candidates[i]['location']} · {result.candidates[i]['BHK']} BHK · {result.candidates[i]['area_sqft']} sq ft",
            )
            baseline = result.candidates[baseline_index]
            alternative_rows = []
            for index, alternative in enumerate(result.candidates):
                if index == baseline_index:
                    continue
                comparison = tradeoff_display(baseline, alternative, saved_request.budget_lakhs)
                row = {
                    "Alternative": f"{alternative['location']} · {alternative['BHK']} BHK · {alternative['area_sqft']} sq ft",
                    "Changed dimensions": ", ".join(comparison["dimensions"]) or "No change",
                    "Alternative − baseline price (₹ L)": comparison["price_difference_display"],
                    "Price difference (%)": comparison["price_difference_pct_display"],
                    "Alternative − baseline area": comparison["area_difference_display"],
                    "Alternative − baseline BHK": comparison["bhk_difference_display"],
                    "Budget remaining (baseline → alternative)": comparison["budget_remaining_display"],
                    "Features gained by alternative": ", ".join(comparison["features_gained"]) or "None",
                    "Features given up by alternative": ", ".join(comparison["features_given_up"]) or "None",
                }
                row.update(comparison["category_differences"])
                alternative_rows.append(row)
            st.caption("Differences compare model estimates and input configurations; they do not establish equal availability, location quality, or property condition.")
            st.dataframe(pd.DataFrame(alternative_rows), use_container_width=True, hide_index=True)
        else:
            st.info("At least two recommendations are needed to compare trade-offs. Try widening the locality scope or using flexible BHK.")

with st.expander("How these recommendations work"):
    st.write("The search generates a bounded set of hypothetical configurations from model-supported categories and documented feature ranges. The existing EstateML model estimates each candidate's asking price. The search filters by your budget and hard requirements, then ranks feasible results by your selected priority and soft preferences. Trade-off insights compare those model predictions and configuration inputs.")
    st.caption("These estimates are based on the saved model and its training data. They are not listings, confirmed asking prices, sale prices, or exact market values. The bundled artifacts cannot verify whether a particular feature combination is represented in the training data.")
