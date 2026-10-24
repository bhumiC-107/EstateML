"""Shared Streamlit inputs for the property estimate and comparison pages."""
from __future__ import annotations

from typing import Any
from datetime import date

import streamlit as st

from estate_ml.property_model import FEATURE_SCHEMA, normalize_property_location


OTHER_LOCALITY = "Other locality"


def friendly_input_message(error: Exception) -> str:
    """Turn validation details into a short message a home buyer can act on."""
    details = str(error).lower()
    if "floor_number" in details and "total_floors" in details:
        return "The property floor can’t be higher than the building’s total floors. Please check those two values."
    if "area_sqft" in details or "area" in details:
        return "Please enter a property area between 400 and 4,719 square feet."
    if "bhk" in details or "bedroom" in details:
        return "Please choose a bedroom count from the options shown."
    if "bathroom" in details:
        return "Please choose a bathroom count from the options shown."
    if "balcon" in details:
        return "Please choose a balcony count from the options shown."
    if "location" in details or "locality" in details:
        return "Please choose a locality or enter its name, then try again."
    return "Some property details need a quick check. Please review the fields and try again."


def render_property_inputs(
    *,
    title: str,
    key: str,
    localities: list[str],
    categories: dict[str, list[str]],
    area_default: float = 1000,
) -> dict[str, Any]:
    """Render essential details first and optional details in a collapsed expander."""
    st.subheader(title)

    locality_choice = st.selectbox(
        "Bengaluru locality",
        ["Unknown", *localities, OTHER_LOCALITY],
        key=f"{key}_locality",
    )
    if locality_choice == OTHER_LOCALITY:
        locality = normalize_property_location(
            st.text_input("Enter locality", key=f"{key}_locality_text")
        )
    else:
        locality = locality_choice

    area, bedrooms = st.columns(2)
    with area:
        area_sqft = st.number_input(
            "Area (sq ft)", min_value=400, max_value=4719,
            value=int(area_default), step=50, key=f"{key}_area",
        )
    with bedrooms:
        bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1, key=f"{key}_bhk")

    bathrooms, balconies = st.columns(2)
    with bathrooms:
        bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1, key=f"{key}_bath")
    with balconies:
        balcony = st.selectbox("Balconies", [0, 1, 2, 3, 4], index=1, key=f"{key}_balcony")

    inputs: dict[str, Any] = {
        "location": locality,
        "latitude": None,
        "longitude": None,
        "area_sqft": area_sqft,
        "BHK": bhk,
        "bathrooms": bath,
        "balconies": balcony,
        "listing_date": date.today().isoformat(),
        "property_age_years": None,
        "condition": "Unknown",
        "floor_number": None,
        "total_floors": None,
        "property_type": "Unknown",
        "furnishing_status": "Unknown",
        "parking_spaces": None,
        "lift": None,
        "security": None,
        "gym": None,
        "swimming_pool": None,
        "power_backup": None,
        "gated_community": None,
    }

    with st.expander("Add more property details (optional)"):
        st.caption("If you leave these blank, the model uses values learned from its training data.")
        age_col, condition_col, type_col = st.columns(3)
        with age_col:
            age_known = st.checkbox("I know the property age", key=f"{key}_age_known")
            if age_known:
                inputs["property_age_years"] = st.number_input(
                    "Property age (years)", 0, 100, 0, key=f"{key}_age"
                )
        with condition_col:
            inputs["condition"] = st.selectbox(
                "Condition", ["Unknown", *categories.get("condition", [])],
                key=f"{key}_condition",
            )
        with type_col:
            inputs["property_type"] = st.selectbox(
                "Property type", ["Unknown", *categories.get("property_type", [])],
                key=f"{key}_type",
            )

        furnish_col, floor_col, parking_col = st.columns(3)
        with furnish_col:
            inputs["furnishing_status"] = st.selectbox(
                "Furnishing", ["Unknown", *categories.get("furnishing_status", [])],
                key=f"{key}_furnishing",
            )
        with floor_col:
            floor_known = st.checkbox("I know the floor details", key=f"{key}_floor_known")
            if floor_known:
                inputs["floor_number"] = st.number_input(
                    "Property floor", 0, 200, 0, key=f"{key}_floor"
                )
                inputs["total_floors"] = st.number_input(
                    "Floors in building", 1, 200, 1, key=f"{key}_floors"
                )
        with parking_col:
            parking_known = st.checkbox("I know the parking spaces", key=f"{key}_parking_known")
            if parking_known:
                inputs["parking_spaces"] = st.number_input(
                    "Parking spaces", 0, 50, 0, key=f"{key}_parking"
                )

        amenity_columns = st.columns(3)
        amenity_labels = {
            "lift": "Lift",
            "security": "Security",
            "gym": "Gym",
            "swimming_pool": "Swimming pool",
            "power_backup": "Power backup",
            "gated_community": "Gated community",
        }
        for index, (feature, label) in enumerate(amenity_labels.items()):
            with amenity_columns[index % len(amenity_columns)]:
                amenity_value = st.selectbox(
                    label, ["Unknown", "No", "Yes"], key=f"{key}_{feature}"
                )
            inputs[feature] = {"Unknown": None, "No": 0, "Yes": 1}[amenity_value]

    if list(inputs) != FEATURE_SCHEMA:
        raise RuntimeError("Property input form no longer matches the trained model schema.")
    return inputs
