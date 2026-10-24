"""Scenario calculations for future property values and sell timing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from estate_ml.property_model import normalize_property_location, predict_property

MAX_ANNUAL_GROWTH = 0.15
MIN_ANNUAL_GROWTH = -0.10
SCENARIO_SPREAD = 0.02


def _annual_log_trend(monthly_values: pd.Series) -> float:
    """Fit a simple log-price trend using actual month offsets."""
    months = monthly_values.index
    offsets = np.asarray([month.ordinal for month in months], dtype=float)
    offsets -= offsets.min()
    values = np.log(monthly_values.to_numpy(dtype=float))
    slope = float(np.polyfit(offsets, values, 1)[0])
    return float(np.exp(slope * 12) - 1)


def estimate_locality_growth(features: pd.DataFrame, prices: pd.Series) -> dict:
    """Estimate annual price-per-square-foot trends from training rows only.

    Local trends with fewer than 18 observed months fall back to the city trend.
    Other locality trends are shrunk toward the city trend to reduce noise.
    The stored growth values are bounded to avoid extreme long-range extrapolation.
    """
    dates = pd.to_datetime(features["listing_date"], errors="coerce", utc=True)
    area = pd.to_numeric(features["area_sqft"], errors="coerce")
    target = pd.to_numeric(prices, errors="coerce")
    valid = dates.notna() & area.gt(0) & target.gt(0)
    if not valid.any():
        return {"city_annual_growth": 0.0, "locality_annual_growth": {}, "method": "no_valid_dates"}

    rows = pd.DataFrame({
        "location": features.loc[valid, "location"].map(normalize_property_location),
        "month": dates.loc[valid].dt.tz_localize(None).dt.to_period("M"),
        "price_per_sqft": target.loc[valid] * 100_000 / area.loc[valid],
    })
    city_monthly = rows.groupby("month")["price_per_sqft"].median().sort_index()
    city_rate = _annual_log_trend(city_monthly) if len(city_monthly) >= 18 else 0.0
    city_rate = float(np.clip(city_rate, MIN_ANNUAL_GROWTH, MAX_ANNUAL_GROWTH))

    localities = {}
    for location, group in rows.groupby("location"):
        monthly = group.groupby("month")["price_per_sqft"].median().sort_index()
        observations = int(len(monthly))
        if observations < 18:
            continue
        raw_rate = _annual_log_trend(monthly)
        local_weight = observations / (observations + 12)
        adjusted_rate = city_rate + local_weight * (raw_rate - city_rate)
        localities[str(location)] = {
            "annual_growth": float(np.clip(adjusted_rate, MIN_ANNUAL_GROWTH, MAX_ANNUAL_GROWTH)),
            "monthly_observations": observations,
        }
    return {
        "city_annual_growth": city_rate,
        "locality_annual_growth": localities,
        "method": "linear_trend_of_monthly_median_asking_price_per_sqft; local trends shrunk to city trend",
        "scenario_spread": SCENARIO_SPREAD,
        "growth_bounds": [MIN_ANNUAL_GROWTH, MAX_ANNUAL_GROWTH],
    }


def training_age_median(model) -> float:
    """Read the age median fitted inside the saved model pipeline."""
    numeric_pipeline = model.named_steps["preprocess"].named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    columns = list(imputer.feature_names_in_)
    return float(imputer.statistics_[columns.index("property_age_years")])


def create_value_scenarios(
    model,
    property_inputs: dict,
    metadata: dict,
    *,
    horizon_years: int = 10,
    alternative_return: float = 0.07,
) -> tuple[pd.DataFrame, dict]:
    """Project modelled building aging plus locality market-growth scenarios.

    The listing date stays fixed for each counterfactual, so date effects from
    the price model are not counted again as market growth.
    """
    if not 1 <= int(horizon_years) <= 10:
        raise ValueError("Forecast horizon must be between 1 and 10 years.")
    if not 0 <= float(alternative_return) <= 0.50:
        raise ValueError("Alternative return must be between 0% and 50%.")

    inputs = dict(property_inputs)
    entered_age = inputs.get("property_age_years")
    if entered_age is None or pd.isna(entered_age):
        current_age = training_age_median(model)
        age_source = "training_median"
    else:
        current_age = float(entered_age)
        age_source = "entered"
    current_age = max(0.0, current_age)
    inputs["property_age_years"] = current_age

    growth_data = metadata.get("locality_growth", {})
    city_rate = float(growth_data.get("city_annual_growth", 0.0))
    location = normalize_property_location(inputs.get("location"))
    locality_data = growth_data.get("locality_annual_growth", {}).get(location)
    if locality_data:
        base_rate = float(locality_data["annual_growth"])
        rate_source = "locality"
        monthly_observations = int(locality_data.get("monthly_observations", 0))
    else:
        base_rate = city_rate
        rate_source = "city_fallback"
        monthly_observations = 0

    low, high = growth_data.get("growth_bounds", [MIN_ANNUAL_GROWTH, MAX_ANNUAL_GROWTH])
    spread = float(growth_data.get("scenario_spread", SCENARIO_SPREAD))
    rates = {
        "Conservative": float(np.clip(base_rate - spread, low, high)),
        "Base": float(np.clip(base_rate, low, high)),
        "Optimistic": float(np.clip(base_rate + spread, low, high)),
    }

    aging_prices = [predict_property(model, inputs)]
    for year in range(1, int(horizon_years) + 1):
        future_inputs = dict(inputs)
        future_inputs["property_age_years"] = current_age + year
        aging_prices.append(predict_property(model, future_inputs))

    rows = []
    first_below_benchmark = {}
    for year in range(1, int(horizon_years) + 1):
        row = {"Year": year}
        for scenario, rate in rates.items():
            value_now = aging_prices[year] * (1 + rate) ** year
            value_previous = aging_prices[year - 1] * (1 + rate) ** (year - 1)
            annual_gain = value_now / value_previous - 1
            row[f"{scenario} value (₹ lakh)"] = float(value_now)
            row[f"{scenario} annual change (%)"] = float(annual_gain * 100)
            if scenario not in first_below_benchmark and annual_gain < alternative_return:
                first_below_benchmark[scenario] = year
        rows.append(row)

    details = {
        "current_age": current_age,
        "current_model_value": aging_prices[0],
        "age_source": age_source,
        "growth_rates": rates,
        "rate_source": rate_source,
        "monthly_observations": monthly_observations,
        "city_growth_rate": city_rate,
        "alternative_return": float(alternative_return),
        "first_year_below_alternative_return": first_below_benchmark,
    }
    return pd.DataFrame(rows), details
