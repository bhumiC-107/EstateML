"""Streamlit display for understandable future-value scenarios."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from estate_ml.forecast import create_value_scenarios

SCENARIO_COLORS = {
    "Conservative": "#718096",
    "Base": "#1a1a2e",
    "Optimistic": "#c9922a",
}


def render_future_value_section(model, metadata: dict, property_inputs: dict) -> None:
    st.markdown("#### What could this home be worth later?")
    st.caption(
        "See a range of possible values based on past Bengaluru listings and the property's age. "
        "Past prices do not guarantee future prices."
    )
    horizon_col, return_col = st.columns(2)
    with horizon_col:
        horizon = st.slider("How many years ahead?", 1, 10, 5, key="future_horizon")
    with return_col:
        alternative_return = st.number_input(
            "Growth from another investment per year (%)",
            min_value=0.0,
            max_value=30.0,
            value=7.0,
            step=0.5,
            help=(
                "This is only a comparison rate. For example, 7% means ₹100 invested elsewhere "
                "could become ₹107 after one year, before fees or taxes. It does not change the "
                "property estimate or guarantee an investment return."
            ),
            key="future_alternative_return",
        ) / 100

    try:
        forecast, details = create_value_scenarios(
            model, property_inputs, metadata,
            horizon_years=horizon,
            alternative_return=alternative_return,
        )
    except (ValueError, KeyError, TypeError) as exc:
        st.info(f"We couldn't make a projection from these property details: {exc}")
        return

    if details["age_source"] == "training_median":
        st.caption(
            f"Property age wasn't entered. We used the typical age in the listings "
            f"({details['current_age']:.0f} years)."
        )
    observed_age_max = metadata.get("training_property_age_range", {}).get("maximum")
    if observed_age_max is not None and details["current_age"] + horizon > float(observed_age_max):
        st.warning(
            "The later years go beyond the property ages available in the listings, "
            "so those estimates are less certain."
        )

    rates = details["growth_rates"]
    area_source = "this area" if details["rate_source"] == "locality" else "Bengaluru overall"
    rates_text = " · ".join(
        f"{name}: {rate * 100:+.1f}% per year" for name, rate in rates.items()
    )
    st.caption(f"Past listing-price trend for {area_source}; the scenarios allow for variation: {rates_text}.")

    figure = go.Figure()
    for scenario, color in SCENARIO_COLORS.items():
        column_name = f"{scenario} value (₹ lakh)"
        values = [details["current_model_value"], *forecast[column_name].tolist()]
        figure.add_trace(go.Scatter(
            x=list(range(0, horizon + 1)),
            y=values,
            mode="lines+markers",
            name=scenario,
            line={"color": color, "width": 3 if scenario == "Base" else 2},
            hovertemplate="Year %{x}: ₹%{y:,.1f} lakh<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1a1a2e"},
        xaxis_title="Years from today",
        yaxis_title="Possible value (₹ lakh)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(dtick=1, gridcolor="#e6e9ef")
    figure.update_yaxes(rangemode="tozero", gridcolor="#e6e9ef")
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    value_columns = [
        f"{name} value (₹ lakh)" for name in ("Conservative", "Base", "Optimistic")
    ]
    display_years = sorted({year for year in (1, 3, 5, horizon) if year <= horizon})
    summary = forecast.loc[forecast["Year"].isin(display_years), ["Year", *value_columns]].copy()
    summary.columns = [
        "In this many years", "Lower estimate (₹ lakh)",
        "Middle estimate (₹ lakh)", "Higher estimate (₹ lakh)",
    ]
    st.dataframe(summary.round(1), width="stretch", hide_index=True)

    st.markdown(f"**Compared with a {alternative_return:.1%} yearly return**")
    review_years = details["first_year_below_alternative_return"]
    if len(review_years) == len(rates) and len(set(review_years.values())) == 1:
        year = next(iter(review_years.values()))
        st.info(
            f"Under these assumptions, estimated yearly value growth is below "
            f"{alternative_return:.1%} from Year {year}. This is a comparison point, "
            "not a recommendation to sell."
        )
    else:
        review_columns = st.columns(3)
        for column, scenario in zip(review_columns, ("Conservative", "Base", "Optimistic")):
            first_year = review_years.get(scenario)
            value = f"Year {first_year}" if first_year is not None else "Not reached in this projection"
            column.metric(f"{scenario} case", value)
        st.caption("This shows when projected growth first falls below your comparison rate; it does not decide when you should sell.")

    st.caption(
        "The low, middle, and high lines are what-if scenarios, not guaranteed sale prices. "
        "They don't include rent, taxes, loan balance, or sale costs."
    )

    st.session_state["future_value_scenarios"] = forecast
    st.session_state["future_value_horizon"] = horizon
    st.session_state["future_value_current_price_lakhs"] = details["current_model_value"]
    st.session_state["future_value_base_return"] = rates["Base"]
    st.session_state["future_value_alternative_return"] = alternative_return
    st.page_link("pages/finance.py", label="Compare buying with renting", icon="🏦")
