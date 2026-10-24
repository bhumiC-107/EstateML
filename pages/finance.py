import streamlit as st
import math
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Financial Tools", layout="wide")

# A comparison can pass its estimate into the EMI calculator.
comparison_price = st.session_state.pop("comparison_property_price_lakhs", None)
if comparison_price is not None:
    st.session_state["emi_property_price_lakhs"] = float(comparison_price)

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
[data-testid="stAppViewContainer"] .main h4,
[data-testid="stHeadingWithActionElements"] * {
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

/* ── Tabs ── */
button[data-baseweb="tab"] {
    color: #596174 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a1a2e !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #e2b96f !important;
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

/* ── Sliders ── */
[data-testid="stSlider"] div[data-testid="stWidgetLabel"] p {
    color: #1a1a2e !important;
    font-weight: 600 !important;
}
[data-testid="stSlider"] div[role="slider"] {
    background-color: #1a1a2e !important;
    border-color: #e2b96f !important;
}
[data-testid="stSlider"] .st-bx {
    background: #e2b96f !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d9deea !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #596174 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #1a1a2e !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background-color: #1a1a2e !important;
    color: #e2b96f !important;
    border: 1px solid #1a1a2e !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #0f3460 !important;
    color: #ffffff !important;
    border-color: #0f3460 !important;
}

/* ── Result Cards ── */
.result-card {
    background: #1a1a2e !important;
    border-radius: 12px !important;
    padding: 1.5rem 2rem !important;
    margin-top: 0.5rem !important;
    border: 1px solid #2d3748 !important;
}
.result-card * {
    color: #f7fafc !important;
}
.result-title {
    color: #e2b96f !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.2rem !important;
}
.result-sub {
    color: #c0c8d8 !important;
    font-size: 0.88rem !important;
    margin-bottom: 1rem !important;
}
.result-divider {
    border-top: 1px solid #2d3748 !important;
    margin: 0.8rem 0 !important;
}
.result-row {
    display: flex !important;
    justify-content: space-between !important;
    padding: 0.25rem 0 !important;
}
.result-key {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
}
.result-val {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #d9deea !important;
    background: #ffffff !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
}
div[data-testid="stAlert"] * {
    color: #1a1a2e !important;
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

st.title("🏦 Financial Planning Tools")
st.markdown("Smart tools to help buyers make data-driven financial decisions.")

tab1, tab2, tab3, tab4 = st.tabs(["💰 EMI Calculator", "✅ Affordability Checker", "📈 Rental Yield", "🏠 Buy vs Rent"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — EMI Calculator
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Home Loan EMI Calculator")
    st.markdown("Calculate your monthly EMI and understand the full cost of a home loan.")

    c1, c2 = st.columns([1, 1])

    with c1:
        price    = st.number_input(
            "Property Price (₹ Lakhs)", 5.0, 2000.0, 50.0, 5.0,
            key="emi_property_price_lakhs",
        )
        down_pct = st.slider("Down Payment (%)", 10, 50, 20, 5,
                              help="RBI mandates min 10–20% down payment")
        rate     = st.number_input("Annual Interest Rate (%)", 5.0, 20.0, 8.5, 0.1)
        tenure   = st.slider("Loan Tenure (Years)", 5, 30, 20)

    loan_amt   = price * (1 - down_pct / 100)
    loan_rs    = loan_amt * 100000
    r          = rate / (12 * 100)
    n          = tenure * 12
    emi        = loan_rs * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_paid = emi * n
    total_int  = total_paid - loan_rs
    down_rs    = price * (down_pct / 100)   # in lakhs

    with c2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">₹{emi:,.0f} / month</div>
            <div class="result-sub">Monthly EMI for ₹{loan_amt:.1f}L loan @ {rate}% for {tenure} yrs</div>
            <div class="result-divider"></div>
            <div class="result-row"><span class="result-key">Down Payment</span><span class="result-val">₹{down_rs:.2f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Loan Amount</span><span class="result-val">₹{loan_amt:.2f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Total Interest</span><span class="result-val">₹{total_int/100000:.2f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Total Amount Paid</span><span class="result-val">₹{total_paid/100000:.2f} Lakhs</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Donut chart
    st.markdown("#### Loan Breakdown")
    fig = go.Figure(go.Pie(
        labels=['Principal', 'Total Interest', 'Down Payment'],
        values=[loan_rs, total_int, down_rs * 100000],
        hole=0.55,
        marker_colors=['#1a1a2e', '#e2b96f', '#a0aec0']
    ))
    fig.update_layout(paper_bgcolor='#f5f5f0', height=300,
                      legend=dict(bgcolor='#f5f5f0', font=dict(color='#1a1a2e')),
                      font_color='#1a1a2e')
    st.plotly_chart(fig, width="stretch")

    # Amortization schedule
    st.markdown("#### Yearly Amortization Schedule")
    balance, rows = loan_rs, []
    for yr in range(1, tenure + 1):
        principal_yr = interest_yr = 0
        for _ in range(12):
            int_pay      = balance * r
            prin_pay     = emi - int_pay
            balance      = max(0, balance - prin_pay)
            principal_yr += prin_pay
            interest_yr  += int_pay
        rows.append({
            "Year":           yr,
            "Principal (L)":  f"₹{principal_yr/100000:.2f}",
            "Interest (L)":   f"₹{interest_yr/100000:.2f}",
            "Balance (L)":    f"₹{balance/100000:.2f}",
        })
    sched_df = pd.DataFrame(rows)
    st.dataframe(sched_df, width="stretch", hide_index=True)
    st.download_button("⬇️ Download Schedule as CSV",
                       data=sched_df.to_csv(index=False),
                       file_name="emi_schedule.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Affordability Checker
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Affordability Checker")
    st.markdown("Find out what property price range fits your income profile.")

    salary    = st.number_input("Monthly Take-Home Salary (₹)", 10000, 2000000, 80000, 5000)
    other_emi = st.number_input("Existing Monthly EMIs (₹)", 0, 500000, 0, 1000,
                                 help="Car loan, personal loan, etc.")
    prop_rate = st.number_input("Expected Loan Interest Rate (%)", 5.0, 20.0, 8.75, 0.25, key="aff_rate")
    prop_ten  = st.slider("Preferred Loan Tenure (Years)", 5, 30, 20, key="aff_ten")
    down_pay  = st.slider("Down Payment You Can Arrange (%)", 10, 50, 20, 5, key="aff_down")

    max_emi  = max(0, salary * 0.50 - other_emi)
    r2       = prop_rate / (12 * 100)
    n2       = prop_ten * 12
    max_loan = max_emi * ((1 + r2) ** n2 - 1) / (r2 * (1 + r2) ** n2) if r2 > 0 and max_emi > 0 else 0
    max_price_l  = max_loan / (1 - down_pay / 100) / 100000
    max_loan_l   = max_loan / 100000
    budget_comfort = max_price_l * 0.80

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Monthly EMI",    f"₹{max_emi:,.0f}")
    col2.metric("Max Loan Amount",    f"₹{max_loan_l:.1f}L")
    col3.metric("Max Property Price", f"₹{max_price_l:.1f}L")

    # Gauge
    st.markdown("#### Budget Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_price_l,
        number={"suffix": " L", "font": {"color": "#1a1a2e", "size": 28}},
        gauge={
            "axis":  {"range": [0, max_price_l * 1.5],
                      "tickcolor": "#1a1a2e", "tickfont": {"color": "#1a1a2e"}},
            "bar":   {"color": "#1a1a2e"},
            "steps": [
                {"range": [0,               budget_comfort],  "color": "#c6f6d5"},
                {"range": [budget_comfort,  max_price_l],     "color": "#fefcbf"},
                {"range": [max_price_l,     max_price_l*1.5], "color": "#fed7d7"},
            ],
            "threshold": {"line": {"color": "#e2b96f", "width": 4}, "value": max_price_l}
        }
    ))
    fig.update_layout(paper_bgcolor='#f5f5f0', height=280, font_color='#1a1a2e')
    st.plotly_chart(fig, width="stretch")

    if max_price_l > 0:
        st.success(f"""
         **Comfortable:** Up to ₹{budget_comfort:.1f} Lakhs  
         **Stretch:** ₹{budget_comfort:.1f}L – ₹{max_price_l:.1f} Lakhs  
         **Above budget:** > ₹{max_price_l:.1f} Lakhs
        """)
    else:
        st.info("Based on these details, there may not be enough monthly income left for a new home loan. Try lowering existing loan payments or adjusting the income details.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Rental Yield
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Rental Yield & ROI Estimator")
    st.markdown("Evaluate a property as an investment using rental yield and ROI metrics.")

    col1, col2 = st.columns([1, 1])

    with col1:
        prop_val      = st.number_input("Property Value (₹ Lakhs)", 5.0, 1000.0, 60.0, 5.0)
        monthly_rent  = st.number_input("Expected Monthly Rent (₹)", 1000, 500000, 18000, 500)
        maint_monthly = st.number_input("Monthly Maintenance (₹)", 0, 50000, 3000, 500)
        vacancy_pct   = st.slider("Expected Vacancy (%/year)", 0, 30, 5)
        appreciate    = st.slider("Expected Annual Appreciation (%)", 0.0, 15.0, 6.0, 0.5)
        hold_years    = st.slider("Holding Period (Years)", 1, 30, 10)

    gross_annual = monthly_rent * 12
    net_annual   = gross_annual * (1 - vacancy_pct / 100) - (maint_monthly * 12)
    prop_val_rs  = prop_val * 100000
    gross_yield  = (gross_annual / prop_val_rs) * 100
    net_yield    = (net_annual  / prop_val_rs) * 100
    future_val   = prop_val * (1 + appreciate / 100) ** hold_years
    capital_gain = future_val - prop_val
    total_rent_l = net_annual * hold_years / 100000
    total_return = capital_gain + total_rent_l
    roi_pct      = (total_return / prop_val) * 100
    color        = "#4caf50" if net_yield >= 3 else "#ff9800" if net_yield >= 2 else "#f44336"
    label        = "Excellent" if net_yield >= 4 else "Good" if net_yield >= 3 else "Average" if net_yield >= 2 else "Poor"

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{net_yield:.2f}% Net Yield</div>
            <div class="result-sub">Gross Yield: {gross_yield:.2f}% &nbsp;|&nbsp;
                <span style="color:{color};font-weight:700">{label} Investment</span>
            </div>
            <div class="result-divider"></div>
            <div class="result-row"><span class="result-key">Net Annual Rent</span><span class="result-val">₹{net_annual/100000:.2f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Future Value ({hold_years}y)</span><span class="result-val">₹{future_val:.1f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Capital Gain</span><span class="result-val">₹{capital_gain:.1f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Cumulative Rental Income</span><span class="result-val">₹{total_rent_l:.1f} Lakhs</span></div>
            <div class="result-row"><span class="result-key">Total ROI over {hold_years}y</span><span class="result-val">{roi_pct:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Projection chart
    st.markdown("#### Property Value & Cumulative Rental Income Over Time")
    years_list = list(range(0, hold_years + 1))
    val_list   = [prop_val * (1 + appreciate / 100) ** y for y in years_list]
    rent_cum   = [net_annual * y / 100000 for y in years_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_list, y=val_list,
                              name="Property Value (L)", line=dict(color="#1a1a2e", width=2.5)))
    fig.add_trace(go.Scatter(x=years_list, y=rent_cum,
                              name="Cumulative Rental (L)", line=dict(color="#e2b96f", width=2.5, dash='dash')))
    fig.update_layout(xaxis_title="Year", yaxis_title="₹ Lakhs",
                      plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
                      font_color='#1a1a2e',
                      legend=dict(bgcolor='#f5f5f0', font=dict(color='#1a1a2e')))
    st.plotly_chart(fig, width="stretch")

    # City benchmarks
    st.markdown("####  Rental Yield Benchmarks — Indian Cities")
    bench_df = pd.DataFrame({
        "City":        ["Mumbai", "Bengaluru", "Delhi NCR", "Pune", "Hyderabad", "Your Property"],
        "Gross Yield": [2.5,       3.5,         2.8,         3.2,    4.0,          round(gross_yield, 2)],
    })
    fig2 = px.bar(bench_df, x="City", y="Gross Yield",
                   color="City", text_auto=True,
                   color_discrete_map={"Your Property": "#e2b96f",
                                        "Mumbai": "#1a1a2e", "Bengaluru": "#1a1a2e",
                                        "Delhi NCR": "#1a1a2e", "Pune": "#1a1a2e",
                                        "Hyderabad": "#1a1a2e"})
    fig2.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
                       showlegend=False, font_color='#1a1a2e')
    st.plotly_chart(fig2, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Buy versus rent
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Buy versus rent comparison")
    st.markdown(
        "Compare estimated home equity with an illustrative renter investment. "
        "If you opened this from a property estimate, its projected base values are prefilled."
    )

    forecast_frame = st.session_state.get("future_value_scenarios")
    estimate_lakhs = st.session_state.get("future_value_current_price_lakhs")
    if estimate_lakhs is None:
        estimate_lakhs = st.session_state.get("emi_property_price_lakhs", 100.0)
        st.info("No property forecast was passed in. Enter an estimated price and appreciation assumption below.")

    buy_col, rent_col = st.columns(2)
    with buy_col:
        buy_price = st.number_input(
            "Estimated purchase price (₹ lakh)", 5.0, 2000.0,
            float(min(max(estimate_lakhs, 5.0), 2000.0)), 1.0,
            key="buy_rent_price",
        )
        down_payment_pct = st.slider("Down payment (%)", 10, 60, 20, 5, key="buy_rent_down")
        loan_rate_pct = st.number_input("Home-loan annual interest (%)", 0.0, 25.0, 8.5, 0.1, key="buy_rent_rate")
        loan_years = st.slider("Loan tenure (years)", 1, 30, 20, key="buy_rent_tenure")
        maintenance = st.number_input("Starting monthly maintenance (₹)", 0, 100000, 3000, 500, key="buy_rent_maintenance")
    with rent_col:
        starting_rent = st.number_input("Starting monthly rent (₹)", 0, 1000000, 25000, 1000, key="buy_rent_monthly_rent")
        rent_increase_pct = st.number_input("Annual rent and maintenance increase (%)", 0.0, 20.0, 5.0, 0.5, key="buy_rent_inflation")
        investment_return_pct = st.number_input("Annual return on renter investments (%)", 0.0, 30.0, 7.0, 0.5, key="buy_rent_return")
        fallback_appreciation_pct = st.number_input("Annual property appreciation if no estimate was passed (%)", -10.0, 20.0, 6.0, 0.5, key="buy_rent_appreciation")

    horizon = int(st.session_state.get("future_value_horizon", 5))
    horizon = min(max(horizon, 1), 10)
    monthly_rate = loan_rate_pct / 1200
    loan_principal = buy_price * 100000 * (1 - down_payment_pct / 100)
    month_count = loan_years * 12
    if monthly_rate == 0:
        monthly_emi = loan_principal / month_count
    else:
        monthly_emi = loan_principal * monthly_rate * (1 + monthly_rate) ** month_count / ((1 + monthly_rate) ** month_count - 1)

    scenario_rows = []
    renter_portfolio = buy_price * 100000 * down_payment_pct / 100
    mortgage_balance = loan_principal
    monthly_return = (1 + investment_return_pct / 100) ** (1 / 12) - 1
    annual_cost_increase = rent_increase_pct / 100
    forecast_lookup = {}
    forecast_base_price = float(st.session_state.get("future_value_current_price_lakhs", 0.0) or 0.0)
    if isinstance(forecast_frame, pd.DataFrame) and "Year" in forecast_frame:
        forecast_lookup = forecast_frame.set_index("Year").get("Base value (₹ lakh)", pd.Series(dtype=float)).to_dict()
    for year in range(1, horizon + 1):
        current_rent = starting_rent * (1 + annual_cost_increase) ** (year - 1)
        current_maintenance = maintenance * (1 + annual_cost_increase) ** (year - 1)
        for _ in range(12):
            if mortgage_balance > 0:
                interest_payment = mortgage_balance * monthly_rate
                principal_payment = min(mortgage_balance, monthly_emi - interest_payment)
                mortgage_balance = max(0.0, mortgage_balance - principal_payment)
                actual_loan_payment = principal_payment + interest_payment
            else:
                actual_loan_payment = 0.0
            monthly_owner_cost = actual_loan_payment + current_maintenance
            renter_portfolio = max(0.0, renter_portfolio * (1 + monthly_return) + (monthly_owner_cost - current_rent))

        if year in forecast_lookup:
            scale_to_purchase_price = buy_price / forecast_base_price if forecast_base_price > 0 else 1.0
            property_value_lakhs = float(forecast_lookup[year]) * scale_to_purchase_price
        else:
            property_value_lakhs = buy_price * (1 + fallback_appreciation_pct / 100) ** year
        owner_equity_lakhs = property_value_lakhs * 100000 - mortgage_balance
        renter_wealth_lakhs = renter_portfolio
        scenario_rows.append({
            "Year": year,
            "Owner equity (₹ lakh)": owner_equity_lakhs / 100000,
            "Renter investment balance (₹ lakh)": renter_wealth_lakhs / 100000,
            "Projected home value (₹ lakh)": property_value_lakhs,
        })

    comparison = pd.DataFrame(scenario_rows)
    st.caption(
        "The renter starts by investing the down payment, then invests (or draws from) the "
        "monthly difference between ownership costs and rent. Maintenance and rent rise at the "
        "rate you entered. The calculation excludes taxes, purchase/sale costs, insurance, "
        "rent deposits, and personal circumstances. It is an illustration, not financial advice."
    )
    st.line_chart(
        comparison.set_index("Year")[["Owner equity (₹ lakh)", "Renter investment balance (₹ lakh)"]],
        y_label="Illustrative wealth (₹ lakh)",
        height=330,
    )
    st.dataframe(comparison.round(1), width="stretch", hide_index=True)

    buy_rent_break_even = comparison.loc[
        comparison["Owner equity (₹ lakh)"] >= comparison["Renter investment balance (₹ lakh)"], "Year"
    ]
    if not buy_rent_break_even.empty:
        st.metric("First year projected owner equity reaches renter investments", f"Year {int(buy_rent_break_even.iloc[0])}")
    else:
        st.info("Owner equity does not reach the renter investment balance within this projection horizon under these assumptions.")

st.caption("EstateML Financial Tools · EMI, affordability, rental yield, and buy-versus-rent scenarios")
