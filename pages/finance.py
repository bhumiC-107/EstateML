import streamlit as st
import math
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Financial Tools", layout="wide")

st.markdown("""
<style>
/* ── Base ── */
.stApp { background-color: #f5f5f0; }
h1, h2, h3, h4 { font-family: 'Georgia', serif; color: #1a1a2e; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background-color: #1a1a2e; }
section[data-testid="stSidebar"] * { color: #e2b96f !important; }

/* ── Button ── */
div.stButton > button {
    background-color: #1a1a2e; color: #e2b96f;
    border: none; padding: 0.6rem 2rem;
    border-radius: 8px; font-size: 0.95rem; font-weight: 600;
}
div.stButton > button:hover { background-color: #0f3460; color: #e2b96f; }

/* ── Slider accent ── */
[data-testid="stSlider"] .st-bx { background: #e2b96f !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: #e2b96f !important; }

/* ── Result card ── */
.result-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 0.5rem;
    border: 1px solid #2d3748;
}
.result-title  { color: #e2b96f;  font-size: 1.7rem; font-weight: 700; margin-bottom: 0.2rem; }
.result-sub    { color: #a0aec0;  font-size: 0.88rem; margin-bottom: 1rem; }
.result-divider{ border-top: 1px solid #2d3748; margin: 0.8rem 0; }
.result-row    { display: flex; justify-content: space-between; padding: 0.25rem 0; }
.result-key    { color: #a0aec0; font-size: 0.85rem; }
.result-val    { color: #f7fafc; font-size: 0.95rem; font-weight: 600; }
            
/* ── Sidebar nav page links ── */
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavLink"] span {
    color: #e2b96f !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}
[data-testid="stSidebarNavLink"]:hover {
    background-color: #2d3748 !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background-color: #2d3748 !important;
}
section[data-testid="stSidebar"] a {
    color: #e2b96f !important;
}
/* ── Sidebar nav header/title ── */
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] * {
    color: #e2b96f !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Financial Planning Tools")
st.markdown("Smart tools to help buyers make data-driven financial decisions.")

tab1, tab2, tab3 = st.tabs(["💰 EMI Calculator", "✅ Affordability Checker", "📈 Rental Yield"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — EMI Calculator
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Home Loan EMI Calculator")
    st.markdown("Calculate your monthly EMI and understand the full cost of a home loan.")

    c1, c2 = st.columns([1, 1])

    with c1:
        price    = st.number_input("Property Price (₹ Lakhs)", 5.0, 1000.0, 50.0, 5.0)
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.dataframe(sched_df, use_container_width=True, hide_index=True)
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
    st.plotly_chart(fig, use_container_width=True)

    if max_price_l > 0:
        st.success(f"""
         **Comfortable:** Up to ₹{budget_comfort:.1f} Lakhs  
         **Stretch:** ₹{budget_comfort:.1f}L – ₹{max_price_l:.1f} Lakhs  
         **Above budget:** > ₹{max_price_l:.1f} Lakhs
        """)
    else:
        st.error(" Existing EMIs exceed the 50% FOIR limit. Consider reducing debt first.")


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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

st.caption(" Financial Tools Module  |  Built by Person 4 — Business Analyst")