import streamlit as st
import math
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Financial Tools", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f5f5f0; }
h1, h2, h3 { font-family: 'Georgia', serif; color: #1a1a2e; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; }
div.stButton > button {
    background-color: #1a1a2e; color: #e2b96f;
    border: none; padding: 0.6rem 2rem; border-radius: 8px;
    font-size: 0.95rem; font-weight: 600;
}
.tool-card {
    background: white; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
}
.result-card {
    background: #1a1a2e; border-radius: 12px;
    padding: 1.5rem 2rem; margin-top: 1rem;
}
.result-main { color: #e2b96f; font-size: 1.8rem; font-weight: 700; }
.result-sub  { color: #a0aec0; font-size: 0.9rem; margin-top: 0.3rem; }
.result-meta { color: #e2e8f0; font-size: 0.95rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Financial Planning Tools")
st.markdown("Smart tools to help buyers make data-driven financial decisions.")

# ── Tab Layout ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 EMI Calculator", "✅ Affordability Checker", "📈 Rental Yield"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — EMI Calculator
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Home Loan EMI Calculator")
    st.markdown("Calculate your monthly EMI and understand the full cost of a home loan.")

    c1, c2 = st.columns(2)

    with c1:
        price     = st.number_input("Property Price (₹ Lakhs)", 5.0, 1000.0, 50.0, 5.0)
        down_pct  = st.slider("Down Payment (%)", 10, 50, 20, 5,
                               help="RBI mandates at least 10–20% down payment")
        rate      = st.number_input("Annual Interest Rate (%)", 5.0, 20.0, 8.5, 0.1)
        tenure    = st.slider("Loan Tenure (Years)", 5, 30, 20)

    loan_amt   = price * (1 - down_pct / 100)
    loan_rs    = loan_amt * 100000
    r          = rate / (12 * 100)
    n          = tenure * 12
    emi        = loan_rs * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_paid = emi * n
    total_int  = total_paid - loan_rs
    down_rs    = price * (down_pct / 100) * 100000

    with c2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-main">₹{emi:,.0f} / month</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-sub">Monthly EMI for ₹{loan_amt:.1f}L loan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-meta" style="margin-top:1rem; border-top:1px solid #2d3748; padding-top:0.8rem;">
            <div>Down Payment &nbsp;&nbsp;&nbsp; ₹{down_rs/100000:.2f} Lakhs</div>
            <div>Loan Amount &nbsp;&nbsp;&nbsp;&nbsp; ₹{loan_amt:.2f} Lakhs</div>
            <div>Total Interest &nbsp;&nbsp;&nbsp; ₹{total_int/100000:.2f} Lakhs</div>
            <div>Total Amount &nbsp;&nbsp;&nbsp;&nbsp; ₹{total_paid/100000:.2f} Lakhs</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Amortization donut
    st.markdown("#### Loan Breakdown")
    fig = go.Figure(go.Pie(
        labels=['Principal', 'Interest', 'Down Payment'],
        values=[loan_rs, total_int, down_rs],
        hole=0.55,
        marker_colors=['#1a1a2e', '#e2b96f', '#a0aec0']
    ))
    fig.update_layout(paper_bgcolor='#f5f5f0', height=320,
                       legend=dict(bgcolor='#f5f5f0'))
    st.plotly_chart(fig, use_container_width=True)

    # Amortization schedule (yearly)
    st.markdown("#### Yearly Amortization Schedule")
    balance   = loan_rs
    rows      = []
    for yr in range(1, tenure + 1):
        principal_yr = 0
        interest_yr  = 0
        for _ in range(12):
            int_pay  = balance * r
            prin_pay = emi - int_pay
            balance  = max(0, balance - prin_pay)
            principal_yr += prin_pay
            interest_yr  += int_pay
        rows.append({
            "Year": yr,
            "Principal (₹)": f"₹{principal_yr/100000:.2f}L",
            "Interest (₹)":  f"₹{interest_yr/100000:.2f}L",
            "Balance (₹)":   f"₹{balance/100000:.2f}L"
        })
    sched_df = pd.DataFrame(rows)
    st.dataframe(sched_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Amortization Schedule",
        data=sched_df.to_csv(index=False),
        file_name="emi_schedule.csv",
        mime="text/csv"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Affordability Checker
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Affordability Checker")
    st.markdown("Find out what property price range fits your income profile.")

    salary    = st.number_input("Monthly Take-Home Salary (₹)", 10000, 2000000, 80000, 5000)
    other_emi = st.number_input("Existing Monthly EMIs (₹)", 0, 500000, 0, 1000,
                                 help="Car loan, personal loan, etc.")
    prop_rate = st.number_input("Expected Loan Interest Rate (%)", 5.0, 20.0, 8.75, 0.25)
    prop_ten  = st.slider("Preferred Loan Tenure (Years)", 5, 30, 20)
    down_pay  = st.slider("Down Payment You Can Arrange (%)", 10, 50, 20, 5)

    # FOIR — Fixed Obligation to Income Ratio (banks use max 40-50%)
    foir_limit   = 0.50
    max_emi      = (salary * foir_limit) - other_emi
    max_emi      = max(0, max_emi)

    # Back-calculate max loan from max EMI
    r2           = prop_rate / (12 * 100)
    n2           = prop_ten * 12
    if r2 > 0 and max_emi > 0:
        max_loan = max_emi * ((1 + r2) ** n2 - 1) / (r2 * (1 + r2) ** n2)
    else:
        max_loan = 0

    max_price_rs = max_loan / (1 - down_pay / 100)
    max_price_l  = max_price_rs / 100000
    max_loan_l   = max_loan / 100000

    # Affordability buckets
    budget_comfort = max_price_l * 0.80
    budget_stretch = max_price_l

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Monthly EMI",   f"₹{max_emi:,.0f}", help="50% FOIR minus existing EMIs")
    col2.metric("Max Loan Amount",   f"₹{max_loan_l:.1f}L")
    col3.metric("Max Property Price",f"₹{max_price_l:.1f}L")

    # Affordability gauge
    st.markdown("#### Budget Range")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=max_price_l,
        number={"suffix": " L", "font": {"color": "#1a1a2e"}},
        gauge={
            "axis":      {"range": [0, max_price_l * 1.5]},
            "bar":       {"color": "#1a1a2e"},
            "steps": [
                {"range": [0,               budget_comfort], "color": "#c6f6d5"},
                {"range": [budget_comfort,  budget_stretch], "color": "#fefcbf"},
                {"range": [budget_stretch,  max_price_l*1.5],"color":"#fed7d7"},
            ],
            "threshold": {"line": {"color": "#e2b96f", "width": 4}, "value": max_price_l}
        }
    ))
    fig.update_layout(paper_bgcolor='#f5f5f0', height=280)
    st.plotly_chart(fig, use_container_width=True)

    if max_price_l > 0:
        st.success(f"""
         **Comfortable budget:** Up to ₹{budget_comfort:.1f} Lakhs  
     **Stretch budget:** ₹{budget_comfort:.1f}L – ₹{budget_stretch:.1f} Lakhs  
         **Above budget:** > ₹{budget_stretch:.1f} Lakhs
        """)
    else:
        st.error(" Based on your inputs, existing EMIs exceed the 50% FOIR limit. Consider reducing debt first.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Rental Yield Estimator
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Rental Yield & ROI Estimator")
    st.markdown("Evaluate a property as an investment using rental yield and ROI metrics.")

    col1, col2 = st.columns(2)

    with col1:
        prop_val      = st.number_input("Property Value (₹ Lakhs)",   5.0, 1000.0, 60.0, 5.0)
        monthly_rent  = st.number_input("Expected Monthly Rent (₹)",  1000, 500000, 18000, 500)
        maint_monthly = st.number_input("Monthly Maintenance (₹)",    0, 50000, 3000, 500)
        vacancy_pct   = st.slider("Expected Vacancy (%/year)",         0, 30, 5,
                                   help="% of the year the property may be vacant")
        appreciate    = st.slider("Expected Annual Appreciation (%)",  0.0, 15.0, 6.0, 0.5)
        hold_years    = st.slider("Holding Period (Years)",            1, 30, 10)

    # Yield calculations
    gross_annual     = monthly_rent * 12
    effective_rent   = gross_annual * (1 - vacancy_pct / 100)
    annual_maint     = maint_monthly * 12
    net_annual       = effective_rent - annual_maint
    prop_val_rs      = prop_val * 100000

    gross_yield      = (gross_annual / prop_val_rs) * 100
    net_yield        = (net_annual  / prop_val_rs) * 100

    # Future value with appreciation
    future_val       = prop_val * (1 + appreciate / 100) ** hold_years
    capital_gain     = future_val - prop_val
    total_rental_inc = net_annual * hold_years / 100000  # in lakhs
    total_return     = capital_gain + total_rental_inc
    roi_pct          = (total_return / prop_val) * 100

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-main">{net_yield:.2f}% Net Yield</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-sub">Gross Yield: {gross_yield:.2f}%</div>', unsafe_allow_html=True)
        color = "#4caf50" if net_yield >= 3 else "#ff9800" if net_yield >= 2 else "#f44336"
        label = "Excellent" if net_yield >= 4 else "Good" if net_yield >= 3 else "Average" if net_yield >= 2 else "Poor"
        st.markdown(f"""
        <div style="color:{color};font-weight:700;margin-top:0.5rem;font-size:1.1rem">{label} Investment</div>
        <div class="result-meta" style="margin-top:1rem;border-top:1px solid #2d3748;padding-top:0.8rem;">
            <div>Net Annual Rent &nbsp;&nbsp;&nbsp; ₹{net_annual/100000:.2f}L</div>
            <div>Future Value ({hold_years}y) &nbsp; ₹{future_val:.1f}L</div>
            <div>Capital Gain &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ₹{capital_gain:.1f}L</div>
            <div>Total ROI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {roi_pct:.1f}% over {hold_years}y</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Year-by-year projection chart
    st.markdown("#### Property Value & Cumulative Rent Over Time")
    years_list  = list(range(0, hold_years + 1))
    val_list    = [prop_val * (1 + appreciate / 100) ** y for y in years_list]
    rent_cum    = [net_annual * y / 100000 for y in years_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_list, y=val_list,
                              name="Property Value (L)", line=dict(color="#1a1a2e", width=2)))
    fig.add_trace(go.Scatter(x=years_list, y=rent_cum,
                              name="Cumulative Rental Income (L)", line=dict(color="#e2b96f", width=2, dash='dash')))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="₹ Lakhs",
        plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
        legend=dict(bgcolor='#f5f5f0')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Benchmark
    st.markdown("####  Yield Benchmarks (India)")
    bench_df = pd.DataFrame({
        "City":        ["Mumbai", "Bengaluru", "Delhi NCR", "Pune", "Hyderabad", "Your Property"],
        "Gross Yield": [2.5,       3.5,         2.8,         3.2,    4.0,          round(gross_yield, 2)],
    })
    fig2 = px.bar(bench_df, x="City", y="Gross Yield",
                   color="City", text_auto=True,
                   color_discrete_sequence=['#1a1a2e','#1a1a2e','#1a1a2e','#1a1a2e','#1a1a2e','#e2b96f'])
    fig2.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0', showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.caption(" Financial Tools Module | Built by Person 4 — Business Analyst")