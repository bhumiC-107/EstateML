import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

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

RMSE = 37.58

@st.cache_resource
def load_assets():
    scaler = joblib.load(BASE_DIR / 'scaler.joblib')
    model  = joblib.load(BASE_DIR / 'final_model.joblib')
    return scaler, model

scaler, model = load_assets()

def predict_price(sqft, bhk, bath, balcony):
    df = pd.DataFrame([{
        'total_sqft':   sqft,
        'bhk':          bhk,
        'bath':         bath,
        'balcony':      balcony,
        'price_per_sqft': 0,
        'sqft_per_bhk': sqft / bhk,
        'bath_per_bhk': bath / bhk
    }])
    df = df[scaler.feature_names_in_]
    scaled = pd.DataFrame(scaler.transform(df), columns=scaler.feature_names_in_)
    return round(model.predict(scaled)[0], 2)

# ── Page Header ───────────────────────────────────────────────────────────
st.title("⚖️ Property Comparator")
st.markdown("Enter details for two properties and compare their estimated prices side by side.")

st.divider()

# ── Input Panels ──────────────────────────────────────────────────────────
col1, div, col2 = st.columns([5, 1, 5])

with col1:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0;font-size:1.2rem;font-weight:700;color:#1a1a2e;'>🏠 Property A</h3>", unsafe_allow_html=True)
        a_sqft   = st.number_input("Total Sqft",  100.0, 10000.0, 1000.0, 50.0, key="a_sqft")
        a_bhk    = st.number_input("BHK",         1,     10,      2,             key="a_bhk")
        a_bath   = st.number_input("Bathrooms",   1,     10,      2,             key="a_bath")
        a_bal    = st.number_input("Balconies",   0,     5,       1,             key="a_bal")

with div:
    st.markdown("<div style='text-align:center;font-size:1.6rem;color:#596174;font-weight:800;margin-top:7rem;'>VS</div>", unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0;font-size:1.2rem;font-weight:700;color:#1a1a2e;'>🏢 Property B</h3>", unsafe_allow_html=True)
        b_sqft   = st.number_input("Total Sqft",  100.0, 10000.0, 1500.0, 50.0, key="b_sqft")
        b_bhk    = st.number_input("BHK",         1,     10,      3,             key="b_bhk")
        b_bath   = st.number_input("Bathrooms",   1,     10,      3,             key="b_bath")
        b_bal    = st.number_input("Balconies",   0,     5,       2,             key="b_bal")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    compare_clicked = st.button("⚖️ Compare Properties", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────
if compare_clicked:
    pa = predict_price(a_sqft, a_bhk, a_bath, a_bal)
    pb = predict_price(b_sqft, b_bhk, b_bath, b_bal)

    pps_a = pa * 100000 / a_sqft
    pps_b = pb * 100000 / b_sqft

    winner = "A" if pa < pb else "B"  # lower price = better value (same features)

    st.markdown("###  Comparison Results")

    r1, r2 = st.columns(2)

    with r1:
        badge = '<div class="winner-badge"> Better Value</div>' if winner == "A" else ''
        st.markdown(f"""
        <div class="prop-card">
            <div class="prop-label">Property A · {int(a_bhk)} BHK · {int(a_sqft)} sqft</div>
            <div class="prop-price">₹{pa:.2f} L</div>
            <div class="prop-range">Range: ₹{max(0, pa-RMSE):.2f} – ₹{pa+RMSE:.2f} Lakhs</div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        badge = '<div class="winner-badge"> Better Value</div>' if winner == "B" else ''
        st.markdown(f"""
        <div class="prop-card">
            <div class="prop-label">Property B · {int(b_bhk)} BHK · {int(b_sqft)} sqft</div>
            <div class="prop-price">₹{pb:.2f} L</div>
            <div class="prop-range">Range: ₹{max(0, pb-RMSE):.2f} – ₹{pb+RMSE:.2f} Lakhs</div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    # Metrics comparison table
    st.markdown("###  Feature-by-Feature Breakdown")
    metrics_df = pd.DataFrame({
        "Metric":          ["Predicted Price (L)", "Price/Sqft (₹)", "Total Area (sqft)", "BHK", "Bathrooms", "Balconies"],
        "Property A":      [f"₹{pa:.2f}",          f"₹{pps_a:,.0f}", f"{a_sqft:.0f}",    str(int(a_bhk)), str(int(a_bath)), str(int(a_bal))],
        "Property B":      [f"₹{pb:.2f}",          f"₹{pps_b:,.0f}", f"{b_sqft:.0f}",    str(int(b_bhk)), str(int(b_bath)), str(int(b_bal))],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Radar chart
    st.markdown("###  Radar Comparison")

    def norm(val, mn, mx):
        return (val - mn) / (mx - mn) if mx != mn else 0.5

    categories = ['Price (L)', 'Area (sqft)', 'BHK', 'Bathrooms', 'Price/Sqft']
    vals_a = [
        norm(pa,    min(pa,pb),    max(pa,pb)),
        norm(a_sqft,min(a_sqft,b_sqft), max(a_sqft,b_sqft)),
        norm(a_bhk, min(a_bhk,b_bhk),   max(a_bhk,b_bhk)),
        norm(a_bath,min(a_bath,b_bath),  max(a_bath,b_bath)),
        norm(pps_a, min(pps_a,pps_b),   max(pps_a,pps_b)),
    ]
    vals_b = [
        norm(pb,    min(pa,pb),    max(pa,pb)),
        norm(b_sqft,min(a_sqft,b_sqft), max(a_sqft,b_sqft)),
        norm(b_bhk, min(a_bhk,b_bhk),   max(a_bhk,b_bhk)),
        norm(b_bath,min(a_bath,b_bath),  max(a_bath,b_bath)),
        norm(pps_b, min(pps_a,pps_b),   max(pps_a,pps_b)),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=categories + [categories[0]],
                                   fill='toself', name='Property A', line_color='#1a1a2e'))
    fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=categories + [categories[0]],
                                   fill='toself', name='Property B', line_color='#e2b96f', opacity=0.7))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#1a1a2e')),
            angularaxis=dict(tickfont=dict(color='#1a1a2e', size=12)),
        ),
        paper_bgcolor='#f5f5f0',
        font=dict(color='#1a1a2e'),
        showlegend=True,
        legend=dict(bgcolor='#ffffff', bordercolor='#d9deea', borderwidth=1, font=dict(color='#1a1a2e'))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download comparison
    st.download_button(
        "⬇ Download Comparison as CSV",
        data=metrics_df.to_csv(index=False),
        file_name="property_comparison.csv",
        mime="text/csv"
    )

st.caption("EstateML Property Intelligence · Side-by-Side Property Comparison")
