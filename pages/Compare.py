import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Property Comparator", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f5f5f0; }
h1, h2, h3 { font-family: 'Georgia', serif; color: #1a1a2e; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; }
div.stButton > button {
    background-color: #1a1a2e; color: #e2b96f;
    border: none; padding: 0.6rem 2.5rem;
    border-radius: 8px; font-size: 0.95rem;
    font-weight: 600; letter-spacing: 0.5px; width: 100%;
}
div.stButton > button:hover { background-color: #0f3460; color: #e2b96f; }
.prop-card {
    background: #1a1a2e; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1rem; text-align: center;
}
.prop-label { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.prop-price { color: #e2b96f; font-size: 2.2rem; font-weight: 700; margin: 0.3rem 0; }
.prop-range { color: #718096; font-size: 0.85rem; }
.winner-badge {
    background: #e2b96f; color: #1a1a2e;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.78rem; font-weight: 700;
    display: inline-block; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

RMSE = 37.58

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.joblib')
    model  = joblib.load('final_model.joblib')
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
    st.markdown("### 🏠 Property A")
    a_sqft   = st.number_input("Total Sqft",  100.0, 10000.0, 1000.0, 50.0, key="a_sqft")
    a_bhk    = st.number_input("BHK",         1,     10,      2,             key="a_bhk")
    a_bath   = st.number_input("Bathrooms",   1,     10,      2,             key="a_bath")
    a_bal    = st.number_input("Balconies",   0,     5,       1,             key="a_bal")

with div:
    st.markdown("<div style='text-align:center;font-size:2rem;color:#9b8ea0;margin-top:5rem'>VS</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🏢 Property B")
    b_sqft   = st.number_input("Total Sqft",  100.0, 10000.0, 1500.0, 50.0, key="b_sqft")
    b_bhk    = st.number_input("BHK",         1,     10,      3,             key="b_bhk")
    b_bath   = st.number_input("Bathrooms",   1,     10,      3,             key="b_bath")
    b_bal    = st.number_input("Balconies",   0,     5,       2,             key="b_bal")

st.divider()

_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    compare_clicked = st.button("⚖️ Compare Properties")

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
        "Property A":      [f"₹{pa:.2f}",          f"₹{pps_a:,.0f}", f"{a_sqft:.0f}",    int(a_bhk), int(a_bath), int(a_bal)],
        "Property B":      [f"₹{pb:.2f}",          f"₹{pps_b:,.0f}", f"{b_sqft:.0f}",    int(b_bhk), int(b_bath), int(b_bal)],
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor='#f5f5f0', showlegend=True,
        legend=dict(bgcolor='#f5f5f0')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download comparison
    st.download_button(
        "⬇ Download Comparison as CSV",
        data=metrics_df.to_csv(index=False),
        file_name="property_comparison.csv",
        mime="text/csv"
    )

st.caption(" Property Comparator | Built by Person 3 — Backend Developer")
