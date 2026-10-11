from pathlib import Path
import datetime
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EstateML — Bengaluru Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System CSS — Clean, high-contrast fintech aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design Tokens ── */
:root {
    --navy:     #1a1a2e;
    --navy2:    #2d3748;
    --gold:     #c9922a;
    --gold-lt:  #f0d080;
    --page-bg:  #f5f5f0;
    --card-bg:  #ffffff;
    --ink:      #1a1a2e;
    --muted:    #596174;
    --border:   #dde2ec;
    --radius:   12px;
    --shadow:   0 2px 10px rgba(26, 26, 46, 0.06);
}

/* ── Base Page Styling ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--page-bg) !important;
    color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] .main,
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] .main h1,
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3,
[data-testid="stAppViewContainer"] .main h4 {
    color: var(--ink) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"],
.stCaption {
    color: var(--muted) !important;
}

/* ── Streamlit Border Container (Used for Form & Result Cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    padding: 1.5rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--gold) !important;
}
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavLink"] span,
section[data-testid="stSidebar"] a {
    color: var(--gold) !important;
    font-weight: 600 !important;
}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: var(--navy2) !important;
}

/* ── Form Inputs ── */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    margin-bottom: 0.25rem !important;
}

[data-baseweb="input"] {
    background: var(--card-bg) !important;
    border-radius: 8px !important;
    border: 1.5px solid #cbd5e1 !important;
}

[data-baseweb="input"]:focus-within {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 1px var(--navy) !important;
}

[data-baseweb="input"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {
    color: var(--ink) !important;
}

/* ── Primary CTA Button ── */
div.stButton > button {
    background: var(--navy) !important;
    color: var(--gold-lt) !important;
    border: 1px solid var(--navy) !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
}

div.stButton > button:hover {
    background: #0f3460 !important;
    color: #ffffff !important;
    border-color: #0f3460 !important;
}

/* ── Tables & DataFrames ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--card-bg) !important;
}

div[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Hero Banner ── */
.hero-band {
    background: var(--navy);
    border-radius: var(--radius);
    padding: 2.2rem 2.4rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-band::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,146,42,.18) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-block;
    background: rgba(201,146,42,.18);
    color: var(--gold-lt) !important;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f7f6f2 !important;
    margin: 0 0 0.4rem;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1rem;
    color: #c0c8d8 !important;
    margin: 0;
    max-width: 620px;
    line-height: 1.6;
}

/* ── Valuation Output Card ── */
.val-card {
    background: var(--navy);
    border-radius: 10px;
    padding: 1.8rem 1.8rem 1.6rem;
    color: #f7fafc !important;
    position: relative;
    overflow: hidden;
}

.val-card::before {
    content: '';
    position: absolute;
    bottom: -30%;
    right: -5%;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,146,42,.14) 0%, transparent 70%);
    pointer-events: none;
}

.val-label {
    color: #c0c8d8 !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
}

.val-price {
    color: var(--gold-lt) !important;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}

.val-disclaimer {
    color: #8895a7 !important;
    font-size: 0.8rem;
    line-height: 1.5;
    margin-top: 0.6rem;
}

.val-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 1.6rem;
    margin-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,.12);
    padding-top: 1.2rem;
}

.val-meta-item {
    min-width: 85px;
}

.val-meta-label {
    color: #8895a7 !important;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 2px;
}

.val-meta-value {
    color: #f0f4f8 !important;
    font-size: 1rem;
    font-weight: 700;
}

/* ── Welcome Placeholder Card ── */
.placeholder-box {
    text-align: center;
    padding: 2.2rem 1.5rem;
    color: var(--muted);
}

.placeholder-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
}

.placeholder-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--ink) !important;
    margin-bottom: 0.4rem;
}

.placeholder-desc {
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--muted) !important;
}

/* ── Responsive adjustments ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.5rem; }
    .val-price { font-size: 2rem; }
    .val-meta { gap: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Database Setup
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "final_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
DB_PATH     = BASE_DIR / "predictions.db"

engine = create_engine(
    f"sqlite:///{DB_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_sqft      REAL,
            bhk             INTEGER,
            bath            INTEGER,
            balcony         INTEGER,
            predicted_price REAL,
            timestamp       TEXT
        )
    """))

# ─────────────────────────────────────────────────────────────────────────────
# Production Model Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model file not found at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_model  = joblib.load(MODEL_PATH)
    return loaded_scaler, loaded_model

try:
    scaler, model = load_assets()
except Exception:
    st.error("The production valuation model could not be loaded. Please ensure model artifacts are present.")
    st.stop()

_feature_names = list(getattr(scaler, "feature_names_in_", [
    "total_sqft", "bath", "balcony", "bhk",
    "price_per_sqft", "sqft_per_bhk", "bath_per_bhk",
]))

def _run_valuation(sqft: float, bhk: int, bath: int, balcony: int) -> float:
    """Compute derived features and return estimated price in Lakhs."""
    row = {
        "total_sqft":     sqft,
        "bhk":            bhk,
        "bath":           bath,
        "balcony":        balcony,
        "price_per_sqft": 0.0,
        "sqft_per_bhk":   sqft / max(bhk, 1),
        "bath_per_bhk":   bath / max(bhk, 1),
    }
    df     = pd.DataFrame([row])[_feature_names]
    scaled = pd.DataFrame(scaler.transform(df), columns=_feature_names)
    return float(model.predict(scaled)[0])

# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-band">
  <div class="hero-badge">🏙 Bengaluru · Residential Valuation</div>
  <div class="hero-title">EstateML — Property Valuation</div>
  <div class="hero-sub">
    Instant, data-driven market price estimates for residential homes in Bengaluru.
    Enter property specifications and click <strong style="color:#f0d080">Estimate Property Value</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Valuation Section: 2 Balanced Columns
# ─────────────────────────────────────────────────────────────────────────────
form_col, result_col = st.columns([1, 1], gap="medium")

with form_col:
    with st.container(border=True):
        st.markdown(
            '<h3 style="margin-top:0;margin-bottom:1.2rem;font-size:1.15rem;font-weight:700;color:#1a1a2e;">'
            '📋 Property Details</h3>',
            unsafe_allow_html=True,
        )

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            total_sqft = st.number_input(
                "Total Built-Up Area (sq ft)",
                min_value=200.0,
                max_value=12000.0,
                value=1000.0,
                step=50.0,
                help="Total built-up or super built-up area in square feet.",
            )
            bath = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )
        with r1c2:
            bhk = st.number_input(
                "Bedrooms (BHK)",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )
            balcony = st.number_input(
                "Balconies",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
            )

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        estimate_clicked = st.button("🏷 Estimate Property Value", type="primary", use_container_width=True)

        if estimate_clicked:
            try:
                predicted_price = _run_valuation(float(total_sqft), int(bhk), int(bath), int(balcony))
                st.session_state["last_prediction"] = {
                    "predicted_price": predicted_price,
                    "inputs": {
                        "total_sqft": float(total_sqft),
                        "bhk": int(bhk),
                        "bath": int(bath),
                        "balcony": int(balcony),
                    },
                }
                with engine.begin() as conn:
                    conn.execute(
                        text("""INSERT INTO predictions
                                (total_sqft, bhk, bath, balcony, predicted_price, timestamp)
                                VALUES (:sqft, :bhk, :bath, :balcony, :price, :ts)"""),
                        {
                            "sqft":    float(total_sqft),
                            "bhk":     int(bhk),
                            "bath":    int(bath),
                            "balcony": int(balcony),
                            "price":   round(predicted_price, 2),
                            "ts":      datetime.datetime.now().isoformat(timespec="seconds"),
                        },
                    )
            except Exception as exc:
                st.error(f"Valuation failed: {exc}")

with result_col:
    with st.container(border=True):
        last = st.session_state.get("last_prediction")
        if last:
            p   = last["predicted_price"]
            inp = last["inputs"]
            pps = p * 100_000 / max(inp["total_sqft"], 1)
            st.markdown(f"""
            <div class="val-card">
              <div class="val-label">Estimated Market Value</div>
              <div class="val-price">₹{p:,.2f} Lakhs</div>
              <div class="val-disclaimer">
                Model-based estimate. Actual closing prices depend on floor level, property age,
                specific locality micro-market, and builder reputation.
              </div>
              <div class="val-meta">
                <div class="val-meta-item">
                  <div class="val-meta-label">Price / sq ft</div>
                  <div class="val-meta-value">₹{pps:,.0f}</div>
                </div>
                <div class="val-meta-item">
                  <div class="val-meta-label">Configuration</div>
                  <div class="val-meta-value">{inp['bhk']} BHK · {inp['bath']} Bath</div>
                </div>
                <div class="val-meta-item">
                  <div class="val-meta-label">Total Area</div>
                  <div class="val-meta-value">{inp['total_sqft']:,.0f} sq ft</div>
                </div>
                <div class="val-meta-item">
                  <div class="val-meta-label">Balconies</div>
                  <div class="val-meta-value">{inp['balcony']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder-box">
              <div class="placeholder-icon">🏠</div>
              <div class="placeholder-title">Your Valuation Appears Here</div>
              <div class="placeholder-desc">
                Fill in the property details on the left and click<br>
                <strong style="color:#1a1a2e">Estimate Property Value</strong> to generate a price estimate.
              </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Recent Valuations History
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

with engine.connect() as conn:
    history_df = pd.read_sql(
        text("SELECT * FROM predictions ORDER BY id DESC LIMIT 20"), conn
    )

if not history_df.empty:
    with st.container(border=True):
        st.markdown(
            '<h3 style="margin-top:0;margin-bottom:1rem;font-size:1.15rem;font-weight:700;color:#1a1a2e;">'
            '📈 Recent Valuations</h3>',
            unsafe_allow_html=True,
        )

        display_df = history_df[
            ["timestamp", "total_sqft", "bhk", "bath", "balcony", "predicted_price"]
        ].copy()
        display_df.columns = [
            "Date & Time", "Area (sq ft)", "BHK", "Bathrooms", "Balconies", "Estimated Value (₹ Lakhs)"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if len(history_df) > 1:
            chart_df = (
                history_df.sort_values("timestamp").tail(20)
                .set_index("timestamp")[["predicted_price"]]
                .rename(columns={"predicted_price": "Estimated Value (₹ Lakhs)"})
            )
            st.line_chart(chart_df, use_container_width=True)
