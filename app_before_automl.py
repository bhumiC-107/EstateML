from pathlib import Path
import datetime
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# --------------------------- Page configuration ---------------------------
st.set_page_config(
    page_title="Bengaluru Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Theme / CSS ----------------------------------
# Explicitly set foreground and background colours on the main app. This
# prevents browser/Streamlit theme differences from making headings white on
# the light background. Sidebar remains dark with gold accents.
st.markdown(
    """
    <style>
    :root {
        --navy: #1a1a2e;
        --navy-light: #2d3748;
        --gold: #e2b96f;
        --page: #f5f5f0;
        --ink: #202238;
        --muted: #596174;
        --border: #d9deea;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--page) !important;
        color: var(--ink) !important;
    }

    /* Main content text: explicit selectors beat inherited theme colours. */
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main li,
    [data-testid="stAppViewContainer"] .main span,
    [data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] .main h1,
    [data-testid="stAppViewContainer"] .main h2,
    [data-testid="stAppViewContainer"] .main h3,
    [data-testid="stAppViewContainer"] .main h4 {
        color: var(--ink) !important;
        font-family: Georgia, 'Times New Roman', serif;
        opacity: 1 !important;
    }
    [data-testid="stAppViewContainer"] .main h1 {
        letter-spacing: -0.5px;
        font-weight: 700;
    }
    [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--navy) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--gold) !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--navy-light) !important;
        color: #f7fafc !important;
        border-color: #4a5568 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] input {
        -webkit-text-fill-color: #f7fafc !important;
    }
    section[data-testid="stSidebar"] button {
        background: var(--navy-light) !important;
        color: var(--gold) !important;
        border-color: #4a5568 !important;
    }
    [data-testid="stSidebarNavLink"] p,
    [data-testid="stSidebarNavLink"] span,
    section[data-testid="stSidebar"] a {
        color: var(--gold) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebarNavLink"]:hover,
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: var(--navy-light) !important;
    }

    /* Buttons */
    div.stButton > button {
        background: var(--navy) !important;
        color: var(--gold) !important;
        border: 1px solid var(--navy) !important;
        padding: 0.6rem 1.6rem;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #0f3460 !important;
        color: #ffffff !important;
        border-color: #0f3460 !important;
    }

    /* Tables, alerts, cards */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    div[data-testid="stAlert"] {
        color: var(--ink) !important;
    }
    .result-card {
        background: var(--navy);
        border-radius: 12px;
        padding: 1.5rem 1.7rem;
        margin: 0.8rem 0 1rem 0;
        color: #f7fafc !important;
    }
    .result-card * { color: #f7fafc !important; }
    .result-main {
        color: var(--gold) !important;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-range { color: #d5dbea !important; font-size: 0.95rem; }
    .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin-top: 1rem;
        border-top: 1px solid #465066;
        padding-top: 1rem;
    }
    .result-meta-label {
        color: #c0c8d8 !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .result-meta-value { color: #ffffff !important; font-size: 1rem; font-weight: 600; }
    .summary-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid var(--border);
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    .summary-item { text-align: center; }
    .summary-label {
        font-size: 0.75rem;
        color: #596174 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .summary-value { font-size: 1.25rem; font-weight: 700; color: var(--navy) !important; }
    .steps {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        margin: 0.8rem 0 2rem 0;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        color: #596174 !important;
    }
    .step.active { color: var(--navy) !important; }
    .step.done { color: #8a641f !important; }
    .step-circle {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        background: #e2e8f0;
        color: #41485b !important;
        flex-shrink: 0;
    }
    .step.active .step-circle { background: var(--navy); color: var(--gold) !important; }
    .step.done .step-circle { background: var(--gold); color: var(--navy) !important; }
    .step-line { flex: 1; height: 1px; background: #d4dae5; min-width: 18px; }
    .step-line.done { background: var(--gold); }

    @media (max-width: 700px) {
        .result-main { font-size: 1.55rem; }
        .result-meta { gap: 1rem; }
        .summary-card { gap: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Constants / paths ---------------------------
RMSE = 37.58
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
DB_PATH = BASE_DIR / "predictions.db"

# SQLite database is used by the supplied original code (not MySQL).
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_sqft REAL,
            bhk INTEGER,
            bath INTEGER,
            balcony INTEGER,
            predicted_price REAL,
            timestamp TEXT
        )
    """))


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    loaded_scaler = joblib.load(MODEL_PATH.parent / SCALER_PATH.name)
    loaded_model = joblib.load(MODEL_PATH)
    return loaded_scaler, loaded_model


try:
    scaler, model = load_assets()
except Exception as exc:
    st.error(f"Could not load the saved model/scaler: {exc}")
    st.info("Make sure final_model.joblib and scaler.joblib are in the same folder as app.py.")
    st.stop()

# --------------------------- Main page ------------------------------------
st.title("Bengaluru's Real Estate Price Predictor")
st.caption("Estimate a property's price using its area and configuration.")

predicted_this_run = bool(st.session_state.get("predicted", False))
st.markdown(
    f"""
    <div class="steps">
      <div class="step done"><div class="step-circle">1</div>Enter Details</div>
      <div class="step-line done"></div>
      <div class="step done"><div class="step-circle">2</div>Review Inputs</div>
      <div class="step-line {'done' if predicted_this_run else ''}"></div>
      <div class="step {'done' if predicted_this_run else 'active'}"><div class="step-circle">3</div>Predict Price</div>
      <div class="step-line {'done' if predicted_this_run else ''}"></div>
      <div class="step {'done' if predicted_this_run else ''}"><div class="step-circle">4</div>View Result</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("User Input Features")
total_sqft = st.sidebar.number_input("Total Sqft", min_value=100.0, max_value=10000.0, value=1000.0, step=50.0)
bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
bath = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
balcony = st.sidebar.number_input("Balconies", min_value=0, max_value=5, value=1, step=1)

user_inputs = {"total_sqft": float(total_sqft), "bhk": int(bhk), "bath": int(bath), "balcony": int(balcony)}
input_df = pd.DataFrame([user_inputs])
input_df["price_per_sqft"] = 0
input_df["sqft_per_bhk"] = input_df["total_sqft"] / input_df["bhk"]
input_df["bath_per_bhk"] = input_df["bath"] / input_df["bhk"]

# Keep the exact training feature order saved in the scaler.
feature_names = list(getattr(scaler, "feature_names_in_", input_df.columns))
missing_features = [name for name in feature_names if name not in input_df.columns]
if missing_features:
    st.error(f"The saved scaler expects features that this app does not create: {missing_features}")
    st.stop()
input_df = input_df[feature_names]

st.subheader("Final Input Features for Prediction")
st.dataframe(input_df, use_container_width=True, hide_index=True)

if st.button("Predict Price", type="primary"):
    try:
        scaled_input = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
        predicted_price = float(model.predict(scaled_input)[0])
        st.session_state["predicted"] = True
        st.session_state["last_prediction"] = {
            "predicted_price": predicted_price,
            "inputs": user_inputs.copy(),
        }

        price_low = max(0.0, predicted_price - RMSE)
        price_high = predicted_price + RMSE
        price_per_sqft = predicted_price * 100000 / user_inputs["total_sqft"]

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-main">₹{predicted_price:,.2f} Lakhs</div>
              <div class="result-range">Illustrative range using the configured RMSE: ₹{price_low:,.2f} – ₹{price_high:,.2f} Lakhs</div>
              <div class="result-meta">
                <div><div class="result-meta-label">Price per Sqft</div><div class="result-meta-value">₹{price_per_sqft:,.0f}</div></div>
                <div><div class="result-meta-label">Configuration</div><div class="result-meta-value">{bhk} BHK · {bath} Bath</div></div>
                <div><div class="result-meta-label">Total Area</div><div class="result-meta-value">{total_sqft:,.0f} sqft</div></div>
                <div><div class="result-meta-label">Configured RMSE</div><div class="result-meta-value">± ₹{RMSE:.2f} Lakhs</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-item"><div class="summary-label">Configuration</div><div class="summary-value">{bhk} BHK</div></div>
              <div class="summary-item"><div class="summary-label">Total Area</div><div class="summary-value">{total_sqft:,.0f} sqft</div></div>
              <div class="summary-item"><div class="summary-label">Bathrooms</div><div class="summary-value">{bath}</div></div>
              <div class="summary-item"><div class="summary-label">Balconies</div><div class="summary-value">{balcony}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO predictions
                (total_sqft, bhk, bath, balcony, predicted_price, timestamp)
                VALUES (:sqft, :bhk, :bath, :balcony, :price, :ts)"""),
                {
                    "sqft": total_sqft,
                    "bhk": bhk,
                    "bath": bath,
                    "balcony": balcony,
                    "price": round(predicted_price, 2),
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                },
            )
        st.success("Prediction completed and saved to prediction history.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# Show the last result again after Streamlit reruns, so it doesn't disappear
# when another widget is changed.
last_prediction = st.session_state.get("last_prediction")
if last_prediction and not st.session_state.get("prediction_shown_this_run", False):
    # The full result is rendered on the button click above; state is retained
    # for the progress indicator. This block intentionally avoids duplicate UI.
    pass

st.subheader("Feature Importance")
if hasattr(model, "feature_importances_"):
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).set_index("Feature")
    st.bar_chart(fi_df)
else:
    st.info("The loaded model does not expose feature_importances_.")

st.subheader("Prediction History")
with engine.connect() as conn:
    history_df = pd.read_sql(text("SELECT * FROM predictions ORDER BY id DESC"), conn)
st.dataframe(history_df, use_container_width=True, hide_index=True)

if not history_df.empty:
    st.subheader("Predicted Prices Over Time")
    chart_df = history_df.sort_values("timestamp").set_index("timestamp")[["predicted_price"]]
    st.line_chart(chart_df)
