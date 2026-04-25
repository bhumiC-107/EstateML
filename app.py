import streamlit as st
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
import datetime

st.set_page_config(page_title="Bengaluru Real Estate Price Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f5f5f0; }
    h1 { font-family: 'Georgia', serif; color: #1a1a2e; letter-spacing: -0.5px; }
    h2, h3 { font-family: 'Georgia', serif; color: #2d2d2d; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] h2 { color: #e2b96f !important; }
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] {
        background-color: #2d3748 !important;
        color: #e2b96f !important;
        border: 1px solid #4a5568 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .stNumberInput label {
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #e2b96f !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        color: #f7fafc !important;
        border-radius: 6px !important;
    }
    div.stButton > button {
        background-color: #1a1a2e;
        color: #e2b96f;
        border: none;
        padding: 0.6rem 2.5rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div.stButton > button:hover {
        background-color: #0f3460;
        color: #e2b96f;
        border: none;
    }
    div[data-testid="stSuccess"] {
        background-color: #1a1a2e;
        color: #e2b96f;
        border: none;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 1rem 1.5rem;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .summary-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border: 1px solid #e2e8f0;
        display: flex;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    .summary-item { text-align: center; }
    .summary-label { font-size: 0.75rem; color: #9b8ea0; text-transform: uppercase; letter-spacing: 1px; }
    .summary-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .steps {
        display: flex;
        align-items: center;
        gap: 0;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #9b8ea0;
    }
    .step.active { color: #1a1a2e; }
    .step.done { color: #e2b96f; }
    .step-circle {
        width: 26px; height: 26px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700;
        background: #e2e8f0; color: #9b8ea0;
    }
    .step.active .step-circle { background: #1a1a2e; color: #e2b96f; }
    .step.done .step-circle { background: #e2b96f; color: #1a1a2e; }
    .step-line { flex: 1; height: 1px; background: #e2e8f0; margin: 0 8px; min-width: 30px; }
    .step-line.done { background: #e2b96f; }
    .result-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }
    .result-main { color: #e2b96f; font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-range { color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.8rem; }
    .result-meta { display: flex; gap: 2rem; margin-top: 0.8rem; border-top: 1px solid #2d3748; padding-top: 0.8rem; }
    .result-meta-item { }
    .result-meta-label { color: #718096; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
    .result-meta-value { color: #e2e8f0; font-size: 1rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
RMSE = 37.58

# ── Database setup ─────────────────────────────────────────────────────────
# ── Database setup ─────────────────────────────────────────────────────────
engine = create_engine('sqlite:///predictions.db')

with engine.connect() as conn:
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_sqft      REAL,
            bhk             INTEGER,
            bath            INTEGER,
            balcony         INTEGER,
            predicted_price REAL,
            timestamp       TEXT
        )
    '''))
    conn.commit()

# ── Load model and scaler ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.joblib')
    model  = joblib.load('final_model.joblib')
    return scaler, model

scaler, model = load_assets()

st.title("Bengaluru's Real Estate Price Predictor")

# ── Step Progress Indicator ────────────────────────────────────────────────
predicted_this_run = st.session_state.get('predicted', False)
step1 = 'done' if True else 'active'
step2 = 'done' if True else 'active'
step3 = 'active' if not predicted_this_run else 'done'
step4 = 'active' if predicted_this_run else ''

st.markdown(f"""
    <div class="steps">
        <div class="step done">
            <div class="step-circle">1</div> Enter Details
        </div>
        <div class="step-line done"></div>
        <div class="step done">
            <div class="step-circle">2</div> Review Inputs
        </div>
        <div class="step-line {'done' if predicted_this_run else ''}"></div>
        <div class="step {'done' if predicted_this_run else 'active'}">
            <div class="step-circle">3</div> Predict Price
        </div>
        <div class="step-line {'done' if predicted_this_run else ''}"></div>
        <div class="step {'done' if predicted_this_run else ''}">
            <div class="step-circle">4</div> View Result
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header('User Input Features')

def user_input_features():
    total_sqft = st.sidebar.number_input('Total Sqft', 100.0, 10000.0, 1000.0, 50.0)
    bhk        = st.sidebar.number_input('BHK', 1, 10, 2)
    bath       = st.sidebar.number_input('Bathrooms', 1, 10, 2)
    balcony    = st.sidebar.number_input('Balconies', 0, 5, 1)
    return {
        'total_sqft': total_sqft,
        'bhk':        bhk,
        'bath':       bath,
        'balcony':    balcony
    }

user_inputs = user_input_features()
input_df    = pd.DataFrame(user_inputs, index=[0])

input_df['price_per_sqft'] = 0
input_df['sqft_per_bhk']   = input_df['total_sqft'] / input_df['bhk']
input_df['bath_per_bhk']   = input_df['bath'] / input_df['bhk']

input_df = input_df[scaler.feature_names_in_]

st.subheader('Final Input Features for Prediction')
st.write(input_df)

# ── Predict and save to MySQL ──────────────────────────────────────────────
if st.button('Predict Price'):
    st.session_state['predicted'] = True

    scaled_input    = pd.DataFrame(scaler.transform(input_df), columns=scaler.feature_names_in_)
    predicted_price = model.predict(scaled_input)[0]

    price_low      = max(0, predicted_price - RMSE)
    price_high     = predicted_price + RMSE
    price_per_sqft = predicted_price * 100000 / user_inputs['total_sqft']

    # ── Result Card ────────────────────────────────────────────────────────
    st.markdown(f"""
        <div class="result-card">
            <div class="result-main">₹{predicted_price:,.2f} Lakhs</div>
            <div class="result-range">Estimated range: ₹{price_low:,.2f} – ₹{price_high:,.2f} Lakhs</div>
            <div class="result-meta">
                <div class="result-meta-item">
                    <div class="result-meta-label">Price per Sqft</div>
                    <div class="result-meta-value">₹{price_per_sqft:,.0f}</div>
                </div>
                <div class="result-meta-item">
                    <div class="result-meta-label">Configuration</div>
                    <div class="result-meta-value">{int(user_inputs['bhk'])} BHK · {int(user_inputs['bath'])} Bath</div>
                </div>
                <div class="result-meta-item">
                    <div class="result-meta-label">Total Area</div>
                    <div class="result-meta-value">{user_inputs['total_sqft']:.0f} sqft</div>
                </div>
                <div class="result-meta-item">
                    <div class="result-meta-label">Model RMSE</div>
                    <div class="result-meta-value">± ₹{RMSE} Lakhs</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Input Summary Card ─────────────────────────────────────────────────
    st.markdown(f"""
        <div class="summary-card">
            <div class="summary-item">
                <div class="summary-label">Configuration</div>
                <div class="summary-value">{int(user_inputs['bhk'])} BHK</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Area</div>
                <div class="summary-value">{user_inputs['total_sqft']:.0f} sqft</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Bathrooms</div>
                <div class="summary-value">{int(user_inputs['bath'])}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Balconies</div>
                <div class="summary-value">{int(user_inputs['balcony'])}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with engine.connect() as conn:
        conn.execute(text('''
            INSERT INTO predictions (total_sqft, bhk, bath, balcony, predicted_price, timestamp)
            VALUES (:sqft, :bhk, :bath, :balcony, :price, :ts)
        '''), {
            'sqft'   : user_inputs['total_sqft'],
            'bhk'    : user_inputs['bhk'],
            'bath'   : user_inputs['bath'],
            'balcony': user_inputs['balcony'],
            'price'  : round(predicted_price, 2),
            'ts'     : datetime.datetime.now()
        })
        conn.commit()

# ── Feature Importance Chart ───────────────────────────────────────────────
st.subheader('Feature Importance')
fi_df = pd.DataFrame({
    'Feature'   : scaler.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).set_index('Feature')
st.bar_chart(fi_df)

# ── Prediction History ─────────────────────────────────────────────────────
st.subheader('Prediction History')
with engine.connect() as conn:
    history_df = pd.read_sql('SELECT * FROM predictions ORDER BY id DESC', conn)
st.dataframe(history_df)

# ── Prediction History Chart ───────────────────────────────────────────────
if not history_df.empty:
    st.subheader('Predicted Prices Over Time')
    chart_df = history_df[['timestamp', 'predicted_price']].set_index('timestamp')
    st.line_chart(chart_df)