"""EstateML Model Performance & Evaluation Insights.

Displays candidate model benchmarks, error metrics, and validation results.
"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance | EstateML",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Unified High-Contrast Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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

.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--page-bg) !important;
    color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] .main,
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] .main h1,
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3,
[data-testid="stAppViewContainer"] .main h4,
[data-testid="stHeadingWithActionElements"] * {
    color: var(--ink) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"],
.stCaption {
    color: var(--muted) !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--navy) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* Border Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    padding: 1.5rem !important;
}

/* DataFrames */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--card-bg) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--gold) !important;
}
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavLink"] span {
    color: var(--gold) !important;
    font-weight: 600 !important;
}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background-color: var(--navy2) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header (Requirement A)
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Model Performance & Evaluation Insights")
st.markdown(
    "EstateML continuously benchmarks candidate machine learning algorithms to ensure robust, "
    "unbiased property valuation across the Bengaluru residential real estate market."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Load Actual Metrics from Artifacts (Requirement B, C, D)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "artifacts" / "automl_metrics.json"

MODEL_FRIENDLY_NAMES = {
    "extra_trees": "Extra Trees Regressor",
    "random_forest": "Random Forest Regressor",
    "hist_gradient_boosting": "HistGradientBoosting Regressor",
}

metrics_loaded = False
if METRICS_PATH.exists():
    try:
        metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        selected_key = metrics_data.get("selected_model", "extra_trees")
        results = metrics_data.get("results", [])
        metrics_loaded = bool(results)
    except Exception as exc:
        st.warning(f"Could not parse metrics file: {exc}")

if not metrics_loaded:
    st.info("Benchmark metrics are not currently available. Run the AutoML pipeline to populate candidate metrics.")
    st.stop()

# Find the best row corresponding to the selected model
best_row = next((r for r in results if r.get("model") == selected_key), None)
if best_row is None and results:
    best_row = min(results, key=lambda x: x.get("test_rmse", float("inf")))

selected_display_name = MODEL_FRIENDLY_NAMES.get(selected_key, selected_key.replace("_", " ").title())

# ─────────────────────────────────────────────────────────────────────────────
# B. Four Metric Cards with Explanations
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🎯 Primary Benchmark Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="Selected Model",
        value=selected_display_name,
    )
    st.caption("Top performer on out-of-sample test split")

with c2:
    val_rmse = best_row.get("test_rmse", 0.0)
    st.metric(
        label="Test RMSE",
        value=f"₹{val_rmse:.2f} L",
    )
    st.caption("Root Mean Squared Error (penalizes large outliers)")

with c3:
    val_mae = best_row.get("test_mae", 0.0)
    st.metric(
        label="Test MAE",
        value=f"₹{val_mae:.2f} L",
    )
    st.caption("Mean Absolute Error (typical price deviation)")

with c4:
    val_r2 = best_row.get("test_r2", 0.0)
    st.metric(
        label="Test R² Score",
        value=f"{val_r2:.4f}",
    )
    st.caption(f"Explains {val_r2*100:.1f}% of price variation")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# C. Model Comparison Table (Candidate Models)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Candidate Model Comparison")
st.markdown(
    "Evaluation on identical 80/20 train/test splits using 5-fold cross-validation. "
    "All candidates use strictly raw property features (`total_sqft`, `bhk`, `bath`, `balcony`)."
)

table_rows = []
for r in results:
    m_key = r.get("model", "")
    m_name = MODEL_FRIENDLY_NAMES.get(m_key, m_key.replace("_", " ").title())
    is_sel = (m_key == selected_key)
    table_rows.append({
        "Model": m_name,
        "5-Fold CV RMSE (₹ Lakhs)": round(float(r.get("cv_rmse", 0)), 2),
        "Test RMSE (₹ Lakhs)": round(float(r.get("test_rmse", 0)), 2),
        "Test MAE (₹ Lakhs)": round(float(r.get("test_mae", 0)), 2),
        "Test R²": round(float(r.get("test_r2", 0)), 4),
        "Status": "★ Selected Production Benchmark" if is_sel else "Evaluated Candidate",
    })

df_table = pd.DataFrame(table_rows).sort_values("Test RMSE (₹ Lakhs)")
st.dataframe(df_table, use_container_width=True, hide_index=True)

# Methodology Disclosure
with st.container(border=True):
    st.markdown(
        "**💡 Methodology Note & Target-Leakage Disclosure:**<br>"
        "All candidate models above were trained strictly without target-derived features. "
        "The original baseline model reported an artificially low RMSE (~₹37.58 Lakhs) because its pipeline "
        "included `price_per_sqft` derived directly from sale price. In real prospective valuation, "
        "`price_per_sqft` cannot be known beforehand. The models above represent genuine predictive generalization.",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# D. Visual Model Performance Comparison Chart
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Visual Performance Comparison")

chart_data = []
for r in results:
    m_key = r.get("model", "")
    m_name = MODEL_FRIENDLY_NAMES.get(m_key, m_key.replace("_", " ").title())
    chart_data.append({
        "Model": m_name,
        "Test RMSE (₹ L)": round(float(r.get("test_rmse", 0)), 2),
        "Test MAE (₹ L)": round(float(r.get("test_mae", 0)), 2),
    })

chart_df = pd.DataFrame(chart_data)
melted_df = pd.melt(
    chart_df,
    id_vars=["Model"],
    value_vars=["Test RMSE (₹ L)", "Test MAE (₹ L)"],
    var_name="Error Metric",
    value_name="Error (₹ Lakhs)",
)

fig = px.bar(
    melted_df,
    x="Model",
    y="Error (₹ Lakhs)",
    color="Error Metric",
    barmode="group",
    text="Error (₹ Lakhs)",
    color_discrete_map={
        "Test RMSE (₹ L)": "#1a1a2e",
        "Test MAE (₹ L)": "#c9922a",
    },
)

fig.update_layout(
    plot_bgcolor="#f5f5f0",
    paper_bgcolor="#f5f5f0",
    font_color="#1a1a2e",
    xaxis=dict(title="Candidate Model", tickfont=dict(color="#1a1a2e")),
    yaxis=dict(title="Error (₹ Lakhs)", tickfont=dict(color="#1a1a2e")),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#dde2ec",
        borderwidth=1,
        font=dict(color="#1a1a2e"),
    ),
    margin=dict(l=40, r=40, t=30, b=40),
)
fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Comparison of Test RMSE and Test MAE across algorithms. Lower values indicate greater accuracy. "
    "Extra Trees and HistGradientBoosting achieve competitive MAE (~₹38.66 Lakhs), while Extra Trees "
    "minimizes peak errors (RMSE ₹95.51 Lakhs)."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# E. Collapsible Technical Details (Requirement E)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🛠 Technical Details & MLflow Experiment Governance"):
    st.markdown("""
    #### Experiment Tracking Architecture
    - **Tracking Backend**: Local SQLite metadata registry (`sqlite:///mlflow.db`).
    - **Artifact Store**: Local directory `./mlruns` storing model weights, feature schemas, and serialization formats.
    - **Hyperparameter Logging**: Preprocessing median imputers, n_estimators, max_depth, and cross-validation splits.
    - **Reproducibility**: DVC stages define deterministic data pipelines from dataset ingestion to artifact release.
    """)
    st.info(
        "For deep model audits, administrators can launch the local MLflow dashboard "
        "via the background CLI interface (`py -m mlflow ui`)."
    )

st.caption("EstateML Model Governance & Benchmark Suite")
