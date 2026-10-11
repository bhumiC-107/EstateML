import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Market Insights", layout="wide")

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

/* ── Dividers ── */
hr, [data-testid="stDivider"] {
    border-color: #d9deea !important;
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

st.title(" Market Insights & Exploratory Data Analysis")
st.markdown("Deep-dive into the Bengaluru housing market trends powering the prediction model.")

@st.cache_data
def load_data():
    csv_path = BASE_DIR / "bengaluru_house_prices.csv"
    if not csv_path.exists():
        csv_path = Path("bengaluru_house_prices.csv")
    df = pd.read_csv(csv_path)

    # Normalize column names FIRST
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # NOW convert to numeric (column names are guaranteed lowercase/clean)
    df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
    df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
    df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
    # Extract BHK — standard Bengaluru dataset has 'size' like "2 BHK"
    if 'size' in df.columns:
        df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)
    elif 'bhk' in df.columns and df['bhk'].dtype == object:
        df['bhk'] = df['bhk'].str.extract(r'(\d+)').astype(float)

    # Rename variants
    df = df.rename(columns={
        'bathrooms': 'bath', 'baths': 'bath',
        'total_area': 'total_sqft', 'balconies': 'balcony',
    })

    # Parse price (handles "45.5" in lakhs OR "₹1.2 Cr" strings)
    if df['price'].dtype == object:
        def parse_price(p):
            try:
                p = str(p).replace('₹','').replace(',','').strip()
                if 'Cr' in p or 'cr' in p:
                    return float(p.replace('Cr','').replace('cr','').strip()) * 100
                elif 'Lac' in p or 'L' in p:
                    return float(p.replace('Lac','').replace('L','').strip())
                return float(p)
            except:
                return None
        df['price'] = df['price'].apply(parse_price)

    # Parse total_sqft ranges like "1000-1200"
    if df['total_sqft'].dtype == object:
        def parse_sqft(s):
            try:
                s = str(s)
                if '-' in s:
                    a, b = s.split('-')[:2]
                    return (float(a) + float(b)) / 2
                return float(s)
            except:
                return None
        df['total_sqft'] = df['total_sqft'].apply(parse_sqft)

    df = df.dropna(subset=['price', 'total_sqft'])
    df = df[(df['price'] > 0) & (df['total_sqft'] > 100)]

    if 'bhk' in df.columns:
        df['bhk'] = df['bhk'].fillna(df['bhk'].median()).astype(int)

    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────
st.markdown("###  Dataset Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Listings",   f"{len(df):,}")
k2.metric("Avg Price",        f"₹{df['price'].mean():.1f} L")
k3.metric("Avg Area",         f"{df['total_sqft'].mean():.0f} sqft")
k4.metric("Avg Price / Sqft", f"₹{df['price_per_sqft'].mean():,.0f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig = px.histogram(
        df[df['price'] < df['price'].quantile(0.95)],
        x='price', nbins=60,
        labels={'price': 'Price (Lakhs)'},
        color_discrete_sequence=['#1a1a2e']
    )
    fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
                      bargap=0.05, showlegend=False, font_color='#1a1a2e')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("BHK Configuration Share")
    if 'bhk' in df.columns:
        bhk_counts = df['bhk'].value_counts().reset_index()
        bhk_counts.columns = ['BHK', 'Count']
        fig = px.pie(bhk_counts, values='Count', names='BHK',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(paper_bgcolor='#f5f5f0', font_color='#1a1a2e')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("BHK column not found in dataset.")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Price Spread by BHK")
    if 'bhk' in df.columns:
        top_bhk = df['bhk'].value_counts().head(5).index.tolist()
        fig = px.box(
            df[df['bhk'].isin(top_bhk)], x='bhk', y='price',
            labels={'bhk': 'BHK', 'price': 'Price (Lakhs)'},
            color='bhk', color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
                          showlegend=False, font_color='#1a1a2e',
                          yaxis_range=[0, df['price'].quantile(0.95)])
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Price vs Area")
    plot_df = df[
        (df['total_sqft'] < df['total_sqft'].quantile(0.97)) &
        (df['price']      < df['price'].quantile(0.97))
    ]
    color_col = 'bhk' if 'bhk' in df.columns else None
    fig = px.scatter(
        plot_df, x='total_sqft', y='price', color=color_col,
        labels={'total_sqft': 'Total Sqft', 'price': 'Price (Lakhs)'},
        opacity=0.5, color_continuous_scale='Blues'
    )
    fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0', font_color='#1a1a2e')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Feature Correlation Heatmap")
num_cols  = ['total_sqft', 'bath', 'balcony', 'bhk', 'price', 'price_per_sqft']
available = [c for c in num_cols if c in df.columns]
corr = df[available].corr().round(2)
fig = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
    colorscale='Blues', text=corr.values, texttemplate='%{text}', showscale=True
))
fig.update_layout(height=380, plot_bgcolor='#f5f5f0',
                  paper_bgcolor='#f5f5f0', font_color='#1a1a2e')
st.plotly_chart(fig, use_container_width=True)

st.divider()

if 'bhk' in df.columns:
    st.subheader("Avg Price per Sqft by BHK Type")
    top_bhk = df['bhk'].value_counts().head(5).index.tolist()
    pps_df  = (
        df[df['bhk'].isin(top_bhk)]
        .groupby('bhk')['price_per_sqft'].mean()
        .reset_index().rename(columns={'price_per_sqft': 'Avg Price/Sqft'})
    )
    fig = px.bar(pps_df, x='bhk', y='Avg Price/Sqft',
                 labels={'bhk': 'BHK', 'Avg Price/Sqft': 'Avg ₹/sqft'},
                 color='Avg Price/Sqft', color_continuous_scale='Blues', text_auto='.0f')
    fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
                      showlegend=False, font_color='#1a1a2e')
    st.plotly_chart(fig, use_container_width=True)

st.caption("EstateML Market Insights · Bengaluru Residential Real Estate Trends")