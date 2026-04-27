import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Market Insights", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f5f5f0; }
h1, h2, h3 { font-family: 'Georgia', serif; color: #1a1a2e; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.title(" Market Insights & Exploratory Data Analysis")
st.markdown("Deep-dive into the Bengaluru housing market trends powering the prediction model.")

@st.cache_data
def load_data():
    df = pd.read_csv("bengaluru_house_prices.csv")
    # Clean price column if needed (handles formats like "₹45 Lac", "1.2 Cr", etc.)
    if df['price'].dtype == object:
        def parse_price(p):
            try:
                p = str(p).replace('₹','').replace(',','').strip()
                if 'Cr' in p:
                    return float(p.replace('Cr','').strip()) * 100
                elif 'Lac' in p or 'L' in p:
                    return float(p.replace('Lac','').replace('L','').strip())
                else:
                    return float(p)
            except:
                return None
        df['price'] = df['price'].apply(parse_price)
    df = df.dropna(subset=['price'])
    # Extract BHK number if stored as string like "2 BHK"
    if df['bhk'].dtype == object:
        df['bhk'] = df['bhk'].str.extract(r'(\d+)').astype(float)
    df['bhk'] = df['bhk'].fillna(df['bhk'].median()).astype(int)
    df = df[(df['price'] > 0) & (df['total_sqft'] > 0)]
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load bengaluru_house_prices.csv: {e}")
    st.stop()

# ── KPI Row ──────────────────────────────────────────────────────────────
st.markdown("### 🔢 Dataset Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Listings",       f"{len(df):,}")
k2.metric("Avg Price",            f"₹{df['price'].mean():.1f} L")
k3.metric("Avg Area",             f"{df['total_sqft'].mean():.0f} sqft")
k4.metric("Avg Price / Sqft",     f"₹{df['price_per_sqft'].mean():,.0f}")

st.divider()

# ── Row 1: Price Distribution + BHK Split ────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig = px.histogram(
        df[df['price'] < df['price'].quantile(0.95)],
        x='price', nbins=60,
        labels={'price': 'Price (Lakhs)'},
        color_discrete_sequence=['#1a1a2e']
    )
    fig.update_layout(
        plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
        bargap=0.05, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("BHK Configuration Share")
    bhk_counts = df['bhk'].value_counts().reset_index()
    bhk_counts.columns = ['BHK', 'Count']
    fig = px.pie(
        bhk_counts, values='Count', names='BHK',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(paper_bgcolor='#f5f5f0')
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Box plots + Scatter ────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Price Spread by BHK")
    top_bhk = df['bhk'].value_counts().head(5).index.tolist()
    fig = px.box(
        df[df['bhk'].isin(top_bhk)],
        x='bhk', y='price',
        labels={'bhk': 'BHK', 'price': 'Price (Lakhs)'},
        color='bhk',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(
        plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0',
        showlegend=False,
        yaxis_range=[0, df['price'].quantile(0.95)]
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Price vs Area")
    plot_df = df[
        (df['total_sqft'] < df['total_sqft'].quantile(0.97)) &
        (df['price'] < df['price'].quantile(0.97))
    ]
    fig = px.scatter(
        plot_df, x='total_sqft', y='price',
        color='bhk',
        labels={'total_sqft': 'Total Sqft', 'price': 'Price (Lakhs)', 'bhk': 'BHK'},
        opacity=0.6,
        color_continuous_scale='Blues'
    )
    fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0')
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Correlation Heatmap ────────────────────────────────────────────
st.subheader("Feature Correlation Heatmap")
num_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'price', 'price_per_sqft']
available = [c for c in num_cols if c in df.columns]
corr = df[available].corr().round(2)

fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale='Blues',
    text=corr.values,
    texttemplate='%{text}',
    showscale=True
))
fig.update_layout(
    height=400,
    plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0'
)
st.plotly_chart(fig, use_container_width=True)

# ── Row 4: Price per Sqft by BHK ─────────────────────────────────────────
st.subheader("Avg Price per Sqft by BHK Type")
pps_df = (
    df[df['bhk'].isin(top_bhk)]
    .groupby('bhk')['price_per_sqft']
    .mean()
    .reset_index()
    .rename(columns={'price_per_sqft': 'Avg Price/Sqft'})
)
fig = px.bar(
    pps_df, x='bhk', y='Avg Price/Sqft',
    labels={'bhk': 'BHK', 'Avg Price/Sqft': 'Avg ₹/sqft'},
    color='Avg Price/Sqft',
    color_continuous_scale='Blues',
    text_auto='.0f'
)
fig.update_layout(plot_bgcolor='#f5f5f0', paper_bgcolor='#f5f5f0', showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(" Data: Bengaluru House Prices Dataset | Analysis by Person 2 — Data Analyst")
