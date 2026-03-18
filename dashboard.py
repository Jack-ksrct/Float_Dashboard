import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.ndimage import uniform_filter1d

st.set_page_config(page_title="ARGO Dashboard", layout="wide", page_icon= "favicon.png")

st.markdown("""
<style>
    /* ── Mobile-first responsive layout ── */
    @media (max-width: 768px) {
        /* Tighter page padding on mobile */
        .block-container { padding: 1rem 0.75rem !important; }

        /* Metric cards stack & shrink on mobile */
        .metric-card { padding: 12px 10px !important; }
        .metric-label { font-size: 11px !important; }
        .metric-value { font-size: 18px !important; }

        /* Sidebar toggle visible on mobile */
        section[data-testid="stSidebar"] { min-width: 80vw !important; }

        /* Tabs scroll horizontally on mobile */
        div[data-testid="stTabs"] button {
            font-size: 11px !important;
            padding: 6px 8px !important;
        }

        /* Title smaller on mobile */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #0f1923);
        border: 1px solid #1f3a5f;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label { color: #7faec8; font-size: 12px; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 22px; font-weight: 700; }

    /* ── Make plotly charts not overflow on mobile ── */
    .js-plotly-plot, .plotly { max-width: 100% !important; }

    /* ── Dataframe horizontal scroll on mobile ── */
    div[data-testid="stDataFrame"] { overflow-x: auto !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 ARGO Float Dashboard")
st.caption("Indian Ocean Argo Float Data — Arabian Sea & Bay of Bengal")

DATA_FILE = "profiles_with_region.csv"

COLORS = {
    "Arabian Sea":   "#F77F00",
    "Bay of Bengal": "#00B4D8",
}

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,25,35,0.85)",
    font=dict(color="#c9d6e3", family="Inter, sans-serif", size=11),
    legend=dict(
        bgcolor="rgba(20,35,55,0.85)", bordercolor="#1f3a5f", borderwidth=1,
        font=dict(size=11),
        orientation="v",
        yanchor="top", y=0.99,
        xanchor="right", x=0.99,
    ),
    margin=dict(l=50, r=15, t=50, b=55),  
)
AXIS_STYLE = dict(gridcolor="#1a3050", linecolor="#2a4a6f",
                  zerolinecolor="#1f3a5f", tickfont=dict(size=10))


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower().str.strip()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["depth_bin"] = (df["depth"] // 50 * 50).astype(int)
    return df


df = load_data()
st.sidebar.header("🔧 Filters")
regions = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique())
)
years = sorted(df["year"].dropna().astype(int).unique())
selected_year = st.sidebar.selectbox("Select Year", options=["All"] + years)
parameter = st.sidebar.radio("Select Parameter", ["Temperature", "Salinity"])

filtered_df = df[df["region"].isin(regions)].copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]

param_col  = "temperature" if parameter == "Temperature" else "salinity"
param_unit = "°C"          if parameter == "Temperature" else "PSU"

st.markdown("### 📌 Overview")
r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)
for col, label, val in zip(
    [r1c1, r1c2, r2c1, r2c2],
    ["Total Records", "Unique Floats", "Unique Profiles", "Year"],
    [f"{filtered_df.shape[0]:,}", f"{filtered_df['float_id'].nunique():,}",
     f"{filtered_df['profile_id'].nunique():,}", str(selected_year)],
):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{val}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("🌍 Float Locations")
map_df = filtered_df.dropna(subset=["latitude", "longitude"])
if len(map_df) > 3000:
    map_df = map_df.sample(3000, random_state=42)

fig_map = px.scatter_map(
    map_df, lat="latitude", lon="longitude", color="region",
    color_discrete_map=COLORS, zoom=3, center={"lat": 5, "lon": 75},
    hover_data={"latitude": ":.3f", "longitude": ":.3f", "year": True,
                "temperature": ":.2f", "salinity": ":.2f", "region": True},
)
fig_map.update_layout(
    map_style="open-street-map",
    height=350,                   
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        orientation="h", yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(15,25,35,0.7)", font=dict(size=11),
    )
)
fig_map.update_traces(marker=dict(size=5, opacity=0.75))
st.plotly_chart(fig_map, use_container_width=True)
st.markdown("---")

st.subheader(f"📊 {parameter} Analysis")
tab1, tab2, tab3 = st.tabs(["🌡 Profile vs Depth", "📦 By Region", "📈 Yearly Trend"])

with tab1:
    fig1 = go.Figure()
    smooth_size = 15 if param_col == "salinity" else 5
    all_means, region_stats = [], {}

    for region in regions:
        rdf = filtered_df[filtered_df["region"] == region]
        grp = rdf.groupby("depth_bin")[param_col]
        stats = grp.agg(
            mean_val="mean",
            p25=lambda x: np.percentile(x.dropna(), 25),
            p75=lambda x: np.percentile(x.dropna(), 75),
        ).reset_index().sort_values("depth_bin")
        sz = min(smooth_size, len(stats))
        stats["mean_s"] = uniform_filter1d(stats["mean_val"].values, size=sz)
        stats["p25_s"]  = uniform_filter1d(stats["p25"].values,      size=sz)
        stats["p75_s"]  = uniform_filter1d(stats["p75"].values,      size=sz)
        region_stats[region] = stats
        all_means.extend(stats["mean_s"].tolist())

    x_min, x_max = min(all_means), max(all_means)
    x_spread = max(x_max - x_min, 0.5 if param_col == "salinity" else 1.0)
    x_pad = x_spread * 0.15

    for region in regions:
        stats   = region_stats[region]
        col_hex = COLORS.get(region, "#aaa")
        depths  = stats["depth_bin"].values
        fig1.add_trace(go.Scatter(
            x=list(depths) + list(depths[::-1]),
            y=list(stats["p75_s"]) + list(stats["p25_s"][::-1]),
            fill="toself", fillcolor=hex_to_rgba(col_hex, 0.18),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig1.add_trace(go.Scatter(
            x=depths, y=stats["mean_s"],
            mode="lines", name=region,
            line=dict(color=col_hex, width=2.5),
            hovertemplate=(
                f"<b>{region}</b><br>Depth: %{{x}} m<br>"
                f"{parameter}: %{{y:.4f}} {param_unit}<extra></extra>"
            ),
        ))

    max_depth = filtered_df["depth_bin"].max()
    fig1.update_layout(
        **CHART_BASE, height=400,
        title=dict(text=f"Mean {parameter} Profile with Depth",
                   font=dict(size=13, color="#e0eaf5")),
        xaxis=dict(**AXIS_STYLE,
                   title=dict(text="Depth (m)", font=dict(size=12)),
                   range=[-20, max_depth + 30]),
        yaxis=dict(**AXIS_STYLE,
                   title=dict(text=f"{parameter} ({param_unit})", font=dict(size=12)),
                   range=[x_min - x_pad, x_max + x_pad]),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("X = Depth | Y = " + parameter + " | Band = IQR")

with tab2:
    fig2 = go.Figure()
    for region in sorted(filtered_df["region"].dropna().unique()):
        vals = filtered_df[filtered_df["region"] == region][param_col].dropna()
        col_hex = COLORS.get(region, "#aaa")
        fig2.add_trace(go.Violin(
            y=vals, name=region,
            box_visible=True, meanline_visible=True,
            fillcolor=hex_to_rgba(col_hex, 0.35),
            line_color=col_hex, opacity=0.9,
        ))
    y_min2 = filtered_df[param_col].dropna().min()
    y_max2 = filtered_df[param_col].dropna().max()
    y_pad2 = (y_max2 - y_min2) * 0.05
    fig2.update_layout(
        **CHART_BASE, height=400,
        title=dict(text=f"{parameter} Distribution by Region",
                   font=dict(size=13, color="#e0eaf5")),
        yaxis=dict(**AXIS_STYLE,
                   title=dict(text=f"{parameter} ({param_unit})", font=dict(size=12)),
                   range=[y_min2 - y_pad2, y_max2 + y_pad2]),
        xaxis=dict(**AXIS_STYLE),
        violingap=0.25, violinmode="overlay",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    yearly = (
        filtered_df.groupby(["year", "region"])[param_col]
        .mean().reset_index().rename(columns={param_col: "mean_val"})
    )
    y_min3, y_max3 = yearly["mean_val"].min(), yearly["mean_val"].max()
    spread = y_max3 - y_min3
    y_pad3 = max(spread * 2.0, 0.5 if param_col == "salinity" else 1.5)

    fig3 = go.Figure()
    for region in sorted(yearly["region"].unique()):
        rdf = yearly[yearly["region"] == region].sort_values("year")
        col_hex = COLORS.get(region, "#aaa")
        fig3.add_trace(go.Scatter(
            x=rdf["year"], y=rdf["mean_val"],
            mode="lines+markers", name=region,
            line=dict(color=col_hex, width=2.5),
            marker=dict(size=5, color=col_hex,
                        line=dict(color="#0f1923", width=1)),
            hovertemplate=(
                f"<b>{region}</b><br>Year: %{{x}}<br>"
                f"{parameter}: %{{y:.3f}} {param_unit}<extra></extra>"
            ),
        ))
    fig3.update_layout(
        **CHART_BASE, height=380,
        title=dict(text=f"Mean {parameter} per Year",
                   font=dict(size=13, color="#e0eaf5")),
        xaxis=dict(**AXIS_STYLE, title=dict(text="Year", font=dict(size=12))),
        yaxis=dict(**AXIS_STYLE,
                   title=dict(text=f"Mean {parameter} ({param_unit})", font=dict(size=12)),
                   range=[y_min3 - y_pad3, y_max3 + y_pad3]),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.subheader("📋 Data Table")
st.caption(f"Showing {min(500, len(filtered_df)):,} of {len(filtered_df):,} records")
st.dataframe(
    filtered_df[["id", "profile_id", "float_id", "time", "latitude",
                 "longitude", "depth", "temperature", "salinity", "region"]]
    .head(500).reset_index(drop=True),
    use_container_width=True, height=300,
)
