import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import hopsworks

st.set_page_config(page_title="AQI Predictor", layout="wide", initial_sidebar_state="collapsed")

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    try:
        HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
    except Exception:
        HOPSWORKS_API_KEY = ""

FG_NAME = "aqi_features"
FG_VERSION = 7

CITY_INFO = {
    "karachi":   {"label": "Karachi"},
    "lahore":    {"label": "Lahore"},
    "islamabad": {"label": "Islamabad"},
    "peshawar":  {"label": "Peshawar"},
    "quetta":    {"label": "Quetta"},
}

FEATURES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "hour",
    "day_of_week", "month", "pm2_5_lag_1h", "pm2_5_lag_24h",
    "aqi_rate_of_change", "city_id",
]

HORIZONS = ["24h", "48h", "72h"]

AQI_CATEGORIES = [
    (0, 50, "Good", "#16A34A", "#DCFCE7"),
    (51, 100, "Moderate", "#D97706", "#FEF3C7"),
    (101, 150, "Unhealthy for Sensitive Groups", "#EA580C", "#FFEDD5"),
    (151, 200, "Unhealthy", "#DC2626", "#FEE2E2"),
    (201, 300, "Very Unhealthy", "#7C3AED", "#EDE9FE"),
    (301, 10000, "Hazardous", "#881337", "#FFE4E6"),
]

def categorize_aqi(value):
    for lo, hi, label, color, bg in AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, color, bg
    return "Unknown", "#64748B", "#F1F5F9"

@st.cache_resource(show_spinner=False)
def get_project():
    return hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=HOPSWORKS_API_KEY)

@st.cache_data(ttl=3600, show_spinner=False)
def load_latest_features():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.select_all().read()
    df["datetime"] = pd.to_datetime(df["event_time"], unit="ms")
    # Stored in UTC; convert to Pakistan time (UTC+5) for display since
    # all 5 cities are in Pakistan.
    df["datetime_local"] = df["datetime"] + pd.Timedelta(hours=5)

    numeric_cols = [c for c in FEATURES if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("datetime")

@st.cache_resource(ttl=86400, show_spinner=False)
def load_city_models(city_name):
    project = get_project()
    mr = project.get_model_registry()
    models = {}
    model_metadata = {}

    for horizon in HORIZONS:
        model_name = f"karachi_aqi_predictor_{horizon}"
        try:
            best_model = mr.get_best_model(model_name, "RMSE", "min")
            model_dir = best_model.download()
            models[horizon] = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))

            metrics = {}
            metrics_json_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_json_path):
                try:
                    with open(metrics_json_path, "r") as f:
                        metrics = json.load(f)
                except Exception:
                    pass

            if not metrics:
                metrics = getattr(best_model, "training_metrics", {}) or {}

            metrics_clean = {str(k).lower(): v for k, v in metrics.items()}
            r2_val = (
                metrics_clean.get("r2_score")
                or metrics_clean.get("r2")
                or metrics_clean.get("r_squared")
                or metrics_clean.get("r_square")
            )
            rmse_val = metrics_clean.get("rmse") or metrics_clean.get("root_mean_squared_error")
            mae_val = metrics_clean.get("mae") or metrics_clean.get("mean_absolute_error")

            model_metadata[horizon] = {
                "RMSE": float(rmse_val) if rmse_val is not None else None,
                "MAE": float(mae_val) if mae_val is not None else None,
                "R2": float(r2_val) if r2_val is not None else None,
            }
        except Exception:
            models[horizon] = None
            model_metadata[horizon] = {"RMSE": None, "MAE": None, "R2": None}

    return models, model_metadata

def predict_forecast(models, latest_row):
    x = latest_row[FEATURES].to_frame().T.apply(pd.to_numeric, errors="coerce")
    res = {}
    for horizon, model in models.items():
        if model is not None:
            res[horizon] = float(model.predict(x)[0])
        else:
            res[horizon] = float(latest_row["us_aqi"])
    return res

def create_gauge(current_val, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_val,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"size": 52, "color": "#FFFFFF", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 300], "tickwidth": 1.5, "tickcolor": "#94A3B8", "tickfont": {"color": "#FFFFFF", "size": 13}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.15)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#DCFCE7"},
                    {"range": [50, 100], "color": "#FEF3C7"},
                    {"range": [100, 150], "color": "#FFEDD5"},
                    {"range": [150, 200], "color": "#FEE2E2"},
                    {"range": [200, 300], "color": "#EDE9FE"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    html, body, [class*="css"], p, span, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background-color: #0B1924 !important;
        background-image: 
            linear-gradient(rgba(11, 25, 36, 0.72), rgba(11, 25, 36, 0.72)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='800' viewBox='0 0 1000 800'%3E%3Cg fill='none' stroke='%2300E5FF' stroke-opacity='0.28'%3E%3Cpath d='M0 180 Q320 280 580 140 T1000 280' stroke-width='6' stroke='%2338BDF8' stroke-opacity='0.45'/%3E%3Cpath d='M240 0 Q280 340 180 580 T380 800' stroke-width='5' stroke='%2338BDF8' stroke-opacity='0.45'/%3E%3Cpath d='M680 0 Q620 380 780 620 T860 800' stroke-width='5.5' stroke='%2338BDF8' stroke-opacity='0.45'/%3E%3Cpath d='M0 680 Q380 560 680 720 T1000 640' stroke-width='6' stroke='%2338BDF8' stroke-opacity='0.45'/%3E%3Cpath d='M0 380 L1000 380 M0 480 L1000 480 M480 0 L480 800 M580 0 L580 800' stroke-width='2.5'/%3E%3Cpath d='M0 90 L1000 90 M0 280 L1000 280 M0 580 L1000 580 M120 0 L120 800 M340 0 L340 800 M820 0 L820 800' stroke-width='1.2' stroke-opacity='0.2'/%3E%3Cpath d='M50 50 L950 750 M950 50 L50 750' stroke-width='1.5' stroke-dasharray='8,8' stroke-opacity='0.2'/%3E%3Ccircle cx='530' cy='430' r='140' stroke-width='2.5' stroke='%2338BDF8' stroke-opacity='0.4'/%3E%3Ccircle cx='530' cy='430' r='280' stroke-width='1.5' stroke-opacity='0.25'/%3E%3C/g%3E%3Cg fill='%2338BDF8' fill-opacity='0.35' font-family='sans-serif' font-size='14' font-weight='bold'%3E%3Ctext x='490' y='425'%3ECENTRAL DISTRICT%3C/text%3E%3Ctext x='180' y='240'%3ENORTH EXPRESSWAY%3C/text%3E%3Ctext x='720' y='600'%3EEAST DIVISION%3C/text%3E%3Ctext x='210' y='710'%3ESOUTH CORRIDOR%3C/text%3E%3C/g%3E%3C/svg%3E");
        background-repeat: repeat;
        background-size: 800px 640px;
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.9);
    }
    
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: #0F172A !important;
        fill: #0F172A !important;
        font-weight: 700 !important;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 800 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .metric-val {
        font-size: 1.6rem;
        font-weight: 900 !important;
        color: #0F172A !important;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800 !important;
        font-size: 0.9rem;
    }

    .forecast-card {
        background: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .forecast-day {
        font-weight: 800 !important;
        color: #1E293B !important;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    .forecast-val {
        font-size: 2.2rem;
        font-weight: 900 !important;
        color: #0F172A !important;
        margin-bottom: 4px;
    }

    .forecast-sub {
        font-size: 0.82rem;
        color: #475569 !important;
        font-weight: 700 !important;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        margin-bottom: 8px;
    }

    .stButton > button {
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: 1.5px solid #94A3B8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stButton > button[kind="primary"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border: 1.5px solid #38BDF8 !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    if not HOPSWORKS_API_KEY:
        st.error("HOPSWORKS_API_KEY is not configured.")
        st.stop()

    try:
        df = load_latest_features()
    except Exception as e:
        st.error(f"Failed to load sensor features: {e}")
        st.stop()

    if "selected_city_label" not in st.session_state:
        st.session_state.selected_city_label = "Karachi"

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown('<div style="font-size: 2.3rem; font-weight: 900; color: #FFFFFF; line-height: 1.1; text-shadow: 0 2px 4px rgba(0,0,0,0.8); margin-bottom: 4px;">AQI Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0; text-shadow: 0 1px 2px rgba(0,0,0,0.8); margin-bottom: 18px;">Machine learning powered 3-day air quality forecasts for major Pakistan cities.</div>', unsafe_allow_html=True)
    with col_t2:
        selected_city_label = st.selectbox(
            "Select City",
            options=[info["label"] for info in CITY_INFO.values()],
            index=[info["label"] for info in CITY_INFO.values()].index(st.session_state.selected_city_label),
            key="city_select_box"
        )
        st.session_state.selected_city_label = selected_city_label
        selected_city = [k for k, v in CITY_INFO.items() if v["label"] == selected_city_label][0]

    models, metadata = load_city_models(selected_city)

    city_df = df[df["city"] == selected_city].sort_values("datetime").copy()
    if city_df.empty:
        st.warning(f"No active sensor data available for {selected_city_label}.")
        st.stop()

    latest_row = city_df.iloc[-1]
    forecast = predict_forecast(models, latest_row)
    current_aqi = float(latest_row["us_aqi"])
    current_cat, current_color, current_bg = categorize_aqi(current_aqi)

    st.markdown(f'<div class="section-title">{selected_city_label} Air Quality Overview</div>', unsafe_allow_html=True)
    h_col1, h_col2 = st.columns([1.2, 1])

    with h_col1:
        st.plotly_chart(create_gauge(current_aqi, current_color), width="stretch")

    with h_col2:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 15px; padding: 22px;">
                <div class="metric-label">Current Air Quality Status</div>
                <div style="margin: 10px 0;">
                    <span class="status-badge" style="background-color: {current_bg}; color: {current_color}; border: 1.5px solid {current_color};">
                        {current_cat}
                    </span>
                </div>
                <div class="forecast-sub">Last updated: {latest_row['datetime_local'].strftime('%b %d, %Y at %I:%M %p')} PKT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top: 20px;">Current Pollutants</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    pollutants = [
        ("PM2.5", latest_row.get("pm2_5", 0), "µg/m³", p1),
        ("PM10", latest_row.get("pm10", 0), "µg/m³", p2),
        ("O3", latest_row.get("ozone", 0), "µg/m³", p3),
        ("NO2", latest_row.get("nitrogen_dioxide", 0), "µg/m³", p4),
        ("SO2", latest_row.get("sulphur_dioxide", 0), "µg/m³", p5),
        ("CO", latest_row.get("carbon_monoxide", 0), "µg/m³", p6),
    ]

    for label, val, unit, col in pollutants:
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center; padding: 12px 6px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val" style="font-size: 1.35rem;">{float(val):.1f}</div>
                    <div class="forecast-sub">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    c_trend, c_cond = st.columns([1.6, 1])
    with c_trend:
        st.markdown('<div class="section-title">24-Hour AQI Trend</div>', unsafe_allow_html=True)
        last_24h = city_df.tail(24).copy()
        fig_24 = px.area(last_24h, x="datetime", y="us_aqi", labels={"us_aqi": "AQI", "datetime": "Time"})
        fig_24.update_traces(line_color="#1E3A8A", fillcolor="rgba(30, 58, 138, 0.15)")
        fig_24.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis=dict(tickfont=dict(color="#0F172A", size=12, family="Inter"), gridcolor="#E2E8F0"),
            yaxis=dict(tickfont=dict(color="#0F172A", size=12, family="Inter"), gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_24, width="stretch")

    with c_cond:
        st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric-card" style="padding: 10px 18px; margin-bottom: 8px;">
                <div class="metric-label">Temperature</div>
                <div class="metric-val" style="font-size: 1.3rem;">{float(latest_row.get('temperature_2m', 0)):.1f} °C</div>
            </div>
            <div class="metric-card" style="padding: 10px 18px; margin-bottom: 8px;">
                <div class="metric-label">Humidity</div>
                <div class="metric-val" style="font-size: 1.3rem;">{float(latest_row.get('relative_humidity_2m', 0)):.0f}%</div>
            </div>
            <div class="metric-card" style="padding: 10px 18px; margin-bottom: 8px;">
                <div class="metric-label">Wind Speed</div>
                <div class="metric-val" style="font-size: 1.3rem;">{float(latest_row.get('wind_speed_10m', 0)):.1f} km/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top: 15px;">3-Day Forecast</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    forecast_days = [("Day 1 (+24h)", "24h", f1), ("Day 2 (+48h)", "48h", f2), ("Day 3 (+72h)", "72h", f3)]

    for label, horizon, col in forecast_days:
        val = forecast[horizon]
        cat, col_c, bg_c = categorize_aqi(val)
        rmse_val = metadata.get(horizon, {}).get("RMSE")
        rmse_str = f"{rmse_val:.2f}" if rmse_val is not None else "Active"
        with col:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-day">{label}</div>
                    <span class="status-badge" style="background-color: {bg_c}; color: {col_c}; border: 1.5px solid {col_c};">
                        {cat}
                    </span>
                    <div class="forecast-val" style="margin-top: 8px;">{val:.1f}</div>
                    <div class="forecast-sub">Model RMSE: ±{rmse_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title" style="margin-top: 20px;">Why This Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; color: #E2E8F0; margin-bottom: 6px;">Select Forecast Horizon:</div>', unsafe_allow_html=True)

    if "selected_horizon" not in st.session_state:
        st.session_state.selected_horizon = "24h"

    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
    with btn_col1:
        if st.button("Day 1 (+24h)", type="primary" if st.session_state.selected_horizon == "24h" else "secondary", width="stretch"):
            st.session_state.selected_horizon = "24h"
            st.rerun()
    with btn_col2:
        if st.button("Day 2 (+48h)", type="primary" if st.session_state.selected_horizon == "48h" else "secondary", width="stretch"):
            st.session_state.selected_horizon = "48h"
            st.rerun()
    with btn_col3:
        if st.button("Day 3 (+72h)", type="primary" if st.session_state.selected_horizon == "72h" else "secondary", width="stretch"):
            st.session_state.selected_horizon = "72h"
            st.rerun()

    current_h = st.session_state.selected_horizon
    selected_model = models.get(current_h)

    if selected_model is not None and hasattr(selected_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": selected_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        top_feature = fi_df.iloc[-1]["Feature"]
        top_importance = fi_df.iloc[-1]["Importance"]

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Top Driving Feature</div>
                    <div class="metric-val">{top_feature.upper()}</div>
                    <div class="forecast-sub">Weight Contribution: {top_importance:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            model_r2 = metadata.get(current_h, {}).get("R2")
            r2_display = f"{model_r2:.3f}" if model_r2 is not None else "Active"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Model Evaluation (R² Confidence)</div>
                    <div class="metric-val">{r2_display}</div>
                    <div class="forecast-sub">{selected_city_label} ({current_h.upper()} Model)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Relative Feature Importance", "Feature": "Input Feature"},
        )
        fig_fi.update_traces(marker_color="#1E3A8A")
        fig_fi.update_layout(
            height=420,
            margin=dict(l=140, r=20, t=10, b=40),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis=dict(
                tickfont=dict(color="#0F172A", size=12, family="Inter"),
                title=dict(font=dict(color="#0F172A", size=13, family="Inter")),
                gridcolor="#E2E8F0",
                showline=True,
                linecolor="#94A3B8"
            ),
            yaxis=dict(
                tickfont=dict(color="#0F172A", size=12, family="Inter"),
                title=dict(font=dict(color="#0F172A", size=13, family="Inter")),
                showline=True,
                linecolor="#94A3B8"
            ),
        )
        st.plotly_chart(fig_fi, width="stretch")

    st.markdown('<div style="font-size: 1.6rem; font-weight: 900; color: #FFFFFF; text-shadow: 0 2px 4px rgba(0,0,0,0.8); margin-top: 25px; margin-bottom: 12px;">Data Visualization and EDA</div>', unsafe_allow_html=True)
    
    if "eda_tab" not in st.session_state:
        st.session_state.eda_tab = "City Trends & Distributions"

    tab_c1, tab_c2, tab_c3, _ = st.columns([1.5, 1.3, 1.4, 1.5])
    with tab_c1:
        if st.button("City Trends & Distributions", type="primary" if st.session_state.eda_tab == "City Trends & Distributions" else "secondary", width="stretch"):
            st.session_state.eda_tab = "City Trends & Distributions"
            st.rerun()
    with tab_c2:
        if st.button("Temporal Analysis", type="primary" if st.session_state.eda_tab == "Temporal Analysis" else "secondary", width="stretch"):
            st.session_state.eda_tab = "Temporal Analysis"
            st.rerun()
    with tab_c3:
        if st.button("Pollutant Correlation", type="primary" if st.session_state.eda_tab == "Pollutant Correlation" else "secondary", width="stretch"):
            st.session_state.eda_tab = "Pollutant Correlation"
            st.rerun()

    st.write("")

    if st.session_state.eda_tab == "City Trends & Distributions":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<div class="section-title">AQI Trend Over Time</div>', unsafe_allow_html=True)
            fig_timeline = px.line(df, x="datetime", y="us_aqi", color="city")
            fig_timeline.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_timeline, width="stretch")

        with col_t2:
            st.markdown('<div class="section-title">PM2.5 Distribution</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(df, x="pm2_5", color="city", opacity=0.6, barmode="overlay")
            fig_hist.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_hist, width="stretch")

    elif st.session_state.eda_tab == "Temporal Analysis":
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown('<div class="section-title">Hourly Diurnal Cycle</div>', unsafe_allow_html=True)
            hourly = df.groupby(["hour", "city"])["us_aqi"].mean().reset_index()
            fig_h = px.line(hourly, x="hour", y="us_aqi", color="city", markers=True)
            fig_h.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_h, width="stretch")

        with col_h2:
            st.markdown('<div class="section-title">Monthly Seasonal Pattern</div>', unsafe_allow_html=True)
            monthly = df.groupby(["month", "city"])["us_aqi"].mean().reset_index()
            fig_m = px.line(monthly, x="month", y="us_aqi", color="city", markers=True)
            fig_m.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_m, width="stretch")

    elif st.session_state.eda_tab == "Pollutant Correlation":
        col_c1, col_c2 = st.columns(2)
        city_corr_df = city_df[[c for c in ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"] if c in city_df.columns]]
        corr_matrix = city_corr_df.corr()

        with col_c1:
            st.markdown(f'<div class="section-title">{selected_city_label} Correlation with AQI</div>', unsafe_allow_html=True)
            aqi_corr = corr_matrix["us_aqi"].drop("us_aqi").reset_index()
            aqi_corr.columns = ["Feature", "Correlation"]
            aqi_corr = aqi_corr.sort_values("Correlation", ascending=True)

            fig_bar = px.bar(
                aqi_corr,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(
                height=420,
                margin=dict(l=140, r=20, t=10, b=40),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
                yaxis=dict(tickfont=dict(color="#0F172A", size=12)),
                font=dict(color="#0F172A"),
            )
            st.plotly_chart(fig_bar, width="stretch")

        with col_c2:
            st.markdown(f'<div class="section-title">{selected_city_label} Correlation Matrix</div>', unsafe_allow_html=True)
            fig_cm = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(
                height=420,
                margin=dict(l=140, r=20, t=10, b=100),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(tickfont=dict(color="#0F172A", size=10), tickangle=-45),
                yaxis=dict(tickfont=dict(color="#0F172A", size=11)),
                font=dict(color="#0F172A"),
            )
            st.plotly_chart(fig_cm, width="stretch")

if __name__ == "__main__":
    main()
