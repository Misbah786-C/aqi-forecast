import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from PIL import Image

# ---------------------------------------
# Page Setup
# ---------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor Dashboard",
    page_icon="",
    layout="wide"
)

# ---------------------------------------
# Hero Header
# ---------------------------------------
st.markdown("""
<div style='text-align: center; padding: 20px 0 10px 0;'>
    <h1 style='color: #F5F5F5; font-size: 42px; font-weight: 700;'>
        Pearls AQI Predictor Dashboard
    </h1>
    <p style='color: #00CED1; font-size: 19px; font-weight: 600;'>
        “Breathe smarter, see tomorrow’s air today.”
    </p>
    <p style='color: #CCCCCC; font-size: 16px; max-width: 750px; margin: 0 auto; line-height: 1.6;'>
        Live <b>Air Quality Index (AQI)</b> updates and <b>3-day forecasts</b> for 
        <span style='color:#00CED1;'>Karachi</span> helping you plan your days and breathe a little better.
    </p>
</div>
<hr style='border: 0.5px solid #333; margin: 20px 0;'>
""", unsafe_allow_html=True)


PRED_PATH = "data/predictions/latest_predictions.csv"
EDA_DIR = "dashboard/eda_outputs"


@st.cache_data
def load_data(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

forecast_df = load_data(PRED_PATH)


def aqi_category(aqi):
    if aqi <= 50:
        return ("Good", "#43A047", "Breathe it in, Karachi’s serving rare, premium grade oxygen today", "Pleasant🌿 ")
    elif aqi <= 100:
        return ("Moderate", "#FDD835", "Decent vibes in the air, not perfect but your lungs won’t complain much", "Normal🙂")
    elif aqi <= 150:
        return ("Unhealthy (Sensitive)", "#FB8C00", "Mild chaos in the air — sensitive folks might notice, the rest won’t care", "Slight Risk⚠️")
    elif aqi <= 200:
        return ("Unhealthy", "#E53935", "Health effects possible for all. Maybe Netflix indoors instead?", "Risky🚨")
    elif aqi <= 300:
        return ("Very Unhealthy", "#8E24AA", "This air’s got texture. Mask up, hydrate, and question your life choices", "Very Dangerous☠️")
    else:
        return ("Hazardous", "#7E0023", "Airpocalypse mode: unlocked. Stay inside, love your air purifier, and pray for rain", "Extreme Danger💀")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 30% 15%, #1A2733 0%, #0A111B 100%);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4, h5 { color: #EAEAEA; }
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #333, transparent);
    margin: 25px 0;
}
.today-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    border-radius: 25px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    margin-bottom: 40px;
    border: 1px solid rgba(255,255,255,0.1);
}
.forecast-card {
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    transition: all 0.3s ease;
}
.forecast-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.45);
}
.aqi-value {
    font-size: 68px;
    font-weight: 800;
    margin: 5px 0;
}
.footer {
    text-align: center;
    color: #aaa;
    font-size: 13px;
    margin-top: 40px;
}
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #00CED1 !important;
    font-size: 16px !important;
}
.streamlit-expanderHeader:hover {
    color: #4DD0E1 !important;
}
</style>
""", unsafe_allow_html=True)

# Today AQI Card
# Today AQI Card
if not forecast_df.empty:
    forecast_df["predicted_for_utc"] = pd.to_datetime(forecast_df["predicted_for_utc"])
    today = forecast_df.iloc[0]
    aqi_val = today["predicted_aqi"]
    cat, color, comment, mood = aqi_category(aqi_val)

    # Use H2 Markdown for consistent font
    st.markdown("## ☁️ Today’s Air Quality Overview")
    st.markdown(f"""
    <div class='today-card'>
        <h2 style='color:#e0e0e0;'>Karachi — {today["predicted_for_utc"].strftime("%A, %b %d")}</h2>
        <p style='color:#bbb;'>Live AQI Measurement</p>
        <div class='aqi-value' style='color:{color};'>{int(aqi_val)} AQI</div>
        <h3 style='color:{color}; margin:0;'>{cat} — {mood}</h3>
        <p style='color:#ccc; font-size:15px;'>{comment}</p>
        <p style='color:#777; font-size:12px; margin-top:10px;'>
            Last Updated: {today["predicted_for_utc"].strftime("%Y-%m-%d %H:%M UTC")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Next 3 Days Forecast
    st.markdown("## Air Quality Forecast")
    next_days = forecast_df[forecast_df["forecast_day"] > 0]
    auto_expand = True if aqi_val > 150 else False

    with st.expander("View Next 3 Days Forecast", expanded=auto_expand):
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, row in enumerate(next_days.itertuples()):
            aqi_val = row.predicted_aqi
            cat, color, comment, mood = aqi_category(aqi_val)
            date_str = row.predicted_for_utc.strftime("%A, %b %d")

            with cols[i]:
                st.markdown(f"""
                    <div class='forecast-card' style='border-top:5px solid {color};'>
                        <h4 style='margin:0; color:#eee;'>{date_str}</h4>
                        <h2 style='color:{color}; margin:5px 0;'>{int(aqi_val)} AQI</h2>
                        <p style='color:{color}; font-weight:600; margin:0;'>{cat} — {mood}</p>
                        <p style='color:#bbb; font-size:13px;'>{comment}</p>
                    </div>
                """, unsafe_allow_html=True)

        # Collapsible AQI Trend Chart ---
        with st.expander("View AQI Forecast Trend", expanded=False):
            st.markdown("<br>", unsafe_allow_html=True)
            fig = px.line(
                forecast_df,
                x="predicted_for_utc",
                y="predicted_aqi",
                title="AQI Forecast Trend (Today + Next 3 Days)",
                markers=True,
                color_discrete_sequence=["#00CED1"]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.05)",
                font=dict(color="#fff"),
                xaxis=dict(title="Date", showgrid=True, gridcolor="#333"),
                yaxis=dict(title="Predicted AQI", showgrid=True, gridcolor="#333"),
                margin=dict(t=60, b=40, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No forecast data found. Please run your `predict.py` script first.")

# EDA Outputs
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## 🔍 Exploratory Data Analysis Insights")

if os.path.exists(EDA_DIR) and len(os.listdir(EDA_DIR)) > 0:
    image_files = sorted([f for f in os.listdir(EDA_DIR) if f.endswith('.png')])
    for img_file in image_files:
        chart_title = os.path.splitext(img_file)[0].replace("_", " ").title()
        img_path = os.path.join(EDA_DIR, img_file)
        with st.expander(f"📊 {chart_title}", expanded=False):
            st.image(Image.open(img_path), use_container_width=True, caption=chart_title, output_format="PNG")
else:
    st.info("No EDA visualizations found. Run `eda.py` to generate charts.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    Data from AQICN & OpenWeather • Models via Hopsworks • Dashboard © 2025 Pearls AQI Project
</div>
""", unsafe_allow_html=True)
