import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
import hopsworks
import numpy as np
from requests.exceptions import RequestException

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY = os.getenv("CITY", "Karachi")
LAT = os.getenv("LAT", "24.8607")
LON = os.getenv("LON", "67.0011")
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

if not OPENWEATHER_KEY or not AQICN_TOKEN or not HOPSWORKS_API_KEY:
    raise ValueError("Missing required API keys in .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXPECTED_COLS = [
    "city", "timestamp_utc",
    "ow_temp", "ow_humidity", "ow_pressure", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_pm10", "ow_pm2_5", "ow_co", "ow_no2", "ow_so2", "ow_o3",
    "aqi_aqicn",
    "hour", "day", "month", "weekday",
    "temp", "humidity", "pressure", "wind_speed",
    "aqi", "pm2_5", "pm10", "no2", "so2", "co"
]

def fetch_openweather():
    """Fetch weather + pollution data from OpenWeather."""
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
        air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"

        weather = requests.get(weather_url, timeout=10)
        air = requests.get(air_url, timeout=10)
        weather.raise_for_status()
        air.raise_for_status()

        weather_data = weather.json()
        air_data = air.json()
        components = air_data.get("list", [{}])[0].get("components", {})

        return {
            "ow_temp": weather_data.get("main", {}).get("temp"),
            "ow_humidity": weather_data.get("main", {}).get("humidity"),
            "ow_pressure": weather_data.get("main", {}).get("pressure"),
            "ow_wind_speed": weather_data.get("wind", {}).get("speed"),
            "ow_wind_deg": weather_data.get("wind", {}).get("deg"),
            "ow_clouds": weather_data.get("clouds", {}).get("all"),
            "ow_pm10": components.get("pm10"),
            "ow_pm2_5": components.get("pm2_5"),
            "ow_co": components.get("co"),
            "ow_no2": components.get("no2"),
            "ow_so2": components.get("so2"),
            "ow_o3": components.get("o3"),
            "aqi_aqicn": air_data.get("list", [{}])[0].get("main", {}).get("aqi"),
        }

    except RequestException as e:
        logging.warning(f"🌩️ OpenWeather fetch failed: {e}")
        return {col: None for col in EXPECTED_COLS if col.startswith("ow_") or col == "aqi_aqicn"}


def fetch_aqicn():
    """Fetch live AQI data from AQICN."""
    try:
        url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logging.warning(f"AQICN returned non-ok status: {data}")
            return {}

        iaqi = data["data"].get("iaqi", {})
        return {
            "aqi": data["data"].get("aqi"),
            "pm2_5": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
        }

    except RequestException as e:
        logging.warning(f"🌫️ AQICN fetch failed: {e}")
        return {}


def fetch_and_upload():
    ow = fetch_openweather()
    aqicn = fetch_aqicn()

    for pollutant in ["pm2_5", "pm10", "no2", "so2", "co", "aqi"]:
        if aqicn.get(pollutant) is None:
            aqicn[pollutant] = ow.get(f"ow_{pollutant}")

    merged = {**ow, **aqicn, "city": CITY}

    merged["timestamp_utc"] = pd.Timestamp(datetime.now(timezone.utc))

    now_dt = merged["timestamp_utc"]
    merged["hour"] = now_dt.hour
    merged["day"] = now_dt.day
    merged["month"] = now_dt.month
    merged["weekday"] = now_dt.weekday()

    df_new = pd.DataFrame([merged])

    mapping = {
        "ow_temp": "temp",
        "ow_humidity": "humidity",
        "ow_pressure": "pressure",
        "ow_wind_speed": "wind_speed",
        "ow_pm2_5": "pm2_5",
        "ow_pm10": "pm10",
        "ow_no2": "no2",
        "ow_so2": "so2",
        "ow_co": "co",
        "aqi_aqicn": "aqi"
    }
    for src, dest in mapping.items():
        if dest not in df_new or df_new[dest].isna().all():
            df_new[dest] = df_new[src]

    for col in EXPECTED_COLS:
        if col not in df_new.columns:
            df_new[col] = np.nan

    df_new = df_new[EXPECTED_COLS]

    for col in df_new.columns:
        if col not in ["city", "timestamp_utc"]:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype(float)

    df_new["timestamp_utc"] = pd.to_datetime(df_new["timestamp_utc"], utc=True)

    logging.info(f"Fetched new data. Columns: {len(df_new.columns)}, Rows: {len(df_new)}")

    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features", version=1)
        fg.insert(df_new, write_options={"wait_for_job": False})
        logging.info("☁️ Successfully uploaded record to Hopsworks Feature Store.")
    except Exception as e:
        logging.warning(f"Upload to Hopsworks failed: {e}")


if __name__ == "__main__":
    logging.info("Starting single AQI + weather fetch and Hopsworks upload...")
    try:
        fetch_and_upload()
        logging.info("Successfully uploaded record to Hopsworks Feature Store. Stopping execution.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
