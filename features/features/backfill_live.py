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
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2

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


def safe_float(val):
    """Convert to float safely, handling '-', None, or NaN."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def map_openweather_aqi(val):
    """Map OpenWeather AQI 1–5 scale to approximate AQICN-like value."""
    if pd.isna(val):
        return np.nan
    mapping = {1: 20, 2: 60, 3: 100, 4: 150, 5: 200}
    return mapping.get(int(val), np.nan)



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
            "ow_temp": safe_float(weather_data.get("main", {}).get("temp")),
            "ow_humidity": safe_float(weather_data.get("main", {}).get("humidity")),
            "ow_pressure": safe_float(weather_data.get("main", {}).get("pressure")),
            "ow_wind_speed": safe_float(weather_data.get("wind", {}).get("speed")),
            "ow_wind_deg": safe_float(weather_data.get("wind", {}).get("deg")),
            "ow_clouds": safe_float(weather_data.get("clouds", {}).get("all")),
            "ow_pm10": safe_float(components.get("pm10")),
            "ow_pm2_5": safe_float(components.get("pm2_5")),
            "ow_co": safe_float(components.get("co")),
            "ow_no2": safe_float(components.get("no2")),
            "ow_so2": safe_float(components.get("so2")),
            "ow_o3": safe_float(components.get("o3")),
            "aqi_aqicn": map_openweather_aqi(
                safe_float(air_data.get("list", [{}])[0].get("main", {}).get("aqi"))
            ),
        }
    except RequestException as e:
        logging.warning(f"OpenWeather fetch failed: {e}")
        return {col: np.nan for col in EXPECTED_COLS if col.startswith("ow_") or col == "aqi_aqicn"}


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
            "aqi": safe_float(data["data"].get("aqi")),
            "pm2_5": safe_float(iaqi.get("pm25", {}).get("v")),
            "pm10": safe_float(iaqi.get("pm10", {}).get("v")),
            "no2": safe_float(iaqi.get("no2", {}).get("v")),
            "so2": safe_float(iaqi.get("so2", {}).get("v")),
            "co": safe_float(iaqi.get("co", {}).get("v")),
            "o3": safe_float(iaqi.get("o3", {}).get("v")),
        }

    except RequestException as e:
        logging.warning(f"AQICN fetch failed: {e}")
        return {}


def fetch_openmeteo():
    """Fetch fallback AQI from Open-Meteo."""
    try:
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={LAT}&longitude={LON}&hourly=us_aqi,pm10,pm2_5,"
            f"carbon_monoxide,nitrogen_dioxide,sulfur_dioxide,ozone"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        if not data or "us_aqi" not in data:
            logging.warning("Open-Meteo returned no usable AQI data.")
            return {}

        last_idx = -1
        return {
            "aqi": safe_float(data["us_aqi"][last_idx]),
            "pm2_5": safe_float(data.get("pm2_5", [np.nan])[last_idx]),
            "pm10": safe_float(data.get("pm10", [np.nan])[last_idx]),
            "co": safe_float(data.get("carbon_monoxide", [np.nan])[last_idx]),
            "no2": safe_float(data.get("nitrogen_dioxide", [np.nan])[last_idx]),
            "so2": safe_float(data.get("sulfur_dioxide", [np.nan])[last_idx]),
            "o3": safe_float(data.get("ozone", [np.nan])[last_idx]),
        }

    except RequestException as e:
        logging.warning(f"Open-Meteo fetch failed: {e}")
        return {}


def fetch_and_upload():
    """Fetch, clean, forward-fill, and upload live AQI + weather data to Hopsworks."""
    ow = fetch_openweather()
    aqicn = fetch_aqicn()

    # Use Open-Meteo fallback if AQICN is invalid
    if not aqicn or pd.isna(aqicn.get("aqi")) or aqicn.get("aqi") in [None, 0, np.nan]:
        logging.warning("AQICN returned invalid AQI — trying Open-Meteo fallback.")
        meteo = fetch_openmeteo()
        if meteo:
            aqicn.update({k: v for k, v in meteo.items() if pd.notna(v)})
            logging.info(f"Open-Meteo fallback AQI used: {aqicn.get('aqi')}")

    # Fill any missing pollutant values with OpenWeather values
    for pollutant in ["pm2_5", "pm10", "no2", "so2", "co", "o3"]:
        if pollutant not in aqicn or pd.isna(aqicn.get(pollutant)):
            aqicn[pollutant] = ow.get(f"ow_{pollutant}", np.nan)

    # Final fallback for AQI
    if pd.isna(aqicn.get("aqi")) or aqicn.get("aqi") is None:
        aqicn["aqi"] = ow.get("aqi_aqicn", np.nan)
        logging.info(f"Final fallback AQI used from OpenWeather: {aqicn['aqi']}")

    merged = {**ow, **aqicn, "city": CITY}
    merged["timestamp_utc"] = pd.Timestamp.now(timezone.utc).floor("H")

    now_dt = merged["timestamp_utc"]
    merged.update({
        "hour": now_dt.hour,
        "day": now_dt.day,
        "month": now_dt.month,
        "weekday": now_dt.weekday()
    })

    df_new = pd.DataFrame([merged])

    # Ensure all expected columns exist
    for col in EXPECTED_COLS:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[EXPECTED_COLS]
    df_new["timestamp_utc"] = pd.to_datetime(df_new["timestamp_utc"], utc=True)

    # Ensure numeric columns are floats
    for col in df_new.columns:
        if col not in ["city", "timestamp_utc"]:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype(float)

    logging.info(f"Prepared new record → Columns: {len(df_new.columns)}, Rows: {len(df_new)}")
    logging.info(df_new.to_string(index=False))

    # ──────────────────────────────
    # Upload to Hopsworks with forward/backward fill
    # ──────────────────────────────
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp_utc"],
            description="Air quality and weather merged dataset (v2, all float types)",
            online_enabled=False,
        )

        # Read existing data
        existing = fg.read().sort_values("timestamp_utc")
        if not existing.empty:
            # Append new row
            combined = pd.concat([existing, df_new], ignore_index=True)
            # Forward/backward fill numeric columns
            num_cols = combined.select_dtypes(include=[np.number]).columns
            combined[num_cols] = combined[num_cols].ffill().bfill()
            # Only keep the new timestamp for insertion
            df_new = combined[combined["timestamp_utc"] == merged["timestamp_utc"]]
        else:
            num_cols = df_new.select_dtypes(include=[np.number]).columns
            df_new[num_cols] = df_new[num_cols].ffill().bfill()

        # Drop plain 'o3' before upload
        if "o3" in df_new.columns:
            df_new = df_new.drop(columns=["o3"])

        # Avoid duplicates
        latest_ts = existing["timestamp_utc"].max() if not existing.empty else None
        new_ts = df_new["timestamp_utc"].iloc[0]
        if latest_ts is not None and new_ts <= latest_ts:
            logging.info(f"Skipping insert — data for {new_ts} already exists in {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION}).")
            return

        logging.info("Uploading columns: %s", list(df_new.columns))
        fg.insert(df_new, write_options={"wait_for_job": False})
        logging.info(f"✅ Inserted new AQI record for {new_ts} into {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION}).")

    except Exception as e:
        logging.warning(f"Upload to Hopsworks failed: {e}")



if __name__ == "__main__":
    logging.info("Starting live AQI + weather fetch and upload...")
    try:
        fetch_and_upload()
    except Exception as e:
        logging.error(f"Error during execution: {e}")
