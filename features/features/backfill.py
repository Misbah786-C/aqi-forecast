import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
import hopsworks
import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

CITY = os.environ.get("CITY", "Karachi")
LAT = float(os.environ.get("LAT", 24.8607))
LON = float(os.environ.get("LON", 67.0011))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "features", "training_dataset.csv")

# Key check
if not OPENWEATHER_API_KEY or not AQICN_TOKEN or not HOPSWORKS_API_KEY:
    raise ValueError("Required secrets are not set in environment variables.")


def fetch_current_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Weather fetch failed: {e}")
        return {}

def fetch_current_aqi():
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("data", {})
    except Exception as e:
        logging.error(f"AQI fetch failed: {e}")
        return {}

def fetch_live_data():
    logging.info("Fetching real-time AQI + weather data...")
    weather = fetch_current_weather()
    aqi = fetch_current_aqi()

    if not weather or not aqi:
        logging.warning("Missing live data, skipping real-time record.")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    main = weather.get("main", {})
    wind = weather.get("wind", {})
    clouds = weather.get("clouds", {})
    iaqi = aqi.get("iaqi", {})

    row = {
        "timestamp_utc": now,
        "ow_temp": main.get("temp"),
        "ow_pressure": main.get("pressure"),
        "ow_humidity": main.get("humidity"),
        "ow_wind_speed": wind.get("speed"),
        "ow_wind_deg": wind.get("deg"),
        "ow_clouds": clouds.get("all"),
        "ow_co": iaqi.get("co", {}).get("v"),
        "ow_no": iaqi.get("no", {}).get("v"),
        "ow_no2": iaqi.get("no2", {}).get("v"),
        "ow_o3": iaqi.get("o3", {}).get("v"),
        "ow_so2": iaqi.get("so2", {}).get("v"),
        "ow_pm2_5": iaqi.get("pm25", {}).get("v"),
        "ow_pm10": iaqi.get("pm10", {}).get("v"),
        "ow_nh3": iaqi.get("nh3", {}).get("v"),
        "aqi_aqicn": aqi.get("aqi"),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),
    }

    df = pd.DataFrame([row])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    logging.info("Live data fetched successfully!")
    return df


def load_training_dataset():
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training dataset not found at {TRAIN_DATA_PATH}")

    df = pd.read_csv(TRAIN_DATA_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    logging.info(f"Loaded training dataset with {len(df)} rows.")
    return df


def backfill():
    logging.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="aqi_features", version=1)
    df_existing = fg.read()
    logging.info(f"Loaded existing feature group with {len(df_existing)} rows.")

    df_train = load_training_dataset()
    df_live = fetch_live_data()

    df_combined = pd.concat([df_existing, df_train, df_live], ignore_index=True)
    df_combined["timestamp_utc"] = pd.to_datetime(df_combined["timestamp_utc"], utc=True, errors="coerce")
    df_combined = df_combined.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df_combined = df_combined.sort_values("timestamp_utc")

    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    df_combined[numeric_cols] = df_combined[numeric_cols].ffill()
    df_combined[numeric_cols] = df_combined[numeric_cols].fillna(0)

    logging.info("Cleaned data with forward fill applied.")
    logging.info(f"Final dataset shape: {df_combined.shape}")

    for attempt in range(1, 4):
        try:
            logging.info(f"Attempt {attempt}/3: Uploading to Hopsworks...")
            fg.insert(df_combined, write_options={"wait_for_job": True})
            logging.info("Successfully updated existing feature group!")
            break
        except Exception as e:
            logging.warning(f"Upload failed (attempt {attempt}): {e}")
            if attempt < 3:
                time.sleep(8)
            else:
                raise

if __name__ == "__main__":
    logging.info(f"Running backfill at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    backfill()
    logging.info("Backfill complete.")
