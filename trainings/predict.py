#!/usr/bin/env python3
import os
import joblib
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import hopsworks
from dotenv import load_dotenv
import pytz
import json

# -------------------------------
# Setup
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(Y-%m-%d %H:%M:%S - %(levelname)s - %(message)s")
log = logging.getLogger("predict")

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = "Karachi"
TIMEZONE = pytz.timezone("Asia/Karachi")

FEATURE_COLS = [
    "ow_temp", "ow_humidity", "ow_pressure", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_pm10", "ow_pm2_5", "ow_co", "ow_no2", "ow_so2", "ow_o3",
    "hour", "day", "month", "weekday", "temp", "humidity", "pressure",
    "wind_speed", "aqi", "pm2_5", "pm10", "no2", "so2", "co"
]

LOCAL_MODEL_DIR = os.path.join("models", "rf_aqi_model")


# -------------------------------
# Helper Functions
# -------------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def get_live_weather(city=CITY):
    """Fetch current weather from OpenWeather (current API)"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10).json()
        return {
            "ow_temp": safe_float(r["main"]["temp"]),
            "ow_humidity": safe_float(r["main"]["humidity"]),
            "ow_pressure": safe_float(r["main"]["pressure"]),
            "ow_wind_speed": safe_float(r.get("wind", {}).get("speed", np.nan)),
            "ow_wind_deg": safe_float(r.get("wind", {}).get("deg", 0.0)),
            "ow_clouds": safe_float(r.get("clouds", {}).get("all", 0.0))
        }
    except Exception as e:
        log.warning(f"OpenWeather current fetch failed: {e}")
        return {
            "ow_temp": 30.0, "ow_humidity": 60.0, "ow_pressure": 1010.0,
            "ow_wind_speed": 3.0, "ow_wind_deg": 180.0, "ow_clouds": 25.0
        }


def get_live_aqi(city=CITY):
    """Fetch live AQI from AQICN"""
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
        r = requests.get(url, timeout=10).json()
        if r.get("status") == "ok":
            data = r["data"]
            iaqi = data.get("iaqi", {}) or {}
            return {
                "aqi": safe_float(data.get("aqi")),
                "pm2_5": safe_float(iaqi.get("pm25", {}).get("v")),
                "pm10": safe_float(iaqi.get("pm10", {}).get("v")),
                "no2": safe_float(iaqi.get("no2", {}).get("v")),
                "so2": safe_float(iaqi.get("so2", {}).get("v")),
                "co": safe_float(iaqi.get("co", {}).get("v")),
            }
    except Exception as e:
        log.warning(f"AQICN fetch failed: {e}")
    return {"aqi": 100.0, "pm2_5": 50.0, "pm10": 60.0, "no2": 30.0, "so2": 15.0, "co": 0.5}


def get_openweather_forecast(city=CITY, days=3):
    """Fetch 3-hourly forecast from OpenWeather and aggregate to daily averages for next `days` days.
    Returns list of dicts with keys matching OW features (ow_temp, ow_humidity, ...).
    If the forecast fails, returns empty list.
    """
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10).json()
        if "list" not in r:
            log.warning("OpenWeather forecast response missing 'list'")
            return []

        df_list = []
        for item in r["list"]:
            ts = datetime.utcfromtimestamp(item["dt"]).replace(tzinfo=pytz.utc)
            df_list.append({
                "ts": ts,
                "temp": safe_float(item.get("main", {}).get("temp")),
                "humidity": safe_float(item.get("main", {}).get("humidity")),
                "pressure": safe_float(item.get("main", {}).get("pressure")),
                "wind_speed": safe_float(item.get("wind", {}).get("speed")),
                "wind_deg": safe_float(item.get("wind", {}).get("deg", 0)),
                "clouds": safe_float(item.get("clouds", {}).get("all", 0))
            })
        dff = pd.DataFrame(df_list)
        results = []
        today = datetime.utcnow().date()
        for day_offset in range(1, days + 1):
            target = today + timedelta(days=day_offset)
            day_mask = dff["ts"].dt.date == target
            if day_mask.any():
                part = dff.loc[day_mask]
                results.append({
                    "ow_temp": part["temp"].mean(),
                    "ow_humidity": part["humidity"].mean(),
                    "ow_pressure": part["pressure"].mean(),
                    "ow_wind_speed": part["wind_speed"].mean(),
                    "ow_wind_deg": part["wind_deg"].mean(),
                    "ow_clouds": part["clouds"].mean(),
                })
            else:
                # no data for that day -> skip
                results.append({})
        return results
    except Exception as e:
        log.warning(f"OpenWeather forecast fetch failed: {e}")
        return []


def get_artifact_files(model_dir):
    """Find model + scaler in local or registry dir"""
    paths = {}
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name in ("model.joblib", "scaler.joblib"):
                paths[name] = os.path.join(root, name)
    return paths


def load_local_model_and_scaler(local_dir=LOCAL_MODEL_DIR):
    """Load model + scaler from local artifacts if present"""
    try:
        files = get_artifact_files(local_dir)
        if "model.joblib" in files and "scaler.joblib" in files:
            model = joblib.load(files["model.joblib"])
            scaler = joblib.load(files["scaler.joblib"])
            log.info(f"Loaded local model artifacts from '{local_dir}'")
            return model, scaler, "local"
    except Exception as e:
        log.warning(f"Loading local artifacts failed: {e}")
    return None, None, None


def load_model_and_scaler(model_registry, model_name="rf_aqi_model"):
    """Load latest trained model and scaler from Hopsworks registry OR local, else dummy."""
    if model_registry:
        try:
            log.info(f"Fetching latest model '{model_name}' from registry...")
            models = model_registry.get_models(model_name)
            if models:
                latest_model = max(models, key=lambda m: m.version)
                model_dir = latest_model.download()
                files = get_artifact_files(model_dir)
                model = joblib.load(files["model.joblib"])
                scaler = joblib.load(files["scaler.joblib"])
                log.info(f"Loaded model version {latest_model.version} from registry")
                return model, scaler, latest_model.version
            else:
                log.warning(f"No models found for '{model_name}' in registry.")
        except Exception as e:
            log.warning(f"Registry load failed: {e}")

    model, scaler, ver = load_local_model_and_scaler()
    if model is not None:
        return model, scaler, ver

    log.warning("Using dummy model/scaler — predictions will be placeholders")
    class DummyModel:
        def predict(self, X): return np.array([120.0 for _ in range(len(X))])
    class DummyScaler:
        def transform(self, X): return np.asarray(X, dtype=float)
    return DummyModel(), DummyScaler(), 0


def prepare_features_for_model(df_row, scaler, feature_cols, hist_means=None):
    """
    Ensure all columns exist, fill NaNs, and scale features.
    df_row: DataFrame (single-row) or dict-like DataFrame
    """
    df = df_row.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_cols]

    if hist_means is not None:
        fill_vals = hist_means.reindex(feature_cols).fillna(0.0).to_dict()
    else:
        fill_vals = {c: 0.0 for c in feature_cols}

    df = df.fillna(value=fill_vals).fillna(0.0)

    X = df.astype(float).to_numpy()
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        log.warning(f"Scaler.transform failed, returning unscaled features: {e}")
        X_scaled = X.astype(float)
    return X_scaled



def main():
    log.info("Starting AQI forecast pipeline...")

    project = fs = mr = None
    hist_means = None
    try:
        if HOPSWORKS_API_KEY:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        else:
            project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        log.info("Connected to Hopsworks project.")
    except Exception as e:
        log.warning(f"Hopsworks connection not available: {e}")
        project = fs = mr = None

    latest_row = None
    if fs:
        try:
            fg = fs.get_feature_group(name="aqi_features", version=2)
            df_hist = fg.read()
            if not df_hist.empty:
                available = [c for c in FEATURE_COLS if c in df_hist.columns]
                if available:
                    hist_means = df_hist[available].mean()
                latest_row = df_hist.tail(1).copy()
                log.info("Loaded latest historical row from aqi_features v2")
        except Exception as e:
            log.warning(f"Failed to fetch historical features: {e}")

    if latest_row is None or latest_row.empty:
        latest_row = pd.DataFrame([{col: np.nan for col in FEATURE_COLS}])

    weather = get_live_weather()
    aqi_data = get_live_aqi()

    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    now_local = now_utc.astimezone(TIMEZONE)

    for k, v in {**weather, **aqi_data}.items():
        latest_row.loc[:, k] = v

    # time features
    latest_row["hour"] = now_local.hour
    latest_row["day"] = now_local.day
    latest_row["month"] = now_local.month
    latest_row["weekday"] = now_local.weekday()
    latest_row["temp"] = weather["ow_temp"]
    latest_row["humidity"] = weather["ow_humidity"]
    latest_row["pressure"] = weather["ow_pressure"]
    latest_row["wind_speed"] = weather["ow_wind_speed"]

    try:
        meta_path = os.path.join(LOCAL_MODEL_DIR, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta_feats = meta.get("features_used")
            if isinstance(meta_feats, list) and len(meta_feats) > 0:
                FEATURE_COLS_local = meta_feats
                log.info(f"Loaded feature list from metadata.json ({len(FEATURE_COLS_local)} features).")
            else:
                FEATURE_COLS_local = FEATURE_COLS
        else:
            FEATURE_COLS_local = FEATURE_COLS
    except Exception as e:
        log.warning(f"Could not load metadata.json: {e}")
        FEATURE_COLS_local = FEATURE_COLS

    model, scaler, version = load_model_and_scaler(mr)

    # Predict Today (real-time)
    forecasts = []
    X_scaled = prepare_features_for_model(latest_row, scaler, FEATURE_COLS_local, hist_means)
    try:
        today_pred = float(model.predict(X_scaled)[0])
    except Exception as e:
        log.warning(f"Model predict failed for today: {e}")
        today_pred = np.nan
    today_pred = float(np.clip(today_pred if not np.isnan(today_pred) else 0.0, 0.0, 500.0))

    forecasts.append({
        "city": CITY,
        "predicted_for_utc": now_utc,
        "predicted_aqi": round(today_pred, 2),
        "model_version": version,
        "note": "Real-time model prediction for today",
        "created_at": now_utc
    })

    # --- Predict Next 3 Days ---
    # Get OW forecast (aggregated per day). If not available, we'll fallback to current weather.
    ow_forecasts = get_openweather_forecast(CITY, days=3)

    for i in range(1, 4):
        future_time_local = now_local + timedelta(days=i)
        future_time_utc = future_time_local.astimezone(pytz.utc)

        # start from latest_row and overwrite OW/weather-related features with forecast values
        df_f = latest_row.copy()
        df_f["hour"] = future_time_local.hour
        df_f["day"] = future_time_local.day
        df_f["month"] = future_time_local.month
        df_f["weekday"] = future_time_local.weekday()

        # apply available forecasted weather for that day (if present)
        ow_vals = ow_forecasts[i - 1] if len(ow_forecasts) >= i and ow_forecasts[i - 1] else None
        if ow_vals:
            df_f["ow_temp"] = ow_vals.get("ow_temp", df_f["ow_temp"]) 
            df_f["ow_humidity"] = ow_vals.get("ow_humidity", df_f["ow_humidity"]) 
            df_f["ow_pressure"] = ow_vals.get("ow_pressure", df_f["ow_pressure"]) 
            df_f["ow_wind_speed"] = ow_vals.get("ow_wind_speed", df_f["ow_wind_speed"]) 
            df_f["ow_wind_deg"] = ow_vals.get("ow_wind_deg", df_f["ow_wind_deg"]) 
            df_f["ow_clouds"] = ow_vals.get("ow_clouds", df_f["ow_clouds"]) 
        else:
            # no forecast data, keep current OW values (already in df_f)
            pass

        # For pollutant features that we can't forecast, leave NaN to be filled by hist_means in prepare_features
        X_future = prepare_features_for_model(df_f, scaler, FEATURE_COLS_local, hist_means)
        try:
            pred = float(model.predict(X_future)[0])
        except Exception as e:
            log.warning(f"Model predict failed for +{i} day(s): {e}")
            pred = np.nan

        if not np.isnan(pred):
            pred = round(float(np.clip(pred + np.random.uniform(-1, 1), 0, 500)), 2)
        else:
            pred = 0.0

        forecasts.append({
            "city": CITY,
            "predicted_for_utc": future_time_utc,
            "predicted_aqi": pred,
            "model_version": version,
            "note": f"Forecast for +{i} day(s)",
            "created_at": now_utc
        })

    df_forecast = pd.DataFrame(forecasts)
    log.info("Forecasts generated:\n" + str(df_forecast))

    if fs:
        try:
            # Ensure timestamp dtypes for Hopsworks
            if "predicted_for_utc" in df_forecast.columns:
                df_forecast["predicted_for_utc"] = pd.to_datetime(df_forecast["predicted_for_utc"]).dt.tz_convert('UTC')
            if "created_at" in df_forecast.columns:
                df_forecast["created_at"] = pd.to_datetime(df_forecast["created_at"]).dt.tz_convert('UTC')

            log.info("Uploading predictions to feature group 'aqi_predictions' (v2)...")
            pred_fg = fs.get_or_create_feature_group(
                name="aqi_predictions",
                version=2,
                description="Real-time + next 3-day AQI forecasts for Karachi (with predicted_for_utc)",
                primary_key=["city", "note"],
                online_enabled=True
            )

            pred_fg.insert(df_forecast, write_options={"wait_for_job": True})
            log.info("Forecasts uploaded successfully to Hopsworks (aqi_predictions v2).")
        except Exception as e:
            log.error(f"Upload failed to 'aqi_predictions': {e}")

    else:
        log.info("Hopsworks not available — skipping upload (local run).")

    log.info("AQI forecast pipeline completed successfully!")


if __name__ == "__main__":
    main()
