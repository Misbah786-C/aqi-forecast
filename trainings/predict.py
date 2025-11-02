import os
import joblib
import logging
from datetime import datetime, timedelta
import pandas as pd
import requests
import hopsworks
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict")

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = "Karachi"

FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

def get_live_weather(city=CITY):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url).json()
    return {
        "ow_temp": r["main"]["temp"],
        "ow_pressure": r["main"]["pressure"],
        "ow_humidity": r["main"]["humidity"],
        "ow_wind_speed": r["wind"]["speed"],
        "ow_wind_deg": r["wind"]["deg"],
        "ow_clouds": r["clouds"]["all"],
        "ow_co": 0.0,
        "ow_no2": 0.0,
        "ow_pm2_5": 0.0,
        "ow_pm10": 0.0
    }

def get_live_aqi(city=CITY):
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    r = requests.get(url).json()
    if r.get("status") == "ok":
        return r["data"]["aqi"]
    else:
        log.warning(f"Could not fetch live AQI, fallback to 0. Status: {r.get('status')}")
        return 0.0

def get_artifact_files(model_dir):
    paths = {}
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name in ("model.joblib", "scaler.joblib"):
                paths[name] = os.path.join(root, name)
    return paths

def load_model_and_scaler(model_registry, model_name="rf_aqi_model"):
    models = model_registry.get_models(model_name)
    latest = max(models, key=lambda m: m.version)
    log.info(f"Loading model '{model_name}' (version {latest.version})...")
    model_dir = latest.download()
    files = get_artifact_files(model_dir)
    model = joblib.load(files["model.joblib"])
    scaler = joblib.load(files["scaler.joblib"])
    log.info("Model and scaler loaded successfully.")
    return model, scaler, latest.version

def main():
    log.info("Starting AQI forecast generation...")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    # Get real-time data
    weather = get_live_weather()
    aqi_now = get_live_aqi()

    # Prepare latest features row
    latest = pd.DataFrame([weather])
    now = datetime.utcnow()
    latest["hour"] = now.hour
    latest["day"] = now.day
    latest["month"] = now.month
    latest["weekday"] = now.weekday()

    model, scaler, version = load_model_and_scaler(mr)

    forecasts = []

    # Add today's real-time AQI
    forecasts.append({
        "forecast_day": 0,
        "predicted_aqi": aqi_now,
        "predicted_for_utc": now.isoformat(sep=" "),
        "model_version": version,
        "note": "real-time AQI"
    })

    # Forecast next 3 days
    current_features = latest.copy()
    for i in range(1, 4):
        future_date = now + timedelta(days=i)
        current_features["hour"] = future_date.hour
        current_features["day"] = future_date.day
        current_features["month"] = future_date.month
        current_features["weekday"] = future_date.weekday()

        X = current_features[FEATURE_COLS].astype("float64")
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])

        forecasts.append({
            "forecast_day": i,
            "predicted_aqi": pred,
            "predicted_for_utc": future_date.isoformat(sep=" "),
            "model_version": version,
            "note": "predicted AQI"
        })

    forecast_df = pd.DataFrame(forecasts)
    os.makedirs("data/predictions", exist_ok=True)
    forecast_df.to_csv("data/predictions/latest_predictions.csv", index=False)
    log.info("AQI forecasts (today + 3 days) saved successfully.")
    print(forecast_df)

if __name__ == "__main__":
    main()
