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
    "ow_temp", "ow_humidity", "ow_pressure", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_pm10", "ow_pm2_5", "ow_co", "ow_no2", "ow_so2", "ow_o3",
    "hour", "day", "month", "weekday", "temp", "humidity", "pressure",
    "wind_speed", "aqi", "pm2_5", "pm10", "no2", "so2", "co"
]


def get_live_weather(city=CITY):
    """Fetch real-time weather from OpenWeather API"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url).json()
    return {
        "ow_temp": r["main"]["temp"],
        "ow_humidity": r["main"]["humidity"],
        "ow_pressure": r["main"]["pressure"],
        "ow_wind_speed": r["wind"]["speed"],
        "ow_wind_deg": r["wind"]["deg"],
        "ow_clouds": r["clouds"]["all"],
        "ow_pm10": 0.0,
        "ow_pm2_5": 0.0,
        "ow_co": 0.0,
        "ow_no2": 0.0,
        "ow_so2": 0.0,
        "ow_o3": 0.0
    }


def get_live_aqi(city=CITY):
    """Fetch real-time AQI data from AQICN API"""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    r = requests.get(url).json()
    if r.get("status") == "ok":
        data = r["data"]
        iaqi = data.get("iaqi", {})
        return {
            "aqi": data["aqi"],
            "pm2_5": iaqi.get("pm25", {}).get("v", 0.0),
            "pm10": iaqi.get("pm10", {}).get("v", 0.0),
            "no2": iaqi.get("no2", {}).get("v", 0.0),
            "so2": iaqi.get("so2", {}).get("v", 0.0),
            "co": iaqi.get("co", {}).get("v", 0.0)
        }
    else:
        log.warning(f"Could not fetch live AQI, fallback to zeros. Status: {r.get('status')}")
        return {"aqi": 0.0, "pm2_5": 0.0, "pm10": 0.0, "no2": 0.0, "so2": 0.0, "co": 0.0}


def get_artifact_files(model_dir):
    paths = {}
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name in ("model.joblib", "scaler.joblib"):
                paths[name] = os.path.join(root, name)
    return paths


def load_model_and_scaler(model_registry, model_name="rf_aqi_model"):
    """Load latest model and scaler from Model Registry"""
    log.info(f"Fetching latest version of model '{model_name}' from Model Registry...")
    models = model_registry.get_models(model_name)
    latest_model = max(models, key=lambda m: m.version)  # get latest version
    model_dir = latest_model.download()

    files = get_artifact_files(model_dir)
    model = joblib.load(files["model.joblib"])
    scaler = joblib.load(files["scaler.joblib"])

    version = latest_model.version
    log.info(f"Model '{model_name}' (version {version}) loaded successfully.")
    return model, scaler, version


def main():
    log.info("Starting AQI forecast generation...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    weather = get_live_weather()
    aqi_data = get_live_aqi()

    features = {**weather, **aqi_data}
    now = datetime.utcnow()
    features.update({
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),
        "temp": weather["ow_temp"],
        "humidity": weather["ow_humidity"],
        "pressure": weather["ow_pressure"],
        "wind_speed": weather["ow_wind_speed"],
    })
    latest = pd.DataFrame([features])

    model, scaler, version = load_model_and_scaler(mr)

    # Real time AQI 
    forecasts = [{
        "forecast_day": 0,
        "predicted_aqi": features["aqi"],
        "predicted_for_utc": now.isoformat(sep=" "),
        "model_version": version,
        "note": "Real-time AQI"
    }]

    # Predict next 3 days
    for i in range(1, 4):
        future_date = now + timedelta(days=i)
        latest["hour"] = future_date.hour
        latest["day"] = future_date.day
        latest["month"] = future_date.month
        latest["weekday"] = future_date.weekday()

        X = latest[FEATURE_COLS].astype(float)
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])

        forecasts.append({
            "forecast_day": i,
            "predicted_aqi": pred,
            "predicted_for_utc": future_date.isoformat(sep=" "),
            "model_version": version,
            "note": "Predicted AQI"
        })

    forecast_df = pd.DataFrame(forecasts)
    os.makedirs("data/predictions", exist_ok=True)
    forecast_df.to_csv("data/predictions/latest_predictions.csv", index=False)

    log.info("AQI forecasts (real-time + next 3 days) generated successfully!")
    print(forecast_df)


if __name__ == "__main__":
    main()
