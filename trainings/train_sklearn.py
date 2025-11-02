import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

logger.info("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()


logger.info("Loading feature group 'aqi_features' (version 1)...")
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
logger.info(f"Data loaded from Hopsworks! Shape: {df.shape}")

logger.info("Cleaning and preparing data...")
df = df.dropna(subset=["aqi_aqicn"])
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]
target_col = "aqi_aqicn"

df = df.dropna(subset=feature_cols)
X = df[feature_cols]
y = df[target_col]

logger.info(f"Features shape: {X.shape}")
logger.info(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logger.info("\nRandom Forest Evaluation (test split):")
logger.info(f"RMSE: {rmse:.2f}")
logger.info(f"MAE: {mae:.2f}")
logger.info(f"R²: {r2:.2f}")


logger.info("Retraining on full dataset for deployment...")
X_scaled_full = scaler.fit_transform(X)
rf_model.fit(X_scaled_full, y)


MODEL_DIR = "models/rf_model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

logger.info(f"Model saved to {MODEL_PATH}")
logger.info(f"Scaler saved to {SCALER_PATH}")

logger.info("Uploading model + scaler to Hopsworks Model Registry...")
mr = project.get_model_registry()

try:
    existing_models = mr.get_models("rf_aqi_model")
    for m in existing_models:
        logger.info(f"Deleting old version {m.version}...")
        m.delete()
except hopsworks.client.exceptions.RestAPIError:
    logger.info("No existing versions found. Creating new model...")

model_meta = mr.python.create_model(
    name="rf_aqi_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="Random Forest model for Karachi AQI forecasting (always fresh version)"
)
model_meta.save(MODEL_DIR)

logger.info("Model successfully uploaded as a fresh version!")
logger.info("Training pipeline completed successfully.")



