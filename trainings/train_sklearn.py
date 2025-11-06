import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import hopsworks
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

logger.info("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

logger.info("Loading feature group 'aqi_features' (version 2)...")
fg = fs.get_feature_group(name="aqi_features", version=2)
df = fg.read()
logger.info(f"Data loaded — shape: {df.shape}")

# -------------------------------
# Data cleaning
# -------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
df = df[(df[numeric_cols] <= 1e6).all(axis=1)]

target_col = "aqi_aqicn"
feature_cols = [c for c in numeric_cols if c != target_col]
df.dropna(subset=[target_col] + feature_cols, inplace=True)

# Log target distribution for debugging prediction scale
logger.info(f"AQI range in training data: min={df[target_col].min()}, max={df[target_col].max()}, mean={df[target_col].mean():.2f}")

X = df[feature_cols]
y = df[target_col]
logger.info(f"Using features: {feature_cols}")

# -------------------------------
# Training setup
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

logger.info("Training Random Forest model...")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_test = rf.predict(X_test)
metrics = {
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 3),
    "MAE": round(mean_absolute_error(y_test, y_pred_test), 3),
    "R2": round(r2_score(y_test, y_pred_test), 3)
}
logger.info(f"Test Set Performance: {metrics}")

# -------------------------------
# Retrain full model
# -------------------------------
logger.info("Retraining on full dataset for deployment...")
rf.fit(X_scaled, y)

MODEL_NAME = "rf_aqi_model"
MODEL_DIR = os.path.join("models", MODEL_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(rf, os.path.join(MODEL_DIR, "model.joblib"), compress=3)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"), compress=3)

metadata = {
    "model_name": MODEL_NAME,
    "trained_at": datetime.now().isoformat(),
    "features_used": feature_cols,
    "target": target_col,
    "metrics": metrics
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

logger.info(f"Artifacts saved locally at: {MODEL_DIR}")

# -------------------------------
# Register model to Hopsworks
# -------------------------------
logger.info("Uploading/updating model in Hopsworks Model Registry...")
try:
    model_meta = mr.get_model(MODEL_NAME)
    logger.info(f"Model '{MODEL_NAME}' exists. Updating files...")
    model_meta.update(MODEL_DIR)
except Exception:
    logger.info(f"Model '{MODEL_NAME}' does not exist. Creating new entry...")
    model_meta = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="Random Forest model for Karachi AQI forecasting"
    )
    model_meta.save(MODEL_DIR)

logger.info("Model successfully saved/updated in Hopsworks.")
logger.info("Training pipeline completed successfully!")

# -------------------------------
# Prediction helpers
# -------------------------------
def predict_current_aqi(model, scaler, latest_features: pd.DataFrame) -> float:
    """
    Predict the AQI for the latest features (real-time)
    """
    latest_features = latest_features.copy()
    latest_features = latest_features[feature_cols]  # ensure consistent order
    X_scaled = scaler.transform(latest_features)
    prediction = float(model.predict(X_scaled)[0])
    return prediction

def predict_next_days_aqi(model, scaler, latest_features: pd.DataFrame, days: int = 3) -> list:
    """
    Predict AQI for the next `days` using recursive strategy
    Simulates simple weather drift for dynamic forecasting
    """
    df_copy = latest_features.copy()
    preds = []

    for i in range(days):
        df_copy = df_copy[feature_cols]  # ensure same order
        X_scaled = scaler.transform(df_copy)
        pred = float(model.predict(X_scaled)[0])
        preds.append(pred)

        # Update simulated values for next day
        df_copy[target_col] = pred
        if "temp" in df_copy.columns:
            df_copy["temp"] += np.random.uniform(-1.5, 1.5)
        if "humidity" in df_copy.columns:
            df_copy["humidity"] += np.random.uniform(-2, 2)
        if "wind_speed" in df_copy.columns:
            df_copy["wind_speed"] += np.random.uniform(-0.5, 0.5)
        if "day_of_week" in df_copy.columns:
            df_copy["day_of_week"] = (df_copy["day_of_week"] + 1) % 7

    return preds

# -------------------------------
# Local debug mode
# -------------------------------
if __name__ == "__main__":
    latest_row = df.tail(1)
    current_aqi = predict_current_aqi(rf, scaler, latest_row)
    next_3_days = predict_next_days_aqi(rf, scaler, latest_row, days=3)

    logger.info(f"Current AQI prediction: {current_aqi:.2f}")
    logger.info(f"Next 3-day forecast: {next_3_days}")
