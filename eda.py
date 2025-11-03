import os
import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from joblib import load
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

OUTPUT_DIR = os.path.join("dashboard", "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")
if not HOPSWORKS_API_KEY:
    logging.error("Missing 'AQI_FORECAST_API_KEY' in environment variables.")
    exit(1)

# Connect to Hopsworks
try:
    logging.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    logging.info("Connected to Hopsworks successfully.")
except Exception as e:
    logging.error(f"Failed to connect to Hopsworks: {e}")
    exit(1)

# Load feature data
try:
    logging.info("Fetching latest feature data...")
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    df = feature_group.read()
    df.sort_values("timestamp_utc", inplace=True)
    logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
except Exception as e:
    logging.error(f"Failed to fetch feature data: {e}")
    exit(1)

# Convert timestamp column
try:
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
except Exception as e:
    logging.warning(f"Timestamp conversion issue: {e}")

# Dataset-only EDA
summary = f"""
EDA SUMMARY REPORT ({datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC)

Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {df.isnull().sum().sum()}
Numeric Columns: {len(df.select_dtypes(include='number').columns)}
Columns List:
{', '.join(df.columns)}
"""
with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w") as f:
    f.write(summary)
logging.info("EDA summary saved to eda_summary.txt")

# AQI trend
try:
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x="timestamp_utc", y="aqi_aqicn", marker="o", linewidth=1.3)
    plt.title("AQI Trend Over Time")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("AQI (AQICN)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aqi_trend.png"))
    plt.close()
    logging.info("AQI trend plot saved.")
except Exception as e:
    logging.warning(f"AQI trend plot failed: {e}")

# Correlation heatmap
try:
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    logging.info("Correlation heatmap saved.")
except Exception as e:
    logging.warning(f"Correlation heatmap failed: {e}")

# Scatter plots
weather_features = ["ow_temp", "ow_humidity", "ow_wind_speed"]
for feature in weather_features:
    if feature in df.columns:
        try:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=feature, y="aqi_aqicn", alpha=0.6)
            plt.title(f"AQI vs {feature}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"aqi_vs_{feature}.png"))
            plt.close()
            logging.info(f"Scatter plot saved: AQI vs {feature}")
        except Exception as e:
            logging.warning(f"Failed to plot AQI vs {feature}: {e}")

# Model-based EDA
try:
    model_name = "rf_aqi_model"
    model_meta = mr.get_model(model_name)
    latest_version = max([v.version for v in model_meta.get_versions()])  # latest version
    model_dir = model_meta.get_version(latest_version).download()

    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    model = load(model_path)
    scaler = load(scaler_path)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model/scaler: {e}")
    exit(1)

# Feature importance
try:
    importances = model.feature_importances_
    feature_names = [
        col for col in df.columns if col not in ["aqi_aqicn", "timestamp_utc", "city"]
    ][:len(importances)]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, orient="h")
    plt.title("Feature Importance in AQI Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    logging.info("Feature importance plot saved.")
except Exception as e:
    logging.warning(f"Skipped feature importance plot: {e}")

# Actual vs Predicted AQI
try:
    from sklearn.metrics import r2_score, mean_absolute_error

    features = df[feature_names]
    labels = df["aqi_aqicn"]

    X_scaled = scaler.transform(features)
    preds = model.predict(X_scaled)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(labels)), y=labels, label="Actual AQI")
    sns.lineplot(x=range(len(preds)), y=preds, label="Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.xlabel("Time Index")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
    plt.close()

    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    logging.info(f"Model Performance — R²: {r2:.3f}, MAE: {mae:.2f}")
except Exception as e:
    logging.warning(f"Skipped actual vs predicted plot: {e}")

logging.info("EDA completed successfully! All outputs saved in 'dashboard/eda_outputs/' folder.")
