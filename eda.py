import os
import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")
if not HOPSWORKS_API_KEY:
    logging.error("Missing 'AQI_FORECAST_API_KEY' in environment variables.")
    exit(1)

try:
    logging.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    logging.info("Connected to Hopsworks successfully.")
except Exception as e:
    logging.error(f"Hopsworks connection failed: {e}")
    exit(1)

try:
    logging.info("Fetching latest feature data...")
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    df = feature_group.read()
    logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    logging.error(f"Failed to fetch feature data: {e}")
    exit(1)

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms")
except Exception as e:
    logging.warning(f"⚠ Failed to convert timestamp: {e}")

#Summary stats
summary = f"""
EDA Summary
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {df.isnull().sum().sum()}
Numeric Columns: {len(df.select_dtypes(include='number').columns)}
"""
with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w") as f:
    f.write(summary)
logging.info("Summary saved.")

#AQI Trend 
try:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="timestamp_utc", y="aqi_aqicn", marker="o")
    plt.title("AQI Over Time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aqi_trend.png"))
    plt.close()
    logging.info("AQI trend saved.")
except Exception as e:
    logging.warning(f"Failed to generate AQI trend: {e}")

#Correlation Heatmap
try:
    plt.figure(figsize=(8, 6))
    correlations = df.corr(numeric_only=True)["aqi_aqicn"].sort_values(ascending=False)
    sns.heatmap(correlations.to_frame(), annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation with AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aqi_correlation.png"))
    plt.close()
    logging.info("AQI correlation heatmap saved.")
except Exception as e:
    logging.warning(f"Failed to generate AQI correlation heatmap: {e}")

#Scatter Plots for AQI vs Weather Parameters
weather_features = ["ow_temp", "ow_humidity", "ow_wind_speed"]
for feature in weather_features:
    try:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=feature, y="aqi_aqicn")
        plt.title(f"AQI vs {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"aqi_vs_{feature}.png"))
        plt.close()
        logging.info(f"Scatter plot saved: AQI vs {feature}")
    except Exception as e:
        logging.warning(f"Failed to plot AQI vs {feature}: {e}")

#Feature Importance 
try:
    from joblib import load
    model = load("trained_model.joblib")  
    importances = model.feature_importances_
    features = df.drop(columns=["aqi_aqicn", "timestamp_utc"]).columns[:len(importances)]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance in AQI Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    logging.info("Feature importance saved.")
except Exception as e:
    logging.warning(f"Feature importance plot skipped: {e}")

#Actual vs Predicted AQI 
try:
    predictions = pd.read_csv("predictions.csv")["AQI_Predicted"]  
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(df)), y=df["aqi_aqicn"], label="Actual AQI")
    sns.lineplot(x=range(len(df)), y=predictions, label="Predicted AQI")
    plt.title("Actual vs Predicted AQI Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
    plt.close()
    logging.info("Prediction comparison saved.")
except Exception as e:
    logging.warning(f"Failed to plot actual vs predicted AQI: {e}")

logging.info("EDA completed. All results saved in 'eda_outputs/' folder.")