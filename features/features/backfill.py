import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import hopsworks
from hsfs.feature_group import FeatureGroup
from hsfs.client.exceptions import FeatureStoreException
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
BASE_DIR = os.getcwd()

OPENMETEO_PATH = os.path.join(BASE_DIR, "data", "raw_openmeteo", "openmeteo_6months.csv")
OPENWEATHER_PATH = os.path.join(BASE_DIR, "data", "raw_openweather", "openweather_data.csv")
AQICN_PATH = os.path.join(BASE_DIR, "data", "raw_aqicn", "aqicn_data.csv")

FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(FEATURES_DIR, "training_dataset.csv")
TRAIN_PARQUET = TRAIN_CSV.replace(".csv", ".parquet")

CITY = "Karachi"

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")


def load_csv(path, name):
    if not os.path.exists(path):
        logging.warning(f"Missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df.drop(columns=["datetime"], inplace=True)
    else:
        logging.warning(f"Missing 'datetime' column in {name}")
        return pd.DataFrame()
    df.dropna(subset=["timestamp_utc"], inplace=True)
    df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
    df.sort_values("timestamp_utc", inplace=True)
    logging.info(f"Loaded {name}: {len(df)} rows")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
        if (df[col] == 0).any() and (df[col] == 0).mean() < 0.8:
            df.loc[df[col] == 0, col] = df[col].median()

    if "city" not in df.columns:
        df["city"] = CITY
    df["city"].fillna(CITY, inplace=True)

    df["hour"] = df["timestamp_utc"].dt.hour.astype(float)
    df["day"] = df["timestamp_utc"].dt.day.astype(float)
    df["month"] = df["timestamp_utc"].dt.month.astype(float)
    df["weekday"] = df["timestamp_utc"].dt.weekday.astype(float)

    df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logging.info(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def merge_local():
    logging.info("Merging Open-Meteo + OpenWeather + AQICN")
    df_meteo = load_csv(OPENMETEO_PATH, "Open-Meteo")
    df_ow = load_csv(OPENWEATHER_PATH, "OpenWeather")
    df_aqi = load_csv(AQICN_PATH, "AQICN")

    dfs = [df for df in [df_meteo, df_ow, df_aqi] if not df.empty]
    if len(dfs) < 2:
        logging.error("Not enough datasets to merge.")
        return None

    df_combined = dfs[0]
    for df_next in dfs[1:]:
        df_combined = pd.merge_asof(
            df_combined, df_next,
            on="timestamp_utc",
            direction="nearest",
            tolerance=pd.Timedelta("15min")
        )

    df_combined.columns = (
        df_combined.columns.str.lower()
        .str.strip()
        .str.replace("_x", "", regex=False)
        .str.replace("_y", "", regex=False)
    )
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated(keep="first")]

    df_combined = clean_dataset(df_combined)

    if "o3" in df_combined.columns:
        df_combined.drop(columns=["o3"], inplace=True)
        logging.info("Dropped 'o3' column (too many NaNs)")

    df_combined.to_csv(TRAIN_CSV, index=False)
    df_combined.to_parquet(TRAIN_PARQUET, index=False)
    logging.info(f"Saved merged dataset → {TRAIN_PARQUET}")
    return df_combined


def upload_to_hopsworks(df: pd.DataFrame):
    logging.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    version = 1
    fg = None
    uploaded = False

    while not uploaded:
        try:
            fg = fs.get_or_create_feature_group(
                name="aqi_features",
                version=version,
                primary_key=["timestamp_utc"],
                description="Air quality and weather merged dataset",
                online_enabled=False,
            )
            fg.insert(df, write_options={"wait_for_job": True})
            logging.info(f"Uploaded to feature group: aqi_features v{version}")
            uploaded = True
        except FeatureStoreException as e:
            if "not compatible" in str(e).lower():
                version += 1
                logging.warning(f"Schema mismatch, retrying with version {version}...")
            else:
                logging.error(f"Hopsworks upload failed: {e}")
                break


def backfill():
    logging.info(f"Running backfill at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    df = merge_local()
    if df is None:
        logging.error("Backfill failed — no merged data.")
        return
    upload_to_hopsworks(df)
    logging.info(f"Backfill complete: {len(df)} rows, {df.shape[1]} columns uploaded.")


if __name__ == "__main__":
    backfill()
