import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import hopsworks
from hsfs.client.exceptions import FeatureStoreException
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
BASE_DIR = os.getcwd()


OPENMETEO_PATH = os.path.join(BASE_DIR, "data", "raw_openmeteo", "openmeteo_6months.csv")
OPENWEATHER_PATH = os.path.join(BASE_DIR, "data", "raw_openweather", "openweather_data.csv")
AQICN_PATH = os.path.join(BASE_DIR, "data", "raw_aqicn", "aqicn_data.csv")

CITY = "Karachi"


load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2


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

    df["city"] = CITY
    df["hour"] = df["timestamp_utc"].dt.hour.astype(float)
    df["day"] = df["timestamp_utc"].dt.day.astype(float)
    df["month"] = df["timestamp_utc"].dt.month.astype(float)
    df["weekday"] = df["timestamp_utc"].dt.weekday.astype(float)

    df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logging.info(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def merge_local():
    logging.info("Merging Open-Meteo + OpenWeather + AQICN datasets...")
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
            df_combined.sort_values("timestamp_utc"),
            df_next.sort_values("timestamp_utc"),
            on="timestamp_utc",
            direction="nearest",
            tolerance=pd.Timedelta("1h")
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
        df_combined["o3"].fillna(df_combined["o3"].median(), inplace=True)
    else:
        df_combined["o3"] = np.nan

    logging.info(f"Merge complete: {len(df_combined)} rows, {df_combined.shape[1]} columns")
    return df_combined


def upload_to_hopsworks(df: pd.DataFrame):
    logging.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(float)

    if "o3" in df.columns:
        logging.info("Dropping plain 'o3' column to match aqi_features v2 schema")
        df = df.drop(columns=["o3"])

    logging.info("Uploading columns: %s", list(df.columns))

    try:
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp_utc"],
            description="Air quality and weather merged dataset (v2, all float types)",
            online_enabled=False,
        )
        fg.insert(df, write_options={"wait_for_job": True})
        logging.info(f"Uploaded {len(df)} rows to '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION})")

    except FeatureStoreException as e:
        logging.error(f"Hopsworks upload failed: {e}")


def backfill():
    logging.info(f"Running backfill at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    df = merge_local()
    if df is None or df.empty:
        logging.error("Backfill failed — no merged data.")
        return
    upload_to_hopsworks(df)
    logging.info(f"Backfill complete: {len(df)} rows, {df.shape[1]} columns uploaded.")


if __name__ == "__main__":
    backfill()
