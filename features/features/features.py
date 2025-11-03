import os
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.getcwd()
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
TRAIN_PARQUET = os.path.join(FEATURES_DIR, "training_dataset.parquet")
TRAIN_CSV = TRAIN_PARQUET.replace(".parquet", ".csv")

load_dotenv()
API_KEY = os.getenv("AQI_FORECAST_API_KEY")
if not API_KEY:
    raise ValueError("Missing AQI_FORECAST_API_KEY in .env file")

def load_training_data():
    """Load local training dataset and clean for Hopsworks upload."""
    if os.path.exists(TRAIN_PARQUET):
        df = pd.read_parquet(TRAIN_PARQUET)
        logger.info(f"Loaded training data from {TRAIN_PARQUET}")
    elif os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
        logger.info(f"Loaded training data from {TRAIN_CSV}")
    else:
        raise FileNotFoundError("No training dataset found in data/features")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp_utc"], inplace=True)

    df.columns = df.columns.str.lower().str.strip()
    for suffix in ["_x", "_y"]:
        df.columns = [c.replace(suffix, "") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]

    int_cols = df.select_dtypes(include=["int", "int32", "int64"]).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype(float)

    if "city" not in df.columns:
        df["city"] = "Karachi"
    df["city"].fillna("Karachi", inplace=True)

    df.drop_duplicates(subset=["timestamp_utc", "city"], inplace=True)
    logger.info(f"Final dataset shape: {df.shape}")
    return df

def upload_to_hopsworks():
    """Upload cleaned data to Hopsworks Feature Store."""
    df = load_training_data()

    logger.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()

    FG_NAME = "aqi_features"
    TARGET_VERSION = 1  

    try:
        existing_groups = fs.get_feature_groups(name=FG_NAME)
        if existing_groups:
            latest_version = max([fg.version for fg in existing_groups])
            TARGET_VERSION = latest_version + 1
    except Exception:
        TARGET_VERSION = 1

    logger.info(f"Creating Feature Group '{FG_NAME}' version {TARGET_VERSION}...")
    fg = fs.create_feature_group(
        name=FG_NAME,
        version=TARGET_VERSION,
        primary_key=["timestamp_utc", "city"],
        description="Merged AQI + weather dataset (cleaned and normalized)",
        online_enabled=False
    )

    logger.info("Uploading data to Hopsworks Feature Store...")
    fg.insert(df, write_options={"wait_for_job": True})
    logger.info(f"Upload complete: {len(df)} rows inserted into '{FG_NAME}' (v{TARGET_VERSION})")

if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Feature upload started at {start_time}")
    upload_to_hopsworks()
    logger.info("Feature upload completed successfully!")
