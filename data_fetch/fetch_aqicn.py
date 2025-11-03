import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY = os.getenv("CITY", "Karachi")

EXPECTED_COLS = ["city", "datetime", "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3"]

def fetch_aqicn():
    """Fetch live AQI data from AQICN API and save to CSV."""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    
    try:
        response = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"Failed to fetch AQICN data: {e}")
        return None

    if response.get("status") != "ok":
        print("Failed to fetch AQICN data:", response)
        return None

    data = response["data"]
    iaqi = data.get("iaqi", {})

    record = {
        "city": CITY,
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": data.get("aqi"),
        "pm2_5": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
    }

    df = pd.DataFrame([record])
    df = df.reindex(columns=EXPECTED_COLS)

    output_dir = os.path.join(os.getcwd(), "data", "raw_aqicn")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "aqicn_data.csv")

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"AQICN data fetched and saved successfully! Total rows now in file: {len(pd.read_csv(output_path))}")
    return df


if __name__ == "__main__":
    fetch_aqicn()
