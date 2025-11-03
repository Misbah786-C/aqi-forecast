import os
import requests
import pandas as pd
from datetime import datetime, timedelta

LAT, LON = 24.8607, 67.0011  # Karachi
CITY = "Karachi"

def fetch_openmeteo_6months():
    # Fetch 6 months of hourly historical weather + air quality data from Open-Meteo
    end = datetime.utcnow().date()
    start = (end - timedelta(days=180)).isoformat()
    print(f"Fetching Open-Meteo data from {start} to {end}")

    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
        f"wind_speed_10m,wind_direction_10m,cloud_cover"
    )

    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone,us_aqi"
    )

    weather = requests.get(weather_url).json()
    aqi = requests.get(aqi_url).json()

    df_weather = pd.DataFrame(weather.get("hourly", {}))
    df_aqi = pd.DataFrame(aqi.get("hourly", {}))

    df = pd.merge(df_weather, df_aqi, on="time", how="outer").sort_values("time").reset_index(drop=True)

    df.rename(columns={
        "time": "datetime",
        "temperature_2m": "ow_temp",
        "relative_humidity_2m": "ow_humidity",
        "pressure_msl": "ow_pressure",
        "wind_speed_10m": "ow_wind_speed",
        "wind_direction_10m": "ow_wind_deg",
        "cloud_cover": "ow_clouds",
        "carbon_monoxide": "ow_co",
        "nitrogen_dioxide": "ow_no2",
        "sulphur_dioxide": "ow_so2",
        "ozone": "ow_o3",
        "pm2_5": "ow_pm2_5",
        "pm10": "ow_pm10",
        "us_aqi": "aqi_aqicn"
    }, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Karachi")

    df.drop_duplicates(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)

    df["city"] = CITY
    df["hour"] = df["datetime"].dt.hour.astype(float)
    df["day"] = df["datetime"].dt.day.astype(float)
    df["month"] = df["datetime"].dt.month.astype(float)
    df["weekday"] = df["datetime"].dt.weekday.astype(float)

    num_cols = df.select_dtypes(include=["float", "int"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    out_dir = os.path.join("data", "raw_openmeteo")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "openmeteo_6months.csv")
    parquet_path = os.path.join(out_dir, "openmeteo_6months.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(df.head())
    print(f"\nSaved {len(df)} unique hourly rows →")
    print(f"  • CSV: {csv_path}")
    print(f"  • Parquet: {parquet_path}")

    return df

if __name__ == "__main__":
    fetch_openmeteo_6months()
