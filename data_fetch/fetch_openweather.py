import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")  # Karachi default
LON = os.getenv("LON", "67.0011")
CITY = os.getenv("CITY", "Karachi")

EXPECTED_COLS = [
    "city", "datetime", "temp", "humidity", "pressure", "wind_speed",
    "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3"
]

def fetch_openweather():
    """Fetch current weather + air pollution data from OpenWeather API and store locally."""
    base_url = "https://api.openweathermap.org/data/2.5/"
    weather_url = f"{base_url}weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    air_url = f"{base_url}air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"

    try:
        weather_data = requests.get(weather_url, timeout=10).json()
        air_data = requests.get(air_url, timeout=10).json()
    except Exception as e:
        print(f"Failed to fetch OpenWeather data: {e}")
        return None

    components = air_data.get("list", [{}])[0].get("components", {})

    data = {
        "city": CITY,
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": weather_data.get("main", {}).get("temp"),
        "humidity": weather_data.get("main", {}).get("humidity"),
        "pressure": weather_data.get("main", {}).get("pressure"),
        "wind_speed": weather_data.get("wind", {}).get("speed"),
        "aqi": air_data.get("list", [{}])[0].get("main", {}).get("aqi"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "o3": components.get("o3"),
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=EXPECTED_COLS)

    output_dir = os.path.join(os.getcwd(), "data", "raw_openweather")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "openweather_data.csv")

    # Append to CSV safely
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, on_bad_lines="skip")
            existing = existing.reindex(columns=EXPECTED_COLS)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_csv(output_path, index=False)
        except Exception as e:
            print(f"CSV read/merge issue, rewriting file fresh: {e}")
            df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    try:
        df_check = pd.read_csv(output_path, on_bad_lines="skip")
        print(f"OpenWeather data fetched and saved successfully! Total rows: {len(df_check)}")
    except Exception as e:
        print(f"Could not verify saved file: {e}")

    return df


if __name__ == "__main__":
    fetch_openweather()
