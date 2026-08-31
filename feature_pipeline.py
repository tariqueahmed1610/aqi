
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import hopsworks

try:
    from google.colab import userdata
    HOPSWORKS_API_KEY = userdata.get("HOPSWORKS_API_KEY")
except Exception:
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise RuntimeError(
        "HOPSWORKS_API_KEY not found. Set it as a Colab Secret or env var."
    )

CITIES = {
    "karachi":   (24.8607, 67.0011),
    "lahore":    (31.5497, 74.3436),
    "islamabad": (33.6844, 73.0479),
    "peshawar":  (34.0151, 71.5249),
    "quetta":    (30.1798, 66.9750),
}

BACKFILL = os.environ.get("BACKFILL", "false").lower() == "true"
BACKFILL_DAYS = 730

FG_NAME = "aqi_features"
FG_VERSION = 9  # Bumped to v9 to cleanly apply disabled statistics/monitoring

def get_with_retries(url, params, max_attempts=4, timeout=90):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            wait = 5 * attempt
            print(f"  Request failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise last_error

def fetch_raw_data(city_name, latitude, longitude, start_date, end_date, use_forecast_api=False):
    print(f"Fetching data for {city_name} ({start_date} to {end_date})...")

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi",
        ],
        "timezone": "UTC",
    }
    aq_resp = get_with_retries(aq_url, aq_params)
    df_aq = pd.DataFrame(aq_resp.json()["hourly"])

    if use_forecast_api:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "past_days": 4,
            "forecast_days": 1,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
            "timezone": "UTC",
        }
    else:
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
            "timezone": "UTC",
        }
    w_resp = get_with_retries(weather_url, weather_params)
    df_weather = pd.DataFrame(w_resp.json()["hourly"])

    df = pd.merge(df_aq, df_weather, on="time", how="inner")

    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] <= now_utc].reset_index(drop=True)

    return df

FEATURE_COLUMNS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "hour",
    "day_of_week", "month", "pm2_5_lag_1h", "pm2_5_lag_24h",
    "aqi_rate_of_change", "city_id",
]

def engineer_features(df: pd.DataFrame, city_name: str, city_id: int) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour.astype(int)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(int)
    df["month"] = df["time"].dt.month.astype(int)

    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["aqi_rate_of_change"] = df["us_aqi"].diff()

    df["target_aqi_24h"] = df["us_aqi"].shift(-24)
    df["target_aqi_48h"] = df["us_aqi"].shift(-48)
    df["target_aqi_72h"] = df["us_aqi"].shift(-72)

    df["city"] = city_name
    df["city_id"] = city_id

    df["id"] = df["city"] + "_" + df["time"].astype(str)
    df["event_time"] = (df["time"].astype("int64") // 10**6).astype("int64")

    df = df.drop(columns=["time"])
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df

def write_to_feature_store(df: pd.DataFrame):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="Multi-city hourly AQI + weather features, targets = AQI at +24h/+48h/+72h",
        primary_key=["id"],
        event_time="event_time",
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config=False,
    )

    max_attempts = 3
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            fg.insert(df, write_options={"wait_for_job": True})
            print(f"Inserted {len(df)} rows into feature group '{FG_NAME}' v{FG_VERSION}.")
            return
        except Exception as e:
            last_error = e
            wait = 15 * attempt
            print(f"  Insert failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise last_error

def main():
    end_date = datetime.now(timezone.utc).date()
    if BACKFILL:
        start_date = end_date - timedelta(days=BACKFILL_DAYS)
    else:
        start_date = end_date - timedelta(days=4)

    all_frames = []
    for city_id, (city_name, (lat, lon)) in enumerate(CITIES.items()):
        raw = fetch_raw_data(city_name, lat, lon, start_date, end_date, use_forecast_api=not BACKFILL)
        features = engineer_features(raw, city_name, city_id)
        print(f"  {city_name}: {len(features)} feature rows.")
        all_frames.append(features)
        time.sleep(1)

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"Prepared {len(combined)} feature rows across {len(CITIES)} cities.")
    write_to_feature_store(combined)

if __name__ == "__main__":
    main()
