import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks

try:
    from google.colab import userdata
    HOPSWORKS_API_KEY = userdata.get("HOPSWORKS_API_KEY")
except Exception:
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise RuntimeError("HOPSWORKS_API_KEY not found. Set it as a Colab Secret or env var.")

FG_NAME = "aqi_features"
FG_VERSION = 9

CITIES = {0: "karachi", 1: "lahore", 2: "islamabad", 3: "peshawar", 4: "quetta"}

os.makedirs("eda_plots", exist_ok=True)
sns.set_style("whitegrid")


def load_data():
    project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.select_all().read()
    df["datetime"] = pd.to_datetime(df["event_time"], unit="ms")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def plot_aqi_over_time(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    for city_id, city_name in CITIES.items():
        city_df = df[df["city_id"] == city_id]
        ax.plot(city_df["datetime"], city_df["us_aqi"], label=city_name, alpha=0.7, linewidth=0.8)
    ax.set_title("AQI Over Time by City")
    ax.set_xlabel("Date")
    ax.set_ylabel("US AQI")
    ax.legend()
    plt.tight_layout()
    plt.savefig("eda_plots/aqi_over_time.png", dpi=120)
    plt.show()


def plot_avg_aqi_by_city(df):
    avg_aqi = df.groupby("city")["us_aqi"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=avg_aqi.index, y=avg_aqi.values, ax=ax)
    ax.set_title("Average AQI by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Average US AQI")
    plt.tight_layout()
    plt.savefig("eda_plots/avg_aqi_by_city.png", dpi=120)
    plt.show()
    print("\nAverage AQI by city:")
    print(avg_aqi.round(1).to_string())


def plot_hourly_pattern(df):
    hourly_avg = df.groupby(["hour", "city"])["us_aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=hourly_avg, x="hour", y="us_aqi", hue="city", marker="o", ax=ax)
    ax.set_title("Average AQI by Hour of Day")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Average US AQI")
    plt.tight_layout()
    plt.savefig("eda_plots/hourly_pattern.png", dpi=120)
    plt.show()


def plot_monthly_pattern(df):
    monthly_avg = df.groupby(["month", "city"])["us_aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_avg, x="month", y="us_aqi", hue="city", marker="o", ax=ax)
    ax.set_title("Average AQI by Month (Seasonal Pattern)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average US AQI")
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig("eda_plots/monthly_pattern.png", dpi=120)
    plt.show()


def plot_correlation_heatmap(df):
    cols = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "us_aqi", "temperature_2m",
        "relative_humidity_2m", "wind_speed_10m",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Pollutants and Weather Features")
    plt.tight_layout()
    plt.savefig("eda_plots/correlation_heatmap.png", dpi=120)
    plt.show()


def plot_aqi_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for city_name in CITIES.values():
        sns.kdeplot(df[df["city"] == city_name]["us_aqi"], label=city_name, ax=ax)
    ax.set_title("AQI Distribution by City")
    ax.set_xlabel("US AQI")
    ax.legend()
    plt.tight_layout()
    plt.savefig("eda_plots/aqi_distribution.png", dpi=120)
    plt.show()


def print_hazardous_summary(df):
    thresholds = {
        "Good (0-50)": (0, 50),
        "Moderate (51-100)": (51, 100),
        "Unhealthy for Sensitive Groups (101-150)": (101, 150),
        "Unhealthy (151-200)": (151, 200),
        "Very Unhealthy (201-300)": (201, 300),
        "Hazardous (301+)": (301, 10000),
    }
    print("\nAQI category breakdown (% of hours), by city:")
    rows = []
    for city_name in CITIES.values():
        city_df = df[df["city"] == city_name]
        row = {"city": city_name}
        for label, (lo, hi) in thresholds.items():
            pct = ((city_df["us_aqi"] >= lo) & (city_df["us_aqi"] <= hi)).mean() * 100
            row[label] = round(pct, 1)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("city")
    print(summary.to_string())
    summary.to_csv("eda_plots/aqi_category_breakdown.csv")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows across {df['city'].nunique()} cities.")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    plot_aqi_over_time(df)
    plot_avg_aqi_by_city(df)
    plot_hourly_pattern(df)
    plot_monthly_pattern(df)
    plot_correlation_heatmap(df)
    plot_aqi_distribution(df)
    print_hazardous_summary(df)

    print("\nAll plots saved to eda_plots/")


if __name__ == "__main__":
    main()
