import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_dataset.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


def create_features(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =====================================================
    # TIME FEATURES
    # =====================================================

    df["hour"] = df["timestamp"].dt.hour

    df["day"] = df["timestamp"].dt.day

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclical day-of-week encoding
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # =====================================================
    # AQI LAG FEATURES
    # =====================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"aqi_lag_{lag}h"] = (
            df["target_aqi"]
            .shift(lag)
        )

    # =====================================================
    # PM2.5 LAG FEATURES
    # =====================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm25_lag_{lag}h"] = (
            df["pm2_5"]
            .shift(lag)
        )

    # =====================================================
    # PM10 LAG FEATURES
    # =====================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm10_lag_{lag}h"] = (
            df["pm10"]
            .shift(lag)
        )

    # =====================================================
    # ROLLING AQI
    # =====================================================

    for window in [3, 6, 12, 24, 48]:

        df[f"aqi_rolling_mean_{window}h"] = (
            df["target_aqi"]
            .shift(1)
            .rolling(window)
            .mean()
        )

        df[f"aqi_rolling_std_{window}h"] = (
            df["target_aqi"]
            .shift(1)
            .rolling(window)
            .std()
        )

    # =====================================================
    # ROLLING PM2.5
    # =====================================================

    for window in [3, 6, 12, 24]:

        df[f"pm25_rolling_mean_{window}h"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(window)
            .mean()
        )

        df[f"pm25_rolling_std_{window}h"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(window)
            .std()
        )

    # =====================================================
    # ROLLING PM10
    # =====================================================

    for window in [3, 6, 12, 24]:

        df[f"pm10_rolling_mean_{window}h"] = (
            df["pm10"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # =====================================================
    # AQI CHANGES
    # =====================================================

    df["aqi_change_1h"] = (
        df["target_aqi"]
        .shift(1)
        -
        df["target_aqi"]
        .shift(2)
    )

    df["aqi_change_3h"] = (
        df["target_aqi"]
        .shift(1)
        -
        df["target_aqi"]
        .shift(4)
    )

    # =====================================================
    # PM2.5 CHANGES
    # =====================================================

    df["pm25_change_1h"] = (
        df["pm2_5"]
        .shift(1)
        -
        df["pm2_5"]
        .shift(2)
    )

    df["pm25_change_3h"] = (
        df["pm2_5"]
        .shift(1)
        -
        df["pm2_5"]
        .shift(4)
    )

    # =====================================================
    # PM10 CHANGES
    # =====================================================

    df["pm10_change_1h"] = (
        df["pm10"]
        .shift(1)
        -
        df["pm10"]
        .shift(2)
    )

    df["pm10_change_3h"] = (
        df["pm10"]
        .shift(1)
        -
        df["pm10"]
        .shift(4)
    )

    return df


def select_features(df):

    feature_columns = [

        # Current weather
        "temperature",
        "humidity",
        "dew_point",
        "feels_like",
        "pressure",
        "surface_pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "precipitation",

        # Current pollutants
        "pm10",
        "pm2_5",
        "co",
        "no2",
        "so2",
        "o3",

        # Time
        "hour",
        "day",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    # AQI lags
    for lag in [1, 3, 6, 12, 24, 48]:
        feature_columns.append(
            f"aqi_lag_{lag}h"
        )

    # PM2.5 lags
    for lag in [1, 3, 6, 12, 24, 48]:
        feature_columns.append(
            f"pm25_lag_{lag}h"
        )

    # PM10 lags
    for lag in [1, 3, 6, 12, 24, 48]:
        feature_columns.append(
            f"pm10_lag_{lag}h"
        )

    # AQI rolling
    for window in [3, 6, 12, 24, 48]:

        feature_columns.append(
            f"aqi_rolling_mean_{window}h"
        )

        feature_columns.append(
            f"aqi_rolling_std_{window}h"
        )

    # PM2.5 rolling
    for window in [3, 6, 12, 24]:

        feature_columns.append(
            f"pm25_rolling_mean_{window}h"
        )

        feature_columns.append(
            f"pm25_rolling_std_{window}h"
        )

    # PM10 rolling
    for window in [3, 6, 12, 24]:

        feature_columns.append(
            f"pm10_rolling_mean_{window}h"
        )

    # Changes
    feature_columns += [
        "aqi_change_1h",
        "aqi_change_3h",
        "pm25_change_1h",
        "pm25_change_3h",
        "pm10_change_1h",
        "pm10_change_3h",
    ]

    return feature_columns


def main():

    print("=" * 60)
    print("LEAKAGE-SAFE FORECAST FEATURE ENGINEERING")
    print("=" * 60)

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nOriginal shape: {df.shape}"
    )

    df = create_features(
        df
    )

    feature_columns = select_features(
        df
    )

    target_columns = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    required_columns = [
        "timestamp",
        "target_aqi",
        *feature_columns,
        *target_columns
    ]

    ml_df = df[
        required_columns
    ].copy()

    print(
        f"\nFeature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Before removing NaN rows: "
        f"{ml_df.shape}"
    )

    ml_df = ml_df.dropna(
        subset=(
            feature_columns
            + target_columns
        )
    )

    ml_df = (
        ml_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"After removing NaN rows: "
        f"{ml_df.shape}"
    )

    print(
        "\nMissing values:"
    )

    missing = (
        ml_df.isnull()
        .sum()
    )

    print(
        missing[
            missing > 0
        ]
    )

    ml_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nML dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        ml_df[
            [
                "timestamp",
                "aqi_lag_1h",
                "aqi_lag_24h",
                "aqi_rolling_mean_24h",
                "pm25_lag_24h",
                "target_aqi_24h",
                "target_aqi_48h",
                "target_aqi_72h",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()