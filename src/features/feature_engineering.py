import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and cyclic time features."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclic encoding of hour
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclic encoding of day of week
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical lag features."""

    df = df.copy()

    # AQI history
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # PM2.5 history
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    # PM10 history
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm10_lag_{lag}"] = df["pm10"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistical features."""

    df = df.copy()

    for window in [3, 6, 12, 24]:

        df[f"aqi_rolling_mean_{window}"] = (
            df["aqi"]
            .rolling(window)
            .mean()
        )

        df[f"aqi_rolling_std_{window}"] = (
            df["aqi"]
            .rolling(window)
            .std()
        )

        df[f"pm2_5_rolling_mean_{window}"] = (
            df["pm2_5"]
            .rolling(window)
            .mean()
        )

        df[f"pm2_5_rolling_std_{window}"] = (
            df["pm2_5"]
            .rolling(window)
            .std()
        )

    return df


def create_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pollutant change features."""

    df = df.copy()

    df["aqi_change_1h"] = (
        df["aqi"] - df["aqi"].shift(1)
    )

    df["pm2_5_change_1h"] = (
        df["pm2_5"] - df["pm2_5"].shift(1)
    )

    df["pm10_change_1h"] = (
        df["pm10"] - df["pm10"].shift(1)
    )

    # Percentage changes
    df["pm2_5_pct_change"] = (
        df["pm2_5"].pct_change()
    )

    df["pm10_pct_change"] = (
        df["pm10"].pct_change()
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""

    df = df.copy()

    # Make sure data is chronological
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Feature groups
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_change_features(df)

    return df


def save_features(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """Save engineered features."""

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Feature dataset saved to: {output_path}"
    )


if __name__ == "__main__":

    input_path = (
        "data/processed/"
        "cleaned_aqi_data.csv"
    )

    output_path = (
        "data/processed/"
        "feature_dataset.csv"
    )

    print("Loading cleaned data...")

    df = pd.read_csv(
        input_path
    )

    print(
        f"Original shape: {df.shape}"
    )

    print(
        "\nCreating features..."
    )

    feature_df = create_features(df)

    print(
        f"Feature shape before cleaning: "
        f"{feature_df.shape}"
    )

    # Remove rows where lag/rolling features
    # cannot yet be calculated.
    feature_df = feature_df.dropna(
        subset=[
            "aqi_lag_24",
            "pm2_5_lag_24",
            "aqi_rolling_mean_24",
            "pm2_5_rolling_mean_24"
        ]
    ).reset_index(drop=True)

    print(
        f"Feature shape after cleaning: "
        f"{feature_df.shape}"
    )

    save_features(
        feature_df,
        output_path
    )

    print("\nFeature engineering completed.")