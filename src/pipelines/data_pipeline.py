"""
Production data preparation pipeline.

Builds the ML dataset from the latest S3 realtime dataset.

Daily production flow:
1. Download latest Lahore hourly CSV from S3
2. Clean/validate the hourly data
3. Calculate AQI
4. Create forecast targets
5. Create forecast features
6. Save the ML dataset for model training

The existing local historical JSON workflow remains supported.
"""

import os
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

from src.data.historical_cleaner import (
    clean_historical_data
)

from src.features.aqi_calculator import (
    calculate_aqi
)

from src.features.forecast_targets import (
    create_forecast_targets
)

from src.features.forecast_features import (
    create_features,
    select_features
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = (
    BASE_DIR / "data" / "historical"
)

PROCESSED_DIR = (
    BASE_DIR / "data" / "processed"
)


# ============================================================
# AWS CONFIGURATION
# ============================================================

S3_BUCKET = os.environ.get(
    "S3_BUCKET",
    "pearls-aqi-predictor-071493957773"
)

S3_REALTIME_KEY = os.environ.get(
    "S3_REALTIME_KEY",
    "realtime/lahore_hourly.csv"
)


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
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
    "pm10",
    "pm2_5",
    "co",
    "no2",
    "so2",
    "o3",
]


# ============================================================
# LOAD S3 REALTIME DATA
# ============================================================

def load_realtime_from_s3():

    print(
        f"\nDownloading latest realtime dataset:"
        f"\ns3://{S3_BUCKET}/{S3_REALTIME_KEY}"
    )

    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=S3_REALTIME_KEY
    )

    df = pd.read_csv(
        BytesIO(
            response["Body"].read()
        )
    )

    print(
        f"S3 dataset shape: {df.shape}"
    )

    return df


# ============================================================
# VALIDATE REALTIME DATA
# ============================================================

def validate_realtime_data(df):

    print(
        "\nValidating realtime dataset..."
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        raise ValueError(
            "Realtime dataset is empty."
        )

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Date range: "
        f"{df['timestamp'].min()} "
        f"-> "
        f"{df['timestamp'].max()}"
    )

    duplicate_count = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate timestamps remain."
        )

    print(
        "Realtime dataset validation: PASSED"
    )

    return df


# ============================================================
# LOAD TRAINING SOURCE
# ============================================================

def load_training_source():

    use_s3 = os.environ.get(
        "USE_S3_TRAINING_DATA",
        "true"
    ).lower() == "true"

    if use_s3:

        print(
            "\nTraining source: S3 realtime dataset"
        )

        df = load_realtime_from_s3()

        return validate_realtime_data(df)

    # --------------------------------------------------------
    # Local fallback
    # --------------------------------------------------------

    print(
        "\nTraining source: local historical JSON"
    )

    files = sorted(
        HISTORICAL_DIR.glob(
            "historical_*.json"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No historical JSON dataset found."
        )

    input_file = files[-1]

    print(
        f"\nUsing historical dataset:\n"
        f"{input_file}"
    )

    df = clean_historical_data(
        input_file
    )

    return df


# ============================================================
# MAIN DATA PIPELINE
# ============================================================

def run_data_pipeline():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------------

    print(
        "\n[1/5] Loading training data..."
    )

    df = load_training_source()

    print(
        f"Training source shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Save cleaned historical data
    # --------------------------------------------------------

    cleaned_file = (
        PROCESSED_DIR
        / "historical_cleaned.csv"
    )

    df.to_csv(
        cleaned_file,
        index=False
    )

    print(
        f"Saved: {cleaned_file}"
    )

    # --------------------------------------------------------
    # 2. Calculate AQI
    # --------------------------------------------------------

    print(
        "\n[2/5] Calculating AQI..."
    )

    df = calculate_aqi(df)

    aqi_file = (
        PROCESSED_DIR
        / "historical_with_aqi.csv"
    )

    df.to_csv(
        aqi_file,
        index=False
    )

    print(
        f"Saved: {aqi_file}"
    )

    # --------------------------------------------------------
    # 3. Create forecast targets
    # --------------------------------------------------------

    print(
        "\n[3/5] Creating forecast targets..."
    )

    df = create_forecast_targets(
        df
    )

    df = df.dropna(
        subset=[
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ).reset_index(drop=True)

    forecast_file = (
        PROCESSED_DIR
        / "forecast_dataset.csv"
    )

    df.to_csv(
        forecast_file,
        index=False
    )

    print(
        f"Saved: {forecast_file}"
    )

    # --------------------------------------------------------
    # 4. Create ML features
    # --------------------------------------------------------

    print(
        "\n[4/5] Creating ML features..."
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
        *target_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing ML columns: "
            + ", ".join(missing_columns)
        )

    ml_df = df[
        required_columns
    ].copy()

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

    if ml_df.empty:

        raise ValueError(
            "ML dataset is empty after removing incomplete rows."
        )

    # --------------------------------------------------------
    # 5. Save ML dataset
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving ML dataset..."
    )

    ml_file = (
        PROCESSED_DIR
        / "ml_dataset.csv"
    )

    ml_df.to_csv(
        ml_file,
        index=False
    )

    print(
        f"Saved: {ml_file}"
    )

    print(
        f"\nFinal ML dataset shape: "
        f"{ml_df.shape}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training date range: "
        f"{ml_df['timestamp'].min()} "
        f"-> "
        f"{ml_df['timestamp'].max()}"
    )

    print(
        "\nDATA PREPARATION COMPLETED"
    )


if __name__ == "__main__":
    run_data_pipeline()