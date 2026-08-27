import json
import os
from io import BytesIO

import boto3
import numpy as np
import pandas as pd


# ============================================================
# AWS CONFIGURATION
# ============================================================

S3_BUCKET = os.environ["S3_BUCKET"]
INPUT_KEY = os.environ.get(
    "INPUT_KEY",
    "realtime/lahore_hourly.csv"
)
OUTPUT_KEY = os.environ.get(
    "OUTPUT_KEY",
    "realtime/latest_features.csv"
)
FEATURE_LIST_KEY = os.environ.get(
    "FEATURE_LIST_KEY",
    "models/production/feature_list.json"
)

s3 = boto3.client("s3")


# ============================================================
# EPA AQI BREAKPOINTS
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 500),
]

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]

O3_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]


# ============================================================
# AQI FUNCTIONS
# ============================================================

def calculate_sub_index(concentration, breakpoints):

    if pd.isna(concentration):
        return np.nan

    for low, high, aqi_low, aqi_high in breakpoints:

        if low <= concentration <= high:

            aqi = (
                (aqi_high - aqi_low)
                / (high - low)
            ) * (
                concentration - low
            ) + aqi_low

            return round(aqi)

    return np.nan


def ugm3_to_ppm(concentration, molecular_weight):

    return (
        concentration * 24.45
        / (molecular_weight * 1000)
    )


def ugm3_to_ppb(concentration, molecular_weight):

    return (
        concentration * 24.45
        / molecular_weight
    )


def calculate_aqi(df):

    df = df.copy()

    df["pm25_aqi"] = df["pm2_5"].apply(
        lambda x: calculate_sub_index(
            x,
            PM25_BREAKPOINTS
        )
    )

    df["pm10_aqi"] = df["pm10"].apply(
        lambda x: calculate_sub_index(
            x,
            PM10_BREAKPOINTS
        )
    )

    df["co_aqi"] = df["co"].apply(
        lambda x: calculate_sub_index(
            ugm3_to_ppm(x, 28.01),
            CO_BREAKPOINTS
        )
    )

    df["no2_aqi"] = df["no2"].apply(
        lambda x: calculate_sub_index(
            ugm3_to_ppb(x, 46.01),
            NO2_BREAKPOINTS
        )
    )

    df["so2_aqi"] = df["so2"].apply(
        lambda x: calculate_sub_index(
            ugm3_to_ppb(x, 64.066),
            SO2_BREAKPOINTS
        )
    )

    df["o3_aqi"] = df["o3"].apply(
        lambda x: calculate_sub_index(
            ugm3_to_ppm(x, 48.00),
            O3_BREAKPOINTS
        )
    )

    aqi_columns = [
        "pm25_aqi",
        "pm10_aqi",
        "co_aqi",
        "no2_aqi",
        "so2_aqi",
        "o3_aqi",
    ]

    df["target_aqi"] = df[aqi_columns].max(axis=1)

    return df


# ============================================================
# FEATURE GENERATION
# ============================================================

def create_features(df):

    df = df.copy()

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
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Time features

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # AQI lags

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"aqi_lag_{lag}h"] = (
            df["target_aqi"].shift(lag)
        )

    # PM2.5 lags

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm25_lag_{lag}h"] = (
            df["pm2_5"].shift(lag)
        )

    # PM10 lags

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm10_lag_{lag}h"] = (
            df["pm10"].shift(lag)
        )

    # AQI rolling

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

    # PM2.5 rolling

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

    # PM10 rolling

    for window in [3, 6, 12, 24]:

        df[f"pm10_rolling_mean_{window}h"] = (
            df["pm10"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # AQI changes

    df["aqi_change_1h"] = (
        df["target_aqi"].shift(1)
        -
        df["target_aqi"].shift(2)
    )

    df["aqi_change_3h"] = (
        df["target_aqi"].shift(1)
        -
        df["target_aqi"].shift(4)
    )

    # PM2.5 changes

    df["pm25_change_1h"] = (
        df["pm2_5"].shift(1)
        -
        df["pm2_5"].shift(2)
    )

    df["pm25_change_3h"] = (
        df["pm2_5"].shift(1)
        -
        df["pm2_5"].shift(4)
    )

    # PM10 changes

    df["pm10_change_1h"] = (
        df["pm10"].shift(1)
        -
        df["pm10"].shift(2)
    )

    df["pm10_change_3h"] = (
        df["pm10"].shift(1)
        -
        df["pm10"].shift(4)
    )

    return df.loc[
        :,
        ~df.columns.duplicated()
    ]


# ============================================================
# S3 HELPERS
# ============================================================

def read_csv_from_s3(bucket, key):

    obj = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    return pd.read_csv(
        BytesIO(
            obj["Body"].read()
        )
    )


def read_feature_list():

    obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key=FEATURE_LIST_KEY
    )

    content = obj["Body"].read().decode("utf-8")

    feature_list = json.loads(content)

    if isinstance(feature_list, dict):

        if "features" in feature_list:

            feature_list = feature_list["features"]

        else:

            raise ValueError(
                "feature_list.json does not contain 'features'."
            )

    if not isinstance(feature_list, list):

        raise ValueError(
            "Invalid feature_list.json format."
        )

    return list(
        dict.fromkeys(feature_list)
    )


def write_csv_to_s3(df, bucket, key):

    buffer = BytesIO()

    df.to_csv(
        buffer,
        index=False
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="text/csv"
    )


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(event, context):

    try:

        print("=" * 60)
        print("PEARLS AQI FEATURE GENERATOR")
        print("=" * 60)

        print(
            f"Reading s3://{S3_BUCKET}/{INPUT_KEY}"
        )

        df = read_csv_from_s3(
            S3_BUCKET,
            INPUT_KEY
        )

        print(
            f"Input shape: {df.shape}"
        )

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ]

        print("Calculating AQI...")

        df = calculate_aqi(df)

        if "target_aqi" not in df.columns:

            raise ValueError(
                "target_aqi was not created."
            )

        print("Creating features...")

        df = create_features(df)

        feature_list = read_feature_list()

        print(
            f"Production features: "
            f"{len(feature_list)}"
        )

        missing = [
            feature
            for feature in feature_list
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing production features: "
                + ", ".join(missing)
            )

        required_columns = list(
            dict.fromkeys(
                [
                    "timestamp",
                    "target_aqi",
                    *feature_list
                ]
            )
        )

        latest = (
            df[required_columns]
            .dropna(
                subset=feature_list
            )
            .sort_values("timestamp")
            .tail(1)
            .copy()
        )

        if latest.empty:

            raise ValueError(
                "No complete feature row available."
            )

        latest = latest.loc[
            :,
            ~latest.columns.duplicated()
        ]

        model_input = latest[
            feature_list
        ]

        if model_input.isnull().any().any():

            raise ValueError(
                "Missing values detected."
            )

        latest_timestamp = str(
            latest["timestamp"].iloc[0]
        )

        current_aqi = float(
            latest["target_aqi"].iloc[0]
        )

        print(
            f"Latest timestamp: {latest_timestamp}"
        )

        print(
            f"Current AQI: {current_aqi:.2f}"
        )

        print(
            f"Features: {len(feature_list)}"
        )

        print(
            f"Writing s3://{S3_BUCKET}/{OUTPUT_KEY}"
        )

        write_csv_to_s3(
            latest,
            S3_BUCKET,
            OUTPUT_KEY
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "success",
                    "timestamp": latest_timestamp,
                    "current_aqi": current_aqi,
                    "features": len(feature_list),
                    "bucket": S3_BUCKET,
                    "key": OUTPUT_KEY
                }
            )
        }

    except Exception as e:

        print(
            f"ERROR: {str(e)}"
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "status": "error",
                    "message": str(e)
                }
            )
        }