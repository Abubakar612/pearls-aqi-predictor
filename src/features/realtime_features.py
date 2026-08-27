import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.aqi_calculator import calculate_aqi


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REALTIME_FILE = (
    BASE_DIR
    / "data"
    / "realtime"
    / "lahore_hourly.csv"
)

FEATURE_LIST_FILE = (
    BASE_DIR
    / "models"
    / "production"
    / "feature_list.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "realtime"
    / "latest_features.csv"
)


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate columns FIRST
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

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

    # ========================================================
    # TIME FEATURES
    # ========================================================

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

    # Hour cyclical encoding

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Day-of-week cyclical encoding

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # ========================================================
    # AQI LAG FEATURES
    # ========================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"aqi_lag_{lag}h"] = (
            df["target_aqi"]
            .shift(lag)
        )

    # ========================================================
    # PM2.5 LAG FEATURES
    # ========================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm25_lag_{lag}h"] = (
            df["pm2_5"]
            .shift(lag)
        )

    # ========================================================
    # PM10 LAG FEATURES
    # ========================================================

    for lag in [1, 3, 6, 12, 24, 48]:

        df[f"pm10_lag_{lag}h"] = (
            df["pm10"]
            .shift(lag)
        )

    # ========================================================
    # AQI ROLLING FEATURES
    # ========================================================

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

    # ========================================================
    # PM2.5 ROLLING FEATURES
    # ========================================================

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

    # ========================================================
    # PM10 ROLLING FEATURES
    # ========================================================

    for window in [3, 6, 12, 24]:

        df[f"pm10_rolling_mean_{window}h"] = (
            df["pm10"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # ========================================================
    # AQI CHANGE FEATURES
    # ========================================================

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

    # ========================================================
    # PM2.5 CHANGE FEATURES
    # ========================================================

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

    # ========================================================
    # PM10 CHANGE FEATURES
    # ========================================================

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

    # --------------------------------------------------------
    # Remove duplicate columns again as a safety measure
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("REALTIME FEATURE GENERATION")
    print("=" * 60)

    # ========================================================
    # LOAD REALTIME DATA
    # ========================================================

    print(
        f"\nLoading:\n{REALTIME_FILE}"
    )

    if not REALTIME_FILE.exists():

        raise FileNotFoundError(
            f"Realtime data file not found:\n"
            f"{REALTIME_FILE}"
        )

    df = pd.read_csv(
        REALTIME_FILE
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Remove duplicate columns BEFORE AQI calculation
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # ========================================================
    # CALCULATE AQI
    # ========================================================

    print(
        "\nCalculating AQI..."
    )

    df = calculate_aqi(
        df
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # calculate_aqi() creates target_aqi.
    # Make absolutely sure there is only one copy.
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    if "target_aqi" not in df.columns:

        raise ValueError(
            "target_aqi was not created by AQI calculation."
        )

    print(
        "AQI calculation completed."
    )

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    print(
        "\nCreating realtime features..."
    )

    df = create_features(
        df
    )

    # ========================================================
    # LOAD PRODUCTION FEATURE LIST
    # ========================================================

    if not FEATURE_LIST_FILE.exists():

        raise FileNotFoundError(
            f"Production feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    with open(
        FEATURE_LIST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feature_list = json.load(f)

    # Support:
    # {"features": [...]}
    # OR
    # [...]

    if isinstance(
        feature_list,
        dict
    ):

        if "features" in feature_list:

            feature_list = (
                feature_list["features"]
            )

        else:

            raise ValueError(
                "feature_list.json does not contain "
                "'features'."
            )

    if not isinstance(
        feature_list,
        list
    ):

        raise ValueError(
            "Invalid feature_list.json format."
        )

    # Remove duplicate feature names

    feature_list = list(
        dict.fromkeys(
            feature_list
        )
    )

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    missing = [
        feature
        for feature in feature_list
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing production features:\n"
            +
            "\n".join(
                f"- {feature}"
                for feature in missing
            )
        )

    # ========================================================
    # SELECT LATEST COMPLETE OBSERVATION
    # ========================================================

    required_columns = [
        "timestamp",
        "target_aqi",
        *feature_list
    ]

    # Remove duplicate column names
    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    latest = (
        df[
            required_columns
        ]
        .dropna(
            subset=feature_list
        )
        .sort_values(
            "timestamp"
        )
        .tail(1)
        .copy()
    )

    if latest.empty:

        raise ValueError(
            "No complete feature row available."
        )

    # --------------------------------------------------------
    # Final duplicate-column protection
    # --------------------------------------------------------

    latest = latest.loc[
        :,
        ~latest.columns.duplicated()
    ]

    # ========================================================
    # GET CURRENT AQI SAFELY
    # ========================================================

    if "target_aqi" not in latest.columns:

        raise ValueError(
            "target_aqi missing from latest feature row."
        )

    current_aqi = latest[
        "target_aqi"
    ].iloc[0]

    # Ensure it is a scalar

    if isinstance(
        current_aqi,
        pd.Series
    ):

        current_aqi = current_aqi.iloc[0]

    current_aqi = float(
        current_aqi
    )

    latest_timestamp = latest[
        "timestamp"
    ].iloc[0]

    # ========================================================
    # VALIDATE MODEL FEATURES
    # ========================================================

    model_input = latest[
        feature_list
    ].copy()

    if model_input.isnull().any().any():

        missing_values = (
            model_input
            .isnull()
            .sum()
        )

        missing_values = (
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Missing values detected:\n"
            f"{missing_values}"
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"\nLatest timestamp: "
        f"{latest_timestamp}"
    )

    print(
        f"Current AQI: "
        f"{current_aqi:.2f}"
    )

    print(
        f"\nFeatures available: "
        f"{len(feature_list)}"
    )

    # ========================================================
    # SAVE LATEST FEATURES
    # ========================================================

    latest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nRealtime feature generation completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()