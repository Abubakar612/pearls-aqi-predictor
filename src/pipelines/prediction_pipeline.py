import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "production"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_list.json"
)

LATEST_DATA = (
    BASE_DIR
    / "data"
    / "realtime"
    / "latest_features.csv"
)


# ============================================================
# MODELS
# ============================================================

MODEL_FILES = {
    "24h": "aqi_model_24h.joblib",
    "48h": "aqi_model_48h.joblib",
    "72h": "aqi_model_72h.joblib",
}


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    models = {}

    for horizon, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[horizon] = joblib.load(
            model_path
        )

    return models


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD LATEST DATA
# ============================================================

def load_latest_data():

    if not LATEST_DATA.exists():

        raise FileNotFoundError(
            f"Feature dataset not found: {LATEST_DATA}"
        )

    df = pd.read_csv(
        LATEST_DATA,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# PREDICT
# ============================================================

def generate_forecast():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("REAL-TIME PREDICTION PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load resources
    # --------------------------------------------------------

    print("\nLoading production models...")

    models = load_models()

    print(
        f"Models loaded: {len(models)}"
    )

    print("\nLoading feature list...")

    features = load_features()

    print(
        f"Features required: {len(features)}"
    )

    # --------------------------------------------------------
    # Load latest observation
    # --------------------------------------------------------

    print("\nLoading latest feature data...")

    df = load_latest_data()

    latest = df.iloc[-1]

    timestamp = latest["timestamp"]

    print(
        f"Latest timestamp: {timestamp}"
    )

    # --------------------------------------------------------
    # Build model input
    # --------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    X = pd.DataFrame(
        [latest[features].values],
        columns=features
    )

    # --------------------------------------------------------
    # Missing-value check
    # --------------------------------------------------------

    if X.isnull().any().any():

        missing_values = (
            X.isnull().sum()
        )

        missing_values = (
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            f"Missing values detected:\n"
            f"{missing_values}"
        )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = {}

    for horizon, model in models.items():

        prediction = model.predict(X)

        prediction = float(
            prediction[0]
        )

        prediction = max(
            0,
            prediction
        )

        predictions[horizon] = {
            "aqi": round(
                prediction,
                2
            ),
            "category": get_aqi_category(
                prediction
            )
        }

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = {

        "city": "Lahore",

        "country": "Pakistan",

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "data_timestamp": str(
            timestamp
        ),

        "current_aqi": float(
            latest["target_aqi"]
        ) if "target_aqi" in latest
        else None,

        "forecast": {

            "24h": predictions["24h"],

            "48h": predictions["48h"],

            "72h": predictions["72h"]

        }

    }

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "AQI FORECAST"
    )

    print(
        "=" * 60
    )

    print(
        f"City: {result['city']}"
    )

    print(
        f"Current AQI: "
        f"{result['current_aqi']}"
    )

    for horizon in ["24h", "48h", "72h"]:

        forecast = result[
            "forecast"
        ][horizon]

        print(
            f"{horizon}: "
            f"{forecast['aqi']} "
            f"({forecast['category']})"
        )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = generate_forecast()

    print(
        "\nPrediction pipeline completed."
    )