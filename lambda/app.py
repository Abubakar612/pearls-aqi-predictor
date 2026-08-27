import json
import os
import boto3
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/var/task")

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "production"
)


# ============================================================
# AWS / S3 CONFIGURATION
# ============================================================

S3_BUCKET = os.environ["S3_BUCKET"]

FEATURES_KEY = os.environ.get(
    "FEATURES_KEY",
    "realtime/latest_features.csv"
)

FORECAST_KEY = os.environ.get(
    "FORECAST_KEY",
    "realtime/latest_forecast.json"
)

s3_client = boto3.client("s3")


# ============================================================
# MODEL FILES
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
# LOAD PRODUCTION MODELS
# ============================================================

print("Loading production models...")

MODELS = {}

for horizon, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    MODELS[horizon] = joblib.load(
        model_path
    )

print(
    f"Loaded {len(MODELS)} production models."
)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

FEATURE_FILE = (
    MODEL_DIR
    / "feature_list.json"
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    FEATURE_LIST = json.load(f)


if isinstance(
    FEATURE_LIST,
    dict
):

    FEATURE_LIST = FEATURE_LIST["features"]


print(
    f"Loaded {len(FEATURE_LIST)} features."
)


# ============================================================
# STANDARD RESPONSE
# ============================================================

def response(
    status_code,
    body
):

    return {

        "statusCode":
        status_code,

        "headers": {

            "Content-Type":
            "application/json",

            "Access-Control-Allow-Origin":
            "*"

        },

        "body":
        json.dumps(
            body
        )

    }


# ============================================================
# LOAD LATEST FEATURES FROM S3
# ============================================================

def load_latest_features():

    print(
        f"Loading latest features from "
        f"s3://{S3_BUCKET}/{FEATURES_KEY}"
    )

    s3_response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=FEATURES_KEY
    )

    df = pd.read_csv(
        s3_response["Body"]
    )

    if df.empty:

        raise ValueError(
            "Latest features file is empty."
        )

    # --------------------------------------------------------
    # Verify required model features
    # --------------------------------------------------------

    missing = [

        feature

        for feature in FEATURE_LIST

        if feature not in df.columns

    ]

    if missing:

        raise ValueError(
            "Missing model features in S3 file: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Use the latest available row
    # --------------------------------------------------------

    latest_row = df.iloc[-1]

    # --------------------------------------------------------
    # Extract model features
    # --------------------------------------------------------

    features = {

        feature:
        latest_row[feature]

        for feature in FEATURE_LIST

    }

    # --------------------------------------------------------
    # Determine current AQI
    # --------------------------------------------------------

    current_aqi = None

    if "current_aqi" in df.columns:

        current_aqi = latest_row[
            "current_aqi"
        ]

    elif "target_aqi" in df.columns:

        current_aqi = latest_row[
            "target_aqi"
        ]

    # --------------------------------------------------------
    # Extract timestamp
    # --------------------------------------------------------

    timestamp = None

    if "timestamp" in df.columns:

        timestamp = str(
            latest_row[
                "timestamp"
            ]
        )

        pollutants = {
        "pm25": float(latest_row["pm25_lag_12h"])
            if "pm25_lag_12h" in df.columns else None,

        "pm10": float(latest_row["pm10_rolling_mean_3h"])
            if "pm10_rolling_mean_3h" in df.columns else None,

        "no2": float(latest_row["no2"])
            if "no2" in df.columns else None,

        "o3": float(latest_row["o3"])
            if "o3" in df.columns else None,

        "co": float(latest_row["co"])
            if "co" in df.columns else None,
         }

    return (
        features,
        current_aqi,
        timestamp,
        pollutants
    )


# ============================================================
# SAVE FORECAST TO S3
# ============================================================

def save_forecast_to_s3(
    result
):

    print(
        f"Saving forecast to "
        f"s3://{S3_BUCKET}/{FORECAST_KEY}"
    )

    forecast_json = json.dumps(
        result,
        indent=2
    )

    s3_client.put_object(

        Bucket=S3_BUCKET,

        Key=FORECAST_KEY,

        Body=forecast_json.encode(
            "utf-8"
        ),

        ContentType="application/json"

    )

    print(
        "Forecast successfully saved to S3."
    )


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(
    event,
    context
):

    try:

        # ----------------------------------------------------
        # Log incoming event
        # ----------------------------------------------------

        print(
            "Received event:"
        )

        print(
            json.dumps(
                event,
                default=str
            )
        )

        # ----------------------------------------------------
        # Load latest production features
        # ----------------------------------------------------

        (
            features,
            current_aqi,
            timestamp,
            pollutants
        ) = load_latest_features()

        print(
            f"Latest feature timestamp: "
            f"{timestamp}"
        )

        print(
            f"Current AQI: "
            f"{current_aqi}"
        )

        # ----------------------------------------------------
        # Build model input
        # ----------------------------------------------------

        X = pd.DataFrame(
            [
                {
                    feature:
                    features[feature]

                    for feature in FEATURE_LIST
                }
            ]
        )

        # ----------------------------------------------------
        # Validate null values
        # ----------------------------------------------------

        if X.isnull().any().any():

            missing_values = (
                X.isnull()
                .sum()
            )

            missing_values = (
                missing_values[
                    missing_values > 0
                ]
            )

            return response(
                500,
                {
                    "error":
                    "Null values detected "
                    "in latest S3 features.",

                    "features":
                    missing_values.to_dict()
                }
            )

        # ----------------------------------------------------
        # Validate numeric model input
        # ----------------------------------------------------

        try:

            X = X.astype(float)

        except Exception as e:

            return response(
                500,
                {
                    "error":
                    "Non-numeric model feature "
                    "detected.",

                    "message":
                    str(e)
                }
            )

        # ----------------------------------------------------
        # Generate forecasts
        # ----------------------------------------------------

        forecasts = {}

        for horizon, model in MODELS.items():

            print(
                f"Generating {horizon} forecast..."
            )

            prediction = model.predict(
                X
            )

            prediction = float(
                prediction[0]
            )

            # Prevent negative AQI
            prediction = max(
                0,
                prediction
            )

            forecasts[horizon] = {

                "aqi":
                round(
                    prediction,
                    2
                ),

                "category":
                get_aqi_category(
                    prediction
                )

            }

            print(
                f"{horizon} AQI: "
                f"{prediction:.2f}"
            )

        # ----------------------------------------------------
        # Build final forecast result
        # ----------------------------------------------------

        result = {

         "city": "Lahore",

        "country": "Pakistan",

        "current_aqi":
            float(current_aqi)
            if current_aqi is not None
            else None,

        "data_timestamp":
            timestamp,

        "pollutants":
            pollutants,

        "forecast":
            forecasts
}

        # ----------------------------------------------------
        # SAVE FORECAST TO S3
        # ----------------------------------------------------

        save_forecast_to_s3(
            result
        )

        print(
            "Prediction pipeline completed successfully."
        )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return response(
            200,
            result
        )

    except Exception as e:

        # ----------------------------------------------------
        # Log error
        # ----------------------------------------------------

        print(
            "ERROR:"
        )

        print(
            str(e)
        )

        # ----------------------------------------------------
        # Return error response
        # ----------------------------------------------------

        return response(
            500,
            {

                "error":
                "Prediction failed.",

                "message":
                str(e)

            }
        )