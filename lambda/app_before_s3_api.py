import json
import os
from io import BytesIO
import joblib
import boto3
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
# AWS CONFIGURATION
# ============================================================

S3_BUCKET = os.environ.get(
    "S3_BUCKET",
    "pearls-aqi-predictor-071493957773"
)

FEATURES_KEY = os.environ.get(
    "FEATURES_KEY",
    "realtime/latest_features.csv"
)

FORECAST_KEY = os.environ.get(
    "FORECAST_KEY",
    "realtime/latest_forecast.json"
)

s3 = boto3.client("s3")


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
# LOAD MODELS
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
# RESPONSE
# ============================================================

def response(
    status_code,
    body
):

    return {
        "statusCode": status_code,

        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },

        "body": json.dumps(
            body
        )
    }


# ============================================================
# LOAD FEATURES FROM S3
# ============================================================

def load_features_from_s3():

    print(
        f"Reading s3://{S3_BUCKET}/{FEATURES_KEY}"
    )

    obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key=FEATURES_KEY
    )

    df = pd.read_csv(
        BytesIO(
            obj["Body"].read()
        )
    )

    if df.empty:

        raise ValueError(
            "latest_features.csv is empty."
        )

    return df


# ============================================================
# GENERATE FORECAST
# ============================================================

def generate_forecast(
    current_aqi,
    timestamp,
    features
):

    missing = [
        feature
        for feature in FEATURE_LIST
        if feature not in features
    ]

    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    X = pd.DataFrame(
        [
            {
                feature:
                features[feature]
                for feature in FEATURE_LIST
            }
        ]
    )

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

        raise ValueError(
            "Null values detected: "
            + str(
                missing_values.to_dict()
            )
        )

    forecasts = {}

    for horizon, model in MODELS.items():

        prediction = model.predict(
            X
        )

        prediction = float(
            prediction[0]
        )

        prediction = max(
            0,
            prediction
        )

        forecasts[horizon] = {

            "aqi": round(
                prediction,
                2
            ),

            "category":
            get_aqi_category(
                prediction
            )
        }

    return {

        "city": "Lahore",

        "country": "Pakistan",

        "current_aqi":
        float(current_aqi)
        if current_aqi is not None
        else None,

        "data_timestamp":
        timestamp,

        "forecast":
        forecasts
    }


# ============================================================
# SAVE FORECAST TO S3
# ============================================================

def save_forecast_to_s3(
    result
):

    s3.put_object(

        Bucket=S3_BUCKET,

        Key=FORECAST_KEY,

        Body=json.dumps(
            result,
            indent=2
        ).encode("utf-8"),

        ContentType="application/json"
    )

    print(
        f"Saved forecast to "
        f"s3://{S3_BUCKET}/{FORECAST_KEY}"
    )


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(
    event,
    context
):

    try:

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
        # Determine whether this is an API request
        # or an automatic S3-based prediction
        # ----------------------------------------------------

        body = event.get(
            "body"
        )

        if body is None:

            body = event

        elif isinstance(
            body,
            str
        ):

            body = json.loads(
                body
            )

        # ----------------------------------------------------
        # Automatic mode
        #
        # EventBridge sends an empty event.
        # In that case load latest_features.csv from S3.
        # ----------------------------------------------------

        if (
            not isinstance(body, dict)
            or
            "features" not in body
        ):

            print(
                "No feature payload detected."
            )

            print(
                "Loading latest features from S3..."
            )

            latest = (
                load_features_from_s3()
            )

            latest = latest.tail(
                1
            )

            if latest.empty:

                raise ValueError(
                    "No latest feature row found."
                )

            row = latest.iloc[0]

            current_aqi = (
                row.get(
                    "target_aqi"
                )
            )

            timestamp = (
                row.get(
                    "timestamp"
                )
            )

            features = {
                feature:
                float(row[feature])
                for feature in FEATURE_LIST
            }

            result = generate_forecast(
                current_aqi,
                timestamp,
                features
            )

            save_forecast_to_s3(
                result
            )

            return response(
                200,
                result
            )

        # ----------------------------------------------------
        # API MODE
        # ----------------------------------------------------

        current_aqi = body.get(
            "current_aqi"
        )

        timestamp = body.get(
            "timestamp"
        )

        features = body.get(
            "features"
        )

        if features is None:

            return response(
                400,
                {
                    "error":
                    "Missing 'features' object."
                }
            )

        result = generate_forecast(
            current_aqi,
            timestamp,
            features
        )

        return response(
            200,
            result
        )

    except Exception as e:

        print(
            f"ERROR: {str(e)}"
        )

        return response(
            500,
            {
                "error":
                "Prediction failed.",

                "message":
                str(e)
            }
        )
