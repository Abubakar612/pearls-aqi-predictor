import json
import os
from io import BytesIO

import boto3
import joblib
import pandas as pd


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

MODEL_PREFIX = os.environ.get(
    "MODEL_PREFIX",
    "models/production/"
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

FEATURE_FILE = "feature_list.json"


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# LOAD MODEL FROM S3
# ============================================================

def load_model_from_s3(filename):

    key = f"{MODEL_PREFIX}{filename}"

    print(
        f"Loading model from "
        f"s3://{S3_BUCKET}/{key}"
    )

    response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )

    model_bytes = response["Body"].read()

    return joblib.load(
        BytesIO(model_bytes)
    )


# ============================================================
# LOAD PRODUCTION MODELS
# ============================================================

def load_production_models():

    models = {}

    for horizon, filename in MODEL_FILES.items():

        models[horizon] = load_model_from_s3(
            filename
        )

    print(
        f"Loaded {len(models)} production models."
    )

    return models


# ============================================================
# LOAD FEATURE LIST
# ============================================================

def load_feature_list():

    key = f"{MODEL_PREFIX}{FEATURE_FILE}"

    print(
        f"Loading feature list from "
        f"s3://{S3_BUCKET}/{key}"
    )

    response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )

    content = response["Body"].read().decode(
        "utf-8"
    )

    features = json.loads(content)

    if isinstance(features, dict):

        if "features" in features:
            features = features["features"]

        else:
            raise ValueError(
                "feature_list.json does not contain "
                "'features'."
            )

    if not isinstance(features, list) or not features:

        raise ValueError(
            "Production feature list is empty or invalid."
        )

    print(
        f"Loaded {len(features)} features."
    )

    return features


# ============================================================
# LOAD LATEST FEATURES FROM S3
# ============================================================

def load_latest_features():

    print(
        f"Loading latest features from "
        f"s3://{S3_BUCKET}/{FEATURES_KEY}"
    )

    response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=FEATURES_KEY
    )

    df = pd.read_csv(
        response["Body"]
    )

    if df.empty:

        raise ValueError(
            "Latest features file is empty."
        )

    return df


# ============================================================
# SAVE FORECAST TO S3
# ============================================================

def save_forecast_to_s3(result):

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
        Body=forecast_json.encode("utf-8"),
        ContentType="application/json"
    )

    print(
        "Forecast successfully saved to S3."
    )


# ============================================================
# LAMBDA RESPONSE
# ============================================================

def response(status_code, body):

    return {

        "statusCode": status_code,

        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },

        "body": json.dumps(body)
    }


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(event, context):

    try:

        print("=" * 60)
        print("PEARLS AQI PREDICTION LAMBDA")
        print("=" * 60)

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
        # Load feature list
        # ----------------------------------------------------

        feature_list = load_feature_list()

        # ----------------------------------------------------
        # Load latest feature data
        # ----------------------------------------------------

        df = load_latest_features()

        print(
            f"Feature dataset shape: {df.shape}"
        )

        # ----------------------------------------------------
        # Validate required features
        # ----------------------------------------------------

        missing = [
            feature
            for feature in feature_list
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features in S3 file: "
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # Use latest available row
        # ----------------------------------------------------

        latest_row = (
            df
            .sort_values("timestamp")
            .iloc[-1]
        )

        # ----------------------------------------------------
        # Build model input
        # ----------------------------------------------------

        X = pd.DataFrame(
            [
                [
                    latest_row[feature]
                    for feature in feature_list
                ]
            ],
            columns=feature_list
        )

        # ----------------------------------------------------
        # Validate null values
        # ----------------------------------------------------

        if X.isnull().any().any():

            missing_values = X.isnull().sum()

            missing_values = (
                missing_values[
                    missing_values > 0
                ]
            )

            raise ValueError(
                "Null values detected in model input: "
                + str(
                    missing_values.to_dict()
                )
            )

        # ----------------------------------------------------
        # Validate numeric input
        # ----------------------------------------------------

        X = X.astype(float)

        # ----------------------------------------------------
        # Current AQI
        # ----------------------------------------------------

        current_aqi = None

        if "current_aqi" in df.columns:

            current_aqi = float(
                latest_row["current_aqi"]
            )

        elif "target_aqi" in df.columns:

            current_aqi = float(
                latest_row["target_aqi"]
            )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = None

        if "timestamp" in df.columns:

            timestamp = str(
                latest_row["timestamp"]
            )

        # ----------------------------------------------------
        # Pollutants
        # ----------------------------------------------------

        pollutants = {

            "pm25": (
                float(latest_row["pm2_5"])
                if "pm2_5" in df.columns
                else None
            ),

            "pm10": (
                float(latest_row["pm10"])
                if "pm10" in df.columns
                else None
            ),

            "no2": (
                float(latest_row["no2"])
                if "no2" in df.columns
                else None
            ),

            "o3": (
                float(latest_row["o3"])
                if "o3" in df.columns
                else None
            ),

            "co": (
                float(latest_row["co"])
                if "co" in df.columns
                else None
            ),
        }

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        models = load_production_models()

        # ----------------------------------------------------
        # Generate forecasts
        # ----------------------------------------------------

        forecasts = {}

        for horizon, model in models.items():

            print(
                f"Generating {horizon} forecast..."
            )

            prediction = model.predict(X)

            prediction = float(
                prediction[0]
            )

            prediction = max(
                0.0,
                prediction
            )

            forecasts[horizon] = {

                "aqi": round(
                    prediction,
                    2
                ),

                "category": get_aqi_category(
                    prediction
                )
            }

            print(
                f"{horizon} AQI: "
                f"{prediction:.2f}"
            )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = {

            "city": "Lahore",

            "country": "Pakistan",

            "current_aqi": current_aqi,

            "data_timestamp": timestamp,

            "pollutants": pollutants,

            "forecast": forecasts
        }

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        save_forecast_to_s3(
            result
        )

        print(
            "Prediction pipeline completed successfully."
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return response(
            200,
            result
        )

    except Exception as e:

        print(
            "ERROR:"
        )

        print(
            str(e)
        )

        return response(
            500,
            {
                "error": "Prediction failed.",
                "message": str(e)
            }
        )