import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "production"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_list.json"
)

MODELS = {
    "24h": "aqi_model_24h.joblib",
    "48h": "aqi_model_48h.joblib",
    "72h": "aqi_model_72h.joblib",
}


def main():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("PRODUCTION MODEL VALIDATION")
    print("=" * 60)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_FILE}"
        )

    print("\nLoading feature list...")

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        features = json.load(f)

    if not isinstance(features, list) or not features:
        raise ValueError(
            "Production feature list is empty or invalid."
        )

    print(
        f"Features loaded: {len(features)}"
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(
        f"Dataset rows: {len(df)}"
    )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing production features: "
            + ", ".join(missing_features)
        )

    print(
        "All required features found."
    )

    # Use the latest complete observation.
    latest_rows = (
        df
        .dropna(subset=features)
        .sort_values("timestamp")
    )

    if latest_rows.empty:
        raise ValueError(
            "No complete observation is available."
        )

    latest = latest_rows.iloc[-1]

    X = pd.DataFrame(
        [latest[features].values],
        columns=features
    )

    print(
        "\nPrediction input timestamp:"
    )

    print(
        latest["timestamp"]
    )

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
            f"Prediction input contains missing values:\n"
            f"{missing_values}"
        )

    print(
        "\nInput validation: PASSED"
    )

    predictions = {}

    for horizon, filename in MODELS.items():

        print(
            "\n" + "-" * 60
        )

        print(
            f"Testing {horizon} model"
        )

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model = joblib.load(
            model_path
        )

        prediction = model.predict(X)

        if len(prediction) != 1:
            raise ValueError(
                f"{horizon} model returned an invalid prediction."
            )

        prediction = float(
            prediction[0]
        )

        prediction = max(
            0,
            prediction
        )

        if not np.isfinite(prediction):
            raise ValueError(
                f"{horizon} model produced an invalid prediction."
            )

        predictions[horizon] = prediction

        print(
            "Model loaded: OK"
        )

        print(
            f"Prediction: {prediction:.2f}"
        )

    if len(predictions) != 3:
        raise ValueError(
            "Not all production models were validated."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "3-DAY AQI FORECAST"
    )

    print(
        "=" * 60
    )

    for horizon in ["24h", "48h", "72h"]:
        print(
            f"{horizon:>3} forecast: "
            f"{predictions[horizon]:.2f}"
        )

    print(
        "\nPrediction sanity check: PASSED"
    )

    output = {
        "input_timestamp": str(
            latest["timestamp"]
        ),
        "predictions": predictions
    }

    output_file = (
        MODEL_DIR
        / "validation_prediction.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(
        f"\nValidation result saved to:\n"
        f"{output_file}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PRODUCTION VALIDATION COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()