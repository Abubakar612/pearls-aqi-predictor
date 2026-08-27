import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = {
    "24h": "aqi_model_24h.joblib",
    "48h": "aqi_model_48h.joblib",
    "72h": "aqi_model_72h.joblib",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("PRODUCTION MODEL VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load feature list
    # --------------------------------------------------------

    print("\nLoading feature list...")

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        features = json.load(f)

    print(
        f"Features loaded: {len(features)}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Verify features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        print(
            "\nERROR: Missing features:"
        )

        for feature in missing_features:
            print(
                f"  - {feature}"
            )

        return

    print(
        "All required features found."
    )

    # --------------------------------------------------------
    # Use latest available observation
    # --------------------------------------------------------

    latest = df.iloc[-1]

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

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if X.isnull().any().any():

        print(
            "\nERROR: Prediction input "
            "contains missing values."
        )

        print(
            X.isnull().sum()[
                X.isnull().sum() > 0
            ]
        )

        return

    print(
        "\nInput validation: PASSED"
    )

    # --------------------------------------------------------
    # Load and test models
    # --------------------------------------------------------

    predictions = {}

    for horizon, filename in MODELS.items():

        print(
            "\n" + "-" * 60
        )

        print(
            f"Testing {horizon} model"
        )

        model_path = (
            MODEL_DIR
            / filename
        )

        if not model_path.exists():

            print(
                f"ERROR: Model not found:"
                f"\n{model_path}"
            )

            continue

        model = joblib.load(
            model_path
        )

        prediction = model.predict(X)

        prediction = float(
            prediction[0]
        )

        # AQI should not be negative
        prediction = max(
            0,
            prediction
        )

        predictions[horizon] = prediction

        print(
            f"Model loaded: OK"
        )

        print(
            f"Prediction: {prediction:.2f}"
        )

    # --------------------------------------------------------
    # Display forecast
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "3-DAY AQI FORECAST"
    )

    print(
        "=" * 60
    )

    for horizon, prediction in predictions.items():

        print(
            f"{horizon:>3} forecast: "
            f"{prediction:.2f}"
        )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    invalid = [
        value
        for value in predictions.values()
        if not np.isfinite(value)
    ]

    if invalid:

        print(
            "\nWARNING: Invalid prediction detected."
        )

    else:

        print(
            "\nPrediction sanity check: PASSED"
        )

    # --------------------------------------------------------
    # Save test prediction
    # --------------------------------------------------------

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
        f"\nValidation result saved to:"
        f"\n{output_file}"
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