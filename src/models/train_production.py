import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor


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

IMPORTANCE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_analysis"
    / "feature_importance_24h.csv"
)

PRODUCTION_DIR = (
    BASE_DIR
    / "models"
    / "production"
)

FEATURE_FILE = (
    PRODUCTION_DIR
    / "feature_list.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}

TOP_FEATURES = 25


# ============================================================
# MODEL
# ============================================================

def create_model():

    return ExtraTreesRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("FINAL PRODUCTION MODEL TRAINING")
    print("=" * 60)

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
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
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Load feature importance
    # --------------------------------------------------------

    importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    importance = importance.sort_values(
        "permutation_importance",
        ascending=False
    )

    excluded = [
        "timestamp",
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h"
    ]

    valid_features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    selected_features = [
        feature
        for feature in importance.head(TOP_FEATURES)["feature"]
        if feature in valid_features
    ]

    print(
        f"\nSelected features: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected_features,
            f,
            indent=4
        )

    print(
        f"\nFeature list saved to:"
        f"\n{FEATURE_FILE}"
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    X = df[
        selected_features
    ]

    for horizon, target in TARGETS.items():

        print(
            "\n" + "=" * 60
        )

        print(
            f"TRAINING FINAL {horizon} MODEL"
        )

        print(
            "=" * 60
        )

        y = df[target]

        model = create_model()

        model.fit(
            X,
            y
        )

        output_file = (
            PRODUCTION_DIR
            / f"aqi_model_{horizon}.joblib"
        )

        joblib.dump(
            model,
            output_file
        )

        print(
            f"Model saved to:"
            f"\n{output_file}"
        )

        print(
            f"Training samples: {len(X)}"
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PRODUCTION TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nProduction models:"
    )

    for horizon in TARGETS:

        print(
            f"  models/production/"
            f"aqi_model_{horizon}.joblib"
        )


if __name__ == "__main__":
    main()