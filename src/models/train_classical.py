import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "models"
    / "experiments"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "classical_model_results.csv"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h"
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# TIME SPLIT
# ============================================================

def time_split(df):

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    return train, test


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "Ridge": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                )
            )
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            loss="huber"
        )
    }

    return models


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("CLASSICAL ML MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"\nDataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Remove timestamp
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGETS
        and column != "timestamp"
    ]

    print(
        f"\nNumber of input features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    train, test = time_split(
        df
    )

    print("\nTime-based split")

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows : {len(test)}"
    )

    print(
        f"\nTraining period:"
    )

    print(
        train["timestamp"].min(),
        "→",
        train["timestamp"].max()
    )

    print(
        f"\nTesting period:"
    )

    print(
        test["timestamp"].min(),
        "→",
        test["timestamp"].max()
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = train[
        feature_columns
    ]

    X_test = test[
        feature_columns
    ]

    models = create_models()

    all_results = []

    # --------------------------------------------------------
    # Train each forecast horizon
    # --------------------------------------------------------

    for target in TARGETS:

        print(
            "\n" + "=" * 60
        )

        print(
            f"TARGET: {target}"
        )

        print(
            "=" * 60
        )

        y_train = train[
            target
        ]

        y_test = test[
            target
        ]

        for model_name, model in models.items():

            print(
                f"\nTraining: {model_name}"
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test,
                predictions
            )

            print(
                f"MAE  : {metrics['MAE']:.4f}"
            )

            print(
                f"RMSE : {metrics['RMSE']:.4f}"
            )

            print(
                f"R²   : {metrics['R2']:.4f}"
            )

            all_results.append({

                "target": target,

                "model": model_name,

                "MAE": metrics["MAE"],

                "RMSE": metrics["RMSE"],

                "R2": metrics["R2"]
            })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        [
            "target",
            "MAE"
        ]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASSICAL MODEL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()