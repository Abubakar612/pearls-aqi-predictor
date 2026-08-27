import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

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
    / "tuned_model_results.csv"
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

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
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

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGETS
        and column != "timestamp"
    ]

    X_train = train[
        feature_columns
    ]

    X_test = test[
        feature_columns
    ]

    print(
        f"\nTraining rows: {len(train)}"
    )

    print(
        f"Testing rows : {len(test)}"
    )

    print(
        f"Features     : {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Time-series cross validation
    # --------------------------------------------------------

    cv = TimeSeriesSplit(
        n_splits=4
    )

    # --------------------------------------------------------
    # Hyperparameter spaces
    # --------------------------------------------------------

    model_configs = {

        "Random Forest": {

            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),

            "params": {

                "n_estimators": [
                    200,
                    400,
                    600
                ],

                "max_depth": [
                    8,
                    12,
                    16,
                    20,
                    None
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "max_features": [
                    0.5,
                    0.7,
                    1.0
                ]
            }
        },

        "Extra Trees": {

            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1
            ),

            "params": {

                "n_estimators": [
                    200,
                    400,
                    600
                ],

                "max_depth": [
                    8,
                    12,
                    16,
                    20,
                    None
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "max_features": [
                    0.5,
                    0.7,
                    1.0
                ]
            }
        },

        "Gradient Boosting": {

            "model": GradientBoostingRegressor(
                random_state=42
            ),

            "params": {

                "n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.1
                ],

                "max_depth": [
                    2,
                    3,
                    4,
                    5
                ],

                "min_samples_leaf": [
                    2,
                    5,
                    10
                ],

                "subsample": [
                    0.7,
                    0.85,
                    1.0
                ]
            }
        }
    }

    all_results = []

    # --------------------------------------------------------
    # Tune each target
    # --------------------------------------------------------

    for target in TARGETS:

        print(
            "\n" + "=" * 60
        )

        print(
            f"TUNING TARGET: {target}"
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

        for model_name, config in model_configs.items():

            print(
                f"\nTuning {model_name}..."
            )

            search = RandomizedSearchCV(

                estimator=config["model"],

                param_distributions=config["params"],

                n_iter=12,

                scoring="neg_mean_absolute_error",

                cv=cv,

                random_state=42,

                n_jobs=-1,

                verbose=1
            )

            search.fit(
                X_train,
                y_train
            )

            best_model = search.best_estimator_

            predictions = best_model.predict(
                X_test
            )

            mae, rmse, r2 = calculate_metrics(
                y_test,
                predictions
            )

            print(
                "\nBest parameters:"
            )

            print(
                search.best_params_
            )

            print(
                f"\nTest MAE  : {mae:.4f}"
            )

            print(
                f"Test RMSE : {rmse:.4f}"
            )

            print(
                f"Test R²   : {r2:.4f}"
            )

            all_results.append({

                "target": target,

                "model": model_name,

                "MAE": mae,

                "RMSE": rmse,

                "R2": r2,

                "best_parameters":
                    str(search.best_params_)
            })

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results = results.sort_values(
        [
            "target",
            "MAE"
        ]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TUNED MODEL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()