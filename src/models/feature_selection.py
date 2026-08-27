import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import (
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

RESULTS_DIR = (
    BASE_DIR
    / "models"
    / "experiments"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "feature_selection_results.csv"
)


TARGETS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h"
]


FEATURE_COUNTS = [
    15,
    25,
    40,
    73
]


# ============================================================
# METRICS
# ============================================================

def metrics(y_true, y_pred):

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
    print("FEATURE SELECTION EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Load importance
    # --------------------------------------------------------

    importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    importance = importance.sort_values(
        "permutation_importance",
        ascending=False
    )

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
    # All valid features
    # --------------------------------------------------------

    excluded = [
        "timestamp",
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h"
    ]

    all_features = [
        c
        for c in df.columns
        if c not in excluded
    ]

    results = []

    # --------------------------------------------------------
    # Feature counts
    # --------------------------------------------------------

    for count in FEATURE_COUNTS:

        if count == 73:

            selected_features = all_features

        else:

            selected_features = (
                importance
                .head(count)
                ["feature"]
                .tolist()
            )

            selected_features = [
                f
                for f in selected_features
                if f in all_features
            ]

        print(
            "\n" + "=" * 60
        )

        print(
            f"FEATURE SET: TOP {len(selected_features)}"
        )

        print(
            "=" * 60
        )

        print(
            selected_features
        )

        X_train = train[
            selected_features
        ]

        X_test = test[
            selected_features
        ]

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        models = {

            "Extra Trees": ExtraTreesRegressor(
                n_estimators=400,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=2,
                min_samples_leaf=10,
                subsample=1.0,
                random_state=42
            )
        }

        # ----------------------------------------------------
        # Targets
        # ----------------------------------------------------

        for target in TARGETS:

            y_train = train[
                target
            ]

            y_test = test[
                target
            ]

            for model_name, model in models.items():

                print(
                    f"\nTraining "
                    f"{model_name} "
                    f"→ {target}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                prediction = model.predict(
                    X_test
                )

                mae, rmse, r2 = metrics(
                    y_test,
                    prediction
                )

                print(
                    f"MAE  : {mae:.4f}"
                )

                print(
                    f"RMSE : {rmse:.4f}"
                )

                print(
                    f"R²   : {r2:.4f}"
                )

                results.append({

                    "feature_count":
                        len(selected_features),

                    "model":
                        model_name,

                    "target":
                        target,

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2
                })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
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
        "FEATURE SELECTION RESULTS"
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
        f"\nSaved to:"
        f"\n{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()