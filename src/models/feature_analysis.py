import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from sklearn.inspection import permutation_importance


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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "feature_importance_24h.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "target_aqi_24h"


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("FEATURE IMPORTANCE ANALYSIS")
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

    target_columns = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns
        and column != "timestamp"
    ]

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET
    ]

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print(
        f"Features        : {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Built-in feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "feature": feature_columns,

        "importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        "\nTop 25 features:"
    )

    print(
        importance.head(25).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    print(
        "\nCalculating permutation importance..."
    )

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({

        "feature": feature_columns,

        "permutation_importance":
            permutation.importances_mean

    })

    permutation_df = permutation_df.sort_values(
        "permutation_importance",
        ascending=False
    )

    print(
        "\nTop 25 permutation features:"
    )

    print(
        permutation_df.head(25).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_df = importance.merge(
        permutation_df,
        on="feature"
    )

    final_df = final_df.sort_values(
        "permutation_importance",
        ascending=False
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nFeature analysis saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\nFeature analysis completed."
    )


if __name__ == "__main__":
    main()