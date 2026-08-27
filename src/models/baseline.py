import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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
    / "baseline_results.csv"
)


TARGETS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h"
]


def calculate_metrics(
    y_true,
    y_pred
):

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


def run_baselines(
    train,
    test
):

    results = []

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

        # -------------------------------------------------
        # Baseline 1
        # Current AQI persistence
        # -------------------------------------------------

        y_true = test[target]

        current_aqi_prediction = (
            test["target_aqi"]
        )

        metrics = calculate_metrics(
            y_true,
            current_aqi_prediction
        )

        print(
            "\nCurrent AQI baseline:"
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

        results.append({
            "target": target,
            "model": "Current AQI Persistence",
            **metrics
        })

        # -------------------------------------------------
        # Baseline 2
        # 24-hour lag
        # -------------------------------------------------

        if "aqi_lag_24h" in test.columns:

            lag_prediction = (
                test["aqi_lag_24h"]
            )

            metrics = calculate_metrics(
                y_true,
                lag_prediction
            )

            print(
                "\n24-hour AQI baseline:"
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

            results.append({
                "target": target,
                "model": "24h AQI Persistence",
                **metrics
            })

    return pd.DataFrame(
        results
    )


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("AQI FORECAST BASELINE EXPERIMENT")
    print("=" * 60)

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    train, test = time_split(
        df
    )

    print(
        "\nTime-based split:"
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows : {len(test)}"
    )

    print(
        "\nTraining period:"
    )

    print(
        train["timestamp"].min(),
        "→",
        train["timestamp"].max()
    )

    print(
        "\nTesting period:"
    )

    print(
        test["timestamp"].min(),
        "→",
        test["timestamp"].max()
    )

    results = run_baselines(
        train,
        test
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BASELINE RESULTS"
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
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()