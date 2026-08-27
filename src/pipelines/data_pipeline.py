"""
Production data preparation pipeline.

Builds the ML dataset from the raw historical source:
1. Historical JSON cleaning
2. AQI calculation
3. Forecast target generation
4. Forecast feature engineering
"""

from src.data.historical_cleaner import (
    clean_historical_data
)

from src.features.aqi_calculator import (
    calculate_aqi
)

from src.features.forecast_targets import (
    create_forecast_targets
)

from src.features.forecast_features import (
    create_features,
    select_features
)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = (
    BASE_DIR / "data" / "historical"
)

PROCESSED_DIR = (
    BASE_DIR / "data" / "processed"
)


def run_data_pipeline():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Find historical source
    # --------------------------------------------------------

    files = sorted(
        HISTORICAL_DIR.glob(
            "historical_*.json"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No historical JSON dataset found."
        )

    input_file = files[-1]

    print(
        f"\nUsing historical dataset:\n"
        f"{input_file}"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 2. Clean historical data
    # --------------------------------------------------------

    print(
        "\n[1/4] Cleaning historical data..."
    )

    df = clean_historical_data(
        input_file
    )

    cleaned_file = (
        PROCESSED_DIR
        / "historical_cleaned.csv"
    )

    df.to_csv(
        cleaned_file,
        index=False
    )

    print(
        f"Saved: {cleaned_file}"
    )

    # --------------------------------------------------------
    # 3. Calculate AQI
    # --------------------------------------------------------

    print(
        "\n[2/4] Calculating AQI..."
    )

    df = calculate_aqi(df)

    aqi_file = (
        PROCESSED_DIR
        / "historical_with_aqi.csv"
    )

    df.to_csv(
        aqi_file,
        index=False
    )

    print(
        f"Saved: {aqi_file}"
    )

    # --------------------------------------------------------
    # 4. Create forecast dataset
    # --------------------------------------------------------

    print(
        "\n[3/4] Creating forecast targets..."
    )

    df = create_forecast_targets(
        df
    )

    df = df.dropna(
        subset=[
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ).reset_index(drop=True)

    forecast_file = (
        PROCESSED_DIR
        / "forecast_dataset.csv"
    )

    df.to_csv(
        forecast_file,
        index=False
    )

    print(
        f"Saved: {forecast_file}"
    )

    # --------------------------------------------------------
    # 5. Create ML dataset
    # --------------------------------------------------------

    print(
        "\n[4/4] Creating ML features..."
    )

    df = create_features(
        df
    )

    feature_columns = select_features(
        df
    )

    target_columns = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    required_columns = [
        "timestamp",
        "target_aqi",
        *feature_columns,
        *target_columns,
    ]

    ml_df = df[
        required_columns
    ].copy()

    ml_df = ml_df.dropna(
        subset=(
            feature_columns
            + target_columns
        )
    )

    ml_df = (
        ml_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    ml_file = (
        PROCESSED_DIR
        / "ml_dataset.csv"
    )

    ml_df.to_csv(
        ml_file,
        index=False
    )

    print(
        f"Saved: {ml_file}"
    )

    print(
        f"\nFinal ML dataset shape: "
        f"{ml_df.shape}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        "\nDATA PREPARATION COMPLETED"
    )


if __name__ == "__main__":
    run_data_pipeline()