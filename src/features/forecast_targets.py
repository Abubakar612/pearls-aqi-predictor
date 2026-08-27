import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "historical_with_aqi.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_dataset.csv"
)


def create_forecast_targets(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create future AQI targets for 1, 2 and 3 days ahead.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Future AQI targets
    # --------------------------------------------------

    df["target_aqi_24h"] = (
        df["target_aqi"].shift(-24)
    )

    df["target_aqi_48h"] = (
        df["target_aqi"].shift(-48)
    )

    df["target_aqi_72h"] = (
        df["target_aqi"].shift(-72)
    )

    return df


def validate_targets(
    df: pd.DataFrame
) -> None:

    print("\nTarget validation")
    print("-" * 40)

    target_columns = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    for column in target_columns:

        print(f"\n{column}")

        print(
            f"Missing: "
            f"{df[column].isna().sum()}"
        )

        print(
            f"Min: "
            f"{df[column].min()}"
        )

        print(
            f"Max: "
            f"{df[column].max()}"
        )


if __name__ == "__main__":

    print("=" * 60)
    print("3-DAY AQI FORECAST TARGET GENERATION")
    print("=" * 60)

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    df = create_forecast_targets(
        df
    )

    validate_targets(
        df
    )

    # The last 72 rows don't have complete
    # 24/48/72-hour future targets.
    df = df.dropna(
        subset=[
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ).reset_index(drop=True)

    print(
        f"\nFinal training shape: "
        f"{df.shape}"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nForecast dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nSample target mapping:"
    )

    print(
        df[
            [
                "timestamp",
                "target_aqi",
                "target_aqi_24h",
                "target_aqi_48h",
                "target_aqi_72h",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )