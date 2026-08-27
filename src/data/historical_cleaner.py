import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = BASE_DIR / "data" / "historical"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_historical_data(file_path: str | Path) -> dict:
    """Load historical Open-Meteo JSON."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Historical file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_weather(data: dict) -> pd.DataFrame:
    """Extract hourly weather observations."""

    hourly = data["weather"]["hourly"]

    df = pd.DataFrame({
        "timestamp": hourly["time"],

        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "dew_point": hourly["dew_point_2m"],
        "feels_like": hourly["apparent_temperature"],

        "pressure": hourly["pressure_msl"],
        "surface_pressure": hourly["surface_pressure"],

        "clouds": hourly["cloud_cover"],

        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "wind_gust": hourly["wind_gusts_10m"],

        "precipitation": hourly["precipitation"],
    })

    return df


def extract_air_quality(data: dict) -> pd.DataFrame:
    """Extract hourly air-quality observations."""

    hourly = data["air_quality"]["hourly"]

    df = pd.DataFrame({
        "timestamp": hourly["time"],

        "pm10": hourly["pm10"],
        "pm2_5": hourly["pm2_5"],

        "co": hourly["carbon_monoxide"],

        "no2": hourly["nitrogen_dioxide"],

        "so2": hourly["sulphur_dioxide"],

        "o3": hourly["ozone"],
    })

    return df


def clean_historical_data(
    file_path: str | Path
) -> pd.DataFrame:
    """Create the complete historical dataset."""

    data = load_historical_data(file_path)

    weather_df = extract_weather(data)

    air_quality_df = extract_air_quality(data)

    # Convert timestamps
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"],
        utc=True
    )

    air_quality_df["timestamp"] = pd.to_datetime(
        air_quality_df["timestamp"],
        utc=True
    )

    # Merge using exact hourly timestamp
    df = pd.merge(
        weather_df,
        air_quality_df,
        on="timestamp",
        how="inner"
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"]
    ).reset_index(drop=True)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate historical dataset."""

    print("\nDataset validation")
    print("-" * 40)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"Date range: "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"\nDuplicate timestamps: "
        f"{df['timestamp'].duplicated().sum()}"
    )

    print("\nMissing values:")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":

    # Automatically find the historical JSON file
    files = sorted(
        HISTORICAL_DIR.glob(
            "historical_*.json"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No historical JSON file found."
        )

    input_file = files[-1]

    print(
        f"Using historical file:\n"
        f"{input_file}"
    )

    df = clean_historical_data(
        input_file
    )

    validate_dataset(df)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DIR
        / "historical_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nHistorical dataset saved to:\n"
        f"{output_file}"
    )