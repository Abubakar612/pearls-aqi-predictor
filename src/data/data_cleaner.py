import json
from pathlib import Path

import pandas as pd


def load_raw_data(file_path: str | Path) -> dict:
    """Load a raw AQI JSON file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_weather_data(data: dict) -> pd.DataFrame:
    """Extract and clean weather forecast data."""

    records = data.get("weather_forecast", {}).get("list", [])

    if not records:
        raise ValueError("No weather forecast data found.")

    rows = []

    for record in records:
        main = record.get("main", {})
        wind = record.get("wind", {})
        clouds = record.get("clouds", {})

        rows.append({
            "timestamp": record.get("dt_txt"),

            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),

            "pressure": main.get("pressure"),
            "humidity": main.get("humidity"),
            "dew_point": main.get("dew_point"),

            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "wind_gust": wind.get("gust"),

            "clouds": clouds.get("all"),

            "visibility": record.get("visibility"),
            "precipitation_probability": record.get("pop"),
        })

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df


def clean_pollution_data(data: dict) -> pd.DataFrame:
    """Extract and clean pollution forecast data."""

    pollution_data = data.get("pollution_forecast", {})
    records = pollution_data.get("list", [])

    if not records:
        raise ValueError("No pollution forecast data found.")

    rows = []

    for record in records:
        components = record.get("components", {})
        main = record.get("main", {})

        rows.append({
            "timestamp": pd.to_datetime(
                record.get("dt"),
                unit="s",
                errors="coerce"
            ),

            "aqi": main.get("aqi"),

            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
        })

    df = pd.DataFrame(rows)

    return df

def merge_weather_pollution(
    weather_df: pd.DataFrame,
    pollution_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge weather and pollution data using nearest timestamp."""

    weather_df = weather_df.copy()
    pollution_df = pollution_df.copy()

    # Convert both timestamps to UTC and force the same
    # nanosecond datetime precision.
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    pollution_df["timestamp"] = pd.to_datetime(
        pollution_df["timestamp"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    # Sort before merge_asof
    weather_df = weather_df.sort_values("timestamp").reset_index(drop=True)
    pollution_df = pollution_df.sort_values("timestamp").reset_index(drop=True)

    # Match each pollution observation with the nearest
    # weather observation within 90 minutes.
    df = pd.merge_asof(
        pollution_df,
        weather_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("90min")
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def clean_data(file_path: str | Path) -> pd.DataFrame:
    """Complete cleaning pipeline."""

    data = load_raw_data(file_path)

    weather_df = clean_weather_data(data)
    pollution_df = clean_pollution_data(data)

    df = merge_weather_pollution(
        weather_df,
        pollution_df
    )

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    raw_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "aqi_raw_20260810_191546.json"
    )

    output_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = clean_data(raw_file)

    output_file = output_dir / "cleaned_aqi_data.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\nData cleaning completed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_file}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())