import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.weather_client import WeatherClient
from src.data.pollution_client import PollutionClient
from src.utils.config import (
    CITY,
    COUNTRY,
    validate_config,
)


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def collect_data():
    """Collect weather and pollution data."""

    validate_config()

    weather_client = WeatherClient()
    pollution_client = PollutionClient()

    print(f"Collecting data for {CITY}, {COUNTRY}...")

    # 1. Get coordinates
    location = weather_client.get_coordinates(
        CITY,
        COUNTRY,
    )

    latitude = location["latitude"]
    longitude = location["longitude"]

    print(
        f"Location found: "
        f"{location['name']} "
        f"({latitude}, {longitude})"
    )

    # 2. Weather forecast
    weather = weather_client.get_forecast(
        latitude,
        longitude,
    )

    # 3. Current pollution
    pollution_current = pollution_client.get_current(
        latitude,
        longitude,
    )

    # 4. Pollution forecast
    pollution_forecast = pollution_client.get_forecast(
        latitude,
        longitude,
    )

    # 5. Combine everything
    data = {
        "collection_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "location": location,

        "weather_forecast": weather,

        "pollution_current": pollution_current,

        "pollution_forecast": pollution_forecast,
    }

    # Create directory if needed
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # File name
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        RAW_DATA_DIR /
        f"aqi_raw_{timestamp}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )

    print(
        f"\nData successfully saved to:\n"
        f"{output_file}"
    )

    return data


if __name__ == "__main__":
    collect_data()