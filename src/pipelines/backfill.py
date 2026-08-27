import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data.historical_client import HistoricalClient
from src.utils.config import CITY


BASE_DIR = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = (
    BASE_DIR / "data" / "historical"
)


LATITUDE = 31.5656822
LONGITUDE = 74.3141829


def get_date_range(days: int = 90):
    """
    Create a historical date range.

    We stop two days before today to avoid requesting
    data that may not yet be available in the archive.
    """

    end_date = (
        datetime.now(timezone.utc).date()
        - timedelta(days=2)
    )

    start_date = (
        end_date
        - timedelta(days=days - 1)
    )

    return (
        start_date.isoformat(),
        end_date.isoformat()
    )


def backfill_historical_data(days: int = 90):

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("Historical Data Backfill")
    print("=" * 60)

    start_date, end_date = get_date_range(days)

    print(f"\nCity: {CITY}")
    print(f"Latitude: {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")

    print(
        f"\nDate range:"
        f"\n{start_date}"
        f"\n{end_date}"
    )

    client = HistoricalClient()

    print("\nDownloading historical weather...")

    weather = client.get_weather(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        end_date=end_date,
    )

    print("Weather download completed.")

    print("\nDownloading historical air quality...")

    air_quality = client.get_air_quality(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        end_date=end_date,
    )

    print("Air-quality download completed.")

    output = {
        "metadata": {
            "city": CITY,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "source": "Open-Meteo",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "weather": weather,
        "air_quality": air_quality,
    }

    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        HISTORICAL_DIR
        / f"historical_{start_date}_{end_date}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETED")
    print("=" * 60)

    print(
        f"\nSaved to:\n{output_file}"
    )

    return output_file


if __name__ == "__main__":
    backfill_historical_data(days=90)