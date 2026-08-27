import os
from datetime import datetime, timezone

import pandas as pd

from src.data.weather_client import WeatherClient
from src.data.pollution_client import PollutionClient
from src.utils.config import CITY, LATITUDE, LONGITUDE


REALTIME_FILE = "data/realtime/lahore_hourly.csv"


def collect_realtime_data():
    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("REAL-TIME DATA COLLECTION")
    print("=" * 60)

    print(f"City: {CITY}")
    print(f"Latitude: {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")
    print()

    weather_client = WeatherClient()
    pollution_client = PollutionClient()

    # ---------------------------------------------------------
    # 1. Fetch weather forecast
    # ---------------------------------------------------------
    print("Fetching latest weather data...")

    weather = weather_client.get_forecast(
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    # ---------------------------------------------------------
    # 2. Fetch current pollution
    # ---------------------------------------------------------
    print("Fetching current air-quality data...")

    pollution = pollution_client.get_current(
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    # ---------------------------------------------------------
    # 3. Extract current weather observation
    # ---------------------------------------------------------
    weather_list = weather.get("list", [])

    if not weather_list:
        raise ValueError("Weather API returned no forecast data.")

    current_weather = weather_list[0]

    main = current_weather.get("main", {})
    wind = current_weather.get("wind", {})
    clouds = current_weather.get("clouds", {})
    rain = current_weather.get("rain", {})

    # ---------------------------------------------------------
    # 4. Extract current pollution observation
    # ---------------------------------------------------------
    pollution_list = pollution.get("list", [])

    if not pollution_list:
        raise ValueError("Pollution API returned no data.")

    current_pollution = pollution_list[0]
    components = current_pollution.get("components", {})

    # Use API timestamp when available.
    if current_pollution.get("dt"):
        timestamp = datetime.fromtimestamp(
            current_pollution["dt"],
            tz=timezone.utc
        ).replace(
            minute=0,
            second=0,
            microsecond=0
        )
    elif current_weather.get("dt"):
        timestamp = datetime.fromtimestamp(
            current_weather["dt"],
            tz=timezone.utc
        ).replace(
            minute=0,
            second=0,
            microsecond=0
        )
    else:
        timestamp = datetime.now(timezone.utc).replace(
            minute=0,
            second=0,
            microsecond=0
        )

    # ---------------------------------------------------------
    # 5. Build standardized observation
    # ---------------------------------------------------------
    row = {
        "timestamp": timestamp,

        # Weather
        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "dew_point": main.get("dew_point"),
        "feels_like": main.get("feels_like"),
        "pressure": main.get("pressure"),
        "surface_pressure": main.get("grnd_level"),
        "clouds": clouds.get("all"),
        "wind_speed": wind.get("speed"),
        "wind_direction": wind.get("deg"),
        "wind_gust": wind.get("gust"),
        "precipitation": rain.get("1h", 0.0),

        # Pollution
        "pm10": components.get("pm10"),
        "pm2_5": components.get("pm2_5"),
        "co": components.get("co"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "o3": components.get("o3"),
    }

    new_data = pd.DataFrame([row])

    # ---------------------------------------------------------
    # 6. Validate observation
    # ---------------------------------------------------------
    required_columns = [
        "timestamp",
        "temperature",
        "humidity",
        "dew_point",
        "feels_like",
        "pressure",
        "surface_pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "precipitation",
        "pm10",
        "pm2_5",
        "co",
        "no2",
        "so2",
        "o3",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in new_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 7. Load existing realtime store
    # ---------------------------------------------------------
    os.makedirs(
        os.path.dirname(REALTIME_FILE),
        exist_ok=True
    )

    if os.path.exists(REALTIME_FILE):

        existing = pd.read_csv(
            REALTIME_FILE,
            parse_dates=["timestamp"]
        )

        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"],
            utc=True
        )

        # Remove same-hour observation if already present
        existing = existing[
            existing["timestamp"] != timestamp
        ]

        df = pd.concat(
            [existing, new_data],
            ignore_index=True
        )

    else:
        df = new_data

    # ---------------------------------------------------------
    # 8. Clean and sort
    # ---------------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    # ---------------------------------------------------------
    # 9. Save
    # ---------------------------------------------------------
    df.to_csv(
        REALTIME_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 10. Display result
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("REAL-TIME DATA COLLECTION COMPLETED")
    print("=" * 60)

    print(f"\nSaved to:")
    print(os.path.abspath(REALTIME_FILE))

    print(f"\nTotal observations: {len(df)}")

    print("\nLatest observation:")
    print(
        df.tail(1).to_string(index=False)
    )


if __name__ == "__main__":
    collect_realtime_data()