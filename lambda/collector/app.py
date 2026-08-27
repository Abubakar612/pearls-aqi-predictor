import json
import os
from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

S3_BUCKET = os.environ["S3_BUCKET"]

S3_KEY = os.environ.get(
    "S3_KEY",
    "realtime/lahore_hourly.csv"
)

OPENWEATHER_API_KEY = os.environ[
    "OPENWEATHER_API_KEY"
]

CITY = os.environ.get(
    "CITY",
    "Lahore"
)

COUNTRY = os.environ.get(
    "COUNTRY",
    "Pakistan"
)

LATITUDE = float(
    os.environ.get(
        "LATITUDE",
        "31.5656822"
    )
)

LONGITUDE = float(
    os.environ.get(
        "LONGITUDE",
        "74.3141829"
    )
)

BASE_URL = "https://api.openweathermap.org"

S3 = boto3.client("s3")


# ============================================================
# OPENWEATHER CLIENTS
# ============================================================

def get_weather():

    url = (
        f"{BASE_URL}"
        "/data/2.5/forecast"
    )

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_pollution():

    url = (
        f"{BASE_URL}"
        "/data/2.5/air_pollution"
    )

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OPENWEATHER_API_KEY,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# BUILD OBSERVATION
# ============================================================

def build_observation():

    print(
        "Fetching latest weather data..."
    )

    weather = get_weather()

    print(
        "Fetching current air-quality data..."
    )

    pollution = get_pollution()

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_list = weather.get(
        "list",
        []
    )

    if not weather_list:

        raise ValueError(
            "Weather API returned no forecast data."
        )

    current_weather = weather_list[0]

    main = current_weather.get(
        "main",
        {}
    )

    wind = current_weather.get(
        "wind",
        {}
    )

    clouds = current_weather.get(
        "clouds",
        {}
    )

    rain = current_weather.get(
        "rain",
        {}
    )

    # --------------------------------------------------------
    # Pollution
    # --------------------------------------------------------

    pollution_list = pollution.get(
        "list",
        []
    )

    if not pollution_list:

        raise ValueError(
            "Pollution API returned no data."
        )

    current_pollution = pollution_list[0]

    components = current_pollution.get(
        "components",
        {}
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

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

        timestamp = datetime.now(
            timezone.utc
        ).replace(
            minute=0,
            second=0,
            microsecond=0
        )

    # --------------------------------------------------------
    # Standardized observation
    # --------------------------------------------------------

    row = {

        "timestamp": timestamp,

        # Weather
        "temperature": main.get(
            "temp"
        ),

        "humidity": main.get(
            "humidity"
        ),

        "dew_point": main.get(
            "dew_point"
        ),

        "feels_like": main.get(
            "feels_like"
        ),

        "pressure": main.get(
            "pressure"
        ),

        "surface_pressure": main.get(
            "grnd_level"
        ),

        "clouds": clouds.get(
            "all"
        ),

        "wind_speed": wind.get(
            "speed"
        ),

        "wind_direction": wind.get(
            "deg"
        ),

        "wind_gust": wind.get(
            "gust"
        ),

        "precipitation": rain.get(
            "1h",
            0.0
        ),

        # Pollution
        "pm10": components.get(
            "pm10"
        ),

        "pm2_5": components.get(
            "pm2_5"
        ),

        "co": components.get(
            "co"
        ),

        "no2": components.get(
            "no2"
        ),

        "so2": components.get(
            "so2"
        ),

        "o3": components.get(
            "o3"
        ),
    }

    return row


# ============================================================
# LOAD S3 CSV
# ============================================================

def load_existing_data():

    try:

        print(
            f"Loading "
            f"s3://{S3_BUCKET}/{S3_KEY}"
        )

        obj = S3.get_object(
            Bucket=S3_BUCKET,
            Key=S3_KEY
        )

        content = obj[
            "Body"
        ].read().decode(
            "utf-8"
        )

        if not content.strip():

            return pd.DataFrame()

        df = pd.read_csv(
            StringIO(content)
        )

        return df

    except S3.exceptions.NoSuchKey:

        print(
            "S3 file does not exist. "
            "Creating a new dataset."
        )

        return pd.DataFrame()

    except Exception as e:

        # S3 may return a generic ClientError
        # for a missing object.

        if "NoSuchKey" in str(e):

            return pd.DataFrame()

        raise


# ============================================================
# SAVE S3 CSV
# ============================================================

def save_to_s3(df):

    csv_buffer = StringIO()

    df.to_csv(
        csv_buffer,
        index=False
    )

    S3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=csv_buffer.getvalue().encode(
            "utf-8"
        ),
        ContentType="text/csv"
    )

    print(
        f"Saved to "
        f"s3://{S3_BUCKET}/{S3_KEY}"
    )


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(
    event,
    context
):

    try:

        print("=" * 60)
        print(
            "PEARLS AQI PREDICTOR"
        )
        print(
            "REAL-TIME DATA COLLECTOR"
        )
        print("=" * 60)

        print(
            f"City: {CITY}"
        )

        print(
            f"Country: {COUNTRY}"
        )

        print(
            f"Latitude: {LATITUDE}"
        )

        print(
            f"Longitude: {LONGITUDE}"
        )

        # ----------------------------------------------------
        # 1. Get new observation
        # ----------------------------------------------------

        row = build_observation()

        print(
            "\nNew observation:"
        )

        print(
            json.dumps(
                {
                    k: str(v)
                    for k, v in row.items()
                },
                indent=2
            )
        )

        # ----------------------------------------------------
        # 2. Load historical data
        # ----------------------------------------------------

        existing = load_existing_data()

        # ----------------------------------------------------
        # 3. Append observation
        # ----------------------------------------------------

        new_data = pd.DataFrame(
            [row]
        )

        df = pd.concat(
            [
                existing,
                new_data
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # 4. Clean timestamps
        # ----------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = (
            df
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=[
                    "timestamp"
                ],
                keep="last"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # 5. Validate columns
        # ----------------------------------------------------

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
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        # ----------------------------------------------------
        # 6. Save
        # ----------------------------------------------------

        save_to_s3(
            df
        )

        latest = df.iloc[-1]

        result = {

            "status": "success",

            "city": CITY,

            "country": COUNTRY,

            "timestamp": str(
                latest[
                    "timestamp"
                ]
            ),

            "rows": len(df),

            "bucket": S3_BUCKET,

            "key": S3_KEY,

        }

        print(
            "\n" + "=" * 60
        )

        print(
            "REAL-TIME DATA COLLECTION COMPLETED"
        )

        print(
            "=" * 60
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                result
            )
        }

    except Exception as e:

        print(
            f"ERROR: {str(e)}"
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "status": "error",
                    "message": str(e)
                }
            )
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    result = lambda_handler(
        {},
        None
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )
