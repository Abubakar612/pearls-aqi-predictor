import requests


class HistoricalClient:
    """Client for Open-Meteo historical APIs."""

    WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

    AIR_QUALITY_URL = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def get_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict:

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "apparent_temperature",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "precipitation",
            ]),
            "timezone": "UTC",
        }

        response = requests.get(
            self.WEATHER_URL,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def get_air_quality(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict:

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join([
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ]),
            "timezone": "UTC",
        }

        response = requests.get(
            self.AIR_QUALITY_URL,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()