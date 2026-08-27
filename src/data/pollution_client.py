from src.data.api_client import APIClient
from src.utils.config import OPENWEATHER_API_KEY


class PollutionClient:
    """Client for OpenWeather Air Pollution APIs."""

    BASE_URL = "https://api.openweathermap.org"

    def __init__(self):
        self.client = APIClient()

    def get_current(
        self,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Get current air pollution data."""

        url = f"{self.BASE_URL}/data/2.5/air_pollution"

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
        }

        return self.client.get(url, params)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Get hourly air pollution forecast."""

        url = f"{self.BASE_URL}/data/2.5/air_pollution/forecast"

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
        }

        return self.client.get(url, params)