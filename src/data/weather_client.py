from src.data.api_client import APIClient
from src.utils.config import OPENWEATHER_API_KEY


class WeatherClient:
    """Client for OpenWeather weather APIs."""

    BASE_URL = "https://api.openweathermap.org"

    def __init__(self):
        self.client = APIClient()

    def get_coordinates(
        self,
        city: str,
        country: str,
    ) -> dict:
        """Convert city name to latitude and longitude."""

        url = f"{self.BASE_URL}/geo/1.0/direct"

        params = {
            "q": f"{city},{country}",
            "limit": 1,
            "appid": OPENWEATHER_API_KEY,
        }

        results = self.client.get(url, params)

        if not results:
            raise ValueError(
                f"Could not find coordinates for {city}, {country}"
            )

        location = results[0]

        return {
            "name": location["name"],
            "country": location["country"],
            "latitude": location["lat"],
            "longitude": location["lon"],
        }

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Get 5-day / 3-hour weather forecast."""

        url = f"{self.BASE_URL}/data/2.5/forecast"

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }

        return self.client.get(url, params)