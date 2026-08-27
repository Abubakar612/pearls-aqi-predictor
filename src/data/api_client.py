import requests


class APIClient:
    """Generic HTTP client for external APIs."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get(self, url: str, params: dict | None = None) -> dict:
        """Send a GET request and return JSON response."""

        response = requests.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()