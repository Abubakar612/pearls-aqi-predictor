import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# .env file
ENV_FILE = BASE_DIR / ".env"

# Load environment variables
load_dotenv(ENV_FILE)


# API keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_API_KEY = os.getenv("AQICN_API_KEY")

# Location
CITY = os.getenv("CITY", "Lahore")
COUNTRY = os.getenv("COUNTRY", "Pakistan")

LATITUDE = float(os.getenv("LATITUDE") or "31.5656822")
LONGITUDE = float(os.getenv("LONGITUDE") or "74.3141829")


def validate_config():
    """Validate required environment variables."""

    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing from the .env file."
        )