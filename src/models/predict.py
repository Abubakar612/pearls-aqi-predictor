import json
import joblib
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "production"
FEATURE_FILE = MODEL_DIR / "feature_list.json"

MODEL_FILES = {
    "24h": "aqi_model_24h.joblib",
    "48h": "aqi_model_48h.joblib",
    "72h": "aqi_model_72h.joblib",
}


def get_aqi_category(aqi: float) -> str:
    """Return the AQI category for a numeric AQI value."""

    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def load_feature_list():
    """Load the production feature list."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_FILE}"
        )

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        features = json.load(file)

    if not isinstance(features, list) or not features:
        raise ValueError("Production feature list is empty or invalid.")

    return features


def load_production_models():
    """Load all production forecasting models."""

    models = {}

    for horizon, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production model not found: {model_path}"
            )

        models[horizon] = joblib.load(model_path)

    return models


def prepare_input(data: pd.DataFrame, features=None) -> pd.DataFrame:
    """Validate and prepare model input."""

    if features is None:
        features = load_feature_list()

    missing = [
        feature
        for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    X = data[features].copy()

    if X.isnull().any().any():
        missing_values = X.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        raise ValueError(
            f"Missing values detected:\n{missing_values}"
        )

    return X


def predict(data: pd.DataFrame):
    """Generate 24h, 48h and 72h AQI predictions."""

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    features = load_feature_list()
    models = load_production_models()

    X = prepare_input(data, features)

    row = X.iloc[[-1]]

    predictions = {}

    for horizon, model in models.items():
        value = float(model.predict(row)[0])
        value = max(0.0, value)

        predictions[horizon] = {
            "aqi": round(value, 2),
            "category": get_aqi_category(value),
        }

    return predictions


def predict_single(data: pd.DataFrame, horizon: str) -> dict:
    """Generate a prediction for one forecast horizon."""

    if horizon not in MODEL_FILES:
        raise ValueError(
            f"Unsupported horizon: {horizon}. "
            f"Expected one of {list(MODEL_FILES)}."
        )

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    features = load_feature_list()
    models = load_production_models()

    X = prepare_input(data, features)
    row = X.iloc[[-1]]

    value = float(models[horizon].predict(row)[0])
    value = max(0.0, value)

    return {
        "aqi": round(value, 2),
        "category": get_aqi_category(value),
    }
