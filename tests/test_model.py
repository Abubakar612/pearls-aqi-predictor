import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import ExtraTreesRegressor

from src.models.predict import (
    get_aqi_category,
    prepare_input,
)

from src.models.evaluate import (
    calculate_metrics,
    evaluate_model,
)

from src.models.train_production import create_model


def test_aqi_category_boundaries():
    assert get_aqi_category(0) == "Good"
    assert get_aqi_category(50) == "Good"
    assert get_aqi_category(51) == "Moderate"
    assert get_aqi_category(100) == "Moderate"
    assert get_aqi_category(101) == "Unhealthy for Sensitive Groups"
    assert get_aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert get_aqi_category(151) == "Unhealthy"
    assert get_aqi_category(200) == "Unhealthy"
    assert get_aqi_category(201) == "Very Unhealthy"
    assert get_aqi_category(300) == "Very Unhealthy"
    assert get_aqi_category(301) == "Hazardous"


def test_create_model():
    model = create_model()

    assert isinstance(
        model,
        ExtraTreesRegressor
    )

    assert model.n_estimators == 400
    assert model.random_state == 42


def test_prepare_input():
    data = pd.DataFrame({
        "feature_a": [1.0],
        "feature_b": [2.0],
        "extra_column": [99.0],
    })

    X = prepare_input(
        data,
        ["feature_a", "feature_b"]
    )

    assert list(X.columns) == [
        "feature_a",
        "feature_b",
    ]

    assert X.shape == (1, 2)


def test_prepare_input_missing_feature():
    data = pd.DataFrame({
        "feature_a": [1.0],
    })

    with pytest.raises(ValueError):
        prepare_input(
            data,
            ["feature_a", "feature_b"]
        )


def test_prepare_input_missing_values():
    data = pd.DataFrame({
        "feature_a": [1.0],
        "feature_b": [np.nan],
    })

    with pytest.raises(ValueError):
        prepare_input(
            data,
            ["feature_a", "feature_b"]
        )


def test_calculate_metrics():
    y_true = [10, 20, 30, 40]
    y_pred = [11, 19, 31, 39]

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert set(metrics.keys()) == {
        "mae",
        "rmse",
        "r2",
    }

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1


def test_evaluate_model():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
    ])

    y = np.array([
        10,
        20,
        30,
        40,
        50,
    ])

    model = ExtraTreesRegressor(
        n_estimators=20,
        random_state=42,
    )

    model.fit(X, y)

    metrics = evaluate_model(
        model,
        X,
        y
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
