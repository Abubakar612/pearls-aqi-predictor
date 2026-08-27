import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate standard regression metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics from empty data."
        )

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        ),
    }


def evaluate_model(model, X, y) -> dict:
    """Generate predictions and evaluate a regression model."""

    predictions = model.predict(X)

    return calculate_metrics(
        y,
        predictions
    )


def evaluate_models(models: dict, X, targets: dict) -> dict:
    """Evaluate multiple models against their corresponding targets."""

    results = {}

    for horizon, model in models.items():
        if horizon not in targets:
            raise ValueError(
                f"No target supplied for horizon: {horizon}"
            )

        results[horizon] = evaluate_model(
            model,
            X,
            targets[horizon]
        )

    return results
