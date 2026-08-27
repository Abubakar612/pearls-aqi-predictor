import json
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed" / "ml_dataset.csv"
MODEL_DIR = BASE_DIR / "models" / "production"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model_analysis" / "shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PEARLS AQI PREDICTOR")
print("SHAP MODEL EXPLAINABILITY ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

# Load production feature list
with open(MODEL_DIR / "feature_list.json", "r", encoding="utf-8") as f:
    features = json.load(f)

print(f"Dataset shape: {df.shape}")
print(f"Production features: {len(features)}")

X = df[features].dropna().copy()

print(f"SHAP input shape: {X.shape}")

# Analyse all three production models
for horizon in ["24h", "48h", "72h"]:

    print("\n" + "=" * 60)
    print(f"SHAP ANALYSIS: {horizon}")
    print("=" * 60)

    model_path = MODEL_DIR / f"aqi_model_{horizon}.joblib"

    model = joblib.load(model_path)

    print("Model loaded.")

    # Use a representative sample to keep SHAP computation manageable
    sample_size = min(500, len(X))
    X_sample = X.tail(sample_size)

    print(f"Samples analysed: {len(X_sample)}")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP importance
    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    output_csv = OUTPUT_DIR / f"shap_importance_{horizon}.csv"

    importance.to_csv(
        output_csv,
        index=False
    )

    print("\nTop 15 SHAP features:")
    print(importance.head(15).to_string(index=False))

    print(f"\nSaved: {output_csv}")

    # SHAP bar plot
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title(
        f"SHAP Feature Importance - {horizon} AQI Forecast"
    )

    plt.tight_layout()

    plot_file = OUTPUT_DIR / f"shap_summary_{horizon}.png"

    plt.savefig(
        plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {plot_file}")

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)



