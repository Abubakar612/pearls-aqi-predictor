"""
Production training pipeline.

This module orchestrates the existing production model
training implementation in src.models.train_production.
"""

from src.models.train_production import main as train_production_models


def run_training_pipeline():
    """Run the complete production training process."""

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("TRAINING PIPELINE")
    print("=" * 60)

    train_production_models()

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
