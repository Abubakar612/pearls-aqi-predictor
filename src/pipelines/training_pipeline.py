"""
Production AQI model training pipeline.

Runs the complete model-development sequence:
1. Baseline evaluation
2. Feature importance analysis
3. Production model training
4. Production model validation
"""

from src.models.baseline import main as run_baseline
from src.models.feature_analysis import main as run_feature_analysis
from src.models.train_production import main as run_production_training
from src.models.validate_production import main as run_production_validation


def run_training_pipeline():
    """Run the complete production training workflow."""

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1/4] Running baseline evaluation...")
    run_baseline()

    print("\n[2/4] Running feature importance analysis...")
    run_feature_analysis()

    print("\n[3/4] Training production models...")
    run_production_training()

    print("\n[4/4] Validating production models...")
    run_production_validation()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()