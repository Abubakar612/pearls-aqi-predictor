"""
Production AQI model training pipeline.

Runs the complete model-development sequence:
1. Data preparation
2. Baseline evaluation
3. Feature importance analysis
4. Production model training
5. Production model validation
"""

from src.pipelines.data_pipeline import (
    run_data_pipeline
)

from src.models.baseline import (
    main as run_baseline
)

from src.models.feature_analysis import (
    main as run_feature_analysis
)

from src.models.train_production import (
    main as run_production_training
)

from src.models.validate_production import (
    main as run_production_validation
)


def run_training_pipeline():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1/5] Preparing training data...")
    run_data_pipeline()

    print("\n[2/5] Running baseline evaluation...")
    run_baseline()

    print("\n[3/5] Running feature importance analysis...")
    run_feature_analysis()

    print("\n[4/5] Training production models...")
    run_production_training()

    print("\n[5/5] Validating production models...")
    run_production_validation()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()