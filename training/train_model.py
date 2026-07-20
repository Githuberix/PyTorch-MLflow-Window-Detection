"""Baseline training entry point.

This project keeps the MLflow version as the main training script. The baseline
entry point delegates to it so local usage stays simple.
"""

from train_model_mlflow import main


if __name__ == "__main__":
    main()
