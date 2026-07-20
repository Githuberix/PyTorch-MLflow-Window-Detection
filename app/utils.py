from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_FILE = SAMPLE_DATA_DIR / "labeled_temperature_features.csv"
PREDICTION_FILE = SAMPLE_DATA_DIR / "window_predictions_bz_pytorch.csv"
MODEL_FILE = MODEL_DIR / "window_model.pt"
SCALER_FILE = MODEL_DIR / "scaler.pt"

TIME_COLUMN = "time"
TARGET_COLUMN = "label_window_BZ"

FEATURE_COLUMNS = [
    "temp_BZ",
    "delta_BZ",
    "roll_mean_BZ",
    "roll_std_BZ",
    "temp_EZ",
    "delta_EZ",
    "roll_mean_EZ",
    "roll_std_EZ",
    "temp_SZ",
    "delta_SZ",
    "roll_mean_SZ",
    "roll_std_SZ",
    "temp_WZ",
    "delta_WZ",
    "roll_mean_WZ",
    "roll_std_WZ",
    "fusion_mean_temp",
    "fusion_temp_spread",
]
