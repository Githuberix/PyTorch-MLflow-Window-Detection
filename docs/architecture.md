# Architecture

The project has three parts:

1. Feature data and labels in `data/sample/`
2. PyTorch training with MLflow tracking in `training/train_model_mlflow.py`
3. Streamlit inference dashboard in `app/app.py`

## Flow

```text
sample CSV data
    -> training/train_model_mlflow.py
    -> models/window_model.pt + models/scaler.pt
    -> data/sample/window_predictions_bz_pytorch.csv
    -> app/app.py
```

The model is a binary classifier for room `BZ`. It uses room temperatures,
temperature deltas, rolling statistics, and fused cross-room features.

MLflow stores parameters, metrics, model artifacts, and prediction artifacts.
