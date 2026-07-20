# PyTorch + MLflow Window Detection

Automatic window-open detection using indoor temperature data.

## Features

- PyTorch binary classifier
- MLflow experiment tracking
- Interactive Streamlit dashboard
- Adjustable decision threshold
- Prediction visualization
- Export-ready detected events table

## Project Structure

```text
PyTorch-MLflow-Window-Detection/
├── app/
│   ├── app.py
│   ├── inference.py
│   └── utils.py
├── training/
│   ├── train_model.py
│   ├── train_model_mlflow.py
│   └── labels.py
├── models/
│   ├── window_model.pt
│   ├── scaler.pt
│   └── window_model_bz.json
├── data/
│   ├── sample/
│   │   ├── fused_temperature_features.csv
│   │   ├── labeled_temperature_features.csv
│   │   └── window_predictions_bz_pytorch.csv
│   └── README.md
├── mlruns/
├── screenshots/
├── docs/
│   └── architecture.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── docker-compose.yml
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Training

```powershell
python training/train_model_mlflow.py
```

Open MLflow:

```powershell
mlflow ui
```

Then visit `http://127.0.0.1:5000`.

## Run Dashboard

```powershell
streamlit run app/app.py
```

Then visit `http://localhost:8501`.

## Model

- Target: `label_window_BZ`
- Type: PyTorch logistic regression
- Features: room temperatures, deltas, rolling statistics, and fused temperature features
- Output: window-open probability and binary decision

## Screenshots

Place dashboard and MLflow screenshots in:

- `screenshots/dashboard.png`
- `screenshots/mlflow.png`
