from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
MODEL_DIR = PROJECT_ROOT / "models"

INPUT_FILE = DATA_DIR / "labeled_temperature_features.csv"
PREDICTIONS_FILE = DATA_DIR / "window_predictions_bz_pytorch.csv"
MODEL_FILE = MODEL_DIR / "window_model.pt"
SCALER_FILE = MODEL_DIR / "scaler.pt"

TARGET = "label_window_BZ"
EXPERIMENT_NAME = "temperature-window-detection"

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


class WindowClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(dim=1)


def read_dataset() -> tuple[list[str], torch.Tensor, torch.Tensor]:
    with INPUT_FILE.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    times = [row["time"] for row in rows]
    features = [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]
    labels = [float(row[TARGET]) for row in rows]

    return (
        times,
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def chronological_split(
    times: list[str],
    features: torch.Tensor,
    labels: torch.Tensor,
    split_time: str = "2026-05-03 19:50:00",
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    train_indices = [index for index, time in enumerate(times) if time < split_time]
    test_indices = [index for index, time in enumerate(times) if time >= split_time]

    return (
        [times[index] for index in train_indices],
        features[train_indices],
        labels[train_indices],
        [times[index] for index in test_indices],
        features[test_indices],
        labels[test_indices],
    )


def fit_scaler(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    means = features.mean(dim=0)
    stds = features.std(dim=0, unbiased=False)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)
    return means, stds


def transform(features: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return (features - means) / stds


def train_model(train_x: torch.Tensor, train_y: torch.Tensor) -> WindowClassifier:
    model = WindowClassifier(input_size=train_x.shape[1])
    positives = train_y.sum()
    negatives = len(train_y) - positives
    pos_weight = negatives / positives if positives > 0 else torch.tensor(1.0)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)

    model.train()
    for _ in range(500):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict(model: WindowClassifier, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(model(features))
    predictions = (probabilities >= 0.5).float()
    return probabilities, predictions


def evaluate(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float | int]:
    true_positive = int(((labels == 1) & (predictions == 1)).sum().item())
    true_negative = int(((labels == 0) & (predictions == 0)).sum().item())
    false_positive = int(((labels == 0) & (predictions == 1)).sum().item())
    false_negative = int(((labels == 1) & (predictions == 0)).sum().item())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(labels) if len(labels) else 0.0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def write_predictions(
    times: list[str],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["time", TARGET, "prediction_window_BZ", "probability_window_BZ"],
        )
        writer.writeheader()
        for time, label, probability, prediction in zip(times, labels, probabilities, predictions):
            writer.writerow(
                {
                    "time": time,
                    TARGET: int(label.item()),
                    "prediction_window_BZ": int(prediction.item()),
                    "probability_window_BZ": round(float(probability.item()), 4),
                }
            )


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    times, features, labels = read_dataset()
    train_times, train_x, train_y, test_times, test_x, test_y = chronological_split(times, features, labels)

    means, stds = fit_scaler(train_x)
    train_x_scaled = transform(train_x, means, stds)
    test_x_scaled = transform(test_x, means, stds)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="bz-window-pytorch-logistic-regression"):
        mlflow.log_params(
            {
                "target": TARGET,
                "model_type": "pytorch_logistic_regression",
                "epochs": 500,
                "learning_rate": 0.01,
                "batch_size": 16,
                "feature_count": len(FEATURE_COLUMNS),
            }
        )

        model = train_model(train_x_scaled, train_y)
        train_probabilities, train_predictions = predict(model, train_x_scaled)
        test_probabilities, test_predictions = predict(model, test_x_scaled)

        train_metrics = evaluate(train_y, train_predictions)
        test_metrics = evaluate(test_y, test_predictions)
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        write_predictions(
            train_times + test_times,
            torch.cat([train_y, test_y]),
            torch.cat([train_probabilities, test_probabilities]),
            torch.cat([train_predictions, test_predictions]),
        )

        torch.save(model.state_dict(), MODEL_FILE)
        torch.save({"means": means, "stds": stds, "features": FEATURE_COLUMNS}, SCALER_FILE)
        mlflow.pytorch.log_model(model, name="model")
        mlflow.log_artifact(str(PREDICTIONS_FILE))
        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(SCALER_FILE))

    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
