from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from utils import FEATURE_COLUMNS, MODEL_FILE, SCALER_FILE


class WindowClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(dim=1)


@torch.no_grad()
def predict_probabilities(frame: pd.DataFrame) -> pd.Series:
    scaler = torch.load(SCALER_FILE, weights_only=False)
    features = torch.tensor(frame[FEATURE_COLUMNS].astype(float).values, dtype=torch.float32)
    scaled_features = (features - scaler["means"]) / scaler["stds"]

    model = WindowClassifier(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()

    probabilities = torch.sigmoid(model(scaled_features)).numpy()
    return pd.Series(probabilities, index=frame.index, name="probability_window_BZ")
