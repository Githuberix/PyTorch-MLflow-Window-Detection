from __future__ import annotations

import pandas as pd
import streamlit as st

from inference import predict_probabilities
from utils import FEATURE_FILE, PREDICTION_FILE, TARGET_COLUMN, TIME_COLUMN


st.set_page_config(page_title="Window Detection", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURE_FILE, parse_dates=[TIME_COLUMN])

    if PREDICTION_FILE.exists():
        predictions = pd.read_csv(PREDICTION_FILE, parse_dates=[TIME_COLUMN])
        return features.merge(predictions, on=[TIME_COLUMN, TARGET_COLUMN], how="left")

    features["probability_window_BZ"] = predict_probabilities(features)
    features["prediction_window_BZ"] = (features["probability_window_BZ"] >= 0.5).astype(int)
    return features


data = load_data()

st.title("PyTorch + MLflow Window Detection")
st.caption("Automatic window-open detection using indoor temperature features.")

threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
data["threshold_prediction"] = (data["probability_window_BZ"] >= threshold).astype(int)
threshold_accuracy = (data[TARGET_COLUMN] == data["threshold_prediction"]).mean()

metrics = st.columns(4)
metrics[0].metric("Rows", f"{len(data):,}")
metrics[1].metric("Positive labels", f"{data[TARGET_COLUMN].mean():.1%}")
metrics[2].metric("Detected events", f"{data['threshold_prediction'].sum():,}")
metrics[3].metric("Accuracy", f"{threshold_accuracy:.1%}")

chart_frame = data.set_index(TIME_COLUMN)[["temp_BZ", "probability_window_BZ", TARGET_COLUMN]]
st.line_chart(chart_frame, height=420)

left, right = st.columns([1, 1])

with left:
    st.subheader("Detected Events")
    events = data.loc[
        data["threshold_prediction"] == 1,
        [TIME_COLUMN, "temp_BZ", TARGET_COLUMN, "probability_window_BZ"],
    ]
    st.dataframe(events, use_container_width=True, hide_index=True)

with right:
    st.subheader("Prediction Details")
    detail_columns = [
        TIME_COLUMN,
        "temp_BZ",
        "delta_BZ",
        "roll_mean_BZ",
        TARGET_COLUMN,
        "threshold_prediction",
        "probability_window_BZ",
    ]
    st.dataframe(data[detail_columns], use_container_width=True, hide_index=True)
