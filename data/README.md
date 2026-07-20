# Data

This repository contains a small anonymized sample under `data/sample/` so the
training script and dashboard can run without private raw sensor exports.

Raw files matching `Temperatures-data-*.csv` are intentionally excluded from Git
because they may contain private indoor-temperature traces.

## Files

- `fused_temperature_features.csv`: engineered temperature features
- `labeled_temperature_features.csv`: feature rows with binary window-open labels
- `window_predictions_bz_pytorch.csv`: prediction output for the Streamlit demo
