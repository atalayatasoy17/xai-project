"""Prediction utilities for the ICU mortality pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    """Load the trained mortality prediction model."""
    return joblib.load(model_path)


def load_threshold(threshold_path: str | Path) -> float:
    """Load the selected classification threshold."""
    with open(threshold_path) as f:
        threshold_info = json.load(f)

    return float(threshold_info["threshold"])


def predict_mortality(model, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return mortality probabilities and threshold-based predictions."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "death_probability": probabilities,
            "prediction": predictions,
            "threshold": threshold,
        },
        index=X.index,
    )
