"""End-to-end pipeline utilities for ICU mortality prediction and explanation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.evidence import build_evidence_packet
from src.explainability import explain_patient
from src.prediction import predict_mortality
from src.preprocessing import ICUPreprocessor


def classify_prediction(y_true: int, y_pred: int) -> str:
    """Return TP/FN/FP/TN label when true and predicted labels are known."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"

    raise ValueError(f"Unexpected y_true/y_pred combination: {y_true}, {y_pred}")


def run_patient_pipeline(
    raw_patient: pd.DataFrame,
    preprocessor: ICUPreprocessor,
    model: Any,
    threshold: float,
    patient_label: str = "patient",
    test_row_index: int | None = None,
    y_true: int | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Run preprocessing, prediction, SHAP, and evidence construction for one patient."""
    if len(raw_patient) != 1:
        raise ValueError("raw_patient must contain exactly one row.")

    X_patient = preprocessor.transform(raw_patient)

    prediction_row = predict_mortality(
        model,
        X_patient,
        threshold=threshold,
    ).iloc[0]

    local_explanation = explain_patient(model, X_patient)
    if hasattr(preprocessor, "get_display_values"):
        display_values = preprocessor.get_display_values(raw_patient)
        local_explanation["model_value"] = local_explanation["value"]
        local_explanation["value"] = local_explanation.apply(
            lambda row: display_values.get(row["feature"], row["value"]),
            axis=1,
        )

    prediction_type = None
    if y_true is not None:
        prediction_type = classify_prediction(
            y_true=int(y_true),
            y_pred=int(prediction_row["prediction"]),
        )

    evidence_packet = build_evidence_packet(
        local_explanation=local_explanation,
        prediction_row=prediction_row,
        patient_label=patient_label,
        test_row_index=test_row_index,
        y_true=y_true,
        prediction_type=prediction_type,
        top_n=top_n,
    )

    return {
        "processed_patient": X_patient,
        "prediction": prediction_row.to_dict(),
        "local_explanation": local_explanation,
        "evidence_packet": evidence_packet,
    }
