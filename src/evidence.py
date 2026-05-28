"""Evidence packet construction for model explanations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CLINICAL_MEANING_MAP = {
    "age": "older age is associated with higher mortality risk",
    "ventilated_apache": "mechanical ventilation indicates severe respiratory failure or critical illness",
    "d1_spo2_min": "low minimum oxygen saturation indicates hypoxemia",
    "d1_spo2_max": "low maximum oxygen saturation may indicate persistent oxygenation impairment",
    "gcs_motor_apache": "low GCS motor score indicates impaired neurological response",
    "gcs_verbal_apache": "low GCS verbal score indicates impaired neurological response",
    "gcs_eyes_apache": "low GCS eye score indicates impaired neurological response",
    "d1_sysbp_min": "low systolic blood pressure indicates hemodynamic instability",
    "d1_mbp_min": "low mean blood pressure indicates hemodynamic instability",
    "d1_heartrate_min": "extreme or very low heart rate may indicate severe instability or data quality issue",
    "d1_heartrate_max": "abnormal maximum heart rate may indicate physiological stress",
    "d1_resprate_min": "extreme or very low respiratory rate may indicate respiratory instability or data quality issue",
    "d1_resprate_max": "high respiratory rate may indicate respiratory distress",
    "d1_bun_max": "elevated BUN may indicate renal dysfunction or metabolic stress",
    "d1_bun_min": "BUN reflects renal function and metabolic status",
    "d1_hco3_min": "low bicarbonate may indicate metabolic acidosis",
    "d1_hco3_max": "low bicarbonate range may indicate metabolic acidosis",
    "d1_platelets_min": "low platelet count may indicate severe illness or coagulation abnormalities",
    "d1_wbc_min": "white blood cell count reflects inflammatory or immune status",
    "pre_icu_los_days": "time between hospital admission and ICU admission; negative values require caution",
    "icu_id": "ICU unit identifier; may reflect unit-level patterns rather than patient-level clinical status",
    "apache_3j_diagnosis": "diagnosis category contributes baseline clinical severity information",
}


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar values into JSON-safe Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def get_caution_flags(feature: str, value: Any) -> list[str]:
    """Return caution flags for features that need careful interpretation."""
    flags = []

    if feature == "icu_id":
        flags.append("Non-clinical unit/location identifier; interpret cautiously.")

    if feature == "pre_icu_los_days" and value is not None and value < 0:
        flags.append("Negative pre-ICU length of stay; may reflect timing or data quality issue.")

    zero_vital_features = {
        "d1_heartrate_min",
        "d1_resprate_min",
        "h1_resprate_min",
    }

    if feature in zero_vital_features and value == 0:
        flags.append("Zero-valued vital sign; may reflect extreme clinical event or recording artifact.")

    return flags


def _format_evidence_records(local_df: pd.DataFrame, direction: str) -> list[dict[str, Any]]:
    records = []

    for _, row in local_df.iterrows():
        feature = row["feature"]
        value = make_json_safe(row["value"])
        shap_value = float(row["shap_value"])

        records.append(
            {
                "feature": feature,
                "value": value,
                "shap_value": shap_value,
                "direction": direction,
                "clinical_meaning": CLINICAL_MEANING_MAP.get(
                    feature,
                    "No predefined clinical interpretation available.",
                ),
                "caution_flags": get_caution_flags(feature, value),
            }
        )

    return records


def build_evidence_packet(
    local_explanation: pd.DataFrame,
    prediction_row: pd.Series | dict[str, Any],
    patient_label: str,
    test_row_index: int | None = None,
    y_true: int | None = None,
    prediction_type: str | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Build a structured evidence packet from local SHAP and prediction outputs."""
    prediction = dict(prediction_row)

    y_pred = int(prediction["prediction"])
    y_proba = float(prediction["death_probability"])

    if prediction_type is None and y_true is not None:
        if y_true == 1 and y_pred == 1:
            prediction_type = "TP"
        elif y_true == 1 and y_pred == 0:
            prediction_type = "FN"
        elif y_true == 0 and y_pred == 1:
            prediction_type = "FP"
        elif y_true == 0 and y_pred == 0:
            prediction_type = "TN"

    risk_increasing_df = (
        local_explanation
        .sort_values("shap_value", ascending=False)
        .head(top_n)
        [["feature", "value", "shap_value"]]
    )

    risk_decreasing_df = (
        local_explanation
        .sort_values("shap_value", ascending=True)
        .head(top_n)
        [["feature", "value", "shap_value"]]
    )

    packet = {
        "patient_label": patient_label,
        "prediction": {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "threshold": float(prediction["threshold"]),
        },
        "risk_increasing_evidence": _format_evidence_records(
            risk_increasing_df,
            direction="risk_increasing",
        ),
        "risk_decreasing_evidence": _format_evidence_records(
            risk_decreasing_df,
            direction="risk_decreasing",
        ),
    }

    if test_row_index is not None:
        packet["test_row_index"] = int(test_row_index)

    if y_true is not None:
        packet["prediction"]["y_true"] = int(y_true)

    if prediction_type is not None:
        packet["prediction"]["prediction_type"] = prediction_type

    return packet
