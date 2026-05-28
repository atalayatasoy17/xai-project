"""SHAP utilities for the ICU mortality pipeline."""
from __future__ import annotations

import pandas as pd
import shap


def build_tree_explainer(model):
    """Create a SHAP TreeExplainer for the trained tree-based model."""
    return shap.TreeExplainer(model)


def get_positive_class_shap_values(explainer, X: pd.DataFrame):
    """Return SHAP values for the positive class when SHAP returns a class list."""
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        return shap_values[1]

    return shap_values


def explain_patient(model, X_patient: pd.DataFrame) -> pd.DataFrame:
    """Create a local SHAP explanation table for one processed patient row."""
    if len(X_patient) != 1:
        raise ValueError("X_patient must contain exactly one row.")

    explainer = build_tree_explainer(model)
    shap_values = get_positive_class_shap_values(explainer, X_patient)

    explanation = pd.DataFrame(
        {
            "feature": X_patient.columns,
            "value": X_patient.iloc[0].values,
            "shap_value": shap_values[0],
        }
    )

    explanation["abs_shap_value"] = explanation["shap_value"].abs()
    explanation["direction"] = explanation["shap_value"].apply(
        lambda value: "risk_increasing" if value > 0 else "risk_decreasing"
    )

    return explanation.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)