"""Prompt construction utilities for evidence-based LLM explanations."""
from __future__ import annotations

import json
from typing import Any


def build_explanation_prompt(evidence_packet: dict[str, Any]) -> str:
    """Build an evidence-grounded LLM prompt from an evidence packet."""
    prediction = evidence_packet["prediction"]

    case_type = prediction.get("prediction_type", "unknown")
    true_label = prediction.get("y_true", "unknown")
    predicted_label = prediction["y_pred"]
    probability = prediction["y_proba"]

    risk_increasing = json.dumps(
        evidence_packet["risk_increasing_evidence"],
        indent=2,
        ensure_ascii=False,
    )
    risk_decreasing = json.dumps(
        evidence_packet["risk_decreasing_evidence"],
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are a clinical AI explanation assistant.

Your task is to explain why the mortality prediction model made its prediction for this ICU patient.

Use only the provided evidence.
Do not invent clinical facts.
Do not mention or use the true label in the explanation.
Do not say the prediction was correct, incorrect, consistent with the true outcome, or inconsistent with the true outcome.
Do not add measurement units unless they are explicitly present in the evidence.
Do not invent normal ranges, diagnoses, mechanisms, or clinical details beyond the provided clinical_meaning.
If clinical_meaning is "No predefined clinical interpretation available.", do not infer a clinical interpretation for that feature.
For features without explicit clinical_meaning, only state that they increased or decreased the model's predicted risk.
For features without explicit clinical_meaning, do not describe the value as low, high, normal, adequate, stable, unstable, elevated, reduced, or moderate.
For features with "No predefined clinical interpretation available.", use this exact style: "<feature> = <value> increased/decreased the model's predicted risk."
Do not paraphrase feature names when clinical_meaning is not available.
Do not use interpretive adjectives such as protective, adequate, stable, unstable, normal, abnormal, elevated, low, high, moderate, favorable, or unfavorable unless the provided clinical_meaning explicitly supports that wording.
Correctly distinguish risk-increasing and risk-decreasing evidence.
If a feature has a caution flag, mention it carefully.

Patient prediction:
- Case type: {case_type}
- True label: {true_label}
- Predicted label: {predicted_label}
- Predicted mortality probability: {probability:.4f}

Risk-increasing evidence:
{risk_increasing}

Risk-decreasing evidence:
{risk_decreasing}

Write the explanation with the following structure:

1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation

Keep the explanation clinically grounded and concise.
""".strip()

    return prompt
