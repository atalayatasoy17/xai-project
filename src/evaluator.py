"""GPT-based subjective evaluation utilities for LLM explanations."""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from src.llm import load_openai_client


SUBJECTIVE_EVALUATOR_SYSTEM_MESSAGE = """
You are an evaluator of ICU mortality model explanations.

Your role is narrow:
- Evaluate only clinical plausibility and clarity.
- Do not evaluate deterministic safety checks such as label leakage, feature grounding, probability consistency, caution flags, or SHAP direction. Those are handled by a separate deterministic validator.
- Use the provided evidence packet and deterministic validation report as context.
- Do not use or infer the true outcome.
- Do not decide whether the model prediction is correct.
- Return only valid JSON.
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in evaluator response.")

    return json.loads(match.group(0))


def build_subjective_evaluation_prompt(
    explanation: str,
    evidence_packet: dict[str, Any],
    deterministic_validation_report: dict[str, Any],
) -> str:
    """Build a prompt for subjective clinical plausibility and clarity scoring."""
    compact_evidence = {
        "prediction": evidence_packet.get("prediction", {}),
        "risk_increasing_evidence": evidence_packet.get("risk_increasing_evidence", []),
        "risk_decreasing_evidence": evidence_packet.get("risk_decreasing_evidence", []),
    }
    compact_validation = {
        "passed": deterministic_validation_report.get("passed"),
        "revision_required": deterministic_validation_report.get("revision_required"),
        "deterministic_validation_score": deterministic_validation_report.get(
            "deterministic_validation_score"
        ),
        "dimension_scores": deterministic_validation_report.get("dimension_scores", {}),
        "revision_feedback": deterministic_validation_report.get("revision_feedback", []),
    }

    return f"""
Evaluate the following ICU mortality explanation.

Important:
- The deterministic validator has already checked label leakage, unsupported wording, feature grounding, SHAP direction, caution mentions, prediction consistency, and section structure.
- Your task is only to score clinical plausibility and clarity.
- Score each dimension from 1 to 5.
- 5 means excellent; 1 means poor.
- Be concise.
- Return only JSON with the requested schema.

Evidence packet:
{json.dumps(compact_evidence, indent=2)}

Deterministic validation report:
{json.dumps(compact_validation, indent=2)}

Explanation:
{explanation}

Return JSON exactly in this schema:
{{
  "clinical_plausibility": {{
    "score": 1,
    "rationale": "short rationale"
  }},
  "clarity": {{
    "score": 1,
    "rationale": "short rationale"
  }},
  "overall_comments": "short summary",
  "schema_version": "1.0"
}}
""".strip()


def evaluate_subjective_quality(
    explanation: str,
    evidence_packet: dict[str, Any],
    deterministic_validation_report: dict[str, Any],
    client: OpenAI | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Evaluate clinical plausibility and clarity using a GPT evaluator."""
    if client is None:
        client = load_openai_client()

    prompt = build_subjective_evaluation_prompt(
        explanation=explanation,
        evidence_packet=evidence_packet,
        deterministic_validation_report=deterministic_validation_report,
    )

    response = client.responses.create(
        model=model,
        instructions=SUBJECTIVE_EVALUATOR_SYSTEM_MESSAGE,
        input=prompt,
        temperature=temperature,
    )

    result = _extract_json_object(response.output_text)
    result["evaluator_model"] = model

    return result


def compute_hybrid_quality_score(
    deterministic_validation_report: dict[str, Any],
    subjective_evaluation_report: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """Combine deterministic and subjective rubric dimensions into one score."""
    dimension_scores = deterministic_validation_report["dimension_scores"]

    scores = {
        "faithfulness_no_hallucination": dimension_scores[
            "faithfulness_no_hallucination"
        ],
        "clinical_plausibility": subjective_evaluation_report[
            "clinical_plausibility"
        ]["score"],
        "caution_awareness": dimension_scores["caution_awareness"],
        "completeness": dimension_scores["completeness"],
        "clarity": subjective_evaluation_report["clarity"]["score"],
    }

    weighted_sum = sum(weights[dimension] * scores[dimension] for dimension in scores)
    total_weight = sum(weights[dimension] for dimension in scores)

    return round(weighted_sum / total_weight, 3)
