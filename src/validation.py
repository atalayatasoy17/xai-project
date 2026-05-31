"""Deterministic validation utilities for LLM-generated explanations."""
from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = [
    "Prediction summary",
    "Main risk-increasing factors",
    "Main risk-decreasing factors",
    "Caution notes",
    "Overall interpretation",
]


FORBIDDEN_PHRASES = [
    "stable",
    "stability",
    "adequate",
    "protective",
    "normal",
    "abnormal",
    "moderate",
    "favorable",
    "unfavorable",
]


TRUE_LABEL_LEAKAGE_PHRASES = [
    "true label",
    "true outcome",
    "correct prediction",
    "incorrect prediction",
    "consistent with the true outcome",
    "inconsistent with the true outcome",
    "patient survived",
    "patient died",
    "survived",
    "died",
]


def _find_phrases(text: str, phrases: list[str]) -> list[str]:
    """Find phrase matches using word-boundary regex."""
    found = []

    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, text.lower()):
            found.append(phrase)

    return found


def check_forbidden_phrases(text: str) -> dict[str, Any]:
    """Check unsupported or risky wording in an explanation."""
    found = _find_phrases(text, FORBIDDEN_PHRASES)

    return {
        "passed": len(found) == 0,
        "found": found,
    }


def check_true_label_leakage(text: str) -> dict[str, Any]:
    """Check whether an explanation references true outcome information."""
    found = _find_phrases(text, TRUE_LABEL_LEAKAGE_PHRASES)

    return {
        "passed": len(found) == 0,
        "found": found,
    }
def check_section_structure(text: str) -> dict[str, Any]:
    """Check whether the explanation contains the required section headings."""
    missing = []

    lowered_text = text.lower()

    for section in REQUIRED_SECTIONS:
        if section.lower() not in lowered_text:
            missing.append(section)

    return {
        "passed": len(missing) == 0,
        "missing": missing,
        "required": REQUIRED_SECTIONS,
    }

def _extract_probability_candidates(text: str) -> list[float]:
    """Extract decimal and percentage probability candidates from text."""
    candidates = []

    percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    for match in percent_matches:
        value = float(match) / 100
        if 0 <= value <= 1:
            candidates.append(value)

    decimal_matches = re.findall(r"\b0\.\d+\b|\b1\.0+\b", text)
    for match in decimal_matches:
        value = float(match)
        if 0 <= value <= 1:
            candidates.append(value)

    return candidates


def check_prediction_consistency(
    text: str,
    evidence_packet: dict[str, Any],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Check whether the explanation mentions a probability consistent with the model output."""
    expected_probability = float(evidence_packet["prediction"]["y_proba"])
    candidates = _extract_probability_candidates(text)

    matched_probability = None
    for candidate in candidates:
        if abs(candidate - expected_probability) <= tolerance:
            matched_probability = candidate
            break

    return {
        "passed": matched_probability is not None,
        "expected_probability": expected_probability,
        "matched_probability": matched_probability,
        "candidates": candidates,
        "tolerance": tolerance,
    }

CAUTION_LANGUAGE = [
    "caution",
    "cautiously",
    "careful",
    "carefully",
    "interpret",
    "interpreted",
    "non-clinical",
    "unit-level",
    "location",
    "artifact",
    "recording",
    "data quality",
]

CAUTION_FEATURE_ALIASES = {
    "icu_id": [
        "icu_id",
        "icu unit identifier",
        "unit identifier",
        "unit-level",
    ],
    "d1_heartrate_min": [
        "d1_heartrate_min",
        "minimum heart rate",
        "heart rate",
    ],
    "d1_resprate_min": [
        "d1_resprate_min",
        "day 1 minimum respiratory rate",
        "minimum respiratory rate",
    ],
    "h1_resprate_min": [
        "h1_resprate_min",
        "hour 1 minimum respiratory rate",
        "minimum respiratory rate",
    ],
    "pre_icu_los_days": [
        "pre_icu_los_days",
        "pre-icu length of stay",
        "pre icu length of stay",
        "length of stay",
    ],
}


def _get_caution_features(evidence_packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evidence records that contain caution flags."""
    records = []

    for group in ["risk_increasing_evidence", "risk_decreasing_evidence"]:
        for record in evidence_packet.get(group, []):
            if record.get("caution_flags"):
                records.append(record)

    return records

def _split_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like chunks."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _find_caution_alias_match(
    caution_section: str,
    feature: str,
) -> dict[str, str] | None:
    """Find a feature code or alias in the caution section with nearby caution language."""
    aliases = CAUTION_FEATURE_ALIASES.get(feature, [feature])

    for sentence in _split_sentences(caution_section):
        lowered_sentence = sentence.lower()

        matched_alias = None
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias.lower()) + r"\b", lowered_sentence):
                matched_alias = alias
                break

        if matched_alias is None:
            continue

        caution_language_present = any(
            phrase in lowered_sentence for phrase in CAUTION_LANGUAGE
        )

        if caution_language_present:
            return {
                "feature": feature,
                "matched_text": matched_alias,
                "matched_sentence": sentence,
            }

    return None

def check_caution_mentions(text: str, evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Check whether caution-flagged features are mentioned with cautious language."""
    caution_records = _get_caution_features(evidence_packet)
    caution_section = _extract_section_text(
        text=text,
        start_heading="Caution notes",
        following_headings=["Overall interpretation"],
    )

    missing_features = []
    mention_matches = []

    for record in caution_records:
        feature = record["feature"]
        match = _find_caution_alias_match(
            caution_section=caution_section,
            feature=feature,
        )

        if match is not None:
            mention_matches.append(match)
        else:
            missing_features.append(feature)

    return {
        "passed": len(missing_features) == 0,
        "missing_features": missing_features,
        "mention_matches": mention_matches,
        "caution_feature_count": len(caution_records),
        "matching_mode": "alias_aware_caution_section",
    }

def _get_evidence_features(evidence_packet: dict[str, Any]) -> set[str]:
    """Return feature names present in the evidence packet."""
    features = set()

    for group in ["risk_increasing_evidence", "risk_decreasing_evidence"]:
        for record in evidence_packet.get(group, []):
            feature = record.get("feature")
            if feature:
                features.add(str(feature))

    return features


def _extract_feature_like_tokens(text: str) -> set[str]:
    """Extract exact feature-like tokens from text.

    This v1 check focuses on technical feature names such as d1_spo2_min.
    It does not detect paraphrases such as "minimum oxygen saturation".
    """
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b", text))


def check_feature_grounding(text: str, evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Check whether exact feature names mentioned in text are present in evidence."""
    evidence_features = _get_evidence_features(evidence_packet)
    mentioned_features = _extract_feature_like_tokens(text)

    ungrounded_features = sorted(mentioned_features - evidence_features)
    grounded_features = sorted(mentioned_features & evidence_features)

    return {
        "passed": len(ungrounded_features) == 0,
        "ungrounded_features": ungrounded_features,
        "grounded_features": grounded_features,
        "mentioned_features": sorted(mentioned_features),
        "evidence_features": sorted(evidence_features),
        "matching_mode": "exact_feature_name",
    }

def _extract_section_text(
    text: str,
    start_heading: str,
    following_headings: list[str],
) -> str:
    """Extract text under a section heading until the next known heading."""
    lowered_text = text.lower()
    start_index = lowered_text.find(start_heading.lower())

    if start_index == -1:
        return ""

    section_start = start_index + len(start_heading)
    section_end = len(text)

    for heading in following_headings:
        next_index = lowered_text.find(heading.lower(), section_start)
        if next_index != -1:
            section_end = min(section_end, next_index)

    return text[section_start:section_end]


def check_direction_consistency(
    text: str,
    evidence_packet: dict[str, Any],
) -> dict[str, Any]:
    """Check exact feature-name direction consistency against SHAP direction."""
    risk_increasing_features = {
        str(record["feature"])
        for record in evidence_packet.get("risk_increasing_evidence", [])
        if record.get("feature")
    }
    risk_decreasing_features = {
        str(record["feature"])
        for record in evidence_packet.get("risk_decreasing_evidence", [])
        if record.get("feature")
    }

    increasing_section = _extract_section_text(
        text=text,
        start_heading="Main risk-increasing factors",
        following_headings=[
            "Main risk-decreasing factors",
            "Caution notes",
            "Overall interpretation",
        ],
    )
    decreasing_section = _extract_section_text(
        text=text,
        start_heading="Main risk-decreasing factors",
        following_headings=[
            "Caution notes",
            "Overall interpretation",
        ],
    )

    all_evidence_features = risk_increasing_features | risk_decreasing_features

    mentioned_in_increasing = {
        feature
        for feature in all_evidence_features
        if re.search(
            r"\b" + re.escape(feature.lower()) + r"\b",
            increasing_section.lower(),
        )
    }

    mentioned_in_decreasing = {
        feature
        for feature in all_evidence_features
        if re.search(
            r"\b" + re.escape(feature.lower()) + r"\b",
            decreasing_section.lower(),
        )
    }

    direction_errors = []

    for feature in sorted(mentioned_in_increasing):
        if feature in risk_decreasing_features:
            direction_errors.append(
                {
                    "feature": feature,
                    "expected": "risk_decreasing",
                    "mentioned_as": "risk_increasing",
                }
            )

    for feature in sorted(mentioned_in_decreasing):
        if feature in risk_increasing_features:
            direction_errors.append(
                {
                    "feature": feature,
                    "expected": "risk_increasing",
                    "mentioned_as": "risk_decreasing",
                }
            )

    return {
        "passed": len(direction_errors) == 0,
        "direction_errors": direction_errors,
        "mentioned_in_risk_increasing_section": sorted(mentioned_in_increasing),
        "mentioned_in_risk_decreasing_section": sorted(mentioned_in_decreasing),
        "matching_mode": "exact_feature_name",
    }

DETERMINISTIC_SCORE_WEIGHTS = {
    "faithfulness_no_hallucination": 0.30,
    "caution_awareness": 0.20,
    "completeness": 0.15,
}


def _score_from_failures(base_score: int, failures: int) -> int:
    """Convert number of failures into a bounded 1-5 score."""
    return max(1, base_score - failures)


def _compute_dimension_scores(checks: dict[str, Any]) -> dict[str, int]:
    """Compute deterministic rubric dimension scores from check results."""
    faithfulness_failures = 0

    if not checks["forbidden_phrases"]["passed"]:
        faithfulness_failures += 1
    if not checks["true_label_leakage"]["passed"]:
        faithfulness_failures += 2
    if not checks["prediction_consistency"]["passed"]:
        faithfulness_failures += 1
    if not checks["feature_grounding"]["passed"]:
        faithfulness_failures += 1
    if not checks["direction_consistency"]["passed"]:
        faithfulness_failures += 2

    completeness_failures = 0
    if not checks["section_structure"]["passed"]:
        completeness_failures += len(checks["section_structure"]["missing"])

    caution_failures = 0
    if not checks["caution_mentions"]["passed"]:
        caution_failures += len(checks["caution_mentions"]["missing_features"])

    return {
        "faithfulness_no_hallucination": _score_from_failures(5, faithfulness_failures),
        "caution_awareness": _score_from_failures(5, caution_failures),
        "completeness": _score_from_failures(5, completeness_failures),
    }


def _compute_deterministic_validation_score(dimension_scores: dict[str, int]) -> float:
    """Compute normalized weighted deterministic score over covered dimensions."""
    total_weight = sum(DETERMINISTIC_SCORE_WEIGHTS.values())

    weighted_sum = sum(
        (DETERMINISTIC_SCORE_WEIGHTS[dimension] / total_weight) * score
        for dimension, score in dimension_scores.items()
    )

    return round(weighted_sum, 3)


def _build_revision_feedback(checks: dict[str, Any]) -> list[str]:
    """Build actionable revision feedback from failed validation checks."""
    feedback = []

    forbidden = checks["forbidden_phrases"]["found"]
    if forbidden:
        feedback.append(
            "Remove or rephrase unsupported/risky wording: "
            + ", ".join(forbidden)
            + "."
        )

    leakage = checks["true_label_leakage"]["found"]
    if leakage:
        feedback.append(
            "Remove true-label or outcome-related wording: "
            + ", ".join(leakage)
            + "."
        )

    if not checks["prediction_consistency"]["passed"]:
        expected_probability = checks["prediction_consistency"]["expected_probability"]
        feedback.append(
            f"Correct the predicted mortality probability to match the evidence: {expected_probability:.4f}."
        )

    missing_caution = checks["caution_mentions"]["missing_features"]
    if missing_caution:
        feedback.append(
            "Mention caution in the Caution notes section using the exact feature name(s): "
            + ", ".join(missing_caution)
            + "."
        )

    ungrounded = checks["feature_grounding"]["ungrounded_features"]
    if ungrounded:
        feedback.append(
            "Remove features not present in the evidence packet: "
            + ", ".join(ungrounded)
            + "."
        )

    direction_errors = checks["direction_consistency"]["direction_errors"]
    for error in direction_errors:
        feedback.append(
            f"Fix direction for {error['feature']}: expected {error['expected']}, "
            f"but it was mentioned as {error['mentioned_as']}."
        )

    missing_sections = checks["section_structure"]["missing"]
    if missing_sections:
        feedback.append(
            "Restore missing required sections: "
            + ", ".join(missing_sections)
            + "."
        )

    return feedback


def validate_explanation(
    text: str,
    evidence_packet: dict[str, Any],
) -> dict[str, Any]:
    """Run all deterministic validation checks and return a structured report."""
    checks = {
        "forbidden_phrases": check_forbidden_phrases(text),
        "true_label_leakage": check_true_label_leakage(text),
        "section_structure": check_section_structure(text),
        "prediction_consistency": check_prediction_consistency(text, evidence_packet),
        "caution_mentions": check_caution_mentions(text, evidence_packet),
        "feature_grounding": check_feature_grounding(text, evidence_packet),
        "direction_consistency": check_direction_consistency(text, evidence_packet),
    }

    dimension_scores = _compute_dimension_scores(checks)
    deterministic_score = _compute_deterministic_validation_score(dimension_scores)
    revision_feedback = _build_revision_feedback(checks)

    passed = all(check["passed"] for check in checks.values())
    revision_required = not passed

    return {
        "passed": passed,
        "revision_required": revision_required,
        "deterministic_validation_score": deterministic_score,
        "dimension_scores": dimension_scores,
        "checks": checks,
        "revision_feedback": revision_feedback,
        "schema_version": "1.0",
    }
