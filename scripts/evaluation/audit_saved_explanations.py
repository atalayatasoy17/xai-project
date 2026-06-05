"""Audit saved LLM explanations with the deterministic validator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.validation import validate_explanation


AUDIT_TARGETS = [
    ROOT / "reports/07_pipeline_demo",
    ROOT / "reports/08_unlabeled_demo",
]

OUTPUT_DIR = ROOT / "reports/09_validation_audit"
OUTPUT_PATH = OUTPUT_DIR / "validation_audit_summary.csv"


def find_evidence_file(explanation_path: Path) -> Path | None:
    """Find the matching evidence JSON for a saved explanation file."""
    file_name = explanation_path.name

    if file_name.endswith("_revised_explanation.txt"):
        evidence_name = file_name.replace("_revised_explanation.txt", "_evidence.json")
    elif file_name.endswith("_explanation.txt"):
        evidence_name = file_name.replace("_explanation.txt", "_evidence.json")
    else:
        return None

    candidate = explanation_path.with_name(evidence_name)
    if candidate.exists():
        return candidate

    return None


def flatten_validation_result(
    explanation_path: Path,
    evidence_path: Path,
    validation_report: dict,
) -> dict:
    """Convert a nested validation report into one CSV-friendly row."""
    checks = validation_report["checks"]

    return {
        "case_id": explanation_path.stem,
        "source_file": str(explanation_path.relative_to(ROOT)),
        "evidence_file": str(evidence_path.relative_to(ROOT)),
        "passed": validation_report["passed"],
        "revision_required": validation_report["revision_required"],
        "deterministic_validation_score": validation_report[
            "deterministic_validation_score"
        ],
        "faithfulness_no_hallucination": validation_report["dimension_scores"][
            "faithfulness_no_hallucination"
        ],
        "caution_awareness": validation_report["dimension_scores"][
            "caution_awareness"
        ],
        "completeness": validation_report["dimension_scores"]["completeness"],
        "forbidden_phrases": checks["forbidden_phrases"]["found"],
        "true_label_leakage": checks["true_label_leakage"]["found"],
        "missing_sections": checks["section_structure"]["missing"],
        "prediction_consistency_passed": checks["prediction_consistency"]["passed"],
        "missing_caution_mentions": checks["caution_mentions"]["missing_features"],
        "ungrounded_features": checks["feature_grounding"]["ungrounded_features"],
        "direction_errors": checks["direction_consistency"]["direction_errors"],
    }


def audit_explanation(explanation_path: Path) -> dict | None:
    """Validate one explanation file against its matching evidence packet."""
    evidence_path = find_evidence_file(explanation_path)

    if evidence_path is None:
        return None

    with open(explanation_path) as f:
        explanation = f.read()

    with open(evidence_path) as f:
        evidence_packet = json.load(f)

    validation_report = validate_explanation(
        text=explanation,
        evidence_packet=evidence_packet,
    )

    return flatten_validation_result(
        explanation_path=explanation_path,
        evidence_path=evidence_path,
        validation_report=validation_report,
    )


def main() -> None:
    """Run validation audit over saved pipeline explanation files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for target_dir in AUDIT_TARGETS:
        if not target_dir.exists():
            continue

        explanation_files = sorted(target_dir.glob("*_explanation.txt"))

        for explanation_path in explanation_files:
            row = audit_explanation(explanation_path)
            if row is not None:
                rows.append(row)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    print("=== Validation Audit Saved ===")
    print(f"Rows       : {len(audit_df)}")
    print(f"Output     : {OUTPUT_PATH.relative_to(ROOT)}")

    if len(audit_df) > 0:
        print()
        print("=== Audit Summary ===")
        print(audit_df[["case_id", "passed", "revision_required", "deterministic_validation_score"]])


if __name__ == "__main__":
    main()