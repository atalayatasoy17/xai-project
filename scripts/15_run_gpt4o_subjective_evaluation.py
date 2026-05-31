"""Run GPT-4o subjective evaluation for deterministically validated explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.evaluator import compute_hybrid_quality_score, evaluate_subjective_quality
from src.validation import validate_explanation


AUDIT_PATH = ROOT / "reports/09_validation_audit/validation_audit_summary.csv"
WEIGHTS_PATH = ROOT / "reports/05_evaluation/evaluation_weights.json"
OUTPUT_DIR = ROOT / "reports/10_gpt4o_evaluation"
EVALUATIONS_DIR = OUTPUT_DIR / "evaluations"
SUMMARY_PATH = OUTPUT_DIR / "gpt4o_subjective_evaluation_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clinical plausibility and clarity for explanations that "
            "already passed deterministic validation."
        )
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model used as the subjective evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of explanations to evaluate.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also evaluate explanations that failed deterministic validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run cases even when a saved evaluator JSON already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()


def safe_case_id(case_id: str) -> str:
    return case_id.replace("/", "_").replace(" ", "_")


def build_summary_row(
    audit_row: pd.Series,
    deterministic_report: dict,
    subjective_report: dict,
    hybrid_score: float,
    evaluation_path: Path,
) -> dict:
    return {
        "case_id": audit_row["case_id"],
        "source_file": audit_row["source_file"],
        "evidence_file": audit_row["evidence_file"],
        "deterministic_passed": deterministic_report["passed"],
        "deterministic_validation_score": deterministic_report[
            "deterministic_validation_score"
        ],
        "faithfulness_no_hallucination": deterministic_report["dimension_scores"][
            "faithfulness_no_hallucination"
        ],
        "caution_awareness": deterministic_report["dimension_scores"][
            "caution_awareness"
        ],
        "completeness": deterministic_report["dimension_scores"]["completeness"],
        "clinical_plausibility": subjective_report["clinical_plausibility"]["score"],
        "clarity": subjective_report["clarity"]["score"],
        "hybrid_quality_score": hybrid_score,
        "evaluator_model": subjective_report["evaluator_model"],
        "evaluation_file": str(evaluation_path.relative_to(ROOT)),
    }


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(AUDIT_PATH)
    weights = load_json(WEIGHTS_PATH)

    if not args.include_failed:
        audit_df = audit_df[audit_df["passed"] == True]  # noqa: E712

    audit_df = audit_df.sort_values("case_id")

    if args.limit is not None:
        audit_df = audit_df.head(args.limit)

    rows = []

    for _, audit_row in audit_df.iterrows():
        case_id = str(audit_row["case_id"])
        evaluation_path = EVALUATIONS_DIR / f"{safe_case_id(case_id)}_gpt4o_evaluation.json"

        explanation = load_text(ROOT / audit_row["source_file"])
        evidence_packet = load_json(ROOT / audit_row["evidence_file"])
        deterministic_report = validate_explanation(
            text=explanation,
            evidence_packet=evidence_packet,
        )

        if evaluation_path.exists() and not args.overwrite:
            subjective_report = load_json(evaluation_path)
        else:
            subjective_report = evaluate_subjective_quality(
                explanation=explanation,
                evidence_packet=evidence_packet,
                deterministic_validation_report=deterministic_report,
                model=args.model,
            )
            with open(evaluation_path, "w") as f:
                json.dump(subjective_report, f, indent=2)

        hybrid_score = compute_hybrid_quality_score(
            deterministic_validation_report=deterministic_report,
            subjective_evaluation_report=subjective_report,
            weights=weights,
        )

        rows.append(
            build_summary_row(
                audit_row=audit_row,
                deterministic_report=deterministic_report,
                subjective_report=subjective_report,
                hybrid_score=hybrid_score,
                evaluation_path=evaluation_path,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("=== GPT-4o Subjective Evaluation Saved ===")
    print(f"Rows   : {len(summary_df)}")
    print(f"Output : {SUMMARY_PATH.relative_to(ROOT)}")

    if len(summary_df) > 0:
        print()
        print("=== Evaluation Summary ===")
        print(
            summary_df[
                [
                    "case_id",
                    "clinical_plausibility",
                    "clarity",
                    "hybrid_quality_score",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
