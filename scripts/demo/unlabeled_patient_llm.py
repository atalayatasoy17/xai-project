"""Run the saved-artifact pipeline and LLM revision loop on one unlabeled patient."""

import argparse
from pathlib import Path
import json
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.llm import generate_explanation, revise_until_valid
from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.prompts import build_explanation_prompt
from src.validation import validate_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the saved-artifact pipeline and LLM revision loop on one unlabeled patient.",
    )
    parser.add_argument(
        "--patient-position",
        type=int,
        default=0,
        help="Zero-based row position in data/raw/unlabeled.csv.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the LLM explanation without writing report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "reports/08_unlabeled_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_unlabeled = pd.read_csv(ROOT / "data/raw/unlabeled.csv")

    preprocessor = joblib.load(ROOT / "models/icu_preprocessor.pkl")
    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    patient_position = args.patient_position
    if patient_position < 0 or patient_position >= len(raw_unlabeled):
        raise ValueError(
            f"patient_position must be between 0 and {len(raw_unlabeled) - 1}."
        )

    patient_label = f"unlabeled_patient_{patient_position}"
    raw_patient = raw_unlabeled.iloc[[patient_position]]

    result = run_patient_pipeline(
        raw_patient=raw_patient,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        patient_label=patient_label,
        test_row_index=int(raw_patient.index[0]),
        y_true=None,
        top_n=8,
    )

    evidence_packet = result["evidence_packet"]
    prompt = build_explanation_prompt(evidence_packet)
    explanation = generate_explanation(prompt)

    validation_report = validate_explanation(
        text=explanation,
        evidence_packet=evidence_packet,
    )

    revised_explanation, revised_validation_report, revision_rounds = revise_until_valid(
        original_prompt=prompt,
        generated_explanation=explanation,
        evidence_packet=evidence_packet,
        validation_report=validation_report,
    )

    evidence_path = output_dir / f"{patient_label}_llm_evidence.json"
    prompt_path = output_dir / f"{patient_label}_llm_prompt.txt"
    explanation_path = output_dir / f"{patient_label}_llm_explanation.txt"
    validation_path = output_dir / f"{patient_label}_llm_validation.json"
    revised_explanation_path = output_dir / f"{patient_label}_llm_revised_explanation.txt"
    revised_validation_path = output_dir / f"{patient_label}_llm_revised_validation.json"

    if not args.no_save:
        with open(evidence_path, "w") as f:
            json.dump(evidence_packet, f, indent=2)

        with open(prompt_path, "w") as f:
            f.write(prompt)

        with open(explanation_path, "w") as f:
            f.write(explanation)

        with open(validation_path, "w") as f:
            json.dump(validation_report, f, indent=2)

        if revised_explanation is not None:
            with open(revised_explanation_path, "w") as f:
                f.write(revised_explanation)

            with open(revised_validation_path, "w") as f:
                json.dump(
                    {
                        **revised_validation_report,
                        "revision_rounds": revision_rounds,
                    },
                    f,
                    indent=2,
                )
        else:
            revised_explanation_path.unlink(missing_ok=True)
            revised_validation_path.unlink(missing_ok=True)

    title = (
        "=== Unlabeled Patient LLM Demo ==="
        if args.no_save
        else "=== Unlabeled Patient LLM Demo Saved ==="
    )
    print(title)
    print(f"Patient label : {patient_label}")
    if not args.no_save:
        print(f"Evidence      : {evidence_path.relative_to(ROOT)}")
        print(f"Prompt        : {prompt_path.relative_to(ROOT)}")
        print(f"Explanation   : {explanation_path.relative_to(ROOT)}")
        print(f"Validation    : {validation_path.relative_to(ROOT)}")

    print(f"Validation passed: {validation_report['passed']}")
    print(f"Revision required: {validation_report['revision_required']}")
    print(f"Validation score : {validation_report['deterministic_validation_score']}")

    if revised_explanation is not None:
        if not args.no_save:
            print(f"Revised explanation: {revised_explanation_path.relative_to(ROOT)}")
            print(f"Revised validation : {revised_validation_path.relative_to(ROOT)}")
        print(f"Revision rounds    : {revision_rounds}")
        print(f"Revised validation passed: {revised_validation_report['passed']}")
        print(f"Revised revision required: {revised_validation_report['revision_required']}")
        print(
            "Revised validation score : "
            f"{revised_validation_report['deterministic_validation_score']}"
        )

    print()
    print("=== Generated Explanation Preview ===")
    print(explanation)

    if revised_explanation is not None:
        print()
        print("=== Revised Explanation Preview ===")
        print(revised_explanation)


if __name__ == "__main__":
    main()
