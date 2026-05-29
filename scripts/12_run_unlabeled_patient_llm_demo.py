"""Run the saved-artifact pipeline and LLM revision loop on one unlabeled patient."""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.llm import check_forbidden_phrases, generate_explanation, revise_explanation
from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.prompts import build_explanation_prompt


def main() -> None:
    output_dir = ROOT / "reports/08_unlabeled_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_unlabeled = pd.read_csv(ROOT / "data/raw/unlabeled.csv")

    preprocessor = joblib.load(ROOT / "models/icu_preprocessor.pkl")
    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    patient_position = 0
    patient_label = "unlabeled_patient_0"
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

    prompt = build_explanation_prompt(result["evidence_packet"])
    explanation = generate_explanation(prompt)
    forbidden_phrases = check_forbidden_phrases(explanation)

    revised_explanation = None
    revised_forbidden_phrases = None
    if forbidden_phrases:
        revised_explanation = revise_explanation(
            original_prompt=prompt,
            generated_explanation=explanation,
            forbidden_phrases=forbidden_phrases,
        )
        revised_forbidden_phrases = check_forbidden_phrases(revised_explanation)

    evidence_path = output_dir / f"{patient_label}_llm_evidence.json"
    prompt_path = output_dir / f"{patient_label}_llm_prompt.txt"
    explanation_path = output_dir / f"{patient_label}_llm_explanation.txt"
    validation_path = output_dir / f"{patient_label}_llm_validation.json"
    revised_explanation_path = output_dir / f"{patient_label}_llm_revised_explanation.txt"
    revised_validation_path = output_dir / f"{patient_label}_llm_revised_validation.json"

    with open(evidence_path, "w") as f:
        json.dump(result["evidence_packet"], f, indent=2)

    with open(prompt_path, "w") as f:
        f.write(prompt)

    with open(explanation_path, "w") as f:
        f.write(explanation)

    with open(validation_path, "w") as f:
        json.dump({"forbidden_phrases": forbidden_phrases}, f, indent=2)

    if revised_explanation is not None:
        with open(revised_explanation_path, "w") as f:
            f.write(revised_explanation)

        with open(revised_validation_path, "w") as f:
            json.dump(
                {"forbidden_phrases": revised_forbidden_phrases},
                f,
                indent=2,
            )

    print("=== Unlabeled Patient LLM Demo Saved ===")
    print(f"Patient label : {patient_label}")
    print(f"Evidence      : {evidence_path.relative_to(ROOT)}")
    print(f"Prompt        : {prompt_path.relative_to(ROOT)}")
    print(f"Explanation   : {explanation_path.relative_to(ROOT)}")
    print(f"Validation    : {validation_path.relative_to(ROOT)}")
    print(f"Forbidden phrases found: {forbidden_phrases}")
    if revised_explanation is not None:
        print(f"Revised explanation: {revised_explanation_path.relative_to(ROOT)}")
        print(f"Revised validation : {revised_validation_path.relative_to(ROOT)}")
        print(f"Revised forbidden phrases found: {revised_forbidden_phrases}")
    print()
    print("=== Generated Explanation Preview ===")
    print(explanation)
    if revised_explanation is not None:
        print()
        print("=== Revised Explanation Preview ===")
        print(revised_explanation)


if __name__ == "__main__":
    main()
