"""Run the test patient pipeline and generate an LLM explanation."""
from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.llm import check_forbidden_phrases, generate_explanation, revise_explanation
from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.preprocessing import ICUPreprocessor
from src.prompts import build_explanation_prompt


def main() -> None:
    output_dir = ROOT / "reports/pipeline_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data/raw/training_v2.csv")

    raw_train, raw_test = train_test_split(
        raw,
        test_size=0.2,
        random_state=42,
        stratify=raw["hospital_death"],
    )

    preprocessor = ICUPreprocessor()
    preprocessor.fit(raw_train)

    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    patient_position = 0
    patient_label = "test_patient_0"
    raw_patient = raw_test.iloc[[patient_position]]

    result = run_patient_pipeline(
        raw_patient=raw_patient,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        patient_label=patient_label,
        test_row_index=int(raw_patient.index[0]),
        y_true=int(raw_patient["hospital_death"].iloc[0]),
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

    explanation_path = output_dir / f"{patient_label}_llm_explanation.txt"
    prompt_path = output_dir / f"{patient_label}_llm_prompt.txt"
    evidence_path = output_dir / f"{patient_label}_llm_evidence.json"
    validation_path = output_dir / f"{patient_label}_llm_validation.json"
    revised_explanation_path = output_dir / f"{patient_label}_llm_revised_explanation.txt"
    revised_validation_path = output_dir / f"{patient_label}_llm_revised_validation.json"

    with open(explanation_path, "w") as f:
        f.write(explanation)

    with open(prompt_path, "w") as f:
        f.write(prompt)

    with open(evidence_path, "w") as f:
        json.dump(result["evidence_packet"], f, indent=2)

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

    print("=== Test Patient LLM Demo Saved ===")
    print(f"Patient label : {patient_label}")
    print(f"Explanation   : {explanation_path.relative_to(ROOT)}")
    print(f"Prompt        : {prompt_path.relative_to(ROOT)}")
    print(f"Evidence      : {evidence_path.relative_to(ROOT)}")
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
