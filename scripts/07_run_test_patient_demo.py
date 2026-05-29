"""Run the validated patient pipeline on one held-out test patient."""
from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.preprocessing import ICUPreprocessor
from src.prompts import build_explanation_prompt


def main() -> None:
    output_dir = ROOT / "reports/07_pipeline_demo"
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

    evidence_path = output_dir / f"{patient_label}_evidence.json"
    prompt_path = output_dir / f"{patient_label}_prompt.txt"
    prediction_path = output_dir / f"{patient_label}_prediction.json"

    with open(evidence_path, "w") as f:
        json.dump(result["evidence_packet"], f, indent=2)

    with open(prompt_path, "w") as f:
        f.write(prompt)

    with open(prediction_path, "w") as f:
        json.dump(result["prediction"], f, indent=2)

    print("=== Test Patient Pipeline Demo Saved ===")
    print(f"Patient label   : {patient_label}")
    print(f"Original index  : {int(raw_patient.index[0])}")
    print(f"Evidence packet : {evidence_path.relative_to(ROOT)}")
    print(f"Prompt          : {prompt_path.relative_to(ROOT)}")
    print(f"Prediction      : {prediction_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()