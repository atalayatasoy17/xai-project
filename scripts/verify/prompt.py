"""Verify prompt construction from an end-to-end patient evidence packet."""
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.preprocessing import ICUPreprocessor
from src.prompts import build_explanation_prompt


def main() -> None:
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
    raw_patient = raw_test.iloc[[patient_position]]

    result = run_patient_pipeline(
        raw_patient=raw_patient,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        patient_label="test_patient_0",
        test_row_index=int(raw_patient.index[0]),
        y_true=int(raw_patient["hospital_death"].iloc[0]),
        top_n=5,
    )

    prompt = build_explanation_prompt(result["evidence_packet"])

    print("=== Prompt Verification ===")
    print(f"Prompt length: {len(prompt)} characters")
    print()
    print(prompt)


if __name__ == "__main__":
    main()
