from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evidence import build_evidence_packet
from src.explainability import explain_patient
from src.prediction import load_model, load_threshold, predict_mortality
from src.preprocessing import ICUPreprocessor


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
    X_test = preprocessor.transform(raw_test)

    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    patient_position = 0
    X_patient = X_test.iloc[[patient_position]]

    prediction = predict_mortality(model, X_patient, threshold=threshold).iloc[0]
    local_explanation = explain_patient(model, X_patient)

    y_true = int(raw_test["hospital_death"].iloc[patient_position])
    original_test_index = int(raw_test.index[patient_position])

    evidence_packet = build_evidence_packet(
        local_explanation=local_explanation,
        prediction_row=prediction,
        patient_label="test_patient_0",
        test_row_index=original_test_index,
        y_true=y_true,
        top_n=5,
    )

    print("=== Evidence Packet Verification ===")
    print(json.dumps(evidence_packet, indent=2))


if __name__ == "__main__":
    main()