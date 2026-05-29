from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

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

    prediction = predict_mortality(model, X_patient, threshold=threshold)
    explanation = explain_patient(model, X_patient)

    print("=== SHAP Explainability Verification ===")
    print(f"Patient position    : {patient_position}")
    print(f"Death probability   : {prediction['death_probability'].iloc[0]:.4f}")
    print(f"Prediction          : {prediction['prediction'].iloc[0]}")
    print(f"Threshold           : {threshold:.2f}")
    print()
    print("=== Top 10 Local SHAP Features ===")
    print(
        explanation[
            ["feature", "value", "shap_value", "abs_shap_value", "direction"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()