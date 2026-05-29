"""Fit, save, and verify the ICU preprocessing artifact."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import ICUPreprocessor


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/training_v2.csv")
    saved_X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")

    raw_train, raw_test = train_test_split(
        raw,
        test_size=0.2,
        random_state=42,
        stratify=raw["hospital_death"],
    )

    preprocessor = ICUPreprocessor()
    preprocessor.fit(raw_train)

    output_path = ROOT / "models/icu_preprocessor.pkl"
    joblib.dump(preprocessor, output_path)

    loaded_preprocessor = joblib.load(output_path)
    recreated_X_test = loaded_preprocessor.transform(raw_test)

    values_match = recreated_X_test.reset_index(drop=True).equals(saved_X_test)

    print("=== Preprocessor Artifact Saved ===")
    print(f"Path         : {output_path.relative_to(ROOT)}")
    print(f"Feature count: {len(loaded_preprocessor.feature_names_)}")
    print(f"X_test match : {values_match}")

    if not values_match:
        raise RuntimeError("Saved preprocessor does not reproduce saved X_test.")


if __name__ == "__main__":
    main()