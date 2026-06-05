"""Fit, save, and verify the ICU preprocessing artifact."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

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

    output_path = ROOT / "models/icu_preprocessor.pkl"
    joblib.dump(preprocessor, output_path)

    loaded_preprocessor = joblib.load(output_path)
    recreated_X_test = loaded_preprocessor.transform(raw_test)
    reloaded_X_train = loaded_preprocessor.transform(raw_train)

    dropped_cols = (
        loaded_preprocessor.id_cols
        + loaded_preprocessor.leakage_cols
        + [loaded_preprocessor.target_col]
    )
    leaked_columns_present = [
        column
        for column in dropped_cols
        if column in reloaded_X_train.columns or column in recreated_X_test.columns
    ]
    train_test_columns_match = list(reloaded_X_train.columns) == list(recreated_X_test.columns)
    feature_names_match = list(recreated_X_test.columns) == loaded_preprocessor.feature_names_

    print("=== Preprocessor Artifact Saved ===")
    print(f"Path                    : {output_path.relative_to(ROOT)}")
    print(f"Processed train shape   : {reloaded_X_train.shape}")
    print(f"Processed test shape    : {recreated_X_test.shape}")
    print(f"Feature count           : {len(loaded_preprocessor.feature_names_)}")
    print(f"Train/test columns match: {train_test_columns_match}")
    print(f"Feature names match     : {feature_names_match}")
    print(f"Leaked columns present  : {leaked_columns_present}")

    if leaked_columns_present:
        raise RuntimeError("Saved preprocessor retained dropped ID/leakage/target columns.")

    if not train_test_columns_match or not feature_names_match:
        raise RuntimeError("Saved preprocessor has an inconsistent feature schema.")


if __name__ == "__main__":
    main()
