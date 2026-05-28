from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import ICUPreprocessor


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/training_v2.csv")
    saved_X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")
    saved_y_test = pd.read_csv(ROOT / "data/processed/y_test.csv").squeeze()

    raw_train, raw_test = train_test_split(
        raw,
        test_size=0.2,
        random_state=42,
        stratify=raw["hospital_death"],
    )

    preprocessor = ICUPreprocessor()
    preprocessor.fit(raw_train)
    recreated_X_test = preprocessor.transform(raw_test)
    recreated_y_test = raw_test["hospital_death"]

    X_test_matches = recreated_X_test.reset_index(drop=True).equals(saved_X_test)
    y_test_matches = recreated_y_test.reset_index(drop=True).equals(saved_y_test)

    print("=== Preprocessing Verification ===")
    print(f"Recreated X_test shape : {recreated_X_test.shape}")
    print(f"Saved X_test shape     : {saved_X_test.shape}")
    print(f"Columns match          : {list(recreated_X_test.columns) == list(saved_X_test.columns)}")
    print(f"Values match           : {X_test_matches}")
    print(f"y_test matches         : {y_test_matches}")
    print()
    print("=== Learned Preprocessing Metadata ===")
    print(f"High-missing columns dropped : {len(preprocessor.high_missing_cols_)}")
    print(f"Missing indicators added     : {len(preprocessor.missing_indicator_cols_)}")
    print(f"Numeric columns              : {len(preprocessor.numeric_cols_)}")
    print(f"Categorical columns          : {len(preprocessor.categorical_cols_)}")
    print(f"Final feature count          : {len(preprocessor.feature_names_)}")


if __name__ == "__main__":
    main()
