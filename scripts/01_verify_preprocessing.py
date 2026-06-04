from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
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
    X_train = preprocessor.fit_transform(raw_train)
    recreated_X_test = preprocessor.transform(raw_test)
    y_train = raw_train["hospital_death"].astype(int)
    y_test = raw_test["hospital_death"].astype(int)

    dropped_cols = preprocessor.id_cols + preprocessor.leakage_cols + [preprocessor.target_col]
    leaked_columns_present = [
        column
        for column in dropped_cols
        if column in X_train.columns or column in recreated_X_test.columns
    ]
    train_test_columns_match = list(X_train.columns) == list(recreated_X_test.columns)
    feature_names_match = list(recreated_X_test.columns) == preprocessor.feature_names_

    print("=== Preprocessing Verification ===")
    print(f"Raw train shape       : {raw_train.shape}")
    print(f"Raw test shape        : {raw_test.shape}")
    print(f"Processed train shape : {X_train.shape}")
    print(f"Recreated X_test shape : {recreated_X_test.shape}")
    print(f"Train/test columns match: {train_test_columns_match}")
    print(f"Feature names match     : {feature_names_match}")
    print(f"Leaked columns present  : {leaked_columns_present}")
    print(f"Train death rate        : {y_train.mean():.4f}")
    print(f"Test death rate         : {y_test.mean():.4f}")
    print()
    print("=== Learned Preprocessing Metadata ===")
    print(f"ID/leakage columns dropped   : {len(dropped_cols)}")
    print(f"Missing indicators added     : {len(preprocessor.missing_indicator_cols_)}")
    print(f"Numeric columns              : {len(preprocessor.numeric_cols_)}")
    print(f"Binary columns               : {len(preprocessor.binary_cols_)}")
    print(f"Categorical columns          : {len(preprocessor.categorical_cols_)}")
    print(f"Final feature count          : {len(preprocessor.feature_names_)}")

    if leaked_columns_present:
        raise RuntimeError("Dropped ID/leakage/target columns are present after preprocessing.")

    if not train_test_columns_match or not feature_names_match:
        raise RuntimeError("Preprocessing feature schema is inconsistent.")


if __name__ == "__main__":
    main()
