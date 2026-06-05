from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

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
    y_test = raw_test["hospital_death"].reset_index(drop=True)

    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    predictions = predict_mortality(model, X_test, threshold=threshold)

    y_proba = predictions["death_probability"]
    y_pred = predictions["prediction"]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("=== Prediction Verification ===")
    print(f"Threshold  : {threshold:.2f}")
    print(f"AUROC      : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"AUPRC      : {average_precision_score(y_test, y_proba):.4f}")
    print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall     : {recall_score(y_test, y_pred):.4f}")
    print(f"F1         : {f1_score(y_test, y_pred):.4f}")
    print(f"TN         : {tn}")
    print(f"FP         : {fp}")
    print(f"FN         : {fn}")
    print(f"TP         : {tp}")


if __name__ == "__main__":
    main()