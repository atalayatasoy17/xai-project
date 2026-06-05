"""Train and save the final LightGBM artifacts from the model experiment setup."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import ICUPreprocessor


SEED = 42
BEST_LGBM_PARAMS = {
    "max_depth": 5,
    "num_leaves": 57,
    "learning_rate": 0.013868762536313799,
    "reg_lambda": 7.692292078391939,
    "n_estimators": 1921,
    "min_child_samples": 32,
    "subsample": 0.7647377968169441,
    "colsample_bytree": 0.7148623268990745,
    "class_weight": "balanced",
    "subsample_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def evaluate(y_true: pd.Series, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": "LightGBM Tuned Experiment",
        "threshold": threshold,
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/training_v2.csv")

    raw_train, raw_test = train_test_split(
        raw,
        test_size=0.2,
        random_state=SEED,
        stratify=raw["hospital_death"],
    )
    y_train = raw_train["hospital_death"].astype(int)
    y_test = raw_test["hospital_death"].astype(int)

    preprocessor = ICUPreprocessor()
    X_train = preprocessor.fit_transform(raw_train)
    X_test = preprocessor.transform(raw_test)

    threshold_tuner = TunedThresholdClassifierCV(
        estimator=LGBMClassifier(**BEST_LGBM_PARAMS),
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
    )
    threshold_tuner.fit(X_train, y_train)
    threshold = float(threshold_tuner.best_threshold_)

    model = LGBMClassifier(**BEST_LGBM_PARAMS)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_proba, threshold)

    joblib.dump(preprocessor, ROOT / "models/icu_preprocessor.pkl")
    joblib.dump(model, ROOT / "models/lgbm_tuned_clean.pkl")

    with open(ROOT / "models/lgbm_tuned_clean_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    processed_dir = ROOT / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    pd.DataFrame({"feature": preprocessor.feature_names_}).to_csv(
        processed_dir / "feature_names.csv",
        index=False,
        header=False,
    )

    modeling_dir = ROOT / "reports/01_modeling"
    modeling_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        modeling_dir / "selected_lgbm_test_metrics.csv",
        index=False,
    )
    pd.DataFrame({"feature": preprocessor.feature_names_}).to_csv(
        modeling_dir / "final_feature_names.csv",
        index=False,
    )

    print("=== Final LightGBM Experiment Artifacts Saved ===")
    print("Preprocessor : models/icu_preprocessor.pkl")
    print("Model        : models/lgbm_tuned_clean.pkl")
    print("Threshold    : models/lgbm_tuned_clean_threshold.json")
    print("Processed data: data/processed/")
    print(f"Feature count: {len(preprocessor.feature_names_)}")
    print()
    print("=== Test Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:10}: {value:.4f}")
        else:
            print(f"{key:10}: {value}")


if __name__ == "__main__":
    main()
