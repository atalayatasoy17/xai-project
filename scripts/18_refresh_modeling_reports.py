"""Refresh modeling report tables and figures for the final saved LightGBM model."""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.prediction import load_threshold


def main() -> None:
    modeling_dir = ROOT / "reports/01_modeling"
    figures_dir = modeling_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")
    y_test = pd.read_csv(ROOT / "data/processed/y_test.csv").squeeze().astype(int)
    model = joblib.load(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": "LightGBM Tuned Experiment",
        "threshold": threshold,
        "AUROC": roc_auc_score(y_test, y_proba),
        "AUPRC": average_precision_score(y_test, y_proba),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    pd.DataFrame([metrics]).to_csv(
        modeling_dir / "selected_lgbm_test_metrics.csv",
        index=False,
    )

    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_rows = []
    for candidate in thresholds:
        candidate_pred = (y_proba >= candidate).astype(int)
        c_tn, c_fp, c_fn, c_tp = confusion_matrix(y_test, candidate_pred).ravel()
        sweep_rows.append(
            {
                "threshold": candidate,
                "precision": precision_score(y_test, candidate_pred, zero_division=0),
                "recall": recall_score(y_test, candidate_pred, zero_division=0),
                "f1": f1_score(y_test, candidate_pred, zero_division=0),
                "TN": int(c_tn),
                "FP": int(c_fp),
                "FN": int(c_fn),
                "TP": int(c_tp),
            }
        )
    threshold_sweep = pd.DataFrame(sweep_rows)
    threshold_sweep.to_csv(modeling_dir / "threshold_sweep_lgbm.csv", index=False)

    booster = model.booster_
    native_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "split_importance": booster.feature_importance(importance_type="split"),
            "gain_importance": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain_importance", ascending=False)
    native_importance["rank_gain"] = range(1, len(native_importance) + 1)
    native_importance.to_csv(
        modeling_dir / "native_lgbm_feature_importance.csv",
        index=False,
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Survived", "Died"],
        values_format="d",
        cmap="Blues",
    )
    plt.title(f"Selected LightGBM Confusion Matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(figures_dir / "selected_lgbm_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_sweep["threshold"], threshold_sweep["precision"], label="Precision")
    plt.plot(threshold_sweep["threshold"], threshold_sweep["recall"], label="Recall")
    plt.plot(threshold_sweep["threshold"], threshold_sweep["f1"], label="F1")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Selected = {threshold:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: Precision, Recall, F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_sweep_precision_recall_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_sweep["threshold"], threshold_sweep["FP"], label="FP")
    plt.plot(threshold_sweep["threshold"], threshold_sweep["FN"], label="FN")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Selected = {threshold:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Count")
    plt.title("Threshold Sweep: False Positives vs False Negatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_sweep_fp_fn.png", dpi=150)
    plt.close()

    top20_gain = native_importance.head(20).sort_values("gain_importance")
    plt.figure(figsize=(9, 8))
    plt.barh(top20_gain["feature"], top20_gain["gain_importance"])
    plt.xlabel("Gain importance")
    plt.title("Native LightGBM Feature Importance by Gain")
    plt.tight_layout()
    plt.savefig(figures_dir / "native_lgbm_feature_importance_gain_top20.png", dpi=150)
    plt.close()

    comparison = pd.DataFrame(
        [
            {
                "model": "Final LightGBM Experiment",
                "threshold": threshold,
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
            }
        ]
    )
    comparison.to_csv(modeling_dir / "final_model_comparison.csv", index=False)

    metric_names = ["AUROC", "AUPRC", "Precision", "Recall", "F1"]
    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, [metrics[name] for name in metric_names])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Final LightGBM Test Metrics")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_metrics.png", dpi=150)
    plt.close()

    print("=== Modeling Reports Refreshed ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:10}: {value:.4f}")
        else:
            print(f"{key:10}: {value}")
    print()
    print("Top 10 native gain features:")
    print(native_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
