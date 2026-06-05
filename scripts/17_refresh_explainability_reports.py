"""Refresh SHAP explainability report tables and figures for the final model."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.explainability import get_positive_class_shap_values
from src.prediction import load_threshold


SEED = 42


def make_prediction_types(y_test: pd.Series, y_proba: np.ndarray, threshold: float) -> pd.DataFrame:
    y_pred = (y_proba >= threshold).astype(int)
    prediction_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    )
    prediction_df["prediction_type"] = np.select(
        [
            (prediction_df["y_true"] == 1) & (prediction_df["y_pred"] == 1),
            (prediction_df["y_true"] == 1) & (prediction_df["y_pred"] == 0),
            (prediction_df["y_true"] == 0) & (prediction_df["y_pred"] == 1),
            (prediction_df["y_true"] == 0) & (prediction_df["y_pred"] == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="Unknown",
    )
    return prediction_df


def select_local_cases(prediction_df: pd.DataFrame) -> dict[str, int]:
    return {
        "TP": int(
            prediction_df[prediction_df["prediction_type"] == "TP"]
            .sort_values("y_proba", ascending=False)
            .index[0]
        ),
        "FN": int(
            prediction_df[prediction_df["prediction_type"] == "FN"]
            .sort_values("y_proba", ascending=True)
            .index[0]
        ),
        "FP": int(
            prediction_df[prediction_df["prediction_type"] == "FP"]
            .sort_values("y_proba", ascending=False)
            .index[0]
        ),
        "TN": int(
            prediction_df[prediction_df["prediction_type"] == "TN"]
            .sort_values("y_proba", ascending=True)
            .index[0]
        ),
    }


def local_explanation(
    index: int,
    X: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "value": X.iloc[index].values,
            "shap_value": shap_values[index],
        }
    )
    explanation["abs_shap"] = explanation["shap_value"].abs()
    return explanation.sort_values("abs_shap", ascending=False).reset_index(drop=True)


def save_waterfall(
    explainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    index: int,
    path: Path,
    title: str,
) -> None:
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[index],
            base_values=expected_value,
            data=X.iloc[index],
            feature_names=X.columns,
        ),
        max_display=15,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    explainability_dir = ROOT / "reports/02_explainability"
    figures_dir = explainability_dir / "figures"
    tables_dir = explainability_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")
    y_test = pd.read_csv(ROOT / "data/processed/y_test.csv").squeeze().astype(int)
    model = joblib.load(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    y_proba = model.predict_proba(X_test)[:, 1]
    prediction_df = make_prediction_types(y_test, y_proba, threshold)

    explainer = shap.TreeExplainer(model)
    shap_values = get_positive_class_shap_values(explainer, X_test)

    shap_importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_importance["rank"] = shap_importance.index + 1

    shap_importance[["feature", "mean_abs_shap", "rank"]].to_csv(
        tables_dir / "global_shap_importance.csv",
        index=False,
    )
    prediction_df.to_csv(tables_dir / "prediction_types.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(y_proba, bins=50)
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold:.3f}",
    )
    plt.xlabel("Predicted probability of death")
    plt.ylabel("Number of patients")
    plt.title("Distribution of Predicted Mortality Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "predicted_probability_distribution.png", dpi=150)
    plt.close()

    top20 = shap_importance.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, 8))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top 20 Global Feature Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "global_shap_importance_top20.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_top20.png", dpi=150, bbox_inches="tight")
    plt.close()

    grouped_outputs: dict[str, pd.DataFrame] = {}

    age_summary = pd.DataFrame(
        {
            "age": X_test["age"],
            "age_shap": shap_values[:, X_test.columns.get_loc("age")],
        }
    )
    age_summary["age_group"] = pd.cut(
        age_summary["age"],
        bins=[0, 40, 50, 60, 70, 80, 120],
        labels=["<=40", "41-50", "51-60", "61-70", "71-80", "80+"],
    )
    grouped_outputs["age_shap_grouped.csv"] = (
        age_summary.groupby("age_group", observed=False)
        .agg(
            n_patients=("age", "count"),
            mean_age=("age", "mean"),
            mean_shap=("age_shap", "mean"),
            median_shap=("age_shap", "median"),
        )
        .reset_index()
    )

    spo2_summary = pd.DataFrame(
        {
            "d1_spo2_min": X_test["d1_spo2_min"],
            "spo2_shap": shap_values[:, X_test.columns.get_loc("d1_spo2_min")],
        }
    )
    spo2_summary["spo2_group"] = pd.cut(
        spo2_summary["d1_spo2_min"],
        bins=[0, 85, 90, 95, 100],
        labels=["<85", "85-90", "90-95", "95-100"],
    )
    grouped_outputs["spo2_shap_grouped.csv"] = (
        spo2_summary.groupby("spo2_group", observed=False)
        .agg(
            n_patients=("d1_spo2_min", "count"),
            mean_spo2=("d1_spo2_min", "mean"),
            mean_shap=("spo2_shap", "mean"),
            median_shap=("spo2_shap", "median"),
        )
        .reset_index()
    )

    vent_summary = pd.DataFrame(
        {
            "ventilated_apache": X_test["ventilated_apache"],
            "vent_shap": shap_values[:, X_test.columns.get_loc("ventilated_apache")],
        }
    )
    grouped_outputs["ventilated_shap_grouped.csv"] = (
        vent_summary.groupby("ventilated_apache")
        .agg(
            n_patients=("ventilated_apache", "count"),
            mean_shap=("vent_shap", "mean"),
            median_shap=("vent_shap", "median"),
        )
        .reset_index()
    )

    gcs_summary = pd.DataFrame(
        {
            "gcs_motor_apache": X_test["gcs_motor_apache"],
            "gcs_motor_shap": shap_values[:, X_test.columns.get_loc("gcs_motor_apache")],
        }
    )
    grouped_outputs["gcs_motor_shap_grouped.csv"] = (
        gcs_summary.groupby("gcs_motor_apache")
        .agg(
            n_patients=("gcs_motor_apache", "count"),
            mean_shap=("gcs_motor_shap", "mean"),
            median_shap=("gcs_motor_shap", "median"),
        )
        .reset_index()
        .sort_values("gcs_motor_apache")
    )

    for filename, df in grouped_outputs.items():
        df.to_csv(tables_dir / filename, index=False)

    for feature, filename in [
        ("age", "shap_dependence_age.png"),
        ("d1_spo2_min", "shap_dependence_d1_spo2_min.png"),
        ("gcs_motor_apache", "shap_dependence_gcs_motor_apache.png"),
    ]:
        shap.dependence_plot(feature, shap_values, X_test, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [
            vent_summary.loc[vent_summary["ventilated_apache"] == 0, "vent_shap"],
            vent_summary.loc[vent_summary["ventilated_apache"] == 1, "vent_shap"],
        ],
        tick_labels=["Not ventilated", "Ventilated"],
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("SHAP value")
    plt.title("SHAP Effect of Mechanical Ventilation")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_effect_ventilated_apache.png", dpi=150)
    plt.close()

    selected_cases = select_local_cases(prediction_df)
    selected_cases_df = pd.DataFrame(
        [
            {
                "case_type": case_type,
                "row_position": index,
                **prediction_df.loc[index].to_dict(),
            }
            for case_type, index in selected_cases.items()
        ]
    )
    selected_cases_df.to_csv(tables_dir / "selected_local_cases.csv", index=False)

    local_tables = {}
    for case_type, index in selected_cases.items():
        local_df = local_explanation(index, X_test, shap_values)
        local_tables[case_type] = local_df
        filename = f"local_explanation_{case_type.lower()}.csv"
        local_df.to_csv(tables_dir / filename, index=False)
        save_waterfall(
            explainer=explainer,
            shap_values=shap_values,
            X=X_test,
            index=index,
            path=figures_dir / f"local_waterfall_{case_type.lower()}.png",
            title=f"{case_type} Patient",
        )

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df["prediction_type"] = prediction_df["prediction_type"].values
    group_mean_shap = shap_df.groupby("prediction_type").mean()
    group_mean_abs_shap = (
        shap_df.drop(columns="prediction_type")
        .abs()
        .assign(prediction_type=prediction_df["prediction_type"].values)
        .groupby("prediction_type")
        .mean()
    )
    group_mean_shap.to_csv(tables_dir / "group_mean_shap.csv", index=True)
    group_mean_abs_shap.to_csv(tables_dir / "group_mean_abs_shap.csv", index=True)

    group_rows = []
    for group_name in ["TP", "FN", "FP", "TN"]:
        for metric_name, source_df in [
            ("mean_shap", group_mean_shap),
            ("mean_abs_shap", group_mean_abs_shap),
        ]:
            top_group = source_df.loc[group_name].sort_values(ascending=False).head(10)
            for rank, (feature, value) in enumerate(top_group.items(), start=1):
                group_rows.append(
                    {
                        "prediction_type": group_name,
                        "metric": metric_name,
                        "rank": rank,
                        "feature": feature,
                        "value": value,
                    }
                )
    pd.DataFrame(group_rows).to_csv(
        tables_dir / "group_top_shap_summary.csv",
        index=False,
    )

    zero_vital_rows = []
    for feature in ["d1_heartrate_min", "d1_resprate_min", "h1_resprate_min"]:
        if feature not in X_test.columns:
            continue
        zero_mask = X_test[feature] == 0
        zero_vital_rows.append(
            {
                "feature": feature,
                "n_zero": int(zero_mask.sum()),
                "zero_rate": float(zero_mask.mean()),
                "death_rate_zero": float(y_test[zero_mask].mean()) if zero_mask.any() else np.nan,
                "death_rate_nonzero": float(y_test[~zero_mask].mean()) if (~zero_mask).any() else np.nan,
            }
        )
    pd.DataFrame(zero_vital_rows).to_csv(
        tables_dir / "zero_vital_summary.csv",
        index=False,
    )

    top20_features = shap_importance.head(20)["feature"].tolist()
    top20_features_df = shap_importance.head(20)[["rank", "feature", "mean_abs_shap"]]
    top20_features_df.to_csv(tables_dir / "top20_shap_features.csv", index=False)

    interaction_sample = X_test.sample(n=min(300, len(X_test)), random_state=SEED)
    interaction_values = explainer.shap_interaction_values(interaction_sample)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[1]
    top20_indices = [X_test.columns.get_loc(feature) for feature in top20_features]
    top20_interaction_values = interaction_values[:, top20_indices, :][:, :, top20_indices]
    mean_abs_interaction_matrix = np.abs(top20_interaction_values).mean(axis=0)
    np.fill_diagonal(mean_abs_interaction_matrix, 0)
    interaction_matrix_df = pd.DataFrame(
        mean_abs_interaction_matrix,
        index=top20_features,
        columns=top20_features,
    )
    interaction_matrix_df.to_csv(tables_dir / "top20_shap_interaction_matrix.csv")

    interaction_rows = []
    for i in range(len(top20_features)):
        for j in range(i + 1, len(top20_features)):
            pair_values = top20_interaction_values[:, i, j]
            interaction_rows.append(
                {
                    "feature_1": top20_features[i],
                    "feature_2": top20_features[j],
                    "mean_abs_interaction": np.abs(pair_values).mean(),
                    "mean_signed_interaction": pair_values.mean(),
                    "median_signed_interaction": np.median(pair_values),
                }
            )
    top_interactions = (
        pd.DataFrame(interaction_rows)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )
    top_interactions["rank"] = top_interactions.index + 1
    top_interactions = top_interactions[
        [
            "rank",
            "feature_1",
            "feature_2",
            "mean_abs_interaction",
            "mean_signed_interaction",
            "median_signed_interaction",
        ]
    ]
    top_interactions.to_csv(tables_dir / "top20_shap_interactions.csv", index=False)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(interaction_matrix_df, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Mean absolute SHAP interaction value")
    plt.xticks(range(len(top20_features)), top20_features, rotation=90)
    plt.yticks(range(len(top20_features)), top20_features)
    plt.title("Top 20 SHAP Interaction Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "top20_shap_interaction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    top20_corr = X_test[top20_features].corr(method="spearman")
    top20_corr.to_csv(tables_dir / "top20_feature_correlations.csv")
    plt.figure(figsize=(12, 10))
    im = plt.imshow(top20_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="Spearman correlation")
    plt.xticks(range(len(top20_features)), top20_features, rotation=90)
    plt.yticks(range(len(top20_features)), top20_features)
    plt.title("Top 20 Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "top20_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=== Explainability Reports Refreshed ===")
    print(f"X_test shape: {X_test.shape}")
    print(f"Threshold   : {threshold:.4f}")
    print("Prediction type counts:")
    print(prediction_df["prediction_type"].value_counts().to_string())
    print()
    print("Top 10 SHAP features:")
    print(shap_importance.head(10).to_string(index=False))
    print()
    print("Selected local cases:")
    print(selected_cases_df.to_string(index=False))


if __name__ == "__main__":
    main()
