"""Streamlit dashboard for live ICU XAI demo outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_fetch import ensure_data
from src.llm import generate_explanation, revise_until_valid
from src.evaluator import compute_hybrid_quality_score, evaluate_subjective_quality
from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.prompts import build_explanation_prompt
from src.validation import validate_explanation


RAW_UNLABELED_PATH = ROOT / "data" / "raw" / "unlabeled.csv"
PREPROCESSOR_PATH = ROOT / "models" / "icu_preprocessor.pkl"
MODEL_PATH = ROOT / "models" / "lgbm_tuned_clean.pkl"
THRESHOLD_PATH = ROOT / "models" / "lgbm_tuned_clean_threshold.json"
WEIGHTS_PATH = ROOT / "reports" / "05_evaluation" / "evaluation_weights.json"
EDA_DIR = ROOT / "reports" / "00_eda"
MODELING_DIR = ROOT / "reports" / "01_modeling"
EXPLAINABILITY_DIR = ROOT / "reports" / "02_explainability"
VALIDATION_AUDIT_PATH = ROOT / "reports" / "09_validation_audit" / "validation_audit_summary.csv"
GPT4O_EVALUATION_PATH = (
    ROOT / "reports" / "10_gpt4o_evaluation" / "gpt4o_subjective_evaluation_summary.csv"
)
DEFAULT_EVALUATION_WEIGHTS = {
    "faithfulness_no_hallucination": 0.30,
    "clinical_plausibility": 0.25,
    "caution_awareness": 0.20,
    "completeness": 0.15,
    "clarity": 0.10,
}


st.set_page_config(
    page_title="ICU Mortality XAI Dashboard",
    page_icon="",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.stApp {
    background: #eef4f2;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1220px;
}
[data-testid="stSidebar"] {
    background: #1f4d4a;
    border-right: 1px solid #183f3d;
}
[data-testid="stSidebar"] * {
    color: #f4fbf8;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #d7eee8;
}
[data-testid="stSidebar"] div[role="slider"] {
    color: #f4fbf8;
}
[data-testid="stMetric"] {
    background: #fbfdfc;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    padding: 0.95rem 1rem;
    box-shadow: 0 8px 20px rgba(31, 77, 74, 0.08);
}
[data-testid="stMetricLabel"] {
    color: #38524e;
    font-weight: 650;
}
[data-testid="stMetricValue"] {
    color: #16312f;
    font-weight: 750;
}
h1, h2, h3 {
    color: #183331;
}
h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    color: #183331 !important;
}
p, li, label, span {
    color: #203432;
}
.small-note {
    color: #d7eee8;
    font-size: 0.92rem;
}
.status-pass {
    color: #0f7b45;
    font-weight: 700;
}
.status-fail {
    color: #b42318;
    font-weight: 700;
}
.section-caption {
    color: #536864;
    margin-top: -0.4rem;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    box-shadow: 0 8px 20px rgba(31, 77, 74, 0.06);
}
[data-testid="stExpander"] {
    background: #fbfdfc;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
}
[data-testid="stAlert"] {
    border: 1px solid #b6c9c3;
}
hr {
    border-color: #c7d8d2;
}
.hero {
    background: #f9fcfa;
    border: 1px solid #c7d8d2;
    border-left: 6px solid #2f8077;
    border-radius: 10px;
    padding: 1.35rem 1.45rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 28px rgba(31, 77, 74, 0.10);
}
.eyebrow {
    color: #2f8077;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.hero-title {
    color: #16312f;
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 800;
    margin: 0;
}
.hero-subtitle {
    color: #435b57;
    font-size: 1rem;
    margin: 0.55rem 0 0 0;
    max-width: 860px;
}
.pipeline-strip {
    display: inline-block;
    background: #e3f2ed;
    color: #1f4d4a;
    border: 1px solid #b9d8cf;
    border-radius: 999px;
    padding: 0.38rem 0.72rem;
    margin-top: 0.9rem;
    font-size: 0.88rem;
    font-weight: 700;
}
.section-kicker {
    color: #2f8077;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.section-title {
    color: #16312f;
    font-size: 1.28rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}
.risk-banner {
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 0.25rem 0 1rem 0;
    font-weight: 750;
}
.risk-low {
    background: #e8f6ee;
    border: 1px solid #acd9bd;
    color: #145c35;
}
.risk-high {
    background: #fff3df;
    border: 1px solid #efc27f;
    color: #7a4200;
}
.explain-box {
    background: #fbfdfc;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 20px rgba(31, 77, 74, 0.06);
}
.empty-state {
    background: #f9fcfa;
    border: 1px dashed #a8c1ba;
    border-radius: 10px;
    padding: 1.2rem;
    color: #435b57;
}
.insight-box {
    background: #fbfdfc;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.45rem 0 1rem 0;
    color: #334946;
    box-shadow: 0 6px 16px rgba(31, 77, 74, 0.05);
}
.insight-box strong {
    color: #16312f;
}
.step-box {
    background: #f9fcfa;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    padding: 1rem;
    min-height: 145px;
    color: #334946;
    line-height: 1.45;
}
.step-box strong {
    color: #16312f;
    font-weight: 800;
}
.narrative-card {
    background: #fbfdfc;
    border: 1px solid #c7d8d2;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 18px rgba(31, 77, 74, 0.06);
}
.narrative-card h4 {
    color: #16312f;
    margin: 0 0 0.45rem 0;
    font-size: 1rem;
}
.narrative-card p {
    margin: 0;
    color: #435b57;
}
.subsection-rule {
    border-top: 1px solid #c7d8d2;
    margin: 1.4rem 0 1rem 0;
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_unlabeled_data() -> pd.DataFrame:
    return pd.read_csv(RAW_UNLABELED_PATH)


@st.cache_resource(show_spinner=False)
def load_live_artifacts() -> tuple[Any, Any, float]:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = load_model(MODEL_PATH)
    threshold = load_threshold(THRESHOLD_PATH)
    return preprocessor, model, threshold


@st.cache_data(show_spinner=False)
def load_evaluation_weights() -> dict[str, float]:
    if not WEIGHTS_PATH.exists():
        return DEFAULT_EVALUATION_WEIGHTS

    with open(WEIGHTS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_probability(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def status_text(passed: bool | None) -> str:
    if passed is True:
        return "Passed"
    if passed is False:
        return "Failed"
    return "Not available"


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">DS 570 Project Dashboard</div>
            <div class="hero-title">ICU Mortality Explanation Dashboard</div>
            <p class="hero-subtitle">
                A presentation-ready walkthrough of the data, preprocessing, model,
                SHAP explanations, LLM validation architecture, and live patient-level demo.
            </p>
            <div class="pipeline-strip">EDA -> Preprocessing -> LightGBM -> SHAP -> LLM validation -> Live demo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, caption: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-caption'>{caption}</p>", unsafe_allow_html=True)


def render_insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def render_image(path: Path, caption: str | None = None) -> None:
    if not path.exists():
        st.warning(f"Missing figure: {path.relative_to(ROOT)}")
        return

    st.image(str(path), use_container_width=True)
    if caption:
        render_insight(caption)


def render_table(path: Path, columns: list[str] | None = None, n_rows: int = 10) -> None:
    if not path.exists():
        st.warning(f"Missing table: {path.relative_to(ROOT)}")
        return

    df = load_csv(str(path))
    if columns:
        df = df[[column for column in columns if column in df.columns]]
    st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)


def render_target_correlation_chart(path: Path, n_rows: int = 12) -> None:
    """Render signed target correlations as a presentation-friendly bar chart."""
    if not path.exists():
        st.warning(f"Missing table: {path.relative_to(ROOT)}")
        return

    import matplotlib.pyplot as plt

    df = load_csv(str(path)).head(n_rows).copy()
    df = df.sort_values("pearson_corr_with_hospital_death", ascending=True)
    colors = [
        "#b7653f" if value > 0 else "#2f8077"
        for value in df["pearson_corr_with_hospital_death"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(
        df["feature"],
        df["pearson_corr_with_hospital_death"],
        color=colors,
    )
    ax.axvline(0, color="#526661", linewidth=1)
    ax.set_xlabel("Pearson correlation with hospital_death")
    ax.set_ylabel("")
    ax.set_title("Top Target Correlations", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d6e2de", linewidth=0.8)
    ax.set_facecolor("#fbfdfc")
    fig.patch.set_facecolor("#eef4f2")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9bb4ad")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def get_baseline_cv_results() -> pd.DataFrame:
    """Return baseline CV summaries from notebook 09 model experiment."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "cv_f1": 0.5038,
                "cv_recall": 0.5335,
                "cv_precision": 0.4786,
                "cv_auroc": 0.8838,
                "cv_auprc": 0.5153,
                "mean_threshold": 0.2404,
            },
            {
                "model": "Decision Tree",
                "cv_f1": 0.4219,
                "cv_recall": 0.4810,
                "cv_precision": 0.3777,
                "cv_auroc": 0.8161,
                "cv_auprc": 0.3937,
                "mean_threshold": 0.1758,
            },
            {
                "model": "Random Forest",
                "cv_f1": 0.4835,
                "cv_recall": 0.4949,
                "cv_precision": 0.4735,
                "cv_auroc": 0.8644,
                "cv_auprc": 0.4925,
                "mean_threshold": 0.1940,
            },
        ]
    )


def get_rf_lgbm_test_comparison() -> pd.DataFrame:
    """Return Random Forest baseline and final tuned LightGBM test results."""
    lgbm = load_csv(str(MODELING_DIR / "selected_lgbm_test_metrics.csv")).iloc[0]
    return pd.DataFrame(
        [
            {
                "model": "Random Forest baseline",
                "threshold": 0.189,
                "AUROC": 0.8725,
                "AUPRC": 0.5282,
                "Precision": 0.4693,
                "Recall": 0.5357,
                "F1": 0.5003,
                "TN": 15801,
                "FP": 959,
                "FN": 735,
                "TP": 848,
            },
            {
                "model": "Tuned LightGBM final",
                "threshold": float(lgbm["threshold"]),
                "AUROC": float(lgbm["AUROC"]),
                "AUPRC": float(lgbm["AUPRC"]),
                "Precision": float(lgbm["Precision"]),
                "Recall": float(lgbm["Recall"]),
                "F1": float(lgbm["F1"]),
                "TN": int(lgbm["TN"]),
                "FP": int(lgbm["FP"]),
                "FN": int(lgbm["FN"]),
                "TP": int(lgbm["TP"]),
            },
        ]
    )


def render_metric_bar_chart(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    model_column: str = "model",
) -> None:
    """Render a compact grouped metric comparison chart."""
    import matplotlib.pyplot as plt

    plot_df = df[[model_column] + metrics].set_index(model_column)
    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 4.8),
        color=["#2f8077", "#8f6f3e", "#b7653f", "#4b6f91", "#6f6f6f"][: len(metrics)],
        rot=0,
    )
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", color="#d6e2de", linewidth=0.8)
    ax.set_facecolor("#fbfdfc")
    legend_labels = [metric.replace("cv_", "CV ").replace("_", " ").title() for metric in metrics]
    ax.legend(
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(len(metrics), 5),
        frameon=False,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9bb4ad")
    ax.spines["bottom"].set_color("#9bb4ad")
    fig = ax.get_figure()
    fig.patch.set_facecolor("#eef4f2")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_confusion_matrix_chart(
    title: str,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> None:
    """Render a compact confusion matrix for presentation comparison."""
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Survived", "Died"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Survived", "Died"])

    threshold = matrix.max() / 2
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > threshold else "#16312f"
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor("#eef4f2")
    ax.set_facecolor("#fbfdfc")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_shap_group_chart(
    path: Path,
    category_column: str,
    value_column: str,
    title: str,
    x_label: str = "",
) -> None:
    """Render grouped mean SHAP effects as a compact bar chart."""
    if not path.exists():
        st.warning(f"Missing table: {path.relative_to(ROOT)}")
        return

    import matplotlib.pyplot as plt

    df = load_csv(str(path)).copy()
    df[category_column] = df[category_column].astype(str)
    colors = [
        "#b7653f" if value > 0 else "#2f8077"
        for value in df[value_column]
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(df[category_column], df[value_column], color=colors)
    ax.axhline(0, color="#526661", linewidth=1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean SHAP contribution")
    ax.grid(axis="y", color="#d6e2de", linewidth=0.8)
    ax.set_facecolor("#fbfdfc")
    fig.patch.set_facecolor("#eef4f2")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9bb4ad")
    ax.spines["bottom"].set_color("#9bb4ad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_top_interaction_chart(path: Path, n_rows: int = 8) -> None:
    """Render strongest SHAP interaction pairs as a horizontal chart."""
    if not path.exists():
        st.warning(f"Missing table: {path.relative_to(ROOT)}")
        return

    import matplotlib.pyplot as plt

    df = load_csv(str(path)).head(n_rows).copy()
    df["pair"] = df["feature_1"] + " x " + df["feature_2"]
    df = df.sort_values("mean_abs_interaction", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(df["pair"], df["mean_abs_interaction"], color="#4b6f91")
    ax.set_title("Top SHAP Interaction Pairs", loc="left", fontweight="bold")
    ax.set_xlabel("Mean absolute interaction strength")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d6e2de", linewidth=0.8)
    ax.set_facecolor("#fbfdfc")
    fig.patch.set_facecolor("#eef4f2")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9bb4ad")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_eda_page() -> None:
    render_section_header(
        title="Exploratory Data Analysis",
        caption="A presentation walkthrough of how the raw ICU data was understood before modeling.",
        kicker="Part 1",
    )

    st.markdown(
        """
        <div class="narrative-card">
        <h4>EDA goal</h4>
        <p>
        Before training a model, the analysis checked the target distribution, feature families,
        missingness, leakage-prone variables, descriptive feature patterns, categorical mortality rates,
        and outliers. The purpose was not only to describe the data, but to justify later preprocessing
        and evaluation decisions.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question_cols = st.columns(4)
    questions = [
        ("Data shape", "How large is the ICU dataset and what types of variables are present?"),
        ("Target", "Is hospital mortality balanced enough for accuracy to be meaningful?"),
        ("Data quality", "Which variables have missing values, outliers, or leakage risk?"),
        ("Modeling impact", "Which EDA findings should change preprocessing and evaluation?"),
    ]
    for column, (title, body) in zip(question_cols, questions):
        with column:
            st.markdown(
                f"""
                <div class="step-box">
                <strong>{title}</strong><br>
                {body}
                </div>
                """,
                unsafe_allow_html=True,
            )

    overview = load_csv(str(EDA_DIR / "tables" / "dataset_overview.csv"))
    overview_map = dict(zip(overview["metric"], overview["value"]))
    target_distribution = load_csv(str(EDA_DIR / "tables" / "target_distribution.csv"))
    missing_thresholds = load_csv(str(EDA_DIR / "tables" / "missingness_threshold_summary.csv"))
    duplicate_summary = load_csv(str(EDA_DIR / "tables" / "duplicate_summary.csv"))

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Dataset snapshot")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Patients", f"{int(overview_map.get('n_rows', 0)):,}")
    metric_cols[1].metric("Raw columns", str(overview_map.get("n_columns", "N/A")))
    metric_cols[2].metric("Target", str(overview_map.get("target_column", "N/A")))
    metric_cols[3].metric("Death rate", f"{float(overview_map.get('positive_rate_percent', 0)):.2f}%")
    metric_cols[4].metric(
        "Columns >50% missing",
        str(int(missing_thresholds.loc[
            missing_thresholds["missing_pct_threshold"] == 50,
            "n_columns_above_threshold",
        ].iloc[0])),
    )
    duplicate_value = duplicate_summary["duplicate_rows"].iloc[0]
    metric_cols[5].metric("Duplicate rows", str(duplicate_value))

    left, right = st.columns([1.1, 1])
    with left:
        render_image(
            EDA_DIR / "figures" / "target_distribution.png",
            "<strong>Finding:</strong> only 8.63% of patients died in hospital. "
            "This class imbalance directly motivates stratified splitting and metrics beyond accuracy."
        )
    with right:
        st.markdown("**Target distribution table**")
        st.dataframe(target_distribution, use_container_width=True, hide_index=True)
        render_table(EDA_DIR / "tables" / "feature_groups_summary.csv", columns=["group", "n_columns"], n_rows=10)
        render_insight(
            "<strong>Feature families:</strong> the dataset includes demographics, APACHE variables, "
            "hour-1 measurements, day-1 measurements, ICU information, labs, and vital signs."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Missingness and measurement patterns")
    left, right = st.columns([1, 1])
    with left:
        render_image(
            EDA_DIR / "figures" / "missingness_top20.png",
            "<strong>Finding:</strong> missingness is common in ICU data. It is not treated as simple noise, "
            "because whether a measurement was ordered can reflect patient state or clinical workflow."
        )
    with right:
        render_image(
            EDA_DIR / "figures" / "missingness_by_target.png",
            "<strong>Finding:</strong> some key labs are less missing among patients who died. "
            "This supports adding missingness indicators instead of only imputing values."
        )
    st.dataframe(missing_thresholds, use_container_width=True, hide_index=True)

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Numeric signals and target-associated features")
    left, right = st.columns([1.1, 1])
    with left:
        render_image(
            EDA_DIR / "figures" / "feature_distributions_by_target.png",
            "<strong>Finding:</strong> several vital signs and lab values differ by outcome group. "
            "These descriptive differences support predictive modeling, but they are not causal claims."
        )
    with right:
        render_image(
            EDA_DIR / "figures" / "top_target_correlations.png",
            "<strong>Finding:</strong> high lactate, ventilation-related variables, lower GCS, lower blood pressure, "
            "and lower pH appear among the strongest simple associations."
        )
    st.markdown("**Top target correlations**")
    render_target_correlation_chart(
        EDA_DIR / "tables" / "top_target_correlations.csv",
        n_rows=12,
    )
    with st.expander("Show correlation table"):
        render_table(
            EDA_DIR / "tables" / "top_target_correlations.csv",
            columns=["feature", "pearson_corr_with_hospital_death", "abs_corr"],
            n_rows=12,
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Leakage investigation")
    left, right = st.columns([1, 1])
    with left:
        render_image(
            EDA_DIR / "figures" / "apache_leakage_by_target.png",
            "<strong>Finding:</strong> APACHE mortality probability columns already encode risk estimates. "
            "Keeping them would make the task less fair and less educational."
        )
    with right:
        st.markdown("**Leakage-prone APACHE score columns**")
        render_table(
            EDA_DIR / "tables" / "apache_leakage_by_target.csv",
            n_rows=8,
        )
        render_insight(
            "<strong>Decision:</strong> remove <code>apache_4a_hospital_death_prob</code> and "
            "<code>apache_4a_icu_death_prob</code> from the final model inputs."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Categorical patterns and outliers")
    left, right = st.columns([1, 1])
    with left:
        render_image(
            EDA_DIR / "figures" / "categorical_mortality_top_groups.png",
            "<strong>Finding:</strong> mortality rates differ across admission source, ICU type, and APACHE body-system groups. "
            "Categorical variables are therefore retained with controlled encoding."
        )
    with right:
        st.markdown("**Outlier review**")
        render_table(
            EDA_DIR / "tables" / "outlier_summary.csv",
            columns=["feature", "min", "max", "mean", "iqr_outlier_pct"],
            n_rows=10,
        )
        render_insight(
            "<strong>Decision:</strong> extreme ICU values were not blindly removed. In critical care, extremes can be clinically meaningful."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### EDA decisions carried into the final pipeline")
    eda_decisions = pd.DataFrame(
        [
            {
                "EDA finding": "Hospital death is rare",
                "Pipeline decision": "Use stratified split and AUROC/AUPRC/precision/recall/F1 instead of accuracy alone.",
            },
            {
                "EDA finding": "Many ICU variables are missing",
                "Pipeline decision": "Median-impute numeric features and add missingness indicators.",
            },
            {
                "EDA finding": "APACHE death probability columns are leakage-prone",
                "Pipeline decision": "Remove APACHE precomputed death probability columns before modeling.",
            },
            {
                "EDA finding": "Categorical groups show mortality-rate variation",
                "Pipeline decision": "Retain categorical variables with one-hot encoding and infrequent-category handling.",
            },
            {
                "EDA finding": "Outliers may represent true critical illness",
                "Pipeline decision": "Do not blindly remove extreme values; interpret unusual values cautiously in explanation.",
            },
        ]
    )
    st.dataframe(eda_decisions, use_container_width=True, hide_index=True)


def render_modeling_page() -> None:
    render_section_header(
        title="Preprocessing and Modeling",
        caption="Notebook 09-style walkthrough: preprocessing pipeline, baseline models, Random Forest baseline, optimized LightGBM, and final selection.",
        kicker="Part 2",
    )

    st.markdown(
        """
        <div class="narrative-card">
        <h4>Modeling goal</h4>
        <p>
        The modeling stage tests several predictive approaches for ICU mortality, using a full
        preprocessing + model pipeline so that preprocessing is learned only from training folds.
        The final model is selected using imbalanced-class metrics and threshold tuning rather than
        accuracy or the default 0.50 decision threshold.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 1. Preprocessing pipeline")
    prep_cols = st.columns(5)
    prep_steps = [
        (
            "Drop columns",
            "Remove identifiers/location-like fields and APACHE precomputed death probability columns.",
        ),
        (
            "Numeric features",
            "Median imputation with missingness indicators; no StandardScaler for final LightGBM.",
        ),
        (
            "Binary features",
            "Most-frequent imputation followed by ordinal encoding with unknown handling.",
        ),
        (
            "Categorical features",
            "Most-frequent imputation and one-hot encoding with infrequent-category handling.",
        ),
        (
            "Final schema",
            "Saved preprocessor produces 379 aligned model features for train, test, and new patients.",
        ),
    ]
    for column, (title, body) in zip(prep_cols, prep_steps):
        with column:
            st.markdown(
                f"""
                <div class="step-box">
                <strong>{title}</strong><br>
                {body}
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_insight(
        "<strong>Why a pipeline?</strong> preprocessing is fit inside the training fold and then applied to validation/test data. "
        "This prevents train-test leakage from imputation, encoding, or feature alignment."
    )

    feature_names = load_csv(str(MODELING_DIR / "final_feature_names.csv"))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final features", f"{len(feature_names):,}")
    metric_cols[1].metric("Removed ID/location", "4")
    metric_cols[2].metric("Removed leakage columns", "2")
    metric_cols[3].metric("Scaling", "Omitted")

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. Baseline model experiment with cross-validation")
    st.markdown(
        """
        <div class="narrative-card">
        <h4>Baseline setup</h4>
        <p>
        Notebook 09 first compared Logistic Regression, Decision Tree, and Random Forest using
        StratifiedKFold cross-validation. Stratification keeps the rare mortality class proportion
        stable across folds. Each model used threshold tuning to maximize F1, because mortality
        prediction needs a balance between catching deaths and avoiding too many false alarms.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_cv = get_baseline_cv_results()
    render_metric_bar_chart(
        baseline_cv,
        metrics=["cv_f1", "cv_recall", "cv_precision", "cv_auroc"],
        title="Baseline Cross-Validation Metrics",
    )
    st.dataframe(
        baseline_cv,
        use_container_width=True,
        hide_index=True,
        column_config={
            "cv_f1": st.column_config.NumberColumn("CV F1", format="%.4f"),
            "cv_recall": st.column_config.NumberColumn("CV Recall", format="%.4f"),
            "cv_precision": st.column_config.NumberColumn("CV Precision", format="%.4f"),
            "cv_auroc": st.column_config.NumberColumn("CV AUROC", format="%.4f"),
            "cv_auprc": st.column_config.NumberColumn("CV AUPRC", format="%.4f"),
            "mean_threshold": st.column_config.NumberColumn("Mean threshold", format="%.4f"),
        },
    )
    render_insight(
        "<strong>Baseline interpretation:</strong> Logistic Regression was competitive, Decision Tree was weaker, "
        "and Random Forest became the strongest non-boosted baseline to compare against optimized LightGBM."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Optimized LightGBM and final comparison")
    comparison = get_rf_lgbm_test_comparison()
    lgbm_row = comparison[comparison["model"] == "Tuned LightGBM final"].iloc[0]
    rf_row = comparison[comparison["model"] == "Random Forest baseline"].iloc[0]
    cols = st.columns(5)
    cols[0].metric("Final AUROC", format_probability(lgbm_row["AUROC"]))
    cols[1].metric("Final AUPRC", format_probability(lgbm_row["AUPRC"]))
    cols[2].metric("Final F1", format_probability(lgbm_row["F1"]))
    cols[3].metric("Final threshold", format_probability(lgbm_row["threshold"]))
    cols[4].metric("FP reduction vs RF", f"{int(rf_row['FP'] - lgbm_row['FP'])}")

    render_metric_bar_chart(
        comparison,
        metrics=["AUROC", "AUPRC", "Precision", "Recall", "F1"],
        title="Random Forest Baseline vs Tuned LightGBM",
    )
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "threshold": st.column_config.NumberColumn("Threshold", format="%.4f"),
            "AUROC": st.column_config.NumberColumn("AUROC", format="%.4f"),
            "AUPRC": st.column_config.NumberColumn("AUPRC", format="%.4f"),
            "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "Recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "F1": st.column_config.NumberColumn("F1", format="%.4f"),
        },
    )
    render_insight(
        "<strong>Final choice:</strong> tuned LightGBM improves AUROC, AUPRC, precision, F1, TP count, "
        "and reduces false positives compared with the Random Forest baseline. Recall is also slightly higher."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 4. Final model diagnostics")
    left, right = st.columns(2)
    with left:
        render_confusion_matrix_chart(
            title="Random Forest Baseline",
            tn=int(rf_row["TN"]),
            fp=int(rf_row["FP"]),
            fn=int(rf_row["FN"]),
            tp=int(rf_row["TP"]),
        )
        render_insight(
            "<strong>Random Forest baseline:</strong> predicts 848 true deaths, "
            "with 959 false positives and 735 false negatives."
        )
    with right:
        render_confusion_matrix_chart(
            title="Tuned LightGBM Final",
            tn=int(lgbm_row["TN"]),
            fp=int(lgbm_row["FP"]),
            fn=int(lgbm_row["FN"]),
            tp=int(lgbm_row["TP"]),
        )
        render_insight(
            "<strong>LightGBM final:</strong> predicts 903 true deaths, "
            "with 808 false positives and 680 false negatives. Compared with Random Forest, it catches more deaths and reduces false positives."
        )

    left, right = st.columns(2)
    with left:
        render_image(
            MODELING_DIR / "figures" / "threshold_sweep_precision_recall_f1.png",
            "<strong>Threshold tuning:</strong> threshold is tuned explicitly because the best operating point "
            "for imbalanced classification is not necessarily 0.50."
        )
    with right:
        render_image(
            MODELING_DIR / "figures" / "threshold_sweep_fp_fn.png",
            "<strong>Error trade-off:</strong> as threshold changes, false positives and false negatives move in opposite directions."
        )

    with st.container():
        render_image(
            MODELING_DIR / "figures" / "native_lgbm_feature_importance_gain_top20.png",
            "<strong>Native LightGBM importance:</strong> gain importance shows which features most improve tree splits. "
            "This is a model diagnostic; SHAP is used later for explanation."
        )

    with st.expander("Show final feature names"):
        render_table(MODELING_DIR / "final_feature_names.csv", n_rows=30)


def render_shap_page() -> None:
    render_section_header(
        title="SHAP Explainability",
        caption="Notebook 04-style walkthrough: global SHAP, feature effect patterns, local patient explanations, caution findings, and interaction checks.",
        kicker="Part 3",
    )

    st.markdown(
        """
        <div class="narrative-card">
        <h4>SHAP goal</h4>
        <p>
        SHAP was used after model selection to explain how the final LightGBM model uses patient-level
        evidence. Positive SHAP values push the prediction toward higher mortality risk; negative values
        push it toward lower mortality risk. The analysis has two levels: global patterns across the test
        set and local explanations for individual patients.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_types = load_csv(str(EXPLAINABILITY_DIR / "tables" / "prediction_types.csv"))
    type_counts = prediction_types["prediction_type"].value_counts().to_dict()
    selected_cases = load_csv(str(EXPLAINABILITY_DIR / "tables" / "selected_local_cases.csv"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("True negatives", f"{type_counts.get('TN', 0):,}")
    metric_cols[1].metric("True positives", f"{type_counts.get('TP', 0):,}")
    metric_cols[2].metric("False positives", f"{type_counts.get('FP', 0):,}")
    metric_cols[3].metric("False negatives", f"{type_counts.get('FN', 0):,}")

    render_insight(
        "<strong>Explanation context:</strong> SHAP is examined across all prediction types, not only successful cases. "
        "This helps show where the model is clinically plausible, where it is cautious, and where it misses risk."
    )

    left, right = st.columns([1.15, 1])
    with left:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "predicted_probability_distribution.png",
            "<strong>Probability distribution:</strong> most patients receive low predicted mortality probabilities, "
            "while a smaller group is assigned high risk. This reflects the imbalanced ICU mortality target."
        )
    with right:
        st.markdown("**Selected local cases used for interpretation**")
        st.dataframe(selected_cases, use_container_width=True, hide_index=True)
        render_insight(
            "<strong>Local case design:</strong> one true positive, false negative, false positive, and true negative "
            "were selected so the presentation includes both correct and incorrect model behavior."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 1. Global SHAP importance")
    left, right = st.columns(2)
    with left:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "global_shap_importance_top20.png",
            "<strong>Mean absolute SHAP:</strong> age, ventilation, diagnosis category, BUN, oxygen saturation, "
            "GCS scores, heart rate, respiratory rate, and urine output are among the strongest global drivers."
        )
    with right:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "shap_summary_top20.png",
            "<strong>Summary plot:</strong> the color distribution shows whether high or low feature values tend "
            "to increase or decrease predicted risk. This is stronger than a plain feature-importance table."
        )

    top_features = load_csv(str(EXPLAINABILITY_DIR / "tables" / "global_shap_importance.csv")).head(12)
    st.markdown("**Top global SHAP features**")
    st.dataframe(
        top_features[["rank", "feature", "mean_abs_shap"]],
        use_container_width=True,
        hide_index=True,
    )
    render_insight(
        "<strong>Interpretation:</strong> global SHAP tells us what the model relies on most often. "
        "It does not say that a feature causes mortality; it says the feature changes the model output."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. Feature effect patterns")
    st.markdown(
        """
        These plots answer a second question: after a feature is globally important, how does its value
        change the model prediction? This is why dependence plots and grouped SHAP summaries were added.
        """
    )

    left, right = st.columns(2)
    with left:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "shap_dependence_age.png",
            "<strong>Age dependence:</strong> older ages generally move SHAP upward. The grouped summary below "
            "shows the same monotonic pattern in a presentation-friendly chart."
        )
        render_shap_group_chart(
            EXPLAINABILITY_DIR / "tables" / "age_shap_grouped.csv",
            category_column="age_group",
            value_column="mean_shap",
            title="Age Group Mean SHAP",
            x_label="Age group",
        )
    with right:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "shap_dependence_d1_spo2_min.png",
            "<strong>Minimum oxygen saturation:</strong> very low SpO2 values increase predicted risk, while "
            "values above roughly 90 tend to reduce or only weakly affect risk."
        )
        render_shap_group_chart(
            EXPLAINABILITY_DIR / "tables" / "spo2_shap_grouped.csv",
            category_column="spo2_group",
            value_column="mean_shap",
            title="SpO2 Group Mean SHAP",
            x_label="Minimum SpO2 group",
        )

    left, right = st.columns(2)
    with left:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "shap_dependence_gcs_motor_apache.png",
            "<strong>GCS motor score:</strong> lower motor scores have positive SHAP effects, while score 6 is "
            "risk-decreasing on average. This matches the clinical meaning of impaired neurological response."
        )
        render_shap_group_chart(
            EXPLAINABILITY_DIR / "tables" / "gcs_motor_shap_grouped.csv",
            category_column="gcs_motor_apache",
            value_column="mean_shap",
            title="GCS Motor Mean SHAP",
            x_label="GCS motor score",
        )
    with right:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "shap_effect_ventilated_apache.png",
            "<strong>Mechanical ventilation:</strong> ventilated patients have a positive mean SHAP effect, "
            "while non-ventilated patients have a negative effect. The model treats ventilation as a severity marker."
        )
        render_shap_group_chart(
            EXPLAINABILITY_DIR / "tables" / "ventilated_shap_grouped.csv",
            category_column="ventilated_apache",
            value_column="mean_shap",
            title="Ventilation Mean SHAP",
            x_label="ventilated_apache",
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Local patient explanations")
    render_insight(
        "<strong>Waterfall plots:</strong> each plot starts from the model baseline and adds feature contributions "
        "until it reaches the patient-level prediction. These are the local SHAP values later used to build evidence packets."
    )

    local_cases = [
        (
            "True positive",
            "TP",
            "local_waterfall_tp.png",
            "Very high risk was driven by severe instability signals such as zero heart rate, high lactate, low SpO2, low GCS, ventilation, and low blood pressure.",
        ),
        (
            "False negative",
            "FN",
            "local_waterfall_fn.png",
            "The model missed a death because several low-risk signals, especially younger age, no ventilation, low BUN, and urine output, pulled the score downward.",
        ),
        (
            "False positive",
            "FP",
            "local_waterfall_fp.png",
            "The patient survived, but the model assigned high risk because the physiological profile resembled severe illness. This is a clinically understandable false alarm.",
        ),
        (
            "True negative",
            "TN",
            "local_waterfall_tn.png",
            "Low predicted risk was supported by younger age, no ventilation, and generally lower-risk vital/lab patterns.",
        ),
    ]
    tab_tp, tab_fn, tab_fp, tab_tn = st.tabs([case[0] for case in local_cases])
    for tab, (title, case_code, filename, summary) in zip(
        [tab_tp, tab_fn, tab_fp, tab_tn],
        local_cases,
    ):
        with tab:
            left, right = st.columns([1.25, 1])
            with left:
                render_image(EXPLAINABILITY_DIR / "figures" / filename)
            with right:
                render_insight(f"<strong>{title} interpretation:</strong> {summary}")
                st.markdown("**Top local SHAP contributions**")
                render_table(
                    EXPLAINABILITY_DIR / "tables" / f"local_explanation_{case_code.lower()}.csv",
                    columns=["feature", "value", "shap_value", "abs_shap"],
                    n_rows=8,
                )

    render_insight(
        "<strong>Bridge to the LLM:</strong> the production-style pipeline selects top local risk-increasing and "
        "risk-decreasing SHAP evidence, attaches feature values and clinical meanings, then sends only that structured "
        "evidence packet to the explanation model."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 4. Caution findings from SHAP analysis")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("**Zero-valued vital sign review**")
        zero_vitals = load_csv(str(EXPLAINABILITY_DIR / "tables" / "zero_vital_summary.csv")).copy()
        zero_vitals["zero_rate"] = zero_vitals["zero_rate"].map(lambda value: f"{value:.2%}")
        zero_vitals["death_rate_zero"] = zero_vitals["death_rate_zero"].map(lambda value: f"{value:.2%}")
        zero_vitals["death_rate_nonzero"] = zero_vitals["death_rate_nonzero"].map(lambda value: f"{value:.2%}")
        st.dataframe(zero_vitals, use_container_width=True, hide_index=True)
    with right:
        render_insight(
            "<strong>Why caution flags exist:</strong> some extreme values are rare but highly influential. "
            "For example, zero-valued heart rate can represent a true extreme clinical event or a data quality issue. "
            "The project does not automatically remove it; it marks it for careful explanation."
        )
        render_insight(
            "<strong>ID/location decision:</strong> <code>icu_id</code> was removed from the final model inputs. "
            "The final SHAP analysis therefore focuses on patient-level clinical variables rather than unit identifiers."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 5. Interaction and correlation checks")
    left, right = st.columns(2)
    with left:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "top20_shap_interaction_heatmap.png",
            "<strong>SHAP interaction heatmap:</strong> checks whether pairs of important features jointly change "
            "the model output beyond their separate effects."
        )
        render_top_interaction_chart(
            EXPLAINABILITY_DIR / "tables" / "top20_shap_interactions.csv",
            n_rows=8,
        )
    with right:
        render_image(
            EXPLAINABILITY_DIR / "figures" / "top20_feature_correlation_heatmap.png",
            "<strong>Feature correlation heatmap:</strong> identifies related input families, such as BUN min/max, "
            "GCS components, and vital-sign measurements. Correlation is not the same as SHAP interaction."
        )
        render_table(
            EXPLAINABILITY_DIR / "tables" / "top20_shap_interactions.csv",
            columns=["rank", "feature_1", "feature_2", "mean_abs_interaction"],
            n_rows=8,
        )

    render_insight(
        "<strong>Design decision:</strong> interaction analysis was kept as exploratory model understanding. "
        "It was not added to the LLM evidence packet because patient explanations should remain concise, local, "
        "and easy to validate deterministically."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 6. What SHAP contributed to the final system")
    shap_decisions = pd.DataFrame(
        [
            {
                "SHAP finding": "Global drivers were clinically plausible",
                "Project use": "Used as evidence that the final LightGBM model relies on meaningful ICU variables.",
            },
            {
                "SHAP finding": "Age, SpO2, GCS motor, and ventilation showed interpretable effect patterns",
                "Project use": "These became key examples for explaining model behavior in the report and dashboard.",
            },
            {
                "SHAP finding": "Local waterfall plots expose TP, FN, FP, and TN behavior",
                "Project use": "Supported error analysis and justified patient-level evidence packets.",
            },
            {
                "SHAP finding": "Rare extreme values can be influential",
                "Project use": "Added caution language for values that require careful interpretation.",
            },
            {
                "SHAP finding": "Interactions exist but are harder to validate in natural language",
                "Project use": "Kept interactions in exploratory analysis, not in the LLM prompt.",
            },
        ]
    )
    st.dataframe(shap_decisions, use_container_width=True, hide_index=True)


def render_architecture_page() -> None:
    render_section_header(
        title="LLM Explanation and Validation Architecture",
        caption="Presentation flow before the live demo: evidence packet, prompt, deterministic validator, revision loop, and GPT-4o subjective evaluation.",
        kicker="Part 4",
    )

    st.markdown(
        """
        <div class="narrative-card">
        <h4>Why this layer exists</h4>
        <p>
        The model prediction and SHAP values are numerical. The LLM turns them into readable language,
        but the project does not accept that language blindly. Every generated explanation is checked
        against the structured evidence before it is used in the patient demo.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audit_df = load_csv(str(VALIDATION_AUDIT_PATH))
    gpt4o_df = load_csv(str(GPT4O_EVALUATION_PATH))
    passed_count = int(audit_df["passed"].sum())
    audited_count = len(audit_df)
    revision_count = int(audit_df["revision_required"].sum())
    mean_det_score = float(audit_df["deterministic_validation_score"].mean())
    mean_hybrid_score = float(gpt4o_df["hybrid_quality_score"].mean())
    mean_plausibility = float(gpt4o_df["clinical_plausibility"].mean())
    mean_clarity = float(gpt4o_df["clarity"].mean())

    metric_cols = st.columns(5)
    metric_cols[0].metric("Audited explanations", f"{audited_count}")
    metric_cols[1].metric("Passed validator", f"{passed_count}/{audited_count}")
    metric_cols[2].metric("Revision required", f"{revision_count}")
    metric_cols[3].metric("Mean deterministic score", f"{mean_det_score:.2f}/5")
    metric_cols[4].metric("Mean hybrid score", f"{mean_hybrid_score:.2f}/5")

    render_insight(
        "<strong>Presentation point:</strong> the explanation system has two gates. "
        "Deterministic validation handles objective safety and faithfulness checks; GPT-4o is used only after that, "
        "for subjective clinical plausibility and clarity review."
    )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 1. Explanation pipeline")
    cols = st.columns(5)
    steps = [
        (
            "Evidence packet",
            "Local SHAP is converted into structured JSON: prediction, threshold, top risk-increasing features, top risk-decreasing features, clinical meanings, and caution flags.",
        ),
        (
            "Prompt",
            "The prompt receives only the evidence packet. True labels and correctness are excluded so the explanation cannot leak the real outcome.",
        ),
        (
            "LLM explanation",
            "The LLM writes the explanation in five fixed sections: prediction summary, risk-increasing factors, risk-decreasing factors, caution notes, and interpretation.",
        ),
        (
            "Validator",
            "The deterministic validator checks whether the explanation stayed faithful to the evidence and whether revision is required.",
        ),
        (
            "Revision + evaluation",
            "If validation fails, the LLM receives targeted feedback and revises. GPT-4o then reviews only subjective quality dimensions.",
        ),
    ]
    for column, (title, body) in zip(cols, steps):
        with column:
            st.markdown(
                f"""
                <div class="step-box">
                <strong>{title}</strong><br>
                {body}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. What the evidence packet contains")
    evidence_rows = pd.DataFrame(
        [
            {
                "field": "prediction",
                "what it stores": "death probability, binary prediction, and tuned threshold",
                "why it matters": "forces the LLM to report the model output accurately",
            },
            {
                "field": "risk_increasing_evidence",
                "what it stores": "top positive local SHAP features with values",
                "why it matters": "shows which patient factors pushed risk upward",
            },
            {
                "field": "risk_decreasing_evidence",
                "what it stores": "top negative local SHAP features with values",
                "why it matters": "shows which patient factors pushed risk downward",
            },
            {
                "field": "clinical_meaning",
                "what it stores": "short interpretation for known clinical variables",
                "why it matters": "keeps the explanation readable without inventing new facts",
            },
            {
                "field": "caution_flags",
                "what it stores": "warnings for values/features needing careful interpretation",
                "why it matters": "prevents non-clinical or data-quality signals from being overclaimed",
            },
        ]
    )
    st.dataframe(evidence_rows, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        render_insight(
            "<strong>Evidence packet idea:</strong> it is the contract between SHAP and the LLM. "
            "The LLM is not asked to rediscover the model; it is asked to explain only the provided evidence."
        )
    with right:
        render_insight(
            "<strong>Prompt idea:</strong> the prompt adds strict rules: do not mention true labels, do not invent units, "
            "do not invent clinical mechanisms, preserve the five-section structure, and mention caution flags."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Deterministic validator checks")
    check_rows = pd.DataFrame(
        [
            {
                "check": "Forbidden phrases",
                "what it catches": "unsupported wording such as stable, adequate, normal, favorable",
                "example action": "revise wording into neutral evidence-grounded language",
            },
            {
                "check": "True-label leakage",
                "what it catches": "mentions of true outcome, survival/death, or correct/incorrect prediction",
                "example action": "remove outcome information from the explanation",
            },
            {
                "check": "Prediction consistency",
                "what it catches": "wrong probability or threshold statement",
                "example action": "correct probability within the evidence tolerance",
            },
            {
                "check": "Caution mention",
                "what it catches": "missing caution language for flagged features",
                "example action": "add careful interpretation in the Caution notes section",
            },
            {
                "check": "Feature grounding",
                "what it catches": "exact feature names not present in the evidence packet",
                "example action": "remove unsupported feature claims",
            },
            {
                "check": "Direction consistency",
                "what it catches": "feature described as increasing risk when SHAP says decreasing risk, or vice versa",
                "example action": "fix risk direction",
            },
            {
                "check": "Section structure",
                "what it catches": "missing required explanation sections",
                "example action": "restore the five-section explanation format",
            },
        ]
    )
    st.dataframe(check_rows, use_container_width=True, hide_index=True)

    weights = pd.DataFrame(
        [
            {"dimension": "faithfulness_no_hallucination", "measured by": "deterministic validator", "weight": 0.30},
            {"dimension": "clinical_plausibility", "measured by": "GPT-4o subjective evaluator", "weight": 0.25},
            {"dimension": "caution_awareness", "measured by": "deterministic validator", "weight": 0.20},
            {"dimension": "completeness", "measured by": "deterministic validator", "weight": 0.15},
            {"dimension": "clarity", "measured by": "GPT-4o subjective evaluator", "weight": 0.10},
        ]
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Rubric split**")
        st.dataframe(weights, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Score formula**")
        st.latex(
            r"""
            Hybrid =
            0.30F + 0.25P + 0.20C_a + 0.15C_o + 0.10K
            """
        )
        st.markdown(
            """
            F = faithfulness, P = clinical plausibility, C_a = caution awareness,
            C_o = completeness, K = clarity.
            """
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 4. Revision loop")
    revision_rows = pd.DataFrame(
        [
            {
                "validator finding": "forbidden phrase",
                "feedback sent to LLM": "Remove or rephrase unsupported wording.",
            },
            {
                "validator finding": "wrong probability",
                "feedback sent to LLM": "Correct the predicted mortality probability to match the evidence.",
            },
            {
                "validator finding": "missing caution",
                "feedback sent to LLM": "Mention caution for the flagged feature in the Caution notes section.",
            },
            {
                "validator finding": "direction error",
                "feedback sent to LLM": "Fix whether the feature increased or decreased model risk.",
            },
            {
                "validator finding": "missing section",
                "feedback sent to LLM": "Restore the required five-section structure.",
            },
        ]
    )
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(revision_rows, use_container_width=True, hide_index=True)
    with right:
        render_insight(
            "<strong>Revision rule:</strong> revision is not triggered because we dislike the wording. "
            "It is triggered when <code>validation_report['revision_required']</code> is true."
        )
        render_insight(
            "<strong>Why this is safer:</strong> the feedback is targeted. The LLM sees the original evidence, "
            "the generated explanation, and a concise list of validator findings."
        )

    st.markdown('<div class="subsection-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### 5. GPT-4o subjective evaluation")
    gpt_cols = st.columns(4)
    gpt_cols[0].metric("Reviewed explanations", f"{len(gpt4o_df)}")
    gpt_cols[1].metric("Mean plausibility", f"{mean_plausibility:.2f}/5")
    gpt_cols[2].metric("Mean clarity", f"{mean_clarity:.2f}/5")
    gpt_cols[3].metric("Evaluator model", str(gpt4o_df["evaluator_model"].iloc[0]))

    subjective_rows = pd.DataFrame(
        [
            {
                "component": "Deterministic validator",
                "role": "hard gate",
                "covers": "faithfulness, caution awareness, completeness, leakage, probability consistency",
            },
            {
                "component": "GPT-4o evaluator",
                "role": "advisory scorer",
                "covers": "clinical plausibility and clarity only",
            },
            {
                "component": "Human/report interpretation",
                "role": "final explanation of results",
                "covers": "why the scores and failures matter for the project",
            },
        ]
    )
    st.dataframe(subjective_rows, use_container_width=True, hide_index=True)

    render_insight(
        "<strong>Key message before the live demo:</strong> a patient explanation is generated by an LLM, "
        "but it is controlled by structured SHAP evidence, deterministic validation, revision feedback, and a narrow GPT-4o quality review."
    )


def evidence_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["shap_value"] = df["shap_value"].map(lambda x: round(float(x), 4))
    df["caution"] = df["caution_flags"].map(
        lambda flags: "; ".join(flags) if isinstance(flags, list) and flags else ""
    )
    return df[
        [
            "feature",
            "value",
            "shap_value",
            "clinical_meaning",
            "caution",
        ]
    ]


def render_prediction(evidence: dict[str, Any]) -> None:
    prediction = evidence.get("prediction", {})
    probability = float(prediction.get("y_proba", 0.0))
    threshold = float(prediction["threshold"])
    predicted_label = int(prediction.get("y_pred", probability >= threshold))
    risk_label = "High mortality risk" if predicted_label == 1 else "Low mortality risk"

    render_section_header(
        title="Prediction Overview",
        caption="Model output for the selected unlabeled ICU patient.",
        kicker="Step 1",
    )
    risk_class = "risk-high" if predicted_label == 1 else "risk-low"
    st.markdown(
        f"<div class='risk-banner {risk_class}'>{risk_label} at threshold {threshold:.2f}</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Mortality probability", format_probability(probability))
    cols[1].metric("Decision threshold", format_probability(threshold))
    cols[2].metric("Prediction", risk_label)
    cols[3].metric("Patient", evidence.get("patient_label", "N/A"))
    st.progress(min(max(probability, 0.0), 1.0))


def render_evidence(evidence: dict[str, Any]) -> None:
    render_section_header(
        title="SHAP Evidence Packet",
        caption="Local risk-increasing and risk-decreasing features passed to the explanation layer.",
        kicker="Step 2",
    )

    inc_tab, dec_tab = st.tabs(
        ["Risk-increasing evidence", "Risk-decreasing evidence"]
    )

    with inc_tab:
        st.dataframe(
            evidence_to_frame(evidence.get("risk_increasing_evidence", [])),
            use_container_width=True,
            hide_index=True,
            column_config={
                "feature": st.column_config.TextColumn("Feature", width="medium"),
                "value": st.column_config.TextColumn("Value", width="small"),
                "shap_value": st.column_config.NumberColumn("SHAP", format="%.4f"),
                "clinical_meaning": st.column_config.TextColumn("Clinical meaning", width="large"),
                "caution": st.column_config.TextColumn("Caution", width="large"),
            },
        )

    with dec_tab:
        st.dataframe(
            evidence_to_frame(evidence.get("risk_decreasing_evidence", [])),
            use_container_width=True,
            hide_index=True,
            column_config={
                "feature": st.column_config.TextColumn("Feature", width="medium"),
                "value": st.column_config.TextColumn("Value", width="small"),
                "shap_value": st.column_config.NumberColumn("SHAP", format="%.4f"),
                "clinical_meaning": st.column_config.TextColumn("Clinical meaning", width="large"),
                "caution": st.column_config.TextColumn("Caution", width="large"),
            },
        )

def render_validation_report(report: dict[str, Any], title: str) -> None:
    st.markdown(f"#### {title}")
    cols = st.columns(3)
    cols[0].metric("Validation", status_text(report.get("passed")))
    cols[1].metric("Revision required", str(report.get("revision_required", "N/A")))
    cols[2].metric(
        "Deterministic score",
        format_probability(report.get("deterministic_validation_score")),
    )

    checks = report.get("checks", {})
    if checks:
        check_rows = []
        for check_name, check_result in checks.items():
            check_rows.append(
                {
                    "check": check_name,
                    "status": "Pass" if check_result.get("passed") else "Fail",
                    "details": summarize_check(check_name, check_result),
                }
            )
        st.dataframe(
            pd.DataFrame(check_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "check": st.column_config.TextColumn("Check", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "details": st.column_config.TextColumn("Details", width="large"),
            },
        )

    feedback = report.get("revision_feedback", [])
    if feedback:
        st.markdown("**Revision feedback**")
        for item in feedback:
            st.write(f"- {item}")


def render_gpt4o_report(report: dict[str, Any], hybrid_score: float | None) -> None:
    cols = st.columns(3)
    cols[0].metric(
        "Clinical plausibility",
        report.get("clinical_plausibility", {}).get("score", "N/A"),
    )
    cols[1].metric(
        "Clarity",
        report.get("clarity", {}).get("score", "N/A"),
    )
    cols[2].metric(
        "Hybrid quality score",
        format_probability(hybrid_score),
    )

    st.markdown("**Evaluator rationale**")
    rationale_rows = [
        {
            "dimension": "clinical_plausibility",
            "rationale": report.get("clinical_plausibility", {}).get("rationale", "-"),
        },
        {
            "dimension": "clarity",
            "rationale": report.get("clarity", {}).get("rationale", "-"),
        },
        {
            "dimension": "overall",
            "rationale": report.get("overall_comments", "-"),
        },
    ]
    st.dataframe(pd.DataFrame(rationale_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Evaluator model: {report.get('evaluator_model', 'gpt-4o')}. "
        "This score is advisory; deterministic validation remains the hard gate."
    )


def summarize_check(check_name: str, check_result: dict[str, Any]) -> str:
    if check_name == "forbidden_phrases":
        return ", ".join(check_result.get("found", [])) or "-"
    if check_name == "true_label_leakage":
        return ", ".join(check_result.get("found", [])) or "-"
    if check_name == "section_structure":
        return ", ".join(check_result.get("missing", [])) or "-"
    if check_name == "prediction_consistency":
        expected = check_result.get("expected_probability")
        matched = check_result.get("matched_probability")
        return f"expected={format_probability(expected)}, matched={format_probability(matched)}"
    if check_name == "caution_mentions":
        missing = check_result.get("missing_features", [])
        return f"missing={', '.join(missing) if missing else '-'}"
    if check_name == "feature_grounding":
        ungrounded = check_result.get("ungrounded_features", [])
        return f"ungrounded={', '.join(ungrounded) if ungrounded else '-'}"
    if check_name == "direction_consistency":
        errors = check_result.get("direction_errors", [])
        return f"errors={len(errors)}"
    return "-"


def run_live_patient(
    patient_position: int,
    top_n: int,
    run_llm: bool,
    run_gpt4o: bool,
    max_revision_rounds: int,
) -> dict[str, Any]:
    raw_unlabeled = load_unlabeled_data()
    preprocessor, model, threshold = load_live_artifacts()

    if patient_position < 0 or patient_position >= len(raw_unlabeled):
        raise ValueError(
            f"Patient position must be between 0 and {len(raw_unlabeled) - 1}."
        )

    patient_label = f"live_unlabeled_patient_{patient_position}"
    raw_patient = raw_unlabeled.iloc[[patient_position]]

    result = run_patient_pipeline(
        raw_patient=raw_patient,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        patient_label=patient_label,
        test_row_index=int(raw_patient.index[0]),
        y_true=None,
        top_n=top_n,
    )

    evidence_packet = result["evidence_packet"]
    prompt = build_explanation_prompt(evidence_packet)
    live_result = {
        "patient_label": patient_label,
        "evidence_packet": evidence_packet,
        "prompt": prompt,
        "generated_explanation": None,
        "generated_validation": None,
        "revised_explanation": None,
        "revised_validation": None,
        "revision_rounds": 0,
        "gpt4o_evaluation": None,
        "hybrid_quality_score": None,
    }

    if not run_llm:
        return live_result

    generated_explanation = generate_explanation(prompt)
    generated_validation = validate_explanation(
        text=generated_explanation,
        evidence_packet=evidence_packet,
    )
    revised_explanation, revised_validation, revision_rounds = revise_until_valid(
        original_prompt=prompt,
        generated_explanation=generated_explanation,
        evidence_packet=evidence_packet,
        validation_report=generated_validation,
        max_rounds=max_revision_rounds,
    )

    live_result.update(
        {
            "generated_explanation": generated_explanation,
            "generated_validation": generated_validation,
            "revised_explanation": revised_explanation,
            "revised_validation": revised_validation,
            "revision_rounds": revision_rounds,
        }
    )

    if run_gpt4o:
        final_explanation = revised_explanation or generated_explanation
        final_validation = revised_validation or generated_validation
        gpt4o_evaluation = evaluate_subjective_quality(
            explanation=final_explanation,
            evidence_packet=evidence_packet,
            deterministic_validation_report=final_validation,
            model="gpt-4o",
        )
        hybrid_quality_score = compute_hybrid_quality_score(
            deterministic_validation_report=final_validation,
            subjective_evaluation_report=gpt4o_evaluation,
            weights=load_evaluation_weights(),
        )
        live_result.update(
            {
                "gpt4o_evaluation": gpt4o_evaluation,
                "hybrid_quality_score": hybrid_quality_score,
            }
        )

    return live_result


def render_live_result(result: dict[str, Any]) -> None:
    evidence = result["evidence_packet"]

    with st.container(border=True):
        render_prediction(evidence)

    with st.container(border=True):
        render_evidence(evidence)

    with st.container(border=True):
        render_section_header(
            title="Prompt",
            caption="The structured instruction produced from the evidence packet.",
            kicker="Step 3",
        )
        with st.expander("Show prompt sent to the LLM"):
            st.code(result["prompt"], language="text")

    generated_explanation = result.get("generated_explanation")
    if generated_explanation is None:
        st.markdown(
            """
            <div class="empty-state">
                Live LLM generation was not requested. The dashboard ran local preprocessing,
                prediction, SHAP, evidence construction, and prompt generation only.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.container(border=True):
        render_section_header(
            title="LLM Explanation",
            caption="Evidence-grounded narrative generated without access to the true label.",
            kicker="Step 4",
        )
        revised_explanation = result.get("revised_explanation")
        if revised_explanation:
            before_tab, after_tab = st.tabs(
                ["Generated explanation", "Revised final explanation"]
            )
            with before_tab:
                st.markdown(generated_explanation)
            with after_tab:
                st.markdown(revised_explanation)
        else:
            st.markdown(generated_explanation)

    with st.container(border=True):
        render_section_header(
            title="Validation Panel",
            caption="Deterministic checks decide whether revision is required.",
            kicker="Step 5",
        )
        generated_validation = result["generated_validation"]
        revised_validation = result.get("revised_validation")
        if revised_validation is not None and revised_explanation:
            before_tab, after_tab = st.tabs(["Generated validation", "Revised validation"])
            with before_tab:
                render_validation_report(generated_validation, "Generated explanation")
            with after_tab:
                render_validation_report(revised_validation, "Revised final explanation")
                st.caption(f"Revision rounds: {result['revision_rounds']}")
        else:
            render_validation_report(generated_validation, "Generated explanation")

    with st.container(border=True):
        render_section_header(
            title="GPT-4o Subjective Evaluation",
            caption="Advisory scoring for clinical plausibility and clarity only.",
            kicker="Step 6",
        )
        gpt4o_evaluation = result.get("gpt4o_evaluation")
        if gpt4o_evaluation is None:
            st.markdown(
                """
                <div class="empty-state">
                    GPT-4o subjective evaluation was not requested. The deterministic
                    validator above remains the hard acceptance gate.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_gpt4o_report(
                report=gpt4o_evaluation,
                hybrid_score=result.get("hybrid_quality_score"),
            )


def render_live_mode() -> None:
    raw_unlabeled = load_unlabeled_data()

    with st.sidebar:
        st.header("Patient Demo")
        st.caption("Run the final pipeline on one unlabeled ICU row.")
        patient_position = st.number_input(
            "Patient position",
            min_value=0,
            max_value=len(raw_unlabeled) - 1,
            value=0,
            step=1,
        )
        top_n = st.slider("Top SHAP features per direction", 3, 12, 8)
        run_llm = st.checkbox("Run live LLM explanation and revision", value=False)
        run_gpt4o = st.checkbox(
            "Run GPT-4o subjective evaluation",
            value=False,
            disabled=not run_llm,
        )
        max_revision_rounds = st.slider("Max revision rounds", 1, 3, 3)
        run_button = st.button("Run live patient pipeline", type="primary")
        st.markdown("---")
        st.markdown(
            "<p class='small-note'>LLM calls are optional. If unchecked, the dashboard runs only local model and SHAP artifacts.</p>",
            unsafe_allow_html=True,
        )

    if run_button:
        with st.spinner("Running live patient pipeline..."):
            st.session_state["live_result"] = run_live_patient(
                patient_position=int(patient_position),
                top_n=int(top_n),
                run_llm=run_llm,
                run_gpt4o=run_gpt4o,
                max_revision_rounds=int(max_revision_rounds),
            )

    result = st.session_state.get("live_result")
    if result is None:
        st.markdown(
            """
            <div class="empty-state">
                Select a patient position in the sidebar, choose whether to run the LLM,
                then start the live pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    render_live_result(result)


@st.cache_resource(show_spinner="Veri indiriliyor (ilk açılış)...")
def _bootstrap_data() -> bool:
    ensure_data()
    return True


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _bootstrap_data()
    render_hero()

    tabs = st.tabs(
        [
            "EDA",
            "Preprocessing & Modeling",
            "SHAP Explainability",
            "LLM & Validation",
            "Live Patient Demo",
        ]
    )

    with tabs[0]:
        render_eda_page()

    with tabs[1]:
        render_modeling_page()

    with tabs[2]:
        render_shap_page()

    with tabs[3]:
        render_architecture_page()

    with tabs[4]:
        render_live_mode()


if __name__ == "__main__":
    main()
