"""Streamlit dashboard for saved ICU XAI demo outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEMO_DIRS = [
    REPORTS_DIR / "07_pipeline_demo",
    REPORTS_DIR / "08_unlabeled_demo",
]
AUDIT_PATH = REPORTS_DIR / "09_validation_audit" / "validation_audit_summary.csv"
GPT4O_SUMMARY_PATH = (
    REPORTS_DIR / "10_gpt4o_evaluation" / "gpt4o_subjective_evaluation_summary.csv"
)
MODEL_METRICS_PATH = REPORTS_DIR / "01_modeling" / "selected_lgbm_test_metrics.csv"
MODEL_FIGURES_DIR = REPORTS_DIR / "01_modeling" / "figures"
SHAP_FIGURES_DIR = REPORTS_DIR / "02_explainability" / "figures"


st.set_page_config(
    page_title="ICU Mortality XAI Dashboard",
    page_icon="",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.small-note {
    color: #5b6472;
    font-size: 0.92rem;
}
.status-pass {
    color: #157347;
    font-weight: 700;
}
.status-fail {
    color: #b42318;
    font-weight: 700;
}
.section-caption {
    color: #4b5563;
    margin-top: -0.4rem;
}
</style>
"""


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    with open(path) as f:
        return f.read()


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_cases() -> dict[str, dict[str, Path]]:
    cases: dict[str, dict[str, Path]] = {}

    for demo_dir in DEMO_DIRS:
        for evidence_path in sorted(demo_dir.glob("*_llm_evidence.json")):
            case_id = evidence_path.name.removesuffix("_llm_evidence.json")
            case_files = {
                "evidence": evidence_path,
                "generated_explanation": demo_dir / f"{case_id}_llm_explanation.txt",
                "generated_validation": demo_dir / f"{case_id}_llm_validation.json",
                "revised_explanation": demo_dir
                / f"{case_id}_llm_revised_explanation.txt",
                "revised_validation": demo_dir / f"{case_id}_llm_revised_validation.json",
            }
            cases[case_id] = {
                name: path for name, path in case_files.items() if path.exists()
            }

    return cases


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


def status_markdown(label: str, passed: bool | None) -> None:
    css_class = "status-pass" if passed else "status-fail"
    symbol = "OK" if passed else "FAIL"
    st.markdown(f"**{label}:** <span class='{css_class}'>{symbol}</span>", unsafe_allow_html=True)


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


def get_final_files(case_files: dict[str, Path]) -> tuple[Path, Path, str]:
    if "revised_explanation" in case_files and "revised_validation" in case_files:
        return (
            case_files["revised_explanation"],
            case_files["revised_validation"],
            "revised",
        )
    return (
        case_files["generated_explanation"],
        case_files["generated_validation"],
        "generated",
    )


def render_prediction(evidence: dict[str, Any]) -> None:
    prediction = evidence.get("prediction", {})
    probability = float(prediction.get("y_proba", 0.0))
    threshold = float(prediction.get("threshold", 0.5))
    predicted_label = int(prediction.get("y_pred", probability >= threshold))
    risk_label = "High mortality risk" if predicted_label == 1 else "Low mortality risk"

    st.subheader("Prediction")
    cols = st.columns(4)
    cols[0].metric("Mortality probability", format_probability(probability))
    cols[1].metric("Decision threshold", format_probability(threshold))
    cols[2].metric("Prediction", risk_label)
    cols[3].metric("Patient", evidence.get("patient_label", "N/A"))
    st.progress(min(max(probability, 0.0), 1.0))


def render_evidence(evidence: dict[str, Any]) -> None:
    st.subheader("SHAP Evidence Packet")
    st.markdown(
        "<p class='section-caption'>Local SHAP evidence used to ground the LLM explanation.</p>",
        unsafe_allow_html=True,
    )

    inc_tab, dec_tab, raw_tab = st.tabs(
        ["Risk-increasing evidence", "Risk-decreasing evidence", "Raw packet"]
    )

    with inc_tab:
        st.dataframe(
            evidence_to_frame(evidence.get("risk_increasing_evidence", [])),
            use_container_width=True,
            hide_index=True,
        )

    with dec_tab:
        st.dataframe(
            evidence_to_frame(evidence.get("risk_decreasing_evidence", [])),
            use_container_width=True,
            hide_index=True,
        )

    with raw_tab:
        st.json(evidence)


def render_explanation(case_files: dict[str, Path], final_explanation_path: Path) -> None:
    st.subheader("LLM Explanation")
    st.markdown(
        "<p class='section-caption'>Generated from the structured evidence packet; true labels are not shown to the LLM.</p>",
        unsafe_allow_html=True,
    )

    generated = read_text(str(case_files["generated_explanation"]))
    final_explanation = read_text(str(final_explanation_path))

    if final_explanation_path == case_files["generated_explanation"]:
        st.markdown(final_explanation)
        return

    before_tab, after_tab = st.tabs(["Generated explanation", "Revised final explanation"])
    with before_tab:
        st.markdown(generated)
    with after_tab:
        st.markdown(final_explanation)


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
                    "passed": check_result.get("passed"),
                    "details": summarize_check(check_name, check_result),
                }
            )
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

    feedback = report.get("revision_feedback", [])
    if feedback:
        st.markdown("**Revision feedback**")
        for item in feedback:
            st.write(f"- {item}")


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


def render_validation(case_files: dict[str, Path], final_validation_path: Path) -> None:
    st.subheader("Validation Panel")
    st.markdown(
        "<p class='section-caption'>Deterministic hard checks before an explanation is accepted.</p>",
        unsafe_allow_html=True,
    )

    generated_report = read_json(str(case_files["generated_validation"]))
    final_report = read_json(str(final_validation_path))

    if final_validation_path == case_files["generated_validation"]:
        render_validation_report(final_report, "Generated explanation")
        return

    before_tab, after_tab = st.tabs(["Generated validation", "Revised validation"])
    with before_tab:
        render_validation_report(generated_report, "Generated explanation")
    with after_tab:
        render_validation_report(final_report, "Revised final explanation")


def render_gpt4o(case_id: str) -> None:
    if not GPT4O_SUMMARY_PATH.exists():
        return

    summary = read_csv(str(GPT4O_SUMMARY_PATH))
    matches = summary[summary["case_id"].str.startswith(case_id)]
    if matches.empty:
        return

    row = matches.iloc[-1]
    st.subheader("GPT-4o Subjective Evaluation")
    st.markdown(
        "<p class='section-caption'>Advisory evaluation for subjective quality only: clinical plausibility and clarity.</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].metric("Clinical plausibility", int(row["clinical_plausibility"]))
    cols[1].metric("Clarity", int(row["clarity"]))
    cols[2].metric("Hybrid quality score", format_probability(row["hybrid_quality_score"]))


def render_audit_context(case_id: str) -> None:
    if not AUDIT_PATH.exists():
        return

    audit_df = read_csv(str(AUDIT_PATH))
    st.subheader("Validation Audit Context")
    case_rows = audit_df[audit_df["case_id"].str.startswith(case_id)]
    if not case_rows.empty:
        st.dataframe(
            case_rows[
                [
                    "case_id",
                    "passed",
                    "revision_required",
                    "deterministic_validation_score",
                    "forbidden_phrases",
                    "missing_caution_mentions",
                    "ungrounded_features",
                    "direction_errors",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    total = len(audit_df)
    passed = int(audit_df["passed"].sum())
    st.caption(f"Audit summary: {passed}/{total} saved explanations passed deterministic validation.")


def render_model_context() -> None:
    st.subheader("Project Context")
    st.markdown(
        """
        This dashboard displays saved outputs from the ICU mortality XAI pipeline:
        LightGBM prediction, local SHAP evidence, evidence-grounded LLM explanation,
        deterministic validation, revision trace, and optional GPT-4o subjective scoring.
        """
    )

    if MODEL_METRICS_PATH.exists():
        metrics = read_csv(str(MODEL_METRICS_PATH))
        if not metrics.empty:
            row = metrics.iloc[0].to_dict()
            cols = st.columns(6)
            for col, metric in zip(
                cols,
                ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1"],
            ):
                if metric in row:
                    col.metric(metric, format_probability(row[metric]))

    figure_candidates = [
        MODEL_FIGURES_DIR / "selected_lgbm_confusion_matrix.png",
        SHAP_FIGURES_DIR / "global_shap_importance_top20.png",
        SHAP_FIGURES_DIR / "shap_summary_top20.png",
    ]
    existing_figures = [path for path in figure_candidates if path.exists()]
    if existing_figures:
        fig_tabs = st.tabs([path.stem.replace("_", " ").title() for path in existing_figures])
        for tab, path in zip(fig_tabs, existing_figures):
            with tab:
                st.image(str(path), use_container_width=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("ICU Mortality XAI Dashboard")
    st.caption("Patient-level prediction, SHAP evidence, LLM explanation, and validation trace.")

    cases = find_cases()
    if not cases:
        st.error("No saved LLM demo cases were found under reports/07_pipeline_demo or reports/08_unlabeled_demo.")
        return

    with st.sidebar:
        st.header("Demo Controls")
        selected_case = st.selectbox("Select patient case", sorted(cases))
        show_project_context = st.checkbox("Show project context", value=True)
        show_audit_context = st.checkbox("Show audit context", value=True)
        show_gpt4o = st.checkbox("Show GPT-4o subjective score", value=True)
        st.markdown("---")
        st.markdown(
            "<p class='small-note'>This dashboard reads saved report files only. It does not call the OpenAI API.</p>",
            unsafe_allow_html=True,
        )

    case_files = cases[selected_case]
    evidence = read_json(str(case_files["evidence"]))
    final_explanation_path, final_validation_path, final_kind = get_final_files(case_files)

    if show_project_context:
        render_model_context()
        st.divider()

    render_prediction(evidence)
    st.divider()

    render_evidence(evidence)
    st.divider()

    render_explanation(case_files, final_explanation_path)
    st.caption(f"Displayed final explanation version: {final_kind}.")
    st.divider()

    render_validation(case_files, final_validation_path)
    st.divider()

    if show_gpt4o:
        render_gpt4o(selected_case)
        st.divider()

    if show_audit_context:
        render_audit_context(selected_case)


if __name__ == "__main__":
    main()
