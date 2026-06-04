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
            <div class="eyebrow">Live XAI Demo</div>
            <div class="hero-title">ICU Mortality Explanation Dashboard</div>
            <p class="hero-subtitle">
                Run a patient-level mortality prediction, inspect local SHAP evidence,
                generate an evidence-grounded LLM explanation, and validate the result
                with deterministic safety checks.
            </p>
            <div class="pipeline-strip">Patient data -> LightGBM -> SHAP evidence -> LLM explanation -> Validator -> GPT-4o quality judge</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, caption: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-caption'>{caption}</p>", unsafe_allow_html=True)


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
    threshold = float(prediction.get("threshold", 0.5))
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


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    render_live_mode()


if __name__ == "__main__":
    main()
