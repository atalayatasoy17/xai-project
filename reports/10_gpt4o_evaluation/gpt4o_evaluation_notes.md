# GPT-4o Subjective Evaluation Notes

This stage adds an advisory GPT-4o evaluation layer after deterministic validation. It is intentionally separate from the hard validation checks.

## Purpose

The deterministic validator checks objective, evidence-grounded issues such as label leakage, forbidden wording, prediction consistency, caution mentions, feature grounding, SHAP direction consistency, and required section structure.

GPT-4o is used only for the rubric dimensions that are more subjective:

- clinical plausibility
- clarity

This keeps the evaluation design conservative: deterministic checks remain the gatekeeper, while GPT-4o provides an additional quality assessment for dimensions that are difficult to score with regex or exact matching.

## Method

The script `scripts/15_run_gpt4o_subjective_evaluation.py` reads explanations that already passed deterministic validation from:

`reports/09_validation_audit/validation_audit_summary.csv`

For each passing explanation, it loads:

- the saved explanation text
- the matching evidence packet
- the deterministic validation report
- the original rubric weights from `reports/05_evaluation/evaluation_weights.json`

GPT-4o then scores only:

- `clinical_plausibility`
- `clarity`

The final hybrid quality score combines:

- deterministic scores for `faithfulness_no_hallucination`, `caution_awareness`, and `completeness`
- GPT-4o scores for `clinical_plausibility` and `clarity`

## Results

Seven explanations were evaluated. These were the explanations that passed deterministic validation after alias-aware caution matching.

Summary:

- five explanations received `clinical_plausibility = 4` and `clarity = 4`, with hybrid score `4.65`
- two explanations for the high-risk unlabeled case, `unlabeled_patient_15_llm_explanation` and `unlabeled_patient_15_llm_revised_explanation`, received `clinical_plausibility = 5` and `clarity = 4`, with hybrid score `4.90`

This suggests that the validator-approved explanations are also judged by GPT-4o as clinically plausible and clear.

## Example

For `unlabeled_patient_15_llm_explanation`, GPT-4o assigned:

- clinical plausibility: `5`
- clarity: `4`
- hybrid quality score: `4.90`

The rationale noted that the explanation aligned with critical illness indicators such as low heart rate, hypoxemia, and impaired neurological response.

## Interpretation

This creates a two-layer evaluation design:

1. Deterministic validator: hard safety and faithfulness checks.
2. GPT-4o evaluator: advisory subjective scoring for plausibility and clarity.

The GPT-4o evaluator does not decide whether an explanation passes or fails the pipeline. That decision remains with the deterministic validator. Its role is to provide additional reporting evidence about explanation quality.

## Limitation

GPT-4o evaluation is still model-based judgment. It may be useful for subjective quality assessment, but it should not replace deterministic evidence checks or clinical review.
