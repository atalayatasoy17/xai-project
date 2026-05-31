# Explainable ICU Mortality Prediction Pipeline

This project builds an end-to-end explainable AI pipeline for ICU hospital mortality prediction using the WiDS Datathon 2020 dataset. The goal is not only to predict mortality risk, but also to explain each prediction using SHAP evidence, structured clinical reasoning, LLM-generated explanations, deterministic validation, and advisory GPT-4o evaluation.

The final system is designed around one principle:

> Model explanations should be evidence-grounded, auditable, and protected against unsupported LLM reasoning.

## Project Pipeline

```text
raw ICU patient data
→ fitted preprocessing artifact
→ LightGBM mortality prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM explanation prompt
→ GPT-4.1-mini explanation
→ deterministic validation
→ revision if needed
→ validation audit
→ GPT-4o subjective evaluation
```

The deployment-style path uses saved artifacts:

```text
models/icu_preprocessor.pkl
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

This allows a new or unlabeled patient row to be transformed, predicted, explained, and validated without refitting preprocessing on new data.

## Dataset

- Dataset: WiDS Datathon 2020 ICU data
- Target: `hospital_death`
- Task: binary classification
- Positive class: in-hospital mortality

Raw and processed data are excluded from Git:

```text
data/raw/
data/processed/
```

## Methodology Summary

### 1. Preprocessing

Preprocessing decisions were learned from the training split only:

- removed ID and leakage-prone APACHE death probability columns
- dropped columns with more than 50% missingness
- added missingness indicators for remaining missing columns
- imputed numeric features with train medians
- imputed categorical features with `Unknown`
- one-hot encoded categorical variables
- aligned train/test feature schemas

The final preprocessing logic was converted into `ICUPreprocessor` and saved as `models/icu_preprocessor.pkl`.

### 2. Modeling

Several models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- imbalance-weighted XGBoost / LightGBM
- Optuna-tuned XGBoost / LightGBM

Because the positive class is rare, model selection used AUROC, AUPRC, recall, precision, F1, and confusion matrix rather than accuracy alone.

Key metrics:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Final model:

```text
LightGBM Tuned Clean
threshold = 0.50
```

Final test performance:

| Metric | Value |
| --- | ---: |
| AUROC | 0.9019 |
| AUPRC | 0.5824 |
| Accuracy | 0.9013 |
| Precision | 0.4486 |
| Recall | 0.6286 |
| F1 | 0.5235 |
| TN | 15537 |
| FP | 1223 |
| FN | 588 |
| TP | 995 |

Threshold `0.50` was kept because it produced a better recall / false-negative balance than higher thresholds in this clinical risk context.

### 3. SHAP Explainability

The final LightGBM model is explained using SHAP:

```text
SHAP > 0  → increases predicted mortality risk
SHAP < 0  → decreases predicted mortality risk
```

The project includes:

- global SHAP importance
- local patient-level explanations
- TP / FN / FP / TN case analysis
- group-level SHAP error analysis
- caution notes for variables such as `icu_id`, zero-valued vital signs, and negative `pre_icu_los_days`

### 4. Structured Evidence Packets

Raw SHAP tables are converted into structured evidence packets before being passed to the LLM. Each packet contains:

- model prediction
- predicted probability
- threshold
- risk-increasing SHAP evidence
- risk-decreasing SHAP evidence
- feature values
- clinical meaning
- caution flags

This evidence layer prevents the LLM from reasoning directly over raw model internals without structure.

### 5. LLM Explanation Generation

The explanation generator uses `gpt-4.1-mini`. Prompts include only:

- predicted label
- predicted mortality probability
- threshold
- SHAP evidence
- clinical meanings
- caution flags

True labels and TP/FN/FP/TN metadata are intentionally excluded from the LLM prompt to prevent label leakage.

### 6. Deterministic Validation and Revision

LLM explanations are not accepted blindly. The deterministic validator checks:

- unsupported / forbidden wording
- true-label leakage
- required section structure
- prediction probability consistency
- caution mentions
- exact feature grounding
- SHAP direction consistency

The validator returns a structured report:

```text
passed
revision_required
deterministic_validation_score
dimension_scores
checks
revision_feedback
```

The deterministic score covers only rubric dimensions that can be checked reliably:

```text
score =
(0.30 / 0.65) * faithfulness
+ (0.20 / 0.65) * caution_awareness
+ (0.15 / 0.65) * completeness
```

Clinical plausibility and clarity are evaluated separately because they are more subjective.

The caution validator uses limited alias-aware matching for caution-flagged features. For example, `ICU unit identifier` can be accepted as a clinical alias for `icu_id` when used in the `Caution notes` section with caution language.

### 7. Validation Audit and GPT-4o Evaluation

Saved explanations are audited with:

```text
scripts/14_audit_saved_explanations.py
```

Current audit result:

- 10 saved explanations audited
- 7 passed deterministic validation
- 3 failed due to unsupported wording
- all revised explanations passed deterministic validation

GPT-4o is used only as an advisory evaluator for:

- clinical plausibility
- clarity

It does not decide hard pass/fail status. The hard gatekeeper remains the deterministic validator.

Current GPT-4o evaluation:

- 7 validated explanations evaluated
- hybrid scores ranged from `4.65` to `4.90`

## Repository Structure

```text
xai-project/
├── data/
│   ├── raw/                       # excluded from Git
│   └── processed/                 # excluded from Git
├── models/
│   ├── icu_preprocessor.pkl
│   ├── lgbm_tuned_clean.pkl
│   └── lgbm_tuned_clean_threshold.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb
│   ├── 04_shap_explainability.ipynb
│   ├── 05_evidence_construction.ipynb
│   ├── 06_llm_reasoning.ipynb
│   ├── 07_explanation_evaluation.ipynb
│   └── 08_llm_generation_and_agentic_review.ipynb
├── reports/
│   ├── 01_modeling/
│   ├── 02_explainability/
│   ├── 03_evidence/
│   ├── 04_llm_reasoning/
│   ├── 05_evaluation/
│   ├── 06_llm_generation/
│   ├── 07_pipeline_demo/
│   ├── 08_unlabeled_demo/
│   ├── 09_validation_audit/
│   └── 10_gpt4o_evaluation/
├── scripts/
│   ├── 01_verify_preprocessing.py
│   ├── 02_verify_prediction.py
│   ├── 03_verify_explainability.py
│   ├── 04_verify_evidence.py
│   ├── 05_verify_patient_pipeline.py
│   ├── 06_verify_prompt.py
│   ├── 07_run_test_patient_demo.py
│   ├── 08_run_test_patient_llm_demo.py
│   ├── 09_save_preprocessor_artifact.py
│   ├── 10_run_saved_artifact_patient_demo.py
│   ├── 11_run_unlabeled_patient_demo.py
│   ├── 12_run_unlabeled_patient_llm_demo.py
│   ├── 13_verify_validation.py
│   ├── 14_audit_saved_explanations.py
│   └── 15_run_gpt4o_subjective_evaluation.py
└── src/
    ├── preprocessing.py
    ├── prediction.py
    ├── explainability.py
    ├── evidence.py
    ├── prompts.py
    ├── llm.py
    ├── validation.py
    ├── evaluator.py
    └── pipeline.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

For LLM scripts, create a local `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Run core verification:

```bash
python scripts/01_verify_preprocessing.py
python scripts/02_verify_prediction.py
python scripts/03_verify_explainability.py
python scripts/04_verify_evidence.py
python scripts/05_verify_patient_pipeline.py
python scripts/06_verify_prompt.py
```

Run a saved-artifact patient demo:

```bash
python scripts/10_run_saved_artifact_patient_demo.py
```

Run an unlabeled patient prediction demo:

```bash
python scripts/11_run_unlabeled_patient_demo.py --patient-position 15 --no-save
```

Run an unlabeled patient LLM explanation demo:

```bash
python scripts/12_run_unlabeled_patient_llm_demo.py --patient-position 15 --no-save
```

Run validation fixture tests:

```bash
python scripts/13_verify_validation.py
```

Run saved explanation audit:

```bash
python scripts/14_audit_saved_explanations.py
```

Run GPT-4o subjective evaluation:

```bash
python scripts/15_run_gpt4o_subjective_evaluation.py
```

## Key Outputs

| Output | Path |
| --- | --- |
| Model comparison | `reports/01_modeling/final_model_comparison.csv` |
| SHAP outputs | `reports/02_explainability/` |
| Evidence packets | `reports/03_evidence/` |
| Pipeline demo outputs | `reports/07_pipeline_demo/` |
| Unlabeled demo outputs | `reports/08_unlabeled_demo/` |
| Validation audit | `reports/09_validation_audit/validation_audit_summary.csv` |
| GPT-4o evaluation | `reports/10_gpt4o_evaluation/gpt4o_subjective_evaluation_summary.csv` |

## Important Notes

- This project is for research and educational use only.
- Model predictions are not clinical decisions.
- LLM explanations are generated drafts and require validation.
- The deterministic validator is the hard gatekeeper for explanation safety checks.
- GPT-4o is used only as an advisory evaluator for subjective quality dimensions.
- `icu_id` is interpreted cautiously because it may reflect unit-level patterns rather than patient-level clinical status.
- Alias-aware caution matching is intentionally limited to a small set of caution-flagged features.
- Raw data and API keys are intentionally excluded from Git.
