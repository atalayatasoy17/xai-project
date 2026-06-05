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

## Key Contribution

The main contribution of this project is a hybrid explanation-validation architecture:

- deterministic validation acts as the hard gatekeeper for evidence-grounded checks
- LLM revision is triggered only when the deterministic validator finds an issue
- GPT-4o is used only as an advisory evaluator for subjective dimensions
- saved explanations are audited systematically rather than accepted as one-off LLM outputs

This design was motivated by exploratory experiments where LLM explanations and even LLM-based evaluators could produce unsupported or false-positive judgments. The final pipeline therefore keeps objective safety checks deterministic and reproducible.

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

For the containerized dashboard, the required raw demo files are downloaded automatically
from public Google Drive links by `src/data_fetch.py` when the app starts. This avoids
requiring a Kaggle login or manual local data download during the Docker demo.

Users who want to reproduce full training should still respect the original
WiDS/Kaggle dataset terms and licensing. The repository does not commit raw data
or API credentials.

## Methodology Summary

### 1. Preprocessing

Preprocessing decisions were learned from the training split only:

- removed ID/location columns: `encounter_id`, `patient_id`, `hospital_id`, `icu_id`
- removed leakage-prone APACHE death probability columns
- imputed numeric features with train medians and added missingness indicators
- imputed binary/categorical features
- ordinal-encoded binary features
- one-hot encoded categorical variables with infrequent-category handling
- omitted numeric scaling because LightGBM is tree-based and evidence values should remain clinically readable
- aligned train/test/new-patient feature schemas

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
LightGBM Tuned Experiment
threshold = 0.7274
```

Final test performance:

| Metric | Value |
| --- | ---: |
| AUROC | 0.9103 |
| AUPRC | 0.5999 |
| Accuracy | 0.9189 |
| Precision | 0.5278 |
| Recall | 0.5704 |
| F1 | 0.5483 |
| TN | 15952 |
| FP | 808 |
| FN | 680 |
| TP | 903 |

The final threshold was selected through F1-oriented threshold tuning. It improves precision and F1 while accepting lower recall than the earlier low-threshold prototype.

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
- exploratory SHAP interaction and feature-correlation checks
- caution notes for zero-valued vital signs, unusual timing values, and coded diagnosis/category features

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

The caution validator uses limited alias-aware matching for caution-flagged features. This remains useful for readable caution phrasing, although `icu_id` itself is no longer a final model feature because ID/location columns are removed during preprocessing.

### 7. Validation Audit and GPT-4o Evaluation

Saved explanations are audited with:

```text
scripts/evaluation/audit_saved_explanations.py
```

Current audit result:

- 7 current saved explanations audited
- 5 final accepted explanations passed deterministic validation directly or after revision
- 2 initial explanations failed due to unsupported wording and passed after revision

GPT-4o is used only as an advisory evaluator for:

- clinical plausibility
- clarity

It does not decide hard pass/fail status. The hard gatekeeper remains the deterministic validator.

Current GPT-4o evaluation:

- 5 final accepted explanations evaluated
- hybrid scores ranged from `4.65` to `4.90`

## Notebooks vs Final Pipeline

The notebooks document the exploratory development process: EDA, preprocessing decisions, model comparison, SHAP analysis, evidence construction, prompt design, and early LLM evaluation experiments.

The final reproducible pipeline lives in `src/` and `scripts/`:

- `src/` contains reusable modules
- `scripts/` contains runnable verification and demo entry points
- `reports/` stores saved outputs, notes, audits, and evaluation summaries

In other words, notebooks explain how decisions were made, while the Python modules and scripts implement the final pipeline.

## Repository Structure

```text
xai-project/
├── Dockerfile
├── .dockerignore
├── requirements.txt
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
│   ├── 08_llm_generation_and_agentic_review.ipynb
│   └── 09_model_experiment.ipynb
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
├── dashboard/
│   └── app.py
├── scripts/
│   ├── final/                    # final model/report refresh scripts
│   ├── verify/                   # deterministic regression checks
│   ├── demo/                     # patient-level demo scripts
│   ├── evaluation/               # validation audit and GPT-4o scoring
│   └── eski/                     # archived old helper scripts
└── src/
    ├── preprocessing.py
    ├── prediction.py
    ├── explainability.py
    ├── evidence.py
    ├── prompts.py
    ├── llm.py
    ├── validation.py
    ├── evaluator.py
    ├── data_fetch.py
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

Download required raw CSV files for local scripts:

```bash
python src/data_fetch.py
```

Run core verification:

```bash
python scripts/verify/preprocessing.py
python scripts/verify/prediction.py
python scripts/verify/explainability.py
python scripts/verify/evidence.py
python scripts/verify/patient_pipeline.py
python scripts/verify/prompt.py
```

Refresh final model and report artifacts:

```bash
python scripts/final/train_final_lgbm.py
python scripts/final/refresh_explainability_reports.py
python scripts/final/refresh_modeling_reports.py
python scripts/final/refresh_evidence_packets.py
```

Run a saved-artifact patient demo:

```bash
python scripts/demo/saved_artifact_patient.py
```

Run an unlabeled patient prediction demo:

```bash
python scripts/demo/unlabeled_patient.py --patient-position 15 --no-save
```

Run an unlabeled patient LLM explanation demo:

```bash
python scripts/demo/unlabeled_patient_llm.py --patient-position 15 --no-save
```

Run validation fixture tests:

```bash
python scripts/verify/validation.py
```

Run saved explanation audit:

```bash
python scripts/evaluation/audit_saved_explanations.py
```

Run GPT-4o subjective evaluation:

```bash
python scripts/evaluation/gpt4o_subjective_evaluation.py
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501`. LLM and GPT-4o calls are optional controls inside the dashboard.

## Run with Docker

Build the image:

```bash
docker build -t xai-project .
```

Run the dashboard:

```bash
docker run --rm -p 8501:8501 xai-project
```

Then open:

```text
http://localhost:8501
```

On first startup, the container downloads the required raw CSV files automatically
into `data/raw/`. No Kaggle login is required for the dashboard demo.

If port `8501` is already in use, run the same container on another local port:

```bash
docker run --rm -p 8502:8501 xai-project
```

Then open `http://localhost:8502`.

LLM generation and GPT-4o evaluation are optional. To enable live LLM calls, pass an
OpenAI API key:

```bash
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_api_key_here xai-project
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
- An exploratory GPT-4o evaluator produced a false-positive faithfulness concern in one case; this motivated keeping deterministic validation as the hard pass/fail layer.
- Earlier exploratory analysis found that `icu_id` could capture unit/location patterns; the final preprocessing schema removes ID/location columns from the model.
- Alias-aware caution matching is intentionally limited to a small set of caution-flagged features.
- Raw data and API keys are intentionally excluded from Git.
