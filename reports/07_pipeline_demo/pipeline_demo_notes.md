# Pipeline Demo Notes

## Purpose

This stage turns the notebook workflow into reusable `.py` modules. The goal is
to take one raw ICU patient row and run the full final pipeline:

```text
raw patient
→ saved/fitted preprocessing
→ LightGBM prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM prompt
→ optional LLM explanation
→ deterministic validation/revision
```

## Final Pipeline Modules

The production-style code lives in `src/`:

- `src/preprocessing.py`: converts raw WiDS rows into the final 379-feature model schema.
- `src/prediction.py`: loads the model/threshold and produces mortality probabilities.
- `src/explainability.py`: produces local SHAP explanations.
- `src/evidence.py`: converts SHAP rows into structured evidence packets.
- `src/prompts.py`: builds LLM prompts from evidence packets without true-label leakage.
- `src/llm.py`: generates and revises explanations.
- `src/validation.py`: deterministically validates explanation quality.
- `src/pipeline.py`: combines preprocessing, prediction, SHAP, and evidence.

## Preprocessing Verification

The final preprocessing verification script is:

```text
scripts/verify/preprocessing.py
```

It checks that:

- train and test splits use the same 379-feature schema,
- dropped ID/location/leakage/target columns are absent,
- feature names match the fitted preprocessor,
- train/test class balance is preserved by stratified splitting.

Current verification summary:

```text
Processed train shape : (73370, 379)
Processed test shape  : (18343, 379)
Leaked columns present: []
Train death rate      : 0.0863
Test death rate       : 0.0863
```

The old direct comparison to a static `data/processed/X_test.csv` was removed
because the final preprocessing schema is generated from the saved experiment
pipeline. The processed files are now refreshed by
`scripts/final/train_final_lgbm.py`.

## Prediction Verification

Prediction verification is run with:

```text
scripts/verify/prediction.py
```

Final test metrics:

| Metric | Value |
| --- | ---: |
| Threshold | 0.7274 |
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

This confirms that the saved `.py` pipeline reproduces the final model
evaluation outside the notebooks.

## SHAP and Evidence Verification

Single-patient SHAP/evidence verification is run with:

```text
scripts/verify/explainability.py
scripts/verify/evidence.py
scripts/verify/patient_pipeline.py
```

The default held-out test patient is:

```text
patient_label: test_patient_0
test_row_index: 72745
y_true: 0
y_pred: 0
y_proba: 0.0229
threshold: 0.7274
prediction_type: TN
```

Example risk-increasing evidence:

- `d1_wbc_min = 22.0`
- `apache_3j_diagnosis = 306.01`
- `d1_resprate_max = 92.0`
- `d1_heartrate_min = 90.0`

Example risk-decreasing evidence:

- `age = 52.0`
- `urineoutput_apache = 3068.928`
- `ventilated_apache = 0.0`
- `d1_mbp_min = 89.0`

The evidence packet separates general clinical meaning from patient-specific
SHAP direction. For example, mechanical ventilation is generally a severity
indicator, but `ventilated_apache = 0` can decrease risk for a specific patient.

## Prompt Construction and Label Leakage Prevention

Prompt verification is run with:

```text
scripts/verify/prompt.py
```

The prompt includes:

- predicted label,
- predicted mortality probability,
- decision threshold,
- risk-increasing evidence,
- risk-decreasing evidence,
- clinical meanings,
- caution flags,
- strict grounding rules.

The prompt intentionally excludes:

- true label,
- prediction type,
- TP/FN/FP/TN case label,
- any statement about whether the prediction was correct.

This prevents label leakage into generated explanations.

## Saved Test Patient Demo

The non-LLM pipeline demo is:

```text
scripts/demo/test_patient.py
```

It writes:

- `reports/07_pipeline_demo/test_patient_0_prediction.json`
- `reports/07_pipeline_demo/test_patient_0_evidence.json`
- `reports/07_pipeline_demo/test_patient_0_prompt.txt`

The saved-artifact version is:

```text
scripts/demo/saved_artifact_patient.py
```

It verifies that the serialized preprocessor/model/threshold can run the same
single-patient inference workflow.

## LLM Generation and Deterministic Validation

The LLM pipeline demo is:

```text
scripts/demo/test_patient_llm.py
```

For the refreshed final model, the initial explanation required revision:

```text
Validation passed: False
Revision required: True
Validation score : 4.538
Revision rounds  : 2
Revised passed   : True
Revised score    : 5.0
```

This demonstrates the intended agentic review loop:

```text
generate explanation
→ validate deterministically
→ revise if needed
→ validate revised explanation
```

The explanation is not accepted blindly; it must pass deterministic checks for
forbidden wording, probability consistency, caution awareness, section
structure, feature grounding, and SHAP direction consistency.

## Conclusion

The pipeline demo confirms that the final model can be used end-to-end from raw
patient row to validated explanation. The same code path supports held-out test
patients, unlabeled patients, and the Streamlit dashboard.
