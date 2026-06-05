# Unlabeled Patient Demo Notes

## Purpose

This stage runs the final saved-artifact pipeline on `data/raw/unlabeled.csv`.
The goal is to simulate a deployment-like setting where the model receives a raw
patient row without a known `hospital_death` label.

This is not a performance evaluation because unlabeled rows do not contain true
outcomes. It demonstrates inference, explanation, validation, and reporting.

## Saved Artifact Flow

The unlabeled demo uses saved artifacts only:

```text
models/icu_preprocessor.pkl
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

Flow:

```text
raw unlabeled patient
→ saved preprocessor transform
→ saved LightGBM prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM prompt
→ optional LLM explanation
→ deterministic validation/revision
```

Main scripts:

```text
scripts/11_run_unlabeled_patient_demo.py
scripts/12_run_unlabeled_patient_llm_demo.py
```

Both scripts support:

```text
--patient-position <row>
--no-save
```

## Refreshed Demo Patients

The refreshed final model outputs were generated for four unlabeled examples:

| Patient | Probability | Threshold | Prediction | LLM validation result |
| --- | ---: | ---: | ---: | --- |
| `unlabeled_patient_0` | 0.1825 | 0.7274 | 0 | passed directly |
| `unlabeled_patient_3` | 0.5217 | 0.7274 | 0 | passed directly |
| `unlabeled_patient_7` | 0.2141 | 0.7274 | 0 | revised, then passed |
| `unlabeled_patient_15` | 0.9940 | 0.7274 | 1 | passed directly |

Because true labels are unavailable, none of these predictions can be called
correct or incorrect.

## Example: `unlabeled_patient_0`

The model predicted:

```text
death_probability: 0.1825
prediction: 0
threshold: 0.7274
```

The main risk-increasing evidence included impaired GCS components, low
systolic blood pressure, neurological body-system information, Neuro ICU
admission, and other patient-level features.

The main risk-decreasing evidence included younger age, absence of mechanical
ventilation, respiratory-rate values, oxygen saturation, hemoglobin, and white
blood cell count.

The LLM explanation passed deterministic validation directly with score `5.0`.

## Example: `unlabeled_patient_15`

The model predicted:

```text
death_probability: 0.9940
prediction: 1
threshold: 0.7274
```

The explanation described high-risk evidence including:

- `d1_heartrate_min = 0.0`
- low minimum SpO2
- low GCS motor score
- mechanical ventilation
- low systolic blood pressure
- diagnosis category information
- high lactate
- low arterial pH

The zero-valued minimum heart rate received a caution note because it may
represent an extreme clinical event or a recording artifact.

## Validation and Revision

The refreshed deterministic audit shows:

- direct pass for `unlabeled_patient_0`
- direct pass for `unlabeled_patient_3`
- direct pass for `unlabeled_patient_15`
- revision required for `unlabeled_patient_7`, followed by revised pass

For `unlabeled_patient_7`, the initial explanation used unsupported wording
(`abnormal`). The revision loop removed/rephrased the unsupported wording and
the revised explanation passed with deterministic score `5.0`.

## Important Final-Model Difference

Earlier prototype outputs included `icu_id` as evidence and therefore required
explicit caution handling for unit/location identifiers. In the refreshed final
model, ID and location-like columns are removed during preprocessing. Therefore,
unlabeled explanations no longer use `icu_id` as model evidence.

Remaining caution behavior focuses on values such as zero-valued vital signs and
unusual time-related features when they appear in the evidence packet.

## Outputs

Main files are stored in:

```text
reports/08_unlabeled_demo/
```

For each selected patient, the demo may include:

- `*_prediction.json`
- `*_evidence.json`
- `*_prompt.txt`
- `*_llm_evidence.json`
- `*_llm_prompt.txt`
- `*_llm_explanation.txt`
- `*_llm_validation.json`
- `*_llm_revised_explanation.txt` only when revision was needed
- `*_llm_revised_validation.json` only when revision was needed

## Conclusion

The unlabeled demo confirms that the final project can process raw, unlabeled
patient rows with saved artifacts and produce model predictions, SHAP evidence,
LLM explanations, deterministic validation reports, and revised explanations
when needed.
