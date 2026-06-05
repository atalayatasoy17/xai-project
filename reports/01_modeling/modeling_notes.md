# Modeling Notes

## Purpose

The modeling stage builds a hospital mortality classifier for ICU patients and
selects a final model that can be reused by the SHAP, evidence, LLM validation,
and dashboard stages.

Because hospital death is an imbalanced target, the final decision was not based
on accuracy alone. The main evaluation metrics were:

- **AUROC:** overall ranking ability.
- **AUPRC:** performance on the minority positive class.
- **Precision:** how many predicted high-risk patients were true deaths.
- **Recall:** how many true deaths were detected.
- **F1:** balance between precision and recall.
- **Confusion matrix:** direct view of TP, FP, FN, and TN counts.

## Final Preprocessing Decision

The final pipeline uses the preprocessing strategy from
`notebooks/09_model_experiment.ipynb` and `src/preprocessing.py`.

Key decisions:

- ID and location-like columns are removed:
  - `encounter_id`
  - `patient_id`
  - `hospital_id`
  - `icu_id`
- APACHE leakage probability columns are removed:
  - `apache_4a_hospital_death_prob`
  - `apache_4a_icu_death_prob`
- Numeric columns are median-imputed with missing indicators.
- Binary columns are imputed and ordinal-encoded.
- Categorical columns are imputed and one-hot encoded with infrequent-category
  handling.
- Numeric scaling is intentionally omitted because the final model is
  tree-based LightGBM and evidence packets should retain clinically readable
  values.

This produced **379 final model features**.

## Final Model

The final model is a tuned LightGBM classifier trained with the experiment
parameters selected in `notebooks/09_model_experiment.ipynb`.

The saved final artifacts are:

- `models/icu_preprocessor.pkl`
- `models/lgbm_tuned_clean.pkl`
- `models/lgbm_tuned_clean_threshold.json`

The final training/export script is:

- `scripts/16_train_final_lgbm_experiment.py`

This script also refreshes `data/processed/` so downstream notebooks can keep
their original structure while using the final preprocessing schema.

## Threshold Decision

The selected threshold is:

```text
threshold = 0.7274
```

The threshold was selected using F1-oriented threshold tuning. Compared with a
lower threshold, this choice produces fewer false positives and higher
precision, while accepting a lower recall. This is now the explicit final
trade-off of the project:

- fewer patients are incorrectly flagged as high risk,
- some additional true death cases are missed,
- the precision/F1 balance improves for the final selected model.

Threshold sweep outputs are saved in:

- `reports/01_modeling/threshold_sweep_lgbm.csv`
- `reports/01_modeling/figures/threshold_sweep_precision_recall_f1.png`
- `reports/01_modeling/figures/threshold_sweep_fp_fn.png`

## Final Test Metrics

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

The confusion matrix is saved as:

- `reports/01_modeling/figures/selected_lgbm_confusion_matrix.png`

## Native LightGBM Feature Importance

Native LightGBM importance was refreshed for the final saved model. This is not
the same as SHAP:

- `split_importance` counts how often a feature is used in tree splits.
- `gain_importance` measures how much a feature improves the model objective
  when used in splits.

Top native gain features:

```text
ventilated_apache
apache_3j_diagnosis
gcs_motor_apache
age
d1_sysbp_min
d1_lactate_min
d1_spo2_min
d1_bun_max
d1_bun_min
gcs_verbal_apache
```

This supports the SHAP analysis but does not replace it. Native importance is a
model-structure diagnostic, while SHAP is used later to explain prediction-level
feature contributions.

Saved outputs:

- `reports/01_modeling/native_lgbm_feature_importance.csv`
- `reports/01_modeling/figures/native_lgbm_feature_importance_gain_top20.png`

## Saved Modeling Outputs

Main tables:

- `reports/01_modeling/selected_lgbm_test_metrics.csv`
- `reports/01_modeling/final_model_comparison.csv`
- `reports/01_modeling/final_feature_names.csv`
- `reports/01_modeling/threshold_sweep_lgbm.csv`
- `reports/01_modeling/native_lgbm_feature_importance.csv`

Main figures:

- `reports/01_modeling/figures/model_comparison_metrics.png`
- `reports/01_modeling/figures/selected_lgbm_confusion_matrix.png`
- `reports/01_modeling/figures/threshold_sweep_precision_recall_f1.png`
- `reports/01_modeling/figures/threshold_sweep_fp_fn.png`
- `reports/01_modeling/figures/native_lgbm_feature_importance_gain_top20.png`

## Conclusion

The final model is a tuned LightGBM classifier using the experiment preprocessing
schema. The final pipeline removes ID/location and leakage probability columns,
keeps numeric values clinically readable, and uses a tuned threshold of `0.7274`.

This final model is the basis for the refreshed SHAP analysis, evidence packet
construction, LLM explanation generation, deterministic validation, GPT-4o
subjective review, and Streamlit dashboard.
