# Error Analysis and Sensitivity Notes

## Global SHAP Observations

The refreshed SHAP analysis uses the final tuned LightGBM model and the final
379-feature preprocessing schema.

Top global SHAP features by mean absolute SHAP value:

```text
age
ventilated_apache
apache_3j_diagnosis
d1_bun_max
d1_spo2_min
gcs_motor_apache
gcs_verbal_apache
d1_heartrate_min
d1_heartrate_max
d1_resprate_max
```

Compared with the earlier prototype, `icu_id` is no longer part of the final
model input. The earlier `icu_id` analysis was useful because it identified a
unit/location signal that should not be treated as patient-level clinical
evidence. In the final preprocessing schema, ID/location columns are removed.

## Saved Explainability Outputs

The refreshed report artifacts are stored under:

- `reports/02_explainability/figures/`
- `reports/02_explainability/tables/`

Key figures:

- `figures/predicted_probability_distribution.png`
- `figures/global_shap_importance_top20.png`
- `figures/shap_summary_top20.png`
- `figures/shap_dependence_age.png`
- `figures/shap_dependence_d1_spo2_min.png`
- `figures/shap_dependence_gcs_motor_apache.png`
- `figures/shap_effect_ventilated_apache.png`
- `figures/local_waterfall_tp.png`
- `figures/local_waterfall_fn.png`
- `figures/local_waterfall_fp.png`
- `figures/local_waterfall_tn.png`
- `figures/top20_shap_interaction_heatmap.png`
- `figures/top20_feature_correlation_heatmap.png`

Key tables:

- `tables/global_shap_importance.csv`
- `tables/top20_shap_features.csv`
- `tables/prediction_types.csv`
- `tables/selected_local_cases.csv`
- `tables/local_explanation_tp.csv`
- `tables/local_explanation_fn.csv`
- `tables/local_explanation_fp.csv`
- `tables/local_explanation_tn.csv`
- `tables/age_shap_grouped.csv`
- `tables/spo2_shap_grouped.csv`
- `tables/ventilated_shap_grouped.csv`
- `tables/gcs_motor_shap_grouped.csv`
- `tables/zero_vital_summary.csv`
- `tables/top20_shap_interactions.csv`
- `tables/top20_feature_correlations.csv`

Root-level CSV files are retained for compatibility with earlier scripts.

## Prediction Type Counts

Using the final threshold `0.7274`, the held-out test set produced:

```text
TN = 15952
TP = 903
FP = 808
FN = 680
```

This matches the final modeling metrics and provides the case groups used for
local SHAP review.

## Feature-Level Interpretation

### Age

Age showed a clear monotonic pattern:

- `<=40`: mean SHAP `-0.8650`
- `41-50`: mean SHAP `-0.5693`
- `51-60`: mean SHAP `-0.3125`
- `61-70`: mean SHAP `0.1749`
- `71-80`: mean SHAP `0.3799`
- `80+`: mean SHAP `0.6177`

This indicates that the model treats older age as increasing predicted
mortality risk, while younger age decreases predicted risk.

### Minimum SpO2

Minimum oxygen saturation showed clinically plausible behavior:

- `<85`: mean SHAP `0.4482`
- `85-90`: mean SHAP `0.0293`
- `90-95`: mean SHAP `-0.0961`
- `95-100`: mean SHAP `-0.0929`

Low SpO2 therefore increases predicted mortality risk, while higher values are
generally risk-decreasing.

### Mechanical Ventilation

Mechanical ventilation showed strong separation:

- `ventilated_apache = 0`: mean SHAP `-0.2234`
- `ventilated_apache = 1`: mean SHAP `0.3916`

This is clinically plausible because ventilation often reflects severe
respiratory failure or critical illness.

### GCS Motor

Lower GCS motor scores increased predicted risk:

- `gcs_motor_apache = 1`: mean SHAP `0.5355`
- `gcs_motor_apache = 2`: mean SHAP `0.5907`
- `gcs_motor_apache = 3`: mean SHAP `0.5161`
- `gcs_motor_apache = 4`: mean SHAP `0.2821`
- `gcs_motor_apache = 5`: mean SHAP `0.1330`
- `gcs_motor_apache = 6`: mean SHAP `-0.0863`

This supports the interpretation that impaired motor response is a mortality
risk signal in the model.

## Local Case Review

The selected local cases were:

```text
TP row_position = 7773,  p = 0.9989
FN row_position = 7920,  p = 0.0192
FP row_position = 1963,  p = 0.9971
TN row_position = 14835, p = 0.0015
```

### True Positive

The selected true positive case had a very high predicted mortality probability.
The strongest risk-increasing features included:

- `d1_heartrate_min = 0`
- `d1_lactate_min = 8.7`
- `d1_spo2_min = 70`
- `apache_3j_diagnosis = 102.01`
- `gcs_motor_apache = 1`
- low arterial pH values
- mechanical ventilation
- low systolic blood pressure

This is clinically understandable as a high-risk case with severe physiological
instability.

### False Negative

The selected false negative had true mortality but a low predicted probability.
The strongest risk-decreasing features included:

- younger age (`28`)
- absence of mechanical ventilation
- low BUN values
- higher urine output
- near-zero pre-ICU length of stay

Some risk-increasing signals were present, including neurological body-system
category and glucose/calcium features, but they did not outweigh the
risk-decreasing evidence. This suggests the model can miss mortality cases when
classical high-risk signals are weak or offset by lower-risk features.

### False Positive

The selected false positive survived but received a high predicted probability.
Strong risk-increasing features included:

- `d1_heartrate_min = 0`
- high lactate
- very low SpO2
- low systolic and mean blood pressure
- low arterial pH
- mechanical ventilation

Although the prediction was incorrect, the high-risk estimate is clinically
understandable because the early physiological profile was severe.

### True Negative

The selected true negative had a very low predicted probability. Strong
risk-decreasing features included:

- younger age
- metabolic body-system/diagnosis categories
- absence of mechanical ventilation
- lower-risk respiratory and laboratory patterns
- higher urine output

This case is consistent with a low-risk model profile.

## Zero-Valued Vital Signs

Zero-valued vital signs remain caution-worthy even after `icu_id` removal:

| Feature | n_zero | zero_rate | death_rate_zero | death_rate_nonzero |
| --- | ---: | ---: | ---: | ---: |
| `d1_heartrate_min` | 129 | 0.0070 | 0.6512 | 0.0823 |
| `d1_resprate_min` | 767 | 0.0418 | 0.2034 | 0.0812 |
| `h1_resprate_min` | 147 | 0.0080 | 0.1293 | 0.0860 |

These values may represent true extreme clinical events, recording artifacts, or
data quality issues. They are retained as model inputs but should be interpreted
carefully in evidence packets and LLM explanations.

## Exploratory SHAP Interaction Analysis

An exploratory SHAP interaction analysis was performed on the top 20 globally
important SHAP features using a 300-patient sample.

Strongest pairwise interaction patterns included:

- `ventilated_apache` x `elective_surgery`
- `ventilated_apache` x `apache_3j_diagnosis`
- `age` x `d1_heartrate_max`
- `ventilated_apache` x `d1_bun_min`
- `apache_3j_diagnosis` x `gcs_motor_apache`
- `age` x `ventilated_apache`
- `ventilated_apache` x `gcs_motor_apache`

These are model-level interaction patterns, not causal clinical findings. They
suggest that the model combines respiratory support, diagnosis category, age,
neurological status, renal markers, and surgical context in non-additive ways.

Interaction evidence was not added to the LLM evidence packet. The LLM pipeline
intentionally uses patient-specific local SHAP main effects to keep explanations
concise, auditable, and easier to validate deterministically.

## Top-20 Feature Correlation Review

Spearman correlation was also computed for the top 20 SHAP features. Strong
correlations were found among related feature families:

- `d1_bun_max` x `d1_bun_min`
- GCS components with each other
- heart-rate minimum and maximum
- blood pressure-related features

Correlation is a data-level relationship, while SHAP interaction describes how
the model combines feature pairs. They are complementary but not equivalent.

## Variables Requiring Caution

Final model caution emphasis changed after preprocessing:

- `icu_id` and other ID/location columns are removed from the final model.
- Zero-valued vital signs remain caution-worthy.
- Negative or unusual time-related values such as `pre_icu_los_days` should be
  interpreted carefully if they appear in local evidence.
- Coded diagnosis/category features such as `apache_3j_diagnosis` are useful
  model signals, but they should be described as diagnosis category information,
  not decoded into specific diagnoses unless a reliable mapping is provided.

## Conclusion

The refreshed SHAP analysis supports the final model as clinically plausible at
the global and local levels. The dominant signals are age, ventilation,
diagnosis/severity categories, oxygenation, neurological response, renal/lab
markers, and vital signs. The earlier `icu_id` concern was addressed by the
final preprocessing design, while zero-valued vital signs remain explicit
caution features for evidence and LLM interpretation.
