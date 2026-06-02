# Error Analysis and Sensitivity Notes

## Global SHAP Observations

- The global SHAP analysis identified clinically meaningful high-importance features such as `age`, `ventilated_apache`, `d1_spo2_min`, `gcs_motor_apache`, vital signs, and laboratory values.
- `icu_id` also appeared among the top global SHAP features. Since this is likely a unit/location identifier rather than a direct clinical measurement, it should not be interpreted as a patient-level clinical risk factor.

## Saved Explainability Outputs

The main visual and tabular outputs from `notebooks/04_shap_explainability.ipynb`
were archived under:

- `figures/`
- `tables/`

Key saved figures include:

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

Key saved tables include:

- `tables/global_shap_importance.csv`
- `tables/age_shap_grouped.csv`
- `tables/spo2_shap_grouped.csv`
- `tables/ventilated_shap_grouped.csv`
- `tables/gcs_motor_shap_grouped.csv`
- `tables/selected_local_cases.csv`
- `tables/local_explanation_tp.csv`
- `tables/local_explanation_fn.csv`
- `tables/local_explanation_fp.csv`
- `tables/local_explanation_tn.csv`
- `tables/zero_vital_summary.csv`
- `tables/group_top_shap_summary.csv`

These outputs separate reusable report artifacts from exploratory notebook
cells. The root-level CSV files are retained for backward compatibility with
earlier project stages, while the `figures/` and `tables/` folders provide a
cleaner archive for reporting.

## Top-20 SHAP Feature Review

- The top 20 features by mean absolute SHAP value were exported to `top20_shap_features.csv`.
- The highest-ranked features were `age`, `ventilated_apache`, `apache_3j_diagnosis`, `d1_spo2_min`, `gcs_motor_apache`, `d1_heartrate_min`, `d1_resprate_max`, and `icu_id`.
- Selected SHAP dependence plots were inspected to understand how clinically meaningful feature values were associated with model-level SHAP contributions.
- Features such as `age`, `d1_spo2_min`, `gcs_motor_apache`, and `ventilated_apache` showed clinically interpretable patterns. In contrast, coded or non-clinical variables such as `apache_3j_diagnosis` and `icu_id` require cautious interpretation.

## Exploratory SHAP Interaction Analysis

- An exploratory SHAP interaction analysis was performed on the top 20 globally important SHAP features using a 300-patient sample.
- The interaction matrix was saved as `top20_shap_interaction_matrix.csv`, and the ranked pairwise interactions were saved as `top20_shap_interactions.csv`.
- The strongest model-level interaction pairs included:
  - `ventilated_apache` x `apache_3j_diagnosis`
  - `age` x `ventilated_apache`
  - `ventilated_apache` x `d1_resprate_max`
  - `ventilated_apache` x `gcs_verbal_apache`
  - `age` x `d1_resprate_max`
- These patterns suggest that the model may combine respiratory support, diagnosis/severity category, age, respiratory status, and neurological response in non-additive ways.
- The strongest interaction, `ventilated_apache` x `apache_3j_diagnosis`, is clinically plausible as a model-level pattern because the meaning of mechanical ventilation may differ across diagnosis or severity categories. However, `apache_3j_diagnosis` is a coded diagnosis category and was not decoded into specific diagnoses in this project.
- The `age` x `ventilated_apache` interaction suggests that the model may combine age and ventilation status when estimating mortality risk, rather than treating them as fully independent signals.
- These interaction findings were interpreted as model-level patterns only, not causal clinical relationships.
- Interaction evidence was not added to the final LLM evidence packet. The LLM pipeline intentionally remains based on patient-specific local SHAP main effects to keep explanations concise, auditable, and easier to validate deterministically.

## Top-20 Feature Correlation Review

- A supplementary Spearman correlation heatmap was generated for the top 20 SHAP features and saved as `top20_feature_correlation_heatmap.png`. Ranked feature-pair correlations were also saved as `top20_feature_correlations.csv`.
- The strongest correlations were observed among clinically related measurement families:
  - `d1_bun_max` x `d1_bun_min`
  - `d1_sysbp_min` x `d1_mbp_min`
  - `gcs_motor_apache` x `gcs_verbal_apache`
  - `gcs_motor_apache` x `gcs_eyes_apache`
  - `gcs_verbal_apache` x `gcs_eyes_apache`
- This correlation analysis is a data-level analysis, not a model explanation. It complements the SHAP interaction heatmap by showing which high-importance features are naturally related in the input data.
- High feature correlation and high SHAP interaction are not the same. Correlation describes relationships between feature values, while SHAP interaction describes how the model combines feature pairs in its prediction function.

## Local True Positive Case

- The selected true positive case had true mortality (`y_true = 1`) and received an extremely high predicted mortality probability (`p = 0.9995`).
- Risk-increasing signals included severe hypoxemia (`d1_spo2_min = 31`), mechanical ventilation, very low systolic and mean blood pressure, low GCS motor score, low bicarbonate values, and low platelets.
- `d1_heartrate_min = 0` had the largest positive SHAP contribution. This value is physiologically extreme and may represent a true critical event, recording artifact, or data quality issue.
- In the test set, 129 patients had `d1_heartrate_min = 0`, corresponding to 0.7% of the test cohort. Their observed mortality rate was 65.1%, compared with 8.2% among patients with `d1_heartrate_min > 0`.

## False Negative Case

- The selected false negative case had true mortality (`y_true = 1`) but received a very low predicted mortality probability (`p = 0.0037`).
- Risk-increasing signals included older age, high maximum respiratory rate, low diastolic blood pressure, and diagnosis-related information.
- Risk-decreasing signals included absence of mechanical ventilation, normal WBC, relatively non-extreme heart rate values, low/normal BUN, and higher early mean blood pressure values.
- This suggests that the model may miss mortality cases when classical high-risk indicators are not strongly present.

## False Positive Case

- The selected false positive case survived (`y_true = 0`) but received a very high predicted mortality probability (`p = 0.9954`).
- Risk-increasing signals included `d1_heartrate_min = 0`, advanced age, low systolic blood pressure, low maximum SpO2, impaired GCS motor and verbal scores, low body temperature, low respiratory rate, and low bicarbonate values.
- Although the patient survived, the high-risk prediction is clinically understandable because several features suggested severe physiological instability.
- This case may represent a patient with strong early risk signals who survived due to clinical intervention, recovery, or transient instability.
- The case reinforces the need to interpret zero-valued vital signs cautiously, since both `d1_heartrate_min = 0` and `d1_resprate_min = 0` contributed to the high-risk prediction.

## True Negative Case

- The selected true negative case survived (`y_true = 0`) and received a very low predicted mortality probability (`p = 0.0002`).
- Risk-decreasing signals included younger age, absence of mechanical ventilation, stable mean blood pressure, non-extreme respiratory rate, normal minimum heart rate, and diagnosis-related information.
- The explanation also showed strong negative contributions from `icu_id` and `pre_icu_los_days`.
- `pre_icu_los_days` can contain negative values and should be reviewed as a potential data quality or interpretation issue.

## Local Waterfall Summary

- The local waterfall explanations generally supported the tabular SHAP explanations.
- The true positive case was driven by severe clinical signals, including hypoxemia, mechanical ventilation, hypotension, impaired GCS response, low bicarbonate values, and thrombocytopenia.
- The true negative case was driven by lower-risk signals, including younger age, absence of mechanical ventilation, stable vital signs, and less extreme clinical measurements.
- The false negative case suggests that the model may miss mortality cases when strong classical high-risk indicators are absent or outweighed by survival-associated features.
- The false positive case was clinically understandable despite being incorrect, because the patient had multiple severe early risk signals but ultimately survived.
- Across local waterfall explanations, `icu_id`, zero-valued vital signs, and negative `pre_icu_los_days` repeatedly appeared as variables requiring cautious interpretation.

## Group-Level SHAP Patterns

- Group-level SHAP analysis compared average feature contributions across TP, FN, FP, and TN groups.
- TP cases were characterized by strong positive SHAP contributions from clinically severe signals such as mechanical ventilation, impaired GCS motor response, diagnosis information, low SpO2, heart rate abnormalities, low systolic blood pressure, and older age.
- FN cases had risk-increasing signals, but these were weaker on average than in TP cases. This suggests that missed mortality cases may have less obvious or less extreme high-risk patterns.
- FP cases showed risk-increasing patterns similar to TP cases, including ventilation, older age, impaired GCS, low SpO2, low blood pressure, BUN, and heart rate features. This suggests that many false positives may be clinically understandable high-risk survivors rather than arbitrary model errors.
- TN cases had very small positive SHAP values, indicating that the model did not detect strong mortality-increasing signals in correctly predicted survivors.
- Mean SHAP values were used to understand direction of effect, while mean absolute SHAP values were used to understand feature influence regardless of direction.

## Variables Requiring Caution

- `icu_id` appears in local explanations and should be interpreted cautiously because it is likely a unit/location identifier rather than a direct clinical measurement.
- Zero-valued vital signs such as `d1_heartrate_min = 0` and `h1_resprate_min = 0` may represent true extreme clinical events, recording artifacts, or data quality issues.
- Negative values in `pre_icu_los_days` should be interpreted carefully and may require additional preprocessing review.

## Future Sensitivity Analyses

- Compare model performance with and without `icu_id`.
- Evaluate model behavior after treating physiologically implausible zero vital signs as missing or outlier values.
- Review and potentially transform or flag negative `pre_icu_los_days` values.
- Check whether false negative cases share common hidden patterns not captured by the current feature set.
