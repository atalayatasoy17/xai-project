# EDA Notes

This folder archives the exploratory data analysis outputs from `notebooks/01_eda.ipynb`.
The goal is to keep the main EDA evidence outside the notebook so that figures,
tables, and conclusions can be reused in the final report or presentation.

## Dataset Overview

- Dataset: WiDS Datathon 2020 ICU mortality data.
- Rows: 91,713 patients.
- Columns: 186 raw columns.
- Target: `hospital_death`.
- Positive class rate: 8.63%.

The target distribution confirms strong class imbalance:

```text
Survived (0): 83,798 patients, 91.37%
Died (1)   :  7,915 patients,  8.63%
```

This motivated the use of stratified splitting and evaluation metrics beyond
accuracy, including AUROC, AUPRC, recall, precision, F1, and confusion matrix
inspection.

Relevant outputs:

- `figures/target_distribution.png`
- `tables/target_distribution.csv`
- `tables/dataset_overview.csv`

## Missingness

Missing data was a major property of the dataset:

```text
Columns with >10% missingness: 103
Columns with >25% missingness: 74
Columns with >50% missingness: 74
Columns with >70% missingness: 55
Columns with >90% missingness: 6
```

This supported the preprocessing decision to drop very high-missingness
features and add missingness indicators for retained variables. Missingness was
not treated as random noise only; in ICU data, whether a lab was ordered or
recorded can itself reflect clinical workflow or patient severity.

The missingness-by-target analysis showed that several key labs were less
missing in mortality cases. For example:

```text
d1_lactate_max      missing survived: 77.0%, missing died: 49.2%
d1_arterial_ph_min  missing survived: 68.4%, missing died: 35.1%
```

This suggests that missingness patterns may carry useful information and
supports the use of missingness indicators.

Relevant outputs:

- `figures/missingness_top20.png`
- `figures/missingness_by_target.png`
- `tables/missingness_summary.csv`
- `tables/missingness_threshold_summary.csv`
- `tables/missingness_by_target.csv`

## Target-Associated Features

The strongest simple numeric correlations with `hospital_death` included
lactate, GCS components, blood pressure, arterial pH, and ventilation-related
information.

Top examples:

```text
d1_lactate_min                 corr =  0.404
d1_lactate_max                 corr =  0.399
h1_lactate_min                 corr =  0.344
h1_lactate_max                 corr =  0.341
gcs_motor_apache               corr = -0.282
gcs_eyes_apache                corr = -0.260
gcs_verbal_apache              corr = -0.241
ventilated_apache              corr =  0.229
```

The direction of these correlations is clinically plausible: higher lactate,
mechanical ventilation, lower GCS, lower blood pressure, and lower pH are
associated with higher mortality risk. These are simple data-level associations,
not causal effects.

Relevant outputs:

- `figures/top_target_correlations.png`
- `figures/feature_distributions_by_target.png`
- `tables/top_target_correlations.csv`
- `tables/key_numeric_feature_summary.csv`

## APACHE Leakage Investigation

Two APACHE probability columns were identified as leakage-prone:

```text
apache_4a_hospital_death_prob
apache_4a_icu_death_prob
```

These columns already encode mortality probability estimates. Their means were
substantially higher in the mortality group:

```text
apache_4a_hospital_death_prob: survived mean = 0.063, died mean = 0.335
apache_4a_icu_death_prob     : survived mean = 0.025, died mean = 0.243
```

Because these columns act like precomputed outcome-risk scores, they were
removed during preprocessing to avoid leakage and preserve a fair prediction
task.

Relevant outputs:

- `figures/apache_leakage_by_target.png`
- `tables/apache_leakage_basic_stats.csv`
- `tables/apache_leakage_by_target.csv`

## Categorical Feature Patterns

Categorical EDA showed different mortality rates across admission source,
ICU type, and APACHE body system groups. Among categories with at least 100
patients, high-mortality groups included:

```text
hospital_admit_source = Step-Down Unit (SDU): 18.8%
apache_3j_bodysystem = Sepsis              : 15.8%
hospital_admit_source = Other ICU          : 15.0%
icu_type = MICU                            : 12.1%
```

These results supported the inclusion of categorical variables after careful
encoding, while also motivating caution around location or unit-level variables
such as `icu_id`.

Relevant outputs:

- `figures/categorical_mortality_top_groups.png`
- `tables/categorical_features_summary.csv`
- `tables/categorical_mortality_summary.csv`

## Outlier Review

Outliers were inspected for selected numeric features. Several extreme values
were present, especially in ICU-relevant labs and vitals:

```text
temp_apache        IQR outlier rate: 10.00%
creatinine_apache  IQR outlier rate: 10.33%
bun_apache         IQR outlier rate:  7.08%
glucose_apache     IQR outlier rate:  4.80%
```

These were not automatically removed. In ICU data, extreme values may represent
true severe illness rather than invalid records. Later explainability steps
therefore treated physiologically unusual or non-clinical signals cautiously
rather than deleting them by default.

Relevant outputs:

- `tables/outlier_summary.csv`
- `tables/duplicate_summary.csv`

## Feature Group Review

EDA grouped raw columns into broad families: identifiers, demographics, APACHE
features, day-1 measurements, hour-1 measurements, ICU information, and selected
vital/lab signals. This helped organize preprocessing and later interpretation.

Relevant outputs:

- `figures/feature_groups_summary.png`
- `tables/feature_groups_summary.csv`

## Modeling Implications

The EDA directly informed later project decisions:

- Use stratified train/test splitting because `hospital_death` is imbalanced.
- Avoid relying on accuracy alone; include AUROC, AUPRC, recall, precision, F1,
  and confusion matrix.
- Remove APACHE precomputed death probability columns to prevent leakage.
- Preserve informative missingness through missingness indicators.
- Treat extreme ICU values carefully rather than blindly removing outliers.
- Interpret unit/location variables such as `icu_id` cautiously.

## Scope Note

EDA findings are descriptive and associational. They were used to guide
preprocessing, modeling, and explainability design, but they do not establish
causal relationships.
