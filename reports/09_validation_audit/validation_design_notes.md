# Validation Audit Notes

This stage adds a deterministic quality-control layer for LLM-generated ICU mortality explanations. The goal is not to replace clinical review, but to make the explanation pipeline safer and more auditable before outputs are treated as final.

## Why This Step Was Added

Earlier LLM outputs showed that an explanation can look fluent while still containing small but important problems, such as unsupported wording, missing caution notes, or probability inconsistencies. Because these checks are rule-based and evidence-dependent, they should be validated deterministically instead of relying only on another LLM judge.

The validation layer therefore checks whether each explanation is grounded in the evidence packet produced by the model and SHAP pipeline.

## What The Validator Checks

The validator is implemented in `src/validation.py` and returns a structured validation report with pass/fail flags, dimension scores, revision feedback, and an overall deterministic score.

It checks:

- forbidden or unsupported wording, such as `stable`, `adequate`, `favorable`, or `abnormal`
- true-label leakage, such as saying the prediction was correct or referring to the actual outcome
- required section structure: prediction summary, risk-increasing factors, risk-decreasing factors, caution notes, and overall interpretation
- prediction probability consistency with the evidence packet, using a tolerance of `0.01`
- missing caution mentions for flagged features such as `icu_id`; the flagged feature or an approved clinical alias must be discussed in the `Caution notes` section with cautious language
- feature grounding by checking that explicitly named technical features exist in the evidence packet
- direction consistency by checking that risk-increasing and risk-decreasing sections match SHAP direction

## Scoring Logic

The deterministic score uses the rubric dimensions that can be checked reliably without subjective clinical judgment:

- faithfulness / no hallucination
- caution awareness
- completeness

Clinical plausibility and clarity are intentionally not scored here because they require human or LLM-judge interpretation. This keeps the production-style validator conservative and reproducible.

The weighted score is normalized over the deterministic dimensions only. A score of `5.0` means the explanation passed all deterministic checks.

## Fixture Verification

The script `scripts/13_verify_validation.py` validates the validator itself with small controlled examples.

Examples:

- a good explanation passes with score `5.0`
- an invented feature such as `potassium_apache` is flagged as ungrounded
- a direction error such as putting `age` in the risk-increasing section when it belongs to risk-decreasing is flagged
- true-label leakage such as `correct prediction` or `patient died` is flagged
- a missing `Caution notes` section triggers revision
- an incorrect probability such as `0.199` when the evidence probability is `0.9919` is flagged
- a missing caution mention for a flagged feature such as `icu_id` triggers revision
- a valid clinical alias such as `ICU unit identifier` is accepted as a caution mention for `icu_id`

This gives confidence that the validator catches the intended failure modes before being applied to saved LLM outputs.

## Saved Explanation Audit

The script `scripts/14_audit_saved_explanations.py` applies the validator to saved explanation files in:

- `reports/07_pipeline_demo`
- `reports/08_unlabeled_demo`

For each explanation, it loads the matching `*_evidence.json`, validates the explanation, and writes the audit table to:

`reports/09_validation_audit/validation_audit_summary.csv`

## Alias-Aware Caution Matching

An audit of the stricter exact-match caution validator revealed a false-positive issue. Some explanations correctly warned that the ICU unit identifier should be interpreted cautiously, but they used clinical language such as `ICU unit identifier` instead of the literal column name `icu_id`. The exact-match validator marked these explanations as missing caution even though the caution was clinically present.

To fix this, caution matching was made alias-aware for the small set of caution-flagged features. The validator now accepts either the exact feature name or an approved clinical alias, but only inside the `Caution notes` section and only when caution language is present in the same sentence.

This keeps the check deterministic while avoiding unnecessary revisions that would force technical column names into otherwise readable explanations.

Examples of accepted caution identity terms:

- `icu_id`: `icu_id`, `ICU unit identifier`, `unit identifier`, `unit-level`
- `d1_heartrate_min`: `d1_heartrate_min`, `minimum heart rate`, `heart rate`
- `pre_icu_los_days`: `pre_icu_los_days`, `pre-ICU length of stay`, `length of stay`

The feature identity term and the caution language are kept as separate requirements. For example, `ICU unit identifier` identifies the feature, while wording such as `interpreted cautiously`, `non-clinical`, or `unit-level` provides the caution.

## Audit Results

The audit contains 10 saved explanations after alias-aware caution matching. The pattern is clinically and methodologically useful:

- initial explanations sometimes fail deterministic validation
- revised explanations pass after the revision loop
- valid clinical caution phrasing is no longer rejected only because the exact column name is absent
- remaining failures are caused by unsupported wording, not missing caution mentions

Examples:

- `test_patient_0_llm_explanation` failed because it used unsupported wording such as `stable` and `adequate`, and did not state the probability consistently. The revised version passed with score `5.0`.
- `unlabeled_patient_0_llm_explanation` failed because it used unsupported wording such as `favorable`. Its caution wording for the ICU unit identifier is no longer treated as missing after alias-aware matching.
- `unlabeled_patient_7_llm_explanation` failed because it used unsupported wording such as `abnormal`. The revised version passed with score `5.0`.
- `unlabeled_patient_3_llm_explanation` and `unlabeled_patient_15_llm_explanation` now pass directly because their caution language is clinically valid even when it uses aliases instead of exact column names.

The audit also confirmed an important limitation of the exact-match v1 checks. Across the saved real explanations, `ungrounded_features` and `direction_errors` were empty. This should not be interpreted as proof that no natural-language grounding or direction errors are possible. Instead, it shows that the v1 checks mainly evaluate explicit technical feature names. When the LLM uses clinical phrases instead of exact column names, such as "oxygen saturation" instead of `d1_spo2_min`, those phrases are outside the scope of the current exact-match validator.

This finding motivated the alias-aware caution matcher described above. Feature grounding and direction consistency remain exact-match v1 checks because their exact-match limitation is passive: they may miss paraphrased issues, but they do not actively trigger unnecessary revision. Caution matching was prioritized because its exact-match false positives could incorrectly reject otherwise valid explanations.

## Interpretation

This stage demonstrates an agentic review loop:

1. The LLM generates an evidence-based explanation.
2. The deterministic validator checks the output against the evidence packet.
3. If validation fails, structured feedback is sent into the revision step.
4. The revised explanation is validated again.

In the current audit, all revised explanations pass the deterministic validator. Some initial explanations also pass directly after alias-aware caution matching. This supports the use of the validation layer as a practical safety and quality-control step in the explanation pipeline.

## Current Limitations

The validator is intentionally conservative. Feature grounding and direction checks currently work best when the explanation uses exact feature names, such as `d1_spo2_min` or `icu_id`. If an LLM paraphrases a feature as "oxygen saturation" instead of using the technical feature name, this version may not fully evaluate grounding or direction for that phrase.

Alias-aware matching is currently applied only to caution mentions for a small set of caution-flagged features. Broader alias-aware matching for feature grounding and direction consistency can be added later if needed. For now, the prompt encourages exact feature names when predefined clinical meaning is unavailable, which keeps the validation process more reliable.
