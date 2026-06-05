# Validation Audit Notes

## Purpose

This stage adds a deterministic quality-control layer for LLM-generated ICU
mortality explanations. The goal is not to replace clinical review, but to make
the explanation pipeline safer, reproducible, and auditable.

The validator checks whether an explanation stays grounded in the evidence
packet produced by the model and SHAP pipeline.

## Validator Checks

The validator is implemented in:

```text
src/validation.py
```

It checks:

- forbidden or unsupported wording, such as `stable`, `adequate`, `favorable`,
  or `abnormal`
- true-label leakage, such as saying the prediction was correct or referring to
  the actual outcome
- required section structure
- prediction probability consistency with tolerance `0.01`
- caution mention coverage for flagged features
- explicit feature grounding for technical feature names
- SHAP direction consistency for explicitly named technical features

## Deterministic Score

The deterministic score covers only dimensions that can be checked without a
subjective judge:

- faithfulness / no hallucination
- caution awareness
- completeness

Clinical plausibility and clarity are intentionally evaluated separately by the
optional GPT-4o subjective evaluator.

## Fixture Verification

`scripts/13_verify_validation.py` verifies the validator with controlled
fixtures:

- good explanation
- ungrounded feature
- sign flip
- true-label leakage
- missing section
- wrong probability
- missing caution
- alias-based caution pass

This makes the validator itself testable before it is applied to generated
outputs.

## Saved Explanation Audit

`scripts/14_audit_saved_explanations.py` audits saved explanations in:

- `reports/07_pipeline_demo/`
- `reports/08_unlabeled_demo/`

The refreshed final audit contains **7 current explanations**:

| Explanation | Result | Score |
| --- | --- | ---: |
| `test_patient_0_llm_explanation` | failed, revision required | 4.538 |
| `test_patient_0_llm_revised_explanation` | passed | 5.000 |
| `unlabeled_patient_0_llm_explanation` | passed | 5.000 |
| `unlabeled_patient_15_llm_explanation` | passed | 5.000 |
| `unlabeled_patient_3_llm_explanation` | passed | 5.000 |
| `unlabeled_patient_7_llm_explanation` | failed, revision required | 4.538 |
| `unlabeled_patient_7_llm_revised_explanation` | passed | 5.000 |

The audit is saved to:

```text
reports/09_validation_audit/validation_audit_summary.csv
```

## Stale Revision Handling

During the final model refresh, a useful engineering issue was found: if a case
previously required revision but later passed directly under a new model/output,
old `*_revised_*` files could remain in the report directory and pollute the
audit.

The LLM demo scripts were updated so that when no revision is needed, stale
revised files for that patient are removed. This keeps the audit aligned with
the current generated outputs.

## Caution Matching

Alias-aware caution matching remains in the validator for caution-flagged
features. This was originally added because exact matching could incorrectly
reject clinically valid caution phrasing.

In the refreshed final model, `icu_id` is no longer a model feature because
ID/location columns are removed during preprocessing. The alias-aware caution
logic is still kept for robustness and for other caution-flagged features such
as zero-valued vital signs or unusual time-related values.

## Current Limitations

Feature grounding and direction consistency are still exact-match checks. They
work best when explanations explicitly mention technical feature names. If the
LLM uses only clinical paraphrases, this validator may not fully evaluate those
phrases.

This limitation is documented rather than hidden. The deterministic validator is
used as a conservative production-style gate; the optional GPT-4o evaluator is
used only for subjective dimensions.

## Conclusion

The final validation layer demonstrates that LLM explanations are not accepted
blindly. Generated explanations must pass deterministic checks, and failed
outputs are revised and validated again. The refreshed audit shows that all
final accepted explanations pass with deterministic score `5.0`.
