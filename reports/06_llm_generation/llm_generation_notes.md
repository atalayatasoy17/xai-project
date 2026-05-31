# LLM Generation and Agentic Review Notes

This notebook builds the LLM-based explanation generation and review stage of the project. The goal was not only to generate natural-language explanations, but also to test whether those explanations stay faithful to the structured SHAP evidence.

## Purpose

The earlier notebooks produced:

- model predictions,
- local SHAP explanations,
- structured evidence packets,
- prompt templates,
- and an explanation evaluation rubric.

This stage connects those pieces into an LLM workflow:

1. A generator LLM produces a clinical explanation from the structured evidence.
2. An evaluator LLM scores the explanation using the predefined rubric.
3. Rule-based checks are used as guardrails for specific failure modes.
4. Revision loops are tested to see whether explanation quality improves.

The main motivation is that LLM explanations can sound clinically convincing even when they add unsupported details. Therefore, the project treats explanation generation as a controlled and evaluated process, not as a single free-form LLM response.

## Models Used

Two separate LLM roles were used:

- Generator: `gpt-4.1-mini`
- Evaluator: `gpt-4o`

The generator creates explanations. The evaluator reviews those explanations using the project rubric. Separating these roles makes the workflow closer to an agentic review setup: one model writes, another model critiques.

## Initial TP Pilot Case

The TP case was used as the pilot case because it had a clear high-risk prediction and strong SHAP evidence. This allowed the first version of the generator and evaluator prompts to be tested in detail.

The first generated explanation was clinically readable, but it showed important risks:

- it referenced the true outcome,
- it added unsupported measurement units,
- and the evaluator initially failed to penalize these problems strongly enough.

For example, the first generated explanation included language similar to:

```text
consistent with the true outcome
```

This is problematic because the explanation should justify the model prediction using only model evidence, not the true label. The true label can be stored as metadata for analysis, but it should not be used to explain why the model predicted mortality risk.

## Prompt Tightening

After the initial issue was observed, the generator and evaluator prompts were made stricter.

The generator was instructed to:

- use only the provided evidence,
- not mention or use the true label,
- not say whether the prediction was correct or incorrect,
- not add measurement units unless present in the evidence,
- not invent diagnoses or clinical mechanisms,
- preserve risk-increasing and risk-decreasing SHAP directions,
- and mention caution flags when present.

The evaluator was instructed to:

- penalize unsupported clinical claims,
- penalize hallucinated facts,
- penalize invented units,
- penalize true-label leakage,
- penalize incorrect SHAP direction interpretation,
- and require revision when faithfulness was low.

This step improved the behavior of the generated explanations and made the evaluation stricter.

## TP Revision Loop

The TP case was then passed through an agentic revision loop:

1. Initial generated explanation
2. Evaluator review
3. First revised explanation
4. Re-evaluation
5. Stricter second revision
6. Final evaluation and rule-based check

The weighted scores improved across the revision process:

| Version | Weighted Score |
|---|---:|
| Initial generated explanation | 3.8 |
| First revision | 4.0 |
| Strict second revision | 4.7 |

This suggests that the agentic revision process improved explanation quality, especially in caution awareness, completeness, and clinical plausibility.

However, an important limitation was also observed. The final evaluator response still marked the strict second revision as needing revision, claiming that the explanation used the true label. A rule-based forbidden phrase check found no true-label leakage:

```text
Forbidden true-label phrases found: []
```

This suggests a likely evaluator false positive. This is an important methodological finding: LLM evaluators can also make mistakes and should not be treated as perfect judges.

## Rule-Based Guardrail

A simple rule-based check was added to search for forbidden true-label phrases, including:

- `true outcome`
- `true label`
- `actual outcome`
- `correct prediction`
- `incorrect prediction`
- `consistent with the true outcome`

This check does not replace the LLM evaluator, but it provides a transparent guardrail for a high-risk failure mode. In this notebook, it helped identify a possible mismatch between the evaluator's criticism and the actual generated text.

## FN, FP, and TN Cases

After the TP prompt refinement, the final stricter generator-evaluator setup was applied to the remaining case types:

- FN: false negative
- FP: false positive
- TN: true negative

These cases were not passed through the full TP-style revision loop. Instead, they were used to test whether the finalized setup worked across different prediction outcomes:

```text
generate explanation -> evaluate explanation -> compute weighted score -> run forbidden phrase check
```

The final scores were:

| Case | Weighted Score | Revision Needed | Forbidden True-Label Phrases |
|---|---:|---|---|
| TP | 4.7 | True | [] |
| FN | 3.8 | True | [] |
| FP | 3.8 | True | [] |
| TN | 4.0 | True | [] |

The TP case had the highest score because it received multiple revision rounds. FN, FP, and TN were generated using the final prompt setup but did not receive the same revision treatment.

## Observed Issues

Several recurring issues were identified:

1. Unsupported clinical phrasing

   The LLM sometimes used terms like "normal", "stable", "slightly elevated", or broader clinical interpretations that were not explicitly present in the evidence.

2. Over-interpretation of risk-decreasing features

   In one revision, the explanation inferred clinical meanings for risk-decreasing features that did not have explicit `clinical_meaning` fields. This was later reduced by stricter prompting.

3. Evaluator JSON formatting problems

   One evaluator response for the TN case was not valid JSON. A retry function was added that explicitly requested valid JSON only. The retry succeeded and the TN evaluation became parseable.

4. Evaluator false positive risk

   The evaluator claimed true-label leakage in the final TP explanation, but the rule-based forbidden phrase check did not find such leakage. This highlights the need to interpret LLM evaluation results carefully.

## Interpretation

The results support several conclusions:

- LLMs can translate structured SHAP evidence into readable clinical explanations.
- Prompt strictness matters; looser prompts can lead to true-label leakage or unsupported details.
- Agentic revision can improve explanation quality.
- Weighted rubric scores are useful for comparing explanation versions.
- Rule-based checks are helpful for specific safety and faithfulness constraints.
- LLM evaluators can make false-positive judgments, so human review remains important.

Overall, this notebook demonstrates a controlled LLM explanation pipeline rather than a one-shot explanation generator. The strongest setup combines:

```text
structured evidence + strict generation prompt + LLM evaluator + weighted scoring + rule-based checks + human interpretation
```

## Saved Outputs

The notebook saves:

- TP agentic revision summary,
- FN/FP/TN evaluation summary,
- final all-case summary,
- generated explanations,
- evaluator outputs,
- and JSON result objects.

The main saved files are:

- `reports/06_llm_generation/final_llm_generation_summary.csv`
- `reports/06_llm_generation/tp_agentic_summary.csv`
- `reports/06_llm_generation/fn_fp_tn_summary.csv`
- `reports/06_llm_generation/tp_agentic_results.json`
- `reports/06_llm_generation/fn_fp_tn_case_results.json`
- `reports/06_llm_generation/explanations/`
- `reports/06_llm_generation/evaluations/`

## Final Methodological Note

The most important finding from this stage is that explanation generation and explanation evaluation both require guardrails. The generator can hallucinate or over-interpret, but the evaluator can also make imperfect judgments. Therefore, the final explanation workflow should be treated as a decision-support and review pipeline, not as a fully autonomous clinical explanation authority.

## Later Label-Leakage Correction

A later project-level review identified an additional prompt-design issue: the early notebook prompts included `true label` and `case_type` as metadata. Even though the prompt instructed the LLM not to use the true label, a cleaner faithful-explanation design should avoid sending that information to the LLM at all.

The final Python pipeline therefore separates internal analysis metadata from LLM-visible evidence:

- `y_true` and `prediction_type` remain in the evidence packet for evaluation and error analysis.
- `true label`, `case_type`, and TP/FN/FP/TN information are removed from the generated LLM prompt.
- The LLM now receives only the predicted label, predicted probability, decision threshold, SHAP evidence, clinical meanings, and caution flags.

This correction makes the final pipeline more faithful to the project goal: explanations should be based on the model prediction and supporting evidence, not on the observed outcome.

## Current Status In Final Pipeline

This notebook should be interpreted as an exploratory LLM generation and review stage. It was useful for identifying the main risks of LLM explanation generation, including true-label leakage, unsupported clinical phrasing, evaluator false positives, and the need for revision.

The final production-style pipeline is now implemented in the Python modules and later reports:

- deterministic validation: `src/validation.py`
- LLM revision bridge: `src/llm.py`
- validator fixture tests: `scripts/13_verify_validation.py`
- saved explanation audit: `scripts/14_audit_saved_explanations.py`
- validation audit notes: `reports/09_validation_audit/validation_design_notes.md`
- GPT-4o subjective evaluation: `reports/10_gpt4o_evaluation/gpt4o_evaluation_notes.md`

In the final pipeline, GPT-4o is not used as the hard validator. Hard pass/fail decisions are made by the deterministic validator. GPT-4o is used only as an advisory evaluator for the subjective rubric dimensions: clinical plausibility and clarity.

Therefore, this notebook documents the development path and early experiments, while the final validation and evaluation methodology is documented in reports 09 and 10.
