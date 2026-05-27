# Explanation Evaluation Notes

## Purpose

This stage defines the evaluation framework for future LLM-generated explanations before generating or inspecting those explanations.

The goal is to reduce confirmation bias by deciding in advance what makes an explanation good, rather than designing criteria after seeing LLM outputs.

## Evaluation Dimensions

The evaluation rubric contains five dimensions:

1. Faithfulness / No Hallucination
2. Completeness
3. Clinical Plausibility
4. Caution Awareness
5. Clarity

Each dimension is scored from 1 to 5.

## How the Metrics Were Defined

These evaluation dimensions are not classical mathematical ML metrics like AUROC or F1. They are rubric-based explanation quality criteria designed for this project.

The dimensions were selected based on the main risks of LLM-generated clinical explanations:

- unsupported or hallucinated claims
- incomplete coverage of important model evidence
- clinically implausible interpretation
- failure to mention caution flags
- unclear or poorly structured explanation

Because the project goal is not only to generate fluent explanations but to generate faithful and clinically grounded explanations, the highest weights were assigned to faithfulness/no hallucination and clinical plausibility.

Caution awareness was also given a relatively high weight because the SHAP and local error analysis identified variables such as `icu_id`, zero-valued vital signs, and negative `pre_icu_los_days` that could mislead explanation if handled incorrectly.

## Dimension Meanings

### Faithfulness / No Hallucination

The explanation should use only the provided evidence, avoid unsupported claims, and correctly reflect SHAP directions.

### Completeness

The explanation should cover the main risk-increasing evidence, risk-decreasing evidence, caution-relevant evidence, and overall prediction logic.

### Clinical Plausibility

The explanation should interpret the evidence in a clinically reasonable way.

### Caution Awareness

The explanation should correctly mention and handle caution flags such as zero-valued vital signs, non-clinical identifiers, or other potentially problematic variables.

### Clarity

The explanation should be structured, concise, and easy to understand.

## Overall Explanation Quality Score

Each dimension is scored from 1 to 5. The overall explanation quality score is computed as a weighted average:

```text
Overall Score =
0.30 x Faithfulness / No Hallucination
+ 0.25 x Clinical Plausibility
+ 0.20 x Caution Awareness
+ 0.15 x Completeness
+ 0.10 x Clarity
```

The weights prioritize faithfulness and clinical plausibility because an explanation that is fluent but unsupported or clinically misleading should not receive a high score.

Caution awareness is weighted higher than completeness because mishandling caution flags such as `icu_id`, zero-valued vital signs, or negative timing variables can create misleading clinical interpretations.

## Reference Explanation

The manually written TP explanation is treated as a reference-style explanation. It defines the expected structure and quality level for future LLM-generated explanations.

The high scores assigned to this reference explanation reflect its role as a gold/reference example, not an independent evaluation of model-generated text.

## Outputs

- `explanation_evaluation_rubric.csv`
- `evaluation_weights.json`
- `tp_reference_evaluation.csv`
- `tp_reference_explanation.txt`
- `tp_reference_overall_score.json`
