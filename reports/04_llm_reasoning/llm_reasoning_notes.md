# LLM Reasoning Notes

## Purpose

This stage prepares structured evidence packets for the future LLM reasoning layer. The goal is to generate clinically grounded patient-level explanations from model predictions and SHAP-based evidence.

## Inputs

- Structured evidence packets from `reports/03_evidence/evidence_packets.json`
- TP, FN, FP, and TN case-level evidence
- Risk-increasing and risk-decreasing SHAP evidence
- Clinical meaning fields and caution flags from the evidence construction layer

## What Was Built

This notebook creates prompt templates for selected TP, FN, FP, and TN cases. Each prompt includes:

- patient prediction summary
- predicted label as model output
- predicted mortality probability
- risk-increasing evidence
- risk-decreasing evidence
- caution flags for potentially problematic variables
- required explanation structure

The prompt explicitly instructs the LLM to use only the provided evidence and avoid inventing clinical facts.

## Label-Leakage Correction

A later review identified that sending `true label` or `case_type` to the LLM is not necessary for explanation generation and can create label leakage risk. The evidence packets may still retain `y_true` and TP/FN/FP/TN metadata for internal analysis, but the LLM prompt should only expose model outputs and SHAP-based evidence.

The saved prompt files in this stage were therefore updated to remove:

- `True label`
- `Case type`
- TP/FN/FP/TN metadata from the prompt text

The final prompt design keeps only the predicted label, predicted probability, SHAP evidence, clinical meanings, and caution flags.

## Manual Explanation Prototype

A manual explanation was written for the selected TP case to define the expected style and structure of future LLM-generated explanations. The explanation follows this structure:

1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation

## Why This Matters

The LLM reasoning layer should not reason directly from raw model features or raw SHAP tables. Instead, it should use structured evidence packets that already contain SHAP direction, feature values, clinical meanings, and caution flags.

This design reduces hallucination risk and makes the generated explanation more faithful to the model evidence.

## Outputs

- `tp_prompt.txt`
- `fn_prompt.txt`
- `fp_prompt.txt`
- `tn_prompt.txt`
- `tp_manual_explanation.txt`
