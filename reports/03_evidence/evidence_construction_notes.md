# Evidence Construction Notes

## Purpose

This stage converts raw SHAP outputs into structured evidence packets that can be used by the later reasoning layer.

The goal is to move from feature-level SHAP tables to a more interpretable patient-level evidence format.

## Inputs

- Final model prediction outputs from `prediction_types.csv`
- Global SHAP importance from `global_shap_importance.csv`
- Local SHAP explanations for selected TP, FN, FP, and TN cases
- Final decision threshold from the saved model threshold file

## What Was Built

For each selected case type (TP, FN, FP, TN), an evidence packet was created containing:

- patient label
- test row index
- true label
- predicted label
- predicted mortality probability
- prediction type
- top risk-increasing SHAP evidence
- top risk-decreasing SHAP evidence
- clinical meaning for known features
- caution flags for variables requiring careful interpretation

## Why This Matters

Raw SHAP values explain which features push a prediction up or down, but they are not yet easy to use for clinical reasoning or narrative explanation.

The evidence packet makes the explanation more structured by connecting each important feature to:

- its observed value
- its SHAP contribution
- its direction of effect
- a short clinical interpretation
- any caution flags

This creates a bridge between the SHAP layer and the future LLM reasoning layer.

## Example

For the selected TP case, severe hypoxemia, mechanical ventilation, low blood pressure, impaired GCS motor response, low bicarbonate, and low platelet count were captured as risk-increasing evidence.

The feature `d1_heartrate_min = 0` was also captured as risk-increasing evidence, but a caution flag was added because zero-valued vital signs may represent either an extreme clinical event or a data quality issue.

## Outputs

- `evidence_packets.json`
- `evidence_packet_tp.json`
- `evidence_packet_fn.json`
- `evidence_packet_fp.json`
- `evidence_packet_tn.json`
