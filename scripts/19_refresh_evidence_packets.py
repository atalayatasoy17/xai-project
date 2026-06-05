"""Refresh selected TP/FN/FP/TN evidence packets and prompts from final pipeline."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.prompts import build_explanation_prompt


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/training_v2.csv")
    _, raw_test = train_test_split(
        raw,
        test_size=0.2,
        random_state=42,
        stratify=raw["hospital_death"],
    )

    selected_cases = pd.read_csv(
        ROOT / "reports/02_explainability/tables/selected_local_cases.csv"
    )

    preprocessor = joblib.load(ROOT / "models/icu_preprocessor.pkl")
    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    evidence_dir = ROOT / "reports/03_evidence"
    prompt_dir = ROOT / "reports/04_llm_reasoning/prompts"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    evidence_packets = {}
    summary_rows = []

    for row in selected_cases.itertuples(index=False):
        case_type = str(row.case_type)
        row_position = int(row.row_position)
        raw_patient = raw_test.iloc[[row_position]]

        result = run_patient_pipeline(
            raw_patient=raw_patient,
            preprocessor=preprocessor,
            model=model,
            threshold=threshold,
            patient_label=case_type,
            test_row_index=int(raw_patient.index[0]),
            y_true=int(raw_patient["hospital_death"].iloc[0]),
            top_n=8,
        )

        evidence_packet = result["evidence_packet"]
        prompt = build_explanation_prompt(evidence_packet)
        evidence_packets[case_type] = evidence_packet

        with open(evidence_dir / f"{case_type.lower()}_evidence_packet.json", "w") as f:
            json.dump(evidence_packet, f, indent=2)

        with open(prompt_dir / f"{case_type.lower()}_prompt.txt", "w") as f:
            f.write(prompt)

        summary_rows.append(
            {
                "case_type": case_type,
                "row_position": row_position,
                "original_index": int(raw_patient.index[0]),
                "y_true": evidence_packet["prediction"]["y_true"],
                "y_pred": evidence_packet["prediction"]["y_pred"],
                "y_proba": evidence_packet["prediction"]["y_proba"],
                "threshold": evidence_packet["prediction"]["threshold"],
                "prediction_type": evidence_packet["prediction"]["prediction_type"],
            }
        )

    with open(evidence_dir / "evidence_packets.json", "w") as f:
        json.dump(evidence_packets, f, indent=2)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(evidence_dir / "selected_evidence_summary.csv", index=False)

    print("=== Evidence Packets Refreshed ===")
    print(f"Cases: {', '.join(evidence_packets.keys())}")
    print(f"Output: {evidence_dir.relative_to(ROOT)}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
