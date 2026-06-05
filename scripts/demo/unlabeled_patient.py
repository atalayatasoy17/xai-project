"""Run the saved-artifact pipeline on one unlabeled patient."""

import argparse
from pathlib import Path
import json
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.pipeline import run_patient_pipeline
from src.prediction import load_model, load_threshold
from src.prompts import build_explanation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the saved-artifact pipeline on one unlabeled patient.",
    )
    parser.add_argument(
        "--patient-position",
        type=int,
        default=0,
        help="Zero-based row position in data/raw/unlabeled.csv.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the prediction without writing report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "reports/08_unlabeled_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_unlabeled = pd.read_csv(ROOT / "data/raw/unlabeled.csv")

    preprocessor = joblib.load(ROOT / "models/icu_preprocessor.pkl")
    model = load_model(ROOT / "models/lgbm_tuned_clean.pkl")
    threshold = load_threshold(ROOT / "models/lgbm_tuned_clean_threshold.json")

    patient_position = args.patient_position
    if patient_position < 0 or patient_position >= len(raw_unlabeled):
        raise ValueError(
            f"patient_position must be between 0 and {len(raw_unlabeled) - 1}."
        )

    patient_label = f"unlabeled_patient_{patient_position}"
    raw_patient = raw_unlabeled.iloc[[patient_position]]

    result = run_patient_pipeline(
        raw_patient=raw_patient,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        patient_label=patient_label,
        test_row_index=int(raw_patient.index[0]),
        y_true=None,
        top_n=8,
    )

    prompt = build_explanation_prompt(result["evidence_packet"])

    prediction_path = output_dir / f"{patient_label}_prediction.json"
    evidence_path = output_dir / f"{patient_label}_evidence.json"
    prompt_path = output_dir / f"{patient_label}_prompt.txt"

    if not args.no_save:
        with open(prediction_path, "w") as f:
            json.dump(result["prediction"], f, indent=2)

        with open(evidence_path, "w") as f:
            json.dump(result["evidence_packet"], f, indent=2)

        with open(prompt_path, "w") as f:
            f.write(prompt)

    title = "=== Unlabeled Patient Demo ===" if args.no_save else "=== Unlabeled Patient Demo Saved ==="
    print(title)
    print(f"Patient label   : {patient_label}")
    print(f"Original index  : {int(raw_patient.index[0])}")
    if not args.no_save:
        print(f"Prediction      : {prediction_path.relative_to(ROOT)}")
        print(f"Evidence packet : {evidence_path.relative_to(ROOT)}")
        print(f"Prompt          : {prompt_path.relative_to(ROOT)}")
    print()
    print("=== Prediction ===")
    print(json.dumps(result["prediction"], indent=2))


if __name__ == "__main__":
    main()
