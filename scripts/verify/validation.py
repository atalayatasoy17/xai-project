"""Verify deterministic explanation validation with curated fixtures."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.validation import validate_explanation


FIXTURE_PACKET = {
    "prediction": {
        "y_pred": 1,
        "y_proba": 0.9919,
        "threshold": 0.7274,
    },
    "risk_increasing_evidence": [
        {
            "feature": "d1_spo2_min",
            "caution_flags": [],
        },
        {
            "feature": "d1_heartrate_min",
            "caution_flags": [
                "Zero-valued vital sign; may reflect extreme clinical event or recording artifact.",
            ],
        },
    ],
    "risk_decreasing_evidence": [
        {
            "feature": "age",
            "caution_flags": [],
        },
    ],
}


GOOD_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
The d1_heartrate_min feature should be interpreted cautiously because a zero-valued vital sign may reflect a recording artifact.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


UNGROUNDED_FEATURE_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature potassium_apache increased risk. The feature d1_spo2_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
The d1_heartrate_min feature should be interpreted cautiously because a zero-valued vital sign may reflect a recording artifact.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


SIGN_FLIP_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature age increased risk.

3. Main risk-decreasing factors
The feature d1_spo2_min decreased risk.

4. Caution notes
The d1_heartrate_min feature should be interpreted cautiously because a zero-valued vital sign may reflect a recording artifact.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


TRUE_LABEL_LEAK_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99 and this was a correct prediction.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
The d1_heartrate_min feature should be interpreted cautiously because a zero-valued vital sign may reflect a recording artifact.

5. Overall interpretation
The patient died, so the model prediction was correct.
""".strip()


MISSING_SECTION_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


WRONG_PROBABILITY_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.199.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
The d1_heartrate_min feature should be interpreted cautiously because a zero-valued vital sign may reflect a recording artifact.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


MISSING_CAUTION_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
No caution notes were identified.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()

ALIAS_CAUTION_EXPLANATION = """
1. Prediction summary
The model predicted a mortality probability of 0.99.

2. Main risk-increasing factors
The feature d1_spo2_min increased risk. The feature d1_heartrate_min increased risk.

3. Main risk-decreasing factors
The feature age decreased risk.

4. Caution notes
The zero-valued minimum heart rate should be interpreted cautiously because it may reflect a recording artifact.

5. Overall interpretation
The explanation is based on the provided evidence.
""".strip()


def main() -> None:
    fixtures = {
        "good": GOOD_EXPLANATION,
        "ungrounded_feature": UNGROUNDED_FEATURE_EXPLANATION,
        "sign_flip": SIGN_FLIP_EXPLANATION,
        "true_label_leak": TRUE_LABEL_LEAK_EXPLANATION,
        "missing_section": MISSING_SECTION_EXPLANATION,
        "wrong_probability": WRONG_PROBABILITY_EXPLANATION,
        "missing_caution": MISSING_CAUTION_EXPLANATION,
        "alias_caution": ALIAS_CAUTION_EXPLANATION,
    }

    print("=== Validation Fixture Verification ===")

    for name, explanation in fixtures.items():
        report = validate_explanation(explanation, FIXTURE_PACKET)

        print()
        print(f"Fixture: {name}")
        print(f"  passed: {report['passed']}")
        print(f"  revision_required: {report['revision_required']}")
        print(f"  score: {report['deterministic_validation_score']}")
        print(f"  forbidden: {report['checks']['forbidden_phrases']['found']}")
        print(f"  leakage: {report['checks']['true_label_leakage']['found']}")
        print(f"  missing_sections: {report['checks']['section_structure']['missing']}")
        print(f"  prediction_passed: {report['checks']['prediction_consistency']['passed']}")
        print(f"  missing_caution: {report['checks']['caution_mentions']['missing_features']}")
        print(f"  ungrounded: {report['checks']['feature_grounding']['ungrounded_features']}")
        print(f"  direction_errors: {report['checks']['direction_consistency']['direction_errors']}")


if __name__ == "__main__":
    main()
