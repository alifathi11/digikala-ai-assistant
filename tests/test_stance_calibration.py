import pandas as pd

from src.rag.evaluation.stance_calibration import (
    StanceProbabilityCalibrator,
)


def test_calibrator_can_fit_three_classes():
    rows = []

    for query_index in range(
        6
    ):
        for label, probs in [
            (
                "support",
                (
                    0.8,
                    0.15,
                    0.05,
                ),
            ),
            (
                "neutral",
                (
                    0.55,
                    0.40,
                    0.05,
                ),
            ),
            (
                "contradict",
                (
                    0.10,
                    0.15,
                    0.75,
                ),
            ),
        ]:
            rows.append(
                {
                    "query_id": (
                        f"q{query_index}"
                    ),
                    "label": label,
                    "support_prob": (
                        probs[0]
                    ),
                    "neutral_prob": (
                        probs[1]
                    ),
                    "contradict_prob": (
                        probs[2]
                    ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    calibrator = (
        StanceProbabilityCalibrator(
            c_values=(
                0.1,
                1.0,
            ),
            max_cv_splits=3,
        )
        .fit(
            frame
        )
    )

    output = calibrator.predict(
        frame
    )

    assert set(
        output[
            "calibrated_label"
        ]
    ) == {
        "support",
        "neutral",
        "contradict",
    }
