import math

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "correctness": 0.20,
    "groundedness": 0.20,
    "criterion_coverage": 0.15,
    "conflict_handling": 0.10,
    "recommendation_calibration": 0.15,
    "relevance": 0.10,
    "instruction_following": 0.05,
    "safety": 0.05,
}


WINNER_RULE_COLUMNS = {
    "lower_price": (
        "Price",
        "min",
    ),
    "higher_price": (
        "Price",
        "max",
    ),
    "higher_rating": (
        "Rate",
        "max",
    ),
    "higher_rating_count": (
        "Rate_cnt",
        "max",
    ),
    "lower_min_price_last_month": (
        "min_price_last_month",
        "min",
    ),
}


def validate_weights(
    weights,
):
    total = float(
        sum(
            weights.values()
        )
    )

    if not np.isclose(
        total,
        1.0,
    ):
        raise ValueError(
            "Comparison judge weights must sum to 1.0."
        )


def weighted_judge_score(
    judge_payload,
    weights=None,
):
    weights = dict(
        weights
        or DEFAULT_WEIGHTS
    )

    validate_weights(
        weights
    )

    return float(
        sum(
            float(
                weights[
                    dimension
                ]
            )
            * float(
                judge_payload[
                    dimension
                ][
                    "score"
                ]
            )
            for dimension in weights
        )
    )


def citation_ownership_rate(
    result,
):
    retrieved = result.get(
        "retrieved_review_ids_by_product",
        {},
    )

    cited = result.get(
        "evidence_ids_by_product",
        {},
    )

    total = 0
    valid = 0

    for raw_product_id, evidence_ids in cited.items():
        product_id = int(
            raw_product_id
        )

        allowed = {
            int(value)
            for value
            in retrieved.get(
                product_id,
                retrieved.get(
                    str(
                        product_id
                    ),
                    [],
                ),
            )
        }

        for value in evidence_ids:
            total += 1

            if int(
                value
            ) in allowed:
                valid += 1

    if total == 0:
        return 1.0

    return float(
        valid
        / total
    )


def assessment_product_coverage(
    result,
):
    selected = {
        int(value)
        for value
        in result.get(
            "product_ids",
            [],
        )
    }

    criteria = result.get(
        "criteria",
        [],
    )

    if not criteria:
        return 0.0

    covered = 0

    for criterion in criteria:
        assessed = set()

        for assessment in criterion.get(
            "assessments",
            [],
        ):
            try:
                assessed.add(
                    int(
                        assessment.get(
                            "product_id"
                        )
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        if assessed == selected:
            covered += 1

    return float(
        covered
        / len(
            criteria
        )
    )


def expected_metadata_winner(
    product_metadata,
    winner_rule,
):
    winner_rule = str(
        winner_rule
        or ""
    ).strip()

    if not winner_rule:
        return {
            "available": False,
            "expected_product_id": None,
            "column": None,
            "rule": None,
        }

    if winner_rule not in (
        WINNER_RULE_COLUMNS
    ):
        raise ValueError(
            "Unknown deterministic winner rule: "
            f"{winner_rule}"
        )

    column, direction = (
        WINNER_RULE_COLUMNS[
            winner_rule
        ]
    )

    if column not in (
        product_metadata.columns
    ):
        return {
            "available": False,
            "expected_product_id": None,
            "column": column,
            "rule": winner_rule,
        }

    frame = product_metadata[
        [
            "id",
            column,
        ]
    ].copy()

    frame[
        column
    ] = pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )

    frame = frame[
        frame[
            column
        ].notna()
    ].copy()

    if len(
        frame
    ) < 2:
        return {
            "available": False,
            "expected_product_id": None,
            "column": column,
            "rule": winner_rule,
        }

    target = (
        frame[
            column
        ].min()
        if direction
        == "min"
        else frame[
            column
        ].max()
    )

    winners = frame[
        np.isclose(
            frame[
                column
            ].astype(float),
            float(
                target
            ),
            equal_nan=False,
        )
    ]

    if len(
        winners
    ) != 1:
        return {
            "available": True,
            "expected_product_id": None,
            "column": column,
            "rule": winner_rule,
        }

    return {
        "available": True,
        "expected_product_id": int(
            winners.iloc[
                0
            ][
                "id"
            ]
        ),
        "column": column,
        "rule": winner_rule,
    }


def deterministic_winner_accuracy(
    result,
    winner_expectation,
):
    if not winner_expectation.get(
        "available",
        False,
    ):
        return math.nan

    expected = winner_expectation.get(
        "expected_product_id"
    )

    predicted = result.get(
        "overall_winner_product_id"
    )

    if predicted is not None:
        try:
            predicted = int(
                predicted
            )
        except (
            TypeError,
            ValueError,
        ):
            predicted = None

    if expected is None:
        return float(
            predicted is None
        )

    return float(
        predicted
        == int(
            expected
        )
    )


def no_winner_accuracy(
    result,
    expect_no_winner=False,
):
    if not bool(
        expect_no_winner
    ):
        return math.nan

    return float(
        result.get(
            "overall_winner_product_id"
        )
        is None
    )


def summarize_comparison_results(
    frame,
):
    metric_columns = [
        column
        for column in [
            "overall_score",
            "correctness",
            "groundedness",
            "criterion_coverage",
            "conflict_handling",
            "recommendation_calibration",
            "relevance",
            "instruction_following",
            "safety",
            "citation_validity",
            "citation_ownership_rate",
            "assessment_product_coverage",
            "deterministic_winner_accuracy",
            "no_winner_accuracy",
        ]
        if column in frame.columns
    ]

    overall = {}

    for column in metric_columns:
        values = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

        if values.notna().any():
            overall[
                column
            ] = float(
                values.mean()
            )

    latency = {}

    for column in [
        "comparison_total_latency_ms",
        "judge_latency_ms",
        "end_to_end_latency_ms",
        "comparison_total_tokens",
        "judge_total_tokens",
        "end_to_end_tokens",
    ]:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

        if not values.notna().any():
            continue

        latency[
            f"mean_{column}"
        ] = float(
            values.mean()
        )

        latency[
            f"p95_{column}"
        ] = float(
            values.quantile(
                0.95
            )
        )

    by_split = (
        frame.groupby(
            "split"
        )[
            metric_columns
        ]
        .mean(
            numeric_only=True
        )
        .reset_index()
        if metric_columns
        else pd.DataFrame()
    )

    by_type = (
        frame.groupby(
            "case_type"
        )[
            metric_columns
        ]
        .mean(
            numeric_only=True
        )
        .reset_index()
        if metric_columns
        else pd.DataFrame()
    )

    return {
        "overall": overall,
        "telemetry": latency,
        "by_split": by_split,
        "by_type": by_type,
    }
