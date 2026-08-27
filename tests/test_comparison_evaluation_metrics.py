import math

import pandas as pd

from src.rag.evaluation.comparison_metrics import (
    assessment_product_coverage,
    citation_ownership_rate,
    deterministic_winner_accuracy,
    expected_metadata_winner,
    no_winner_accuracy,
    weighted_judge_score,
)


def test_expected_lower_price_winner():
    metadata = pd.DataFrame(
        [
            {
                "id": 1,
                "Price": 200.0,
            },
            {
                "id": 2,
                "Price": 100.0,
            },
        ]
    )

    expectation = expected_metadata_winner(
        metadata,
        "lower_price",
    )

    assert expectation[
        "available"
    ]
    assert expectation[
        "expected_product_id"
    ] == 2


def test_deterministic_winner_accuracy():
    result = {
        "overall_winner_product_id": 2,
    }

    expectation = {
        "available": True,
        "expected_product_id": 2,
    }

    assert deterministic_winner_accuracy(
        result,
        expectation,
    ) == 1.0


def test_no_winner_accuracy():
    result = {
        "overall_winner_product_id": None,
    }

    assert no_winner_accuracy(
        result,
        expect_no_winner=True,
    ) == 1.0


def test_citation_ownership_rate():
    result = {
        "retrieved_review_ids_by_product": {
            10: [101, 102],
            20: [201],
        },
        "evidence_ids_by_product": {
            10: [101],
            20: [201],
        },
    }

    assert citation_ownership_rate(
        result
    ) == 1.0


def test_assessment_product_coverage():
    result = {
        "product_ids": [10, 20],
        "criteria": [
            {
                "assessments": [
                    {"product_id": 10},
                    {"product_id": 20},
                ]
            },
            {
                "assessments": [
                    {"product_id": 10},
                    {"product_id": 20},
                ]
            },
        ],
    }

    assert assessment_product_coverage(
        result
    ) == 1.0


def test_weighted_judge_score():
    payload = {
        name: {
            "score": 5,
            "reason": "ok",
        }
        for name in [
            "correctness",
            "groundedness",
            "criterion_coverage",
            "conflict_handling",
            "recommendation_calibration",
            "relevance",
            "instruction_following",
            "safety",
        ]
    }

    assert math.isclose(
        weighted_judge_score(
            payload
        ),
        5.0,
    )
