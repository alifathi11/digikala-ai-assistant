
import pandas as pd
import pytest

from src.rag.evaluation.analytics_dataset import (
    AnalyticsEvaluationDataset,
)
from src.rag.evaluation.analytics_metrics import (
    expected_fact_values,
    fact_value_accuracy,
    rendered_metric_accuracy,
    weighted_judge_score,
)


def _frame():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "Brand": "برند الف",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "Price": 100,
                "Rate": 80,
                "Rate_cnt": 10,
                "review_count": 2,
                "review_rate_count": 2,
                "review_rate_sum": 8.0,
            },
            {
                "id": 2,
                "Brand": "متفرقه",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "Price": 200,
                "Rate": 0,
                "Rate_cnt": 0,
                "review_count": 0,
                "review_rate_count": 0,
                "review_rate_sum": 0.0,
            },
            {
                "id": 3,
                "Brand": "برند ب",
                "Category1": "دیجیتال",
                "Category2": "ماوس",
                "Price": 300,
                "Rate": 90,
                "Rate_cnt": 20,
                "review_count": 1,
                "review_rate_count": 1,
                "review_rate_sum": 5.0,
            },
        ]
    )


def test_expected_overview_metrics_are_independent():
    case = {
        "filters": {
            "Category2": "کرم"
        },
        "comparison_categories": [],
    }

    values = expected_fact_values(
        product_frame=_frame(),
        case=case,
        generic_brand_values=[
            "متفرقه",
        ],
        rating_max=100,
    )

    assert values[
        "overview.product_count"
    ] == 2

    assert values[
        "overview.brand_count"
    ] == 1

    assert values[
        "overview.median_price"
    ] == 150

    assert values[
        "overview.review_coverage_pct"
    ] == 50

    assert values[
        "overview.rated_product_coverage_pct"
    ] == 50

    assert values[
        "overview.weighted_product_rating_100"
    ] == 80


def test_expected_comparison_metrics_preserve_category_order():
    case = {
        "filters": {},
        "comparison_categories": [
            "ماوس",
            "کرم",
        ],
        "category_field": "Category2",
    }

    values = expected_fact_values(
        product_frame=_frame(),
        case=case,
        generic_brand_values=[
            "متفرقه",
        ],
        rating_max=100,
    )

    assert values[
        "comparison.c0.product_count"
    ] == 1

    assert values[
        "comparison.c1.product_count"
    ] == 2

    assert values[
        "comparison.c0.median_price"
    ] == 300

    assert values[
        "comparison.c1.median_price"
    ] == 150


def test_fact_value_accuracy_detects_mismatch():
    generated = {
        "overview.product_count": {
            "value": 3,
        },
        "overview.median_price": {
            "value": 150,
        },
    }

    expected = {
        "overview.product_count": 2,
        "overview.median_price": 150,
    }

    result = fact_value_accuracy(
        generated_facts=generated,
        expected_values=expected,
    )

    assert result[
        "accuracy"
    ] == pytest.approx(
        0.5
    )

    assert len(
        result[
            "errors"
        ]
    ) == 1


def test_rendered_metric_accuracy_checks_placeholder_output():
    result = {
        "answer_template": (
            "تعداد {{metric:overview.product_count}} است."
        ),
        "answer": (
            "تعداد 2 است."
        ),
        "insights": [],
        "facts": {
            "overview.product_count": {
                "display_value": "2",
            }
        },
    }

    assert (
        rendered_metric_accuracy(
            result
        )
        == 1.0
    )


def test_weighted_score_is_on_five_point_scale():
    score = weighted_judge_score(
        {
            "correctness": 5,
            "groundedness": 5,
            "caveat_compliance": 5,
            "completeness": 5,
            "relevance": 5,
            "managerial_usefulness": 5,
            "instruction_following": 5,
        }
    )

    assert score == 5.0


def test_dataset_rejects_duplicate_case_ids():
    cases = [
        {
            "case_id": "a1",
            "split": "dev",
            "case_type": "overview",
            "question": "x",
        },
        {
            "case_id": "a1",
            "split": "test",
            "case_type": "overview",
            "question": "y",
        },
    ]

    with pytest.raises(
        ValueError
    ):
        AnalyticsEvaluationDataset(
            cases
        )


def test_dataset_rejects_invalid_comparison_size():
    with pytest.raises(
        ValueError
    ):
        AnalyticsEvaluationDataset(
            [
                {
                    "case_id": "a1",
                    "split": "test",
                    "case_type": "comparison",
                    "question": "x",
                    "comparison_categories": [
                        "a",
                    ],
                }
            ]
        )
