
from src.rag.evaluation.analytics_judge_prompt import (
    build_analytics_judge_prompt,
)
from src.rag.evaluation.analytics_evaluator import (
    ManagerAnalyticsEvaluator,
)


def test_judge_prompt_includes_top_brands_support():
    case = {
        "case_id": "a004",
        "case_type": "brand_policy",
        "question": "برندها؟",
        "policy_expectations": [
            "brand_coverage_caveat"
        ],
    }

    result = {
        "facts": {},
        "context": {
            "scope": {
                "filters": {
                    "Category2": "دفتر"
                }
            },
            "data_quality": {
                "brand_usable_coverage": 0.44,
            },
            "top_brands": [
                {
                    "Brand": "برند الف",
                    "product_count": 10,
                }
            ],
            "top_products_by_rating": [],
            "top_products_by_rating_count": [],
            "category_comparison": [],
        },
        "answer": "نمونه",
        "insights": [],
        "caveats": [],
        "confidence": "medium",
        "numeric_faithfulness_valid": True,
    }

    prompt = build_analytics_judge_prompt(
        case=case,
        generated_result=result,
    )

    assert "top_brands" in prompt
    assert "برند الف" in prompt
    assert '"product_count": 10' in prompt


def test_compact_result_preserves_supporting_tables():
    result = {
        "context": {
            "scope": {
                "filters": {}
            },
            "data_quality": {
                "x": 1
            },
            "top_brands": [
                {
                    "Brand": "الف"
                }
            ],
            "top_products_by_rating": [
                {
                    "id": 1
                }
            ],
            "top_products_by_rating_count": [
                {
                    "id": 2
                }
            ],
            "category_comparison": [
                {
                    "Category2": "دفتر"
                }
            ],
        },
        "facts": {},
        "telemetry": {},
    }

    compact = (
        ManagerAnalyticsEvaluator
        ._compact_result(
            result
        )
    )

    assert compact[
        "context"
    ][
        "top_brands"
    ][0][
        "Brand"
    ] == "الف"

    assert compact[
        "context"
    ][
        "top_products_by_rating"
    ][0][
        "id"
    ] == 1

    assert compact[
        "context"
    ][
        "top_products_by_rating_count"
    ][0][
        "id"
    ] == 2

    assert compact[
        "context"
    ][
        "category_comparison"
    ][0][
        "Category2"
    ] == "دفتر"
