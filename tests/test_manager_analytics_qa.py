
import pandas as pd

from src.rag.analytics import (
    AnalyticsRepository,
    AnalyticsService,
)
from src.rag.analytics.validation import (
    render_metric_template,
    validate_manager_response,
)
from src.rag.pipeline.analytics import (
    ManagerAnalyticsPipeline,
)


class FakeGenerator:

    def __init__(
        self,
        payload,
    ):
        self.payload = payload
        self.calls = 0

    def generate(
        self,
        system_prompt,
        user_prompt,
    ):
        self.calls += 1

        return {
            "payload": self.payload,
            "model": "fake",
            "latency_ms": 1.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": None,
        }


def _service():
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "title_fa": "محصول الف",
                "Brand": "برند الف",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "sub_category": "beauty",
                "Price": 100,
                "Rate": 80,
                "Rate_cnt": 10,
            },
            {
                "id": 2,
                "title_fa": "محصول ب",
                "Brand": "متفرقه",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "sub_category": "beauty",
                "Price": 200,
                "Rate": 0,
                "Rate_cnt": 0,
            },
        ]
    )

    reviews = pd.DataFrame(
        [
            {
                "product_id": 1,
                "review_count": 3,
                "review_rate_count": 3,
                "review_rate_sum": 12,
                "avg_review_rate": 4.0,
            }
        ]
    )

    repository = AnalyticsRepository(
        products=products,
        review_stats=reviews,
    )

    return AnalyticsService(
        repository=repository,
        product_rating_max=100,
        min_rating_count_for_leaders=1,
    )


def test_literal_number_is_rejected():
    payload = {
        "answer_template": (
            "این دسته 2 محصول دارد."
        ),
        "insights": [],
        "caveats": [],
        "confidence": "high",
    }

    valid, errors = (
        validate_manager_response(
            payload,
            {
                "overview.product_count"
            },
        )
    )

    assert not valid

    assert any(
        "literal_numeric_digit"
        in value
        for value in errors
    )


def test_metric_placeholder_is_rendered_deterministically():
    text = (
        "تعداد محصولات "
        "{{metric:overview.product_count}} است."
    )

    facts = {
        "overview.product_count": {
            "display_value": "2",
        }
    }

    assert (
        render_metric_template(
            text,
            facts,
        )
        == "تعداد محصولات 2 است."
    )


def test_pipeline_numeric_faithfulness():
    generator = FakeGenerator(
        {
            "answer_template": (
                "این محدوده "
                "{{metric:overview.product_count}} "
                "محصول دارد و میانه قیمت "
                "{{metric:overview.median_price}} است."
            ),
            "insights": [],
            "caveats": [],
            "confidence": "high",
        }
    )

    pipeline = ManagerAnalyticsPipeline(
        analytics_service=_service(),
        generator=generator,
        top_n=2,
    )

    result = pipeline.answer(
        question="وضعیت را خلاصه کن",
        filters={
            "Category2": "کرم"
        },
    )

    assert (
        result[
            "numeric_faithfulness_valid"
        ]
        is True
    )

    assert "2" in result[
        "answer"
    ]

    assert "150" in result[
        "answer"
    ]

    assert generator.calls == 1


def test_rating_count_leader_uses_rate_cnt():
    result = _service().top_products(
        filters={
            "Category2": "کرم"
        },
        sort_by="rating_count",
        top_n=2,
    )

    assert int(
        result.iloc[0][
            "id"
        ]
    ) == 1
