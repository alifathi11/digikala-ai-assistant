
import pandas as pd
import pytest

from src.rag.analytics import (
    AnalyticsDataAuditor,
    AnalyticsRepository,
    AnalyticsService,
)


def _products():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title_fa": "محصول ۱",
                "Brand": "برند الف",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "sub_category": "آبرسان",
                "Price": 100,
                "min_price_last_month": None,
                "Rate": 4.8,
                "Rate_cnt": 50,
            },
            {
                "id": 2,
                "title_fa": "محصول ۲",
                "Brand": "برند ب",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "sub_category": "آبرسان",
                "Price": 200,
                "min_price_last_month": None,
                "Rate": 4.6,
                "Rate_cnt": 5,
            },
            {
                "id": 3,
                "title_fa": "محصول ۳",
                "Brand": "متفرقه",
                "Category1": "زیبایی",
                "Category2": "کرم",
                "sub_category": "آبرسان",
                "Price": 300,
                "min_price_last_month": None,
                "Rate": 5.0,
                "Rate_cnt": 1,
            },
            {
                "id": 4,
                "title_fa": "محصول ۴",
                "Brand": "برند الف",
                "Category1": "دیجیتال",
                "Category2": "ماوس",
                "sub_category": "بی سیم",
                "Price": 400,
                "min_price_last_month": 350,
                "Rate": 4.0,
                "Rate_cnt": 20,
            },
        ]
    )


def _review_stats():
    return pd.DataFrame(
        [
            {
                "product_id": 1,
                "review_count": 10,
                "review_rate_count": 10,
                "review_rate_sum": 45,
                "avg_review_rate": 4.5,
            },
            {
                "product_id": 3,
                "review_count": 5,
                "review_rate_count": 5,
                "review_rate_sum": 15,
                "avg_review_rate": 3.0,
            },
            {
                "product_id": 4,
                "review_count": 2,
                "review_rate_count": 2,
                "review_rate_sum": 8,
                "avg_review_rate": 4.0,
            },
        ]
    )


def _service():
    repository = AnalyticsRepository(
        products=_products(),
        review_stats=_review_stats(),
    )

    return AnalyticsService(
        repository=repository,
        generic_brand_values=[
            "متفرقه",
            "Unknown",
        ],
        min_rating_count_for_leaders=10,
    )


def test_repository_rejects_duplicate_product_ids():
    products = _products()

    duplicated = pd.concat(
        [
            products,
            products.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError
    ):
        AnalyticsRepository(
            duplicated,
            review_stats=(
                _review_stats()
            ),
        )


def test_category_overview_counts_products_once():
    overview = _service().overview(
        {
            "Category2": "کرم"
        }
    )

    assert overview[
        "product_count"
    ] == 3

    assert overview[
        "review_count"
    ] == 15

    assert overview[
        "products_with_reviews"
    ] == 2

    assert overview[
        "review_coverage"
    ] == pytest.approx(
        2 / 3
    )

    assert overview[
        "price"
    ][
        "median"
    ] == 200


def test_top_brands_excludes_generic_brand():
    brands = _service().top_brands(
        {
            "Category2": "کرم"
        },
        top_n=10,
    )

    assert "متفرقه" not in set(
        brands["Brand"]
    )

    assert set(
        brands["Brand"]
    ) == {
        "برند الف",
        "برند ب",
    }


def test_rating_leader_has_minimum_rating_count_guard():
    leaders = _service().top_products(
        {
            "Category2": "کرم"
        },
        sort_by="rating",
        top_n=10,
    )

    assert 3 not in set(
        leaders["id"]
    )

    assert int(
        leaders.iloc[0]["id"]
    ) == 1


def test_category_comparison_is_deterministic():
    comparison = _service().compare_categories(
        [
            "کرم",
            "ماوس",
        ],
        category_field="Category2",
    )

    cream = comparison[
        comparison[
            "Category2"
        ]
        == "کرم"
    ].iloc[0]

    mouse = comparison[
        comparison[
            "Category2"
        ]
        == "ماوس"
    ].iloc[0]

    assert int(
        cream[
            "product_count"
        ]
    ) == 3

    assert int(
        mouse[
            "product_count"
        ]
    ) == 1

    assert float(
        cream[
            "median_price"
        ]
    ) == 200.0



def test_product_rating_excludes_unrated_zero_rows():
    overview = _service().overview()

    # All four synthetic products have Rate_cnt > 0, so all are rated.
    assert (
        overview[
            "product_rating"
        ][
            "rated_product_count"
        ]
        == 4
    )

    assert (
        overview[
            "product_rating"
        ][
            "scale_max"
        ]
        == 100.0
    )


def test_zero_rate_with_zero_count_is_unrated():
    products = _products().copy()

    products.loc[
        products["id"]
        == 2,
        "Rate",
    ] = 0

    products.loc[
        products["id"]
        == 2,
        "Rate_cnt",
    ] = 0

    repository = AnalyticsRepository(
        products=products,
        review_stats=_review_stats(),
    )

    service = AnalyticsService(
        repository=repository,
        product_rating_max=100,
    )

    overview = service.overview()

    assert (
        overview[
            "product_rating"
        ][
            "rated_product_count"
        ]
        == 3
    )

    assert (
        overview[
            "product_rating"
        ][
            "mean"
        ]
        > 4.0
    )



def test_audit_marks_sparse_historical_price_unavailable():
    repository = AnalyticsRepository(
        products=_products(),
        review_stats=_review_stats(),
    )

    audit = AnalyticsDataAuditor(
        repository=repository,
        ready_coverage=0.70,
        limited_coverage=0.30,
        product_rating_max=100,
        generic_brand_values=[
            "متفرقه",
            "Unknown",
        ],
    ).run()

    readiness = (
        audit.metric_readiness
        .set_index(
            "metric"
        )
    )

    assert (
        readiness.loc[
            "historical_price_statistics",
            "status",
        ]
        == "unavailable"
    )

    assert (
        audit.summary[
            "duplicate_product_ids"
        ]
        == 0
    )
