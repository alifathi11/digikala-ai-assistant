
import math
import re

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "correctness": 0.25,
    "groundedness": 0.20,
    "caveat_compliance": 0.20,
    "completeness": 0.15,
    "relevance": 0.10,
    "managerial_usefulness": 0.05,
    "instruction_following": 0.05,
}


PLACEHOLDER_RE = re.compile(
    r"\{\{metric:([A-Za-z0-9_.-]+)\}\}"
)


def validate_weights(
    weights,
):
    missing = (
        set(
            DEFAULT_WEIGHTS
        )
        - set(
            weights
        )
    )

    if missing:
        raise ValueError(
            "Missing analytics judge weights: "
            f"{sorted(missing)}"
        )

    total = sum(
        float(
            weights[
                key
            ]
        )
        for key
        in DEFAULT_WEIGHTS
    )

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Analytics judge weights must sum to 1.0."
        )


def weighted_judge_score(
    scores,
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
                scores[
                    key
                ]
            )
            * float(
                weights[
                    key
                ]
            )
            for key
            in DEFAULT_WEIGHTS
        )
    )


def _normalized_text(
    series,
):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def _filter_frame(
    frame,
    filters,
):
    result = frame

    for field, value in (
        filters
        or {}
    ).items():
        values = (
            list(
                value
            )
            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            )
            else [
                value
            ]
        )

        normalized = {
            str(
                item
            )
            .strip()
            .casefold()
            for item
            in values
        }

        result = result[
            _normalized_text(
                result[
                    field
                ]
            ).isin(
                normalized
            )
        ]

    return result.copy()


def _overview_expected(
    frame,
    generic_brand_values,
    rating_max=100.0,
):
    product_count = int(
        len(frame)
    )

    brands = (
        frame.get(
            "Brand",
            pd.Series(
                "",
                index=frame.index,
            ),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    normalized_brands = (
        brands
        .str.casefold()
    )

    generic = {
        str(
            value
        )
        .strip()
        .casefold()
        for value
        in generic_brand_values
    }

    meaningful = (
        (
            normalized_brands
            != ""
        )
        & (
            ~normalized_brands
            .isin(
                generic
            )
        )
    )

    brand_count = int(
        normalized_brands[
            meaningful
        ].nunique()
    )

    prices = pd.to_numeric(
        frame.get(
            "Price",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    valid_prices = (
        prices.dropna()
    )

    review_counts = (
        pd.to_numeric(
            frame.get(
                "review_count",
                pd.Series(
                    0,
                    index=frame.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
    )

    products_with_reviews = int(
        (
            review_counts
            > 0
        ).sum()
    )

    rate = pd.to_numeric(
        frame.get(
            "Rate",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    rate_count = (
        pd.to_numeric(
            frame.get(
                "Rate_cnt",
                pd.Series(
                    0,
                    index=frame.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
    )

    rated_mask = (
        rate.between(
            0,
            float(
                rating_max
            ),
            inclusive="both",
        )
        & (
            rate_count
            > 0
        )
    )

    weighted_rating = None

    if rated_mask.any():
        denominator = float(
            rate_count[
                rated_mask
            ].sum()
        )

        if denominator > 0:
            weighted_rating = float(
                (
                    rate[
                        rated_mask
                    ]
                    * rate_count[
                        rated_mask
                    ]
                ).sum()
                / denominator
            )

    review_rate_count = (
        pd.to_numeric(
            frame.get(
                "review_rate_count",
                pd.Series(
                    0,
                    index=frame.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
    )

    review_rate_sum = (
        pd.to_numeric(
            frame.get(
                "review_rate_sum",
                pd.Series(
                    0.0,
                    index=frame.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    valid_review_ratings = float(
        review_rate_count.sum()
    )

    weighted_review_rating = (
        float(
            review_rate_sum.sum()
            / valid_review_ratings
        )
        if valid_review_ratings
        > 0
        else None
    )

    return {
        "product_count": (
            product_count
        ),
        "brand_count": (
            brand_count
        ),
        "median_price": (
            float(
                valid_prices.median()
            )
            if len(
                valid_prices
            )
            else None
        ),
        "price_p25": (
            float(
                valid_prices.quantile(
                    0.25
                )
            )
            if len(
                valid_prices
            )
            else None
        ),
        "price_p75": (
            float(
                valid_prices.quantile(
                    0.75
                )
            )
            if len(
                valid_prices
            )
            else None
        ),
        "price_coverage_pct": (
            float(
                len(
                    valid_prices
                )
                / product_count
                * 100
            )
            if product_count
            else 0.0
        ),
        "review_coverage_pct": (
            float(
                products_with_reviews
                / product_count
                * 100
            )
            if product_count
            else 0.0
        ),
        "rated_product_coverage_pct": (
            float(
                rated_mask.sum()
                / product_count
                * 100
            )
            if product_count
            else 0.0
        ),
        "weighted_product_rating_100": (
            weighted_rating
        ),
        "weighted_product_rating_5": (
            (
                weighted_rating
                / (
                    float(
                        rating_max
                    )
                    / 5.0
                )
            )
            if weighted_rating
            is not None
            else None
        ),
        "rating_count_total": int(
            rate_count.sum()
        ),
        "weighted_review_rating_5": (
            weighted_review_rating
        ),
    }


def expected_fact_values(
    product_frame,
    case,
    generic_brand_values,
    rating_max=100.0,
):
    result = {}

    filters = case.get(
        "filters"
    ) or {}

    scope = _filter_frame(
        product_frame,
        filters,
    )

    overview = _overview_expected(
        scope,
        generic_brand_values=(
            generic_brand_values
        ),
        rating_max=rating_max,
    )

    for suffix, value in (
        overview.items()
    ):
        result[
            f"overview.{suffix}"
        ] = value

    categories = (
        case.get(
            "comparison_categories"
        )
        or []
    )

    category_field = case.get(
        "category_field",
        "Category2",
    )

    for index, category in enumerate(
        categories
    ):
        category_scope = _filter_frame(
            product_frame,
            {
                category_field: (
                    category
                )
            },
        )

        values = _overview_expected(
            category_scope,
            generic_brand_values=(
                generic_brand_values
            ),
            rating_max=rating_max,
        )

        prefix = (
            f"comparison.c{index}"
        )

        for suffix in (
            "product_count",
            "median_price",
            "review_coverage_pct",
            "weighted_product_rating_100",
            "rating_count_total",
        ):
            result[
                f"{prefix}.{suffix}"
            ] = values[
                suffix
            ]

    return result


def _numeric_close(
    actual,
    expected,
    rel_tol=1e-8,
    abs_tol=1e-6,
):
    if (
        actual is None
        or expected is None
    ):
        return (
            actual is None
            and expected is None
        )

    try:
        actual_float = float(
            actual
        )

        expected_float = float(
            expected
        )
    except (
        TypeError,
        ValueError,
    ):
        return actual == expected

    return math.isclose(
        actual_float,
        expected_float,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def fact_value_accuracy(
    generated_facts,
    expected_values,
):
    correct = 0
    checked = 0
    errors = []

    for key, expected in (
        expected_values.items()
    ):
        if key not in (
            generated_facts
        ):
            errors.append(
                f"missing_fact:{key}"
            )
            checked += 1
            continue

        checked += 1

        actual = (
            generated_facts[
                key
            ].get(
                "value"
            )
        )

        if _numeric_close(
            actual,
            expected,
        ):
            correct += 1
        else:
            errors.append(
                f"fact_mismatch:{key}:"
                f"expected={expected}:actual={actual}"
            )

    return {
        "accuracy": (
            float(
                correct
                / checked
            )
            if checked
            else 1.0
        ),
        "correct": int(
            correct
        ),
        "checked": int(
            checked
        ),
        "errors": errors,
    }


def scope_product_count_accuracy(
    generated_facts,
    expected_values,
):
    key = (
        "overview.product_count"
    )

    if key not in (
        expected_values
    ):
        return 1.0

    if key not in (
        generated_facts
    ):
        return 0.0

    return float(
        _numeric_close(
            generated_facts[
                key
            ].get(
                "value"
            ),
            expected_values[
                key
            ],
        )
    )


def comparison_fact_accuracy(
    generated_facts,
    expected_values,
):
    comparison_keys = [
        key
        for key in (
            expected_values
        )
        if key.startswith(
            "comparison."
        )
    ]

    if not comparison_keys:
        return None

    subset = {
        key: expected_values[
            key
        ]
        for key
        in comparison_keys
    }

    return fact_value_accuracy(
        generated_facts=(
            generated_facts
        ),
        expected_values=subset,
    )[
        "accuracy"
    ]


def rendered_metric_accuracy(
    result,
):
    facts = result.get(
        "facts"
    ) or {}

    checks = []

    answer_template = str(
        result.get(
            "answer_template",
            "",
        )
    )

    answer = str(
        result.get(
            "answer",
            "",
        )
    )

    for key in (
        PLACEHOLDER_RE.findall(
            answer_template
        )
    ):
        fact = facts.get(
            key
        )

        checks.append(
            bool(
                fact
                and str(
                    fact.get(
                        "display_value"
                    )
                )
                in answer
            )
        )

    for insight in (
        result.get(
            "insights"
        )
        or []
    ):
        template = str(
            insight.get(
                "text_template",
                "",
            )
        )

        rendered = str(
            insight.get(
                "text",
                "",
            )
        )

        for key in (
            PLACEHOLDER_RE.findall(
                template
            )
        ):
            fact = facts.get(
                key
            )

            checks.append(
                bool(
                    fact
                    and str(
                        fact.get(
                            "display_value"
                        )
                    )
                    in rendered
                )
            )

    if not checks:
        return 1.0

    return float(
        sum(
            checks
        )
        / len(
            checks
        )
    )


def policy_guard_configuration(
    result,
    policy_expectations,
):
    data_quality = (
        result.get(
            "context",
            {}
        ).get(
            "data_quality",
            {}
        )
    )

    checks = []

    for policy in (
        policy_expectations
        or []
    ):
        if policy == (
            "historical_price_refusal"
        ):
            checks.append(
                data_quality.get(
                    "historical_price_enabled"
                )
                is False
            )

        elif policy == (
            "review_volume_guard"
        ):
            checks.append(
                data_quality.get(
                    "review_volume_ranking_enabled"
                )
                is False
            )

        elif policy == (
            "brand_coverage_caveat"
        ):
            coverage = float(
                data_quality.get(
                    "brand_usable_coverage",
                    1.0,
                )
            )

            checks.append(
                coverage
                < 0.70
            )

        elif policy == (
            "rating_coverage_caveat"
        ):
            coverage = float(
                data_quality.get(
                    "product_rating_coverage",
                    1.0,
                )
            )

            checks.append(
                coverage
                < 0.70
            )

        elif policy == (
            "zero_review_awareness"
        ):
            facts = result.get(
                "facts",
                {}
            )

            value = (
                facts.get(
                    "overview.review_coverage_pct",
                    {}
                ).get(
                    "value"
                )
            )

            checks.append(
                value is not None
                and float(
                    value
                )
                == 0.0
            )

    if not checks:
        return None

    return float(
        sum(
            checks
        )
        / len(
            checks
        )
    )


def summarize_analytics_results(
    frame,
):
    if len(
        frame
    ) == 0:
        return pd.DataFrame()

    rows = []

    for split, group in (
        frame.groupby(
            "split",
            sort=False,
        )
    ):
        row = {
            "split": split,
            "cases": int(
                len(
                    group
                )
            ),
            "successful_cases": int(
                (
                    group[
                        "status"
                    ]
                    == "ok"
                ).sum()
            ),
        }

        numeric_columns = [
            "overall_judge_score",
            "numeric_faithfulness",
            "fact_value_accuracy",
            "scope_product_count_accuracy",
            "comparison_fact_accuracy",
            "rendered_metric_accuracy",
            "policy_guard_configuration",
            "correctness",
            "groundedness",
            "caveat_compliance",
            "completeness",
            "relevance",
            "managerial_usefulness",
            "instruction_following",
            "answer_latency_ms",
            "judge_latency_ms",
            "evaluation_latency_ms",
            "answer_total_tokens",
            "judge_total_tokens",
        ]

        for column in (
            numeric_columns
        ):
            if column not in (
                group.columns
            ):
                continue

            values = pd.to_numeric(
                group[
                    column
                ],
                errors="coerce",
            )

            if values.notna().any():
                row[
                    column
                ] = float(
                    values.mean()
                )

        latency = pd.to_numeric(
            group.get(
                "answer_latency_ms",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        ).dropna()

        if len(
            latency
        ):
            row[
                "answer_latency_p95_ms"
            ] = float(
                latency.quantile(
                    0.95
                )
            )

        rows.append(
            row
        )

    overall = {
        "split": "ALL",
        "cases": int(
            len(
                frame
            )
        ),
        "successful_cases": int(
            (
                frame[
                    "status"
                ]
                == "ok"
            ).sum()
        ),
    }

    for column in [
        "overall_judge_score",
        "numeric_faithfulness",
        "fact_value_accuracy",
        "scope_product_count_accuracy",
        "comparison_fact_accuracy",
        "rendered_metric_accuracy",
        "policy_guard_configuration",
        "correctness",
        "groundedness",
        "caveat_compliance",
        "completeness",
        "relevance",
        "managerial_usefulness",
        "instruction_following",
        "answer_latency_ms",
        "judge_latency_ms",
        "evaluation_latency_ms",
        "answer_total_tokens",
        "judge_total_tokens",
    ]:
        if column not in (
            frame.columns
        ):
            continue

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

    latency = pd.to_numeric(
        frame.get(
            "answer_latency_ms",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    ).dropna()

    if len(
        latency
    ):
        overall[
            "answer_latency_p95_ms"
        ] = float(
            latency.quantile(
                0.95
            )
        )

    rows.append(
        overall
    )

    return pd.DataFrame(
        rows
    )
