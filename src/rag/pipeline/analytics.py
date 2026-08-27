
import time

import pandas as pd

from ..analytics.prompt import (
    MANAGER_SYSTEM_PROMPT,
    build_manager_prompt,
    build_manager_repair_prompt,
)
from ..analytics.validation import (
    render_metric_template,
    sanitize_manager_response,
    validate_manager_response,
)


class ManagerAnalyticsPipeline:
    """
    Deterministic analytics + grounded LLM explanation.

    The LLM never owns numeric values. It can only reference verified metric
    placeholders, which are rendered after validation.
    """

    def __init__(
        self,
        analytics_service,
        generator,
        brand_usable_coverage=0.441082,
        product_rating_coverage=0.380649,
        historical_price_enabled=False,
        review_volume_ranking_enabled=False,
        top_n=5,
    ):
        self.analytics = (
            analytics_service
        )

        self.generator = (
            generator
        )

        self.brand_usable_coverage = float(
            brand_usable_coverage
        )

        self.product_rating_coverage = float(
            product_rating_coverage
        )

        self.historical_price_enabled = bool(
            historical_price_enabled
        )

        self.review_volume_ranking_enabled = bool(
            review_volume_ranking_enabled
        )

        self.top_n = int(
            top_n
        )


    @staticmethod
    def _format_integer(
        value,
    ):
        if value is None:
            return "—"

        return f"{int(round(float(value))):,}"


    @staticmethod
    def _format_price(
        value,
    ):
        if value is None:
            return "—"

        return (
            f"{float(value):,.0f} تومان"
        )


    @staticmethod
    def _format_percent(
        value,
    ):
        if value is None:
            return "—"

        return (
            f"{float(value):.1f}%"
        )


    @staticmethod
    def _format_score(
        value,
        scale,
    ):
        if value is None:
            return "—"

        return (
            f"{float(value):.2f}/{scale}"
        )


    def _fact(
        self,
        label,
        value,
        display_value,
        unit=None,
        caveat=None,
    ):
        return {
            "label": label,
            "value": value,
            "display_value": (
                display_value
            ),
            "unit": unit,
            "caveat": caveat,
        }


    def _overview_facts(
        self,
        overview,
    ):
        facts = {}

        facts[
            "overview.product_count"
        ] = self._fact(
            "تعداد محصولات",
            overview[
                "product_count"
            ],
            self._format_integer(
                overview[
                    "product_count"
                ]
            ),
        )

        facts[
            "overview.brand_count"
        ] = self._fact(
            "تعداد برندهای قابل شناسایی",
            overview[
                "brand_count"
            ],
            self._format_integer(
                overview[
                    "brand_count"
                ]
            ),
            caveat=(
                "Brand coverage is limited."
            ),
        )

        facts[
            "overview.median_price"
        ] = self._fact(
            "میانه قیمت فعلی",
            overview[
                "price"
            ][
                "median"
            ],
            self._format_price(
                overview[
                    "price"
                ][
                    "median"
                ]
            ),
            unit="تومان",
        )

        facts[
            "overview.price_p25"
        ] = self._fact(
            "چارک اول قیمت",
            overview[
                "price"
            ][
                "p25"
            ],
            self._format_price(
                overview[
                    "price"
                ][
                    "p25"
                ]
            ),
            unit="تومان",
        )

        facts[
            "overview.price_p75"
        ] = self._fact(
            "چارک سوم قیمت",
            overview[
                "price"
            ][
                "p75"
            ],
            self._format_price(
                overview[
                    "price"
                ][
                    "p75"
                ]
            ),
            unit="تومان",
        )

        facts[
            "overview.price_coverage_pct"
        ] = self._fact(
            "پوشش قیمت",
            (
                overview[
                    "price"
                ][
                    "coverage"
                ]
                * 100
            ),
            self._format_percent(
                overview[
                    "price"
                ][
                    "coverage"
                ]
                * 100
            ),
            unit="%",
        )

        facts[
            "overview.review_coverage_pct"
        ] = self._fact(
            "درصد محصولات دارای review در corpus",
            (
                overview[
                    "review_coverage"
                ]
                * 100
            ),
            self._format_percent(
                overview[
                    "review_coverage"
                ]
                * 100
            ),
            unit="%",
            caveat=(
                "Review presence is valid; review-volume ranking "
                "is intentionally not treated as market-wide volume."
            ),
        )

        rating = overview[
            "product_rating"
        ]

        facts[
            "overview.rated_product_coverage_pct"
        ] = self._fact(
            "پوشش محصولات دارای امتیاز",
            (
                rating[
                    "rated_product_coverage"
                ]
                * 100
            ),
            self._format_percent(
                rating[
                    "rated_product_coverage"
                ]
                * 100
            ),
            unit="%",
        )

        facts[
            "overview.weighted_product_rating_100"
        ] = self._fact(
            "میانگین وزنی امتیاز محصول",
            overview[
                "weighted_product_rating"
            ],
            self._format_score(
                overview[
                    "weighted_product_rating"
                ],
                100,
            ),
            unit="/100",
            caveat=(
                "Only products with Rate_cnt > 0 participate."
            ),
        )

        facts[
            "overview.weighted_product_rating_5"
        ] = self._fact(
            "معادل پنج‌نمره‌ای امتیاز محصول",
            overview[
                "weighted_product_rating_5"
            ],
            self._format_score(
                overview[
                    "weighted_product_rating_5"
                ],
                5,
            ),
            unit="/5",
        )

        facts[
            "overview.rating_count_total"
        ] = self._fact(
            "مجموع تعداد امتیازهای ثبت‌شده",
            overview[
                "rating_count_total"
            ],
            self._format_integer(
                overview[
                    "rating_count_total"
                ]
            ),
        )

        facts[
            "overview.weighted_review_rating_5"
        ] = self._fact(
            "میانگین امتیاز reviewهای موجود",
            overview[
                "weighted_review_rating"
            ],
            self._format_score(
                overview[
                    "weighted_review_rating"
                ],
                5,
            ),
            unit="/5",
        )

        return facts


    @staticmethod
    def _records(
        frame,
        columns,
    ):
        if (
            frame is None
            or len(
                frame
            ) == 0
        ):
            return []

        result = []

        for row in frame.head(
            10
        ).to_dict(
            orient="records"
        ):
            record = {}

            for column in columns:
                if column not in row:
                    continue

                value = row[
                    column
                ]

                try:
                    if pd.isna(
                        value
                    ):
                        value = None
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

                record[
                    column
                ] = value

            result.append(
                record
            )

        return result


    def build_context(
        self,
        filters=None,
        comparison_categories=None,
        category_field="Category2",
    ):
        filters = dict(
            filters
            or {}
        )

        overview = (
            self.analytics
            .overview(
                filters=filters
            )
        )

        facts = (
            self._overview_facts(
                overview
            )
        )

        top_brands = (
            self.analytics
            .top_brands(
                filters=filters,
                top_n=(
                    self.top_n
                ),
                include_generic=False,
            )
        )

        top_rating = (
            self.analytics
            .top_products(
                filters=filters,
                sort_by="rating",
                top_n=(
                    self.top_n
                ),
            )
        )

        top_rating_count = (
            self.analytics
            .top_products(
                filters=filters,
                sort_by=(
                    "rating_count"
                ),
                top_n=(
                    self.top_n
                ),
            )
        )

        comparison_records = []

        if comparison_categories:
            comparison = (
                self.analytics
                .compare_categories(
                    category_values=(
                        comparison_categories
                    ),
                    category_field=(
                        category_field
                    ),
                )
            )

            comparison_records = (
                comparison.to_dict(
                    orient="records"
                )
            )

            for index, row in (
                comparison
                .reset_index(
                    drop=True
                )
                .iterrows()
            ):
                prefix = (
                    f"comparison.c{index}"
                )

                facts[
                    f"{prefix}.product_count"
                ] = self._fact(
                    (
                        f"تعداد محصول "
                        f"{row[category_field]}"
                    ),
                    row[
                        "product_count"
                    ],
                    self._format_integer(
                        row[
                            "product_count"
                        ]
                    ),
                )

                facts[
                    f"{prefix}.median_price"
                ] = self._fact(
                    (
                        f"میانه قیمت "
                        f"{row[category_field]}"
                    ),
                    row[
                        "median_price"
                    ],
                    self._format_price(
                        row[
                            "median_price"
                        ]
                    ),
                    unit="تومان",
                )

                facts[
                    f"{prefix}.review_coverage_pct"
                ] = self._fact(
                    (
                        f"پوشش review "
                        f"{row[category_field]}"
                    ),
                    (
                        row[
                            "review_coverage"
                        ]
                        * 100
                    ),
                    self._format_percent(
                        row[
                            "review_coverage"
                        ]
                        * 100
                    ),
                    unit="%",
                )

                facts[
                    f"{prefix}.weighted_product_rating_100"
                ] = self._fact(
                    (
                        f"امتیاز وزنی "
                        f"{row[category_field]}"
                    ),
                    row[
                        "weighted_product_rating"
                    ],
                    self._format_score(
                        row[
                            "weighted_product_rating"
                        ],
                        100,
                    ),
                    unit="/100",
                )

                facts[
                    f"{prefix}.rating_count_total"
                ] = self._fact(
                    (
                        f"تعداد امتیاز ثبت‌شده "
                        f"{row[category_field]}"
                    ),
                    row[
                        "rating_count_total"
                    ],
                    self._format_integer(
                        row[
                            "rating_count_total"
                        ]
                    ),
                )

        scope = {
            "filters": filters,
            "comparison_categories": (
                list(
                    comparison_categories
                    or []
                )
            ),
            "category_field": (
                category_field
            ),
        }

        return {
            "scope": scope,
            "facts": facts,
            "top_brands": self._records(
                top_brands,
                [
                    "Brand",
                    "product_count",
                    "product_share",
                    "review_coverage",
                    "weighted_product_rating",
                    "rating_count_total",
                ],
            ),
            "top_products_by_rating": (
                self._records(
                    top_rating,
                    [
                        "id",
                        "title_fa",
                        "Brand",
                        "Price",
                        "Rate",
                        "Rate_cnt",
                    ],
                )
            ),
            "top_products_by_rating_count": (
                self._records(
                    top_rating_count,
                    [
                        "id",
                        "title_fa",
                        "Brand",
                        "Price",
                        "Rate",
                        "Rate_cnt",
                    ],
                )
            ),
            "category_comparison": (
                comparison_records
            ),
            "data_quality": {
                "brand_usable_coverage": (
                    self.brand_usable_coverage
                ),
                "product_rating_coverage": (
                    self.product_rating_coverage
                ),
                "historical_price_enabled": (
                    self.historical_price_enabled
                ),
                "review_volume_ranking_enabled": (
                    self.review_volume_ranking_enabled
                ),
                "notes": [
                    (
                        "Brand analytics is partial because generic/unknown "
                        "brands cover a large part of the catalog."
                    ),
                    (
                        "Historical price is disabled because coverage is too low."
                    ),
                    (
                        "Review presence/coverage is usable, but review_count is "
                        "not treated as true market-wide popularity."
                    ),
                    (
                        "Use Rate_cnt when discussing engagement and label it "
                        "as number of ratings."
                    ),
                ],
            },
            "overview": overview,
            "tables": {
                "top_brands": top_brands,
                "top_products_by_rating": (
                    top_rating
                ),
                "top_products_by_rating_count": (
                    top_rating_count
                ),
            },
        }


    def answer(
        self,
        question,
        filters=None,
        comparison_categories=None,
        category_field="Category2",
    ):
        start = time.perf_counter()

        question = str(
            question
        ).strip()

        if not question:
            raise ValueError(
                "Manager question must not be empty."
            )

        context = self.build_context(
            filters=filters,
            comparison_categories=(
                comparison_categories
            ),
            category_field=(
                category_field
            ),
        )

        prompt = (
            build_manager_prompt(
                question=question,
                context=context,
            )
        )

        first = (
            self.generator
            .generate(
                system_prompt=(
                    MANAGER_SYSTEM_PROMPT
                ),
                user_prompt=prompt,
            )
        )

        attempts = [
            first
        ]

        payload = first[
            "payload"
        ]

        allowed_keys = set(
            context[
                "facts"
            ]
        )

        valid, errors = (
            validate_manager_response(
                payload=payload,
                allowed_metric_keys=(
                    allowed_keys
                ),
            )
        )

        repaired = False

        if not valid:
            repair_prompt = (
                build_manager_repair_prompt(
                    original_prompt=(
                        prompt
                    ),
                    previous_payload=(
                        payload
                    ),
                    validation_errors=(
                        errors
                    ),
                    allowed_metric_keys=(
                        allowed_keys
                    ),
                )
            )

            second = (
                self.generator
                .generate(
                    system_prompt=(
                        MANAGER_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        repair_prompt
                    ),
                )
            )

            attempts.append(
                second
            )

            candidate = second[
                "payload"
            ]

            second_valid, second_errors = (
                validate_manager_response(
                    payload=candidate,
                    allowed_metric_keys=(
                        allowed_keys
                    ),
                )
            )

            if second_valid:
                payload = candidate
                valid = True
                errors = []
                repaired = True
            else:
                payload = (
                    sanitize_manager_response(
                        payload=candidate,
                        facts=context[
                            "facts"
                        ],
                    )
                )

                valid, errors = (
                    validate_manager_response(
                        payload=payload,
                        allowed_metric_keys=(
                            allowed_keys
                        ),
                    )
                )

        rendered_answer = (
            render_metric_template(
                payload[
                    "answer_template"
                ],
                context[
                    "facts"
                ],
            )
        )

        rendered_insights = []

        for insight in payload.get(
            "insights",
            [],
        ):
            rendered_insights.append(
                {
                    **insight,
                    "text": (
                        render_metric_template(
                            insight[
                                "text_template"
                            ],
                            context[
                                "facts"
                            ],
                        )
                    ),
                }
            )

        rendered_caveats = [
            render_metric_template(
                value,
                context[
                    "facts"
                ],
            )
            for value
            in payload.get(
                "caveats",
                [],
            )
        ]

        total_latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "question": question,
            "answer": rendered_answer,
            "answer_template": (
                payload[
                    "answer_template"
                ]
            ),
            "insights": (
                rendered_insights
            ),
            "caveats": (
                rendered_caveats
            ),
            "confidence": (
                payload.get(
                    "confidence",
                    "low",
                )
            ),
            "numeric_faithfulness_valid": (
                bool(
                    valid
                )
            ),
            "validation_errors": (
                list(
                    errors
                )
            ),
            "repaired": (
                repaired
            ),
            "facts": (
                context[
                    "facts"
                ]
            ),
            "context": context,
            "telemetry": {
                "model": attempts[-1].get(
                    "model"
                ),
                "generation_calls": len(
                    attempts
                ),
                "generation_latency_ms": sum(
                    float(
                        attempt.get(
                            "latency_ms",
                            0.0,
                        )
                    )
                    for attempt
                    in attempts
                ),
                "total_latency_ms": (
                    total_latency_ms
                ),
                "prompt_tokens": sum(
                    int(
                        attempt.get(
                            "prompt_tokens",
                            0,
                        )
                    )
                    for attempt
                    in attempts
                ),
                "completion_tokens": sum(
                    int(
                        attempt.get(
                            "completion_tokens",
                            0,
                        )
                    )
                    for attempt
                    in attempts
                ),
                "total_tokens": sum(
                    int(
                        attempt.get(
                            "total_tokens",
                            0,
                        )
                    )
                    for attempt
                    in attempts
                ),
                "estimated_cost_usd": (
                    sum(
                        float(
                            attempt.get(
                                "estimated_cost_usd",
                                0.0,
                            )
                            or 0.0
                        )
                        for attempt
                        in attempts
                    )
                    if any(
                        attempt.get(
                            "estimated_cost_usd"
                        )
                        is not None
                        for attempt
                        in attempts
                    )
                    else None
                ),
            },
        }
