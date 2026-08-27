
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


def _json_safe(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        if np.isnan(
            value
        ):
            return None

        return float(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    return value


def _normalized(
    series,
):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def _coverage_status(
    coverage,
    ready_threshold,
    limited_threshold,
):
    coverage = float(
        coverage
    )

    if coverage >= float(
        ready_threshold
    ):
        return "ready"

    if coverage >= float(
        limited_threshold
    ):
        return "limited"

    return "unavailable"


@dataclass
class AnalyticsAuditResult:
    summary: dict
    column_quality: pd.DataFrame
    numeric_quality: pd.DataFrame
    review_quality: pd.DataFrame
    metric_readiness: pd.DataFrame
    top_category1: pd.DataFrame
    top_category2: pd.DataFrame
    top_brands: pd.DataFrame

    def to_dict(
        self,
    ):
        return {
            "summary": (
                self.summary
            ),
            "column_quality": (
                self.column_quality
                .to_dict(
                    orient="records"
                )
            ),
            "numeric_quality": (
                self.numeric_quality
                .to_dict(
                    orient="records"
                )
            ),
            "review_quality": (
                self.review_quality
                .to_dict(
                    orient="records"
                )
            ),
            "metric_readiness": (
                self.metric_readiness
                .to_dict(
                    orient="records"
                )
            ),
            "top_category1": (
                self.top_category1
                .to_dict(
                    orient="records"
                )
            ),
            "top_category2": (
                self.top_category2
                .to_dict(
                    orient="records"
                )
            ),
            "top_brands": (
                self.top_brands
                .to_dict(
                    orient="records"
                )
            ),
        }


    def save_json(
        self,
        path,
    ):
        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                _json_safe(
                    self.to_dict()
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )


class AnalyticsDataAuditor:

    TEXT_FIELDS = (
        "title_fa",
        "Brand",
        "Category1",
        "Category2",
        "sub_category",
    )

    NUMERIC_FIELDS = (
        "Price",
        "min_price_last_month",
        "Rate",
        "Rate_cnt",
    )


    def __init__(
        self,
        repository,
        ready_coverage=0.70,
        limited_coverage=0.30,
        minimum_comment_join_rate=0.95,
        product_rating_max=100.0,
        review_cap_min_products=20,
        generic_brand_values=None,
        unknown_category_values=None,
    ):
        self.repository = repository

        self.ready_coverage = float(
            ready_coverage
        )

        self.limited_coverage = float(
            limited_coverage
        )

        self.minimum_comment_join_rate = float(
            minimum_comment_join_rate
        )

        self.product_rating_max = float(
            product_rating_max
        )

        self.review_cap_min_products = int(
            review_cap_min_products
        )

        self.generic_brand_values = {
            str(
                value
            )
            .strip()
            .casefold()
            for value
            in (
                generic_brand_values
                or [
                    "متفرقه",
                    "unknown",
                    "نامشخص",
                ]
            )
        }

        self.unknown_category_values = {
            str(
                value
            )
            .strip()
            .casefold()
            for value
            in (
                unknown_category_values
                or [
                    "unknown",
                    "نامشخص",
                ]
            )
        }


    def _column_quality(
        self,
        products,
    ):
        rows = []

        total = max(
            len(products),
            1,
        )

        for column in (
            self.TEXT_FIELDS
        ):
            if column not in products.columns:
                continue

            normalized = _normalized(
                products[
                    column
                ]
            )

            non_empty = (
                normalized
                != ""
            )

            placeholder_values = (
                self.generic_brand_values
                if column
                == "Brand"
                else (
                    self.unknown_category_values
                    if column
                    in {
                        "Category1",
                        "Category2",
                        "sub_category",
                    }
                    else set()
                )
            )

            placeholder = (
                normalized.isin(
                    placeholder_values
                )
                if placeholder_values
                else pd.Series(
                    False,
                    index=products.index,
                )
            )

            usable = (
                non_empty
                & (
                    ~placeholder
                )
            )

            rows.append(
                {
                    "field": (
                        column
                    ),
                    "rows": int(
                        len(products)
                    ),
                    "non_empty_count": int(
                        non_empty.sum()
                    ),
                    "coverage": float(
                        non_empty.sum()
                        / total
                    ),
                    "unique_values": int(
                        normalized[
                            non_empty
                        ].nunique()
                    ),
                    "placeholder_count": int(
                        placeholder.sum()
                    ),
                    "placeholder_share": float(
                        placeholder.sum()
                        / total
                    ),
                    "usable_coverage": float(
                        usable.sum()
                        / total
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


    def _numeric_quality(
        self,
        products,
    ):
        rows = []

        total = max(
            len(products),
            1,
        )

        rating_counts = (
            pd.to_numeric(
                products.get(
                    "Rate_cnt",
                    pd.Series(
                        0,
                        index=products.index,
                    ),
                ),
                errors="coerce",
            )
            .fillna(0)
        )

        for column in (
            self.NUMERIC_FIELDS
        ):
            if column not in products.columns:
                continue

            values = pd.to_numeric(
                products[
                    column
                ],
                errors="coerce",
            )

            raw_valid = values.dropna()

            zero_count = int(
                (
                    raw_valid
                    == 0
                ).sum()
            )

            negative_count = int(
                (
                    raw_valid
                    < 0
                ).sum()
            )

            out_of_range = 0
            usable_mask = values.notna()
            scale = None

            if column == "Rate":
                out_of_range_mask = (
                    values.notna()
                    & (
                        (
                            values
                            < 0
                        )
                        |
                        (
                            values
                            > self.product_rating_max
                        )
                    )
                )

                out_of_range = int(
                    out_of_range_mask.sum()
                )

                usable_mask = (
                    values.between(
                        0,
                        self.product_rating_max,
                        inclusive="both",
                    )
                    & (
                        rating_counts
                        > 0
                    )
                )

                scale = (
                    f"0-{int(self.product_rating_max)}"
                )

            elif column in {
                "Price",
                "min_price_last_month",
            }:
                usable_mask = (
                    values.notna()
                    & (
                        values
                        > 0
                    )
                )

            elif column == "Rate_cnt":
                usable_mask = (
                    values.notna()
                    & (
                        values
                        >= 0
                    )
                )

            usable = values[
                usable_mask
            ]

            rows.append(
                {
                    "field": column,
                    "raw_valid_count": int(
                        len(raw_valid)
                    ),
                    "raw_coverage": float(
                        len(raw_valid)
                        / total
                    ),
                    "usable_count": int(
                        len(usable)
                    ),
                    "coverage": float(
                        len(usable)
                        / total
                    ),
                    "zero_count": (
                        zero_count
                    ),
                    "negative_count": (
                        negative_count
                    ),
                    "out_of_range_count": (
                        out_of_range
                    ),
                    "scale": scale,
                    "min": (
                        float(
                            usable.min()
                        )
                        if len(usable)
                        else None
                    ),
                    "p25": (
                        float(
                            usable.quantile(
                                0.25
                            )
                        )
                        if len(usable)
                        else None
                    ),
                    "median": (
                        float(
                            usable.median()
                        )
                        if len(usable)
                        else None
                    ),
                    "p75": (
                        float(
                            usable.quantile(
                                0.75
                            )
                        )
                        if len(usable)
                        else None
                    ),
                    "p95": (
                        float(
                            usable.quantile(
                                0.95
                            )
                        )
                        if len(usable)
                        else None
                    ),
                    "max": (
                        float(
                            usable.max()
                        )
                        if len(usable)
                        else None
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


    @staticmethod
    def _top_values(
        products,
        column,
        top_n=20,
    ):
        if column not in (
            products.columns
        ):
            return pd.DataFrame()

        values = (
            products[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        values = values[
            values
            != ""
        ]

        counts = (
            values
            .value_counts()
            .head(
                int(
                    top_n
                )
            )
        )

        total = max(
            len(products),
            1,
        )

        return pd.DataFrame(
            {
                column: (
                    counts.index
                ),
                "product_count": (
                    counts.values
                ),
                "product_share": (
                    counts.values
                    / total
                ),
            }
        )


    def _review_corpus_diagnostics(
        self,
        products,
        review_summary,
    ):
        stats = (
            self.repository
            .review_stats()
        )

        if len(stats) == 0:
            return {
                **review_summary,
                "max_reviews_per_product_in_corpus": 0,
                "products_at_max_review_count": 0,
                "products_at_max_with_rate_cnt_above_max": 0,
                "review_count_cap_suspected": False,
            }

        counts = pd.to_numeric(
            stats[
                "review_count"
            ],
            errors="coerce",
        ).fillna(0)

        max_count = int(
            counts.max()
        )

        at_max = stats[
            counts
            == max_count
        ][
            [
                "product_id",
                "review_count",
            ]
        ].copy()

        products_at_max = int(
            len(at_max)
        )

        rate_count_lookup = (
            products[
                [
                    "id",
                    "Rate_cnt",
                ]
            ]
            .copy()
        )

        rate_count_lookup[
            "Rate_cnt"
        ] = pd.to_numeric(
            rate_count_lookup[
                "Rate_cnt"
            ],
            errors="coerce",
        ).fillna(0)

        at_max = at_max.merge(
            rate_count_lookup,
            left_on="product_id",
            right_on="id",
            how="left",
        )

        max_with_more_votes = int(
            (
                pd.to_numeric(
                    at_max[
                        "Rate_cnt"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                > max_count
            ).sum()
        )

        cap_suspected = bool(
            max_count > 0
            and products_at_max
            >= self.review_cap_min_products
            and max_with_more_votes
            >= self.review_cap_min_products
        )

        return {
            **review_summary,
            "max_reviews_per_product_in_corpus": (
                max_count
            ),
            "products_at_max_review_count": (
                products_at_max
            ),
            "products_at_max_with_rate_cnt_above_max": (
                max_with_more_votes
            ),
            "review_count_cap_suspected": (
                cap_suspected
            ),
        }


    def _metric_readiness(
        self,
        products,
        column_quality,
        numeric_quality,
        review_summary,
    ):
        text_lookup = (
            column_quality
            .set_index(
                "field"
            )
            .to_dict(
                orient="index"
            )
            if len(
                column_quality
            )
            else {}
        )

        numeric_lookup = (
            numeric_quality
            .set_index(
                "field"
            )
            .to_dict(
                orient="index"
            )
            if len(
                numeric_quality
            )
            else {}
        )

        rows = []

        rows.append(
            {
                "metric": (
                    "product_count"
                ),
                "status": "ready",
                "coverage": 1.0,
                "reason": (
                    "Canonical product IDs are the "
                    "counting unit."
                ),
            }
        )

        for metric, field in (
            (
                "current_price_statistics",
                "Price",
            ),
            (
                "product_rating_statistics",
                "Rate",
            ),
            (
                "historical_price_statistics",
                "min_price_last_month",
            ),
        ):
            coverage = float(
                numeric_lookup.get(
                    field,
                    {}
                ).get(
                    "coverage",
                    0.0,
                )
            )

            rows.append(
                {
                    "metric": metric,
                    "status": (
                        _coverage_status(
                            coverage,
                            self.ready_coverage,
                            self.limited_coverage,
                        )
                    ),
                    "coverage": (
                        coverage
                    ),
                    "reason": (
                        (
                            "Rated-product coverage on the native 0..100 "
                            "product score scale; products with Rate_cnt=0 "
                            "are treated as unrated."
                        )
                        if field == "Rate"
                        else f"{field} usable coverage."
                    ),
                }
            )

        for metric, field in (
            (
                "category1_analysis",
                "Category1",
            ),
            (
                "category2_analysis",
                "Category2",
            ),
            (
                "sub_category_analysis",
                "sub_category",
            ),
        ):
            coverage = float(
                text_lookup.get(
                    field,
                    {}
                ).get(
                    "usable_coverage",
                    0.0,
                )
            )

            rows.append(
                {
                    "metric": metric,
                    "status": (
                        _coverage_status(
                            coverage,
                            self.ready_coverage,
                            self.limited_coverage,
                        )
                    ),
                    "coverage": (
                        coverage
                    ),
                    "reason": (
                        f"{field} usable non-placeholder coverage."
                    ),
                }
            )

        brand_coverage = float(
            text_lookup.get(
                "Brand",
                {}
            ).get(
                "usable_coverage",
                0.0,
            )
        )

        rows.append(
            {
                "metric": (
                    "brand_analysis"
                ),
                "status": (
                    _coverage_status(
                        brand_coverage,
                        self.ready_coverage,
                        self.limited_coverage,
                    )
                ),
                "coverage": (
                    brand_coverage
                ),
                "reason": (
                    "Usable brand coverage after excluding "
                    "generic/unknown brand labels."
                ),
            }
        )

        join_rate = float(
            review_summary.get(
                "product_join_rate",
                0.0,
            )
        )

        review_count_status = (
            "ready"
            if join_rate
            >= self.minimum_comment_join_rate
            else (
                "limited"
                if join_rate
                >= self.limited_coverage
                else "unavailable"
            )
        )

        rows.append(
            {
                "metric": (
                    "review_presence_and_coverage"
                ),
                "status": (
                    review_count_status
                ),
                "coverage": (
                    join_rate
                ),
                "reason": (
                    "Share of valid comment product IDs "
                    "that join the canonical product catalog."
                ),
            }
        )

        cap_suspected = bool(
            review_summary.get(
                "review_count_cap_suspected",
                False,
            )
        )

        rows.append(
            {
                "metric": (
                    "review_volume_ranking"
                ),
                "status": (
                    "limited"
                    if cap_suspected
                    else review_count_status
                ),
                "coverage": (
                    join_rate
                ),
                "reason": (
                    (
                        "Per-product review counts appear capped/truncated "
                        f"at {review_summary.get('max_reviews_per_product_in_corpus')}; "
                        "use review presence/coverage, but do not claim true "
                        "'most reviewed' ranking from this corpus."
                    )
                    if cap_suspected
                    else (
                        "No deterministic per-product review-count cap "
                        "was detected in the available corpus."
                    )
                ),
            }
        )

        review_rate_coverage = float(
            review_summary.get(
                "review_rate_coverage",
                0.0,
            )
        )

        rows.append(
            {
                "metric": (
                    "review_rating_statistics"
                ),
                "status": (
                    _coverage_status(
                        review_rate_coverage,
                        self.ready_coverage,
                        self.limited_coverage,
                    )
                ),
                "coverage": (
                    review_rate_coverage
                ),
                "reason": (
                    "Share of matched review rows with "
                    "valid 1..5 review ratings."
                ),
            }
        )

        return pd.DataFrame(
            rows
        )


    def run(
        self,
        top_n=20,
    ):
        products = self.repository.products

        column_quality = (
            self._column_quality(
                products
            )
        )

        numeric_quality = (
            self._numeric_quality(
                products
            )
        )

        review_summary = (
            self._review_corpus_diagnostics(
                products=products,
                review_summary=(
                    self.repository
                    .review_scan_summary()
                ),
            )
        )

        review_quality = (
            pd.DataFrame(
                [
                    review_summary
                ]
            )
        )

        top_category1 = (
            self._top_values(
                products,
                "Category1",
                top_n=top_n,
            )
        )

        top_category2 = (
            self._top_values(
                products,
                "Category2",
                top_n=top_n,
            )
        )

        top_brands = (
            self._top_values(
                products,
                "Brand",
                top_n=top_n,
            )
        )

        metric_readiness = (
            self._metric_readiness(
                products=products,
                column_quality=(
                    column_quality
                ),
                numeric_quality=(
                    numeric_quality
                ),
                review_summary=(
                    review_summary
                ),
            )
        )

        summary = {
            "product_source": (
                self.repository
                .product_source
            ),
            "canonical_product_count": int(
                len(products)
            ),
            "unique_product_ids": int(
                products[
                    "id"
                ].nunique()
            ),
            "duplicate_product_ids": int(
                products[
                    "id"
                ].duplicated().sum()
            ),
            "total_comment_rows": int(
                review_summary.get(
                    "total_comment_rows",
                    0,
                )
            ),
            "matched_comment_rows": int(
                review_summary.get(
                    "matched_product_rows",
                    0,
                )
            ),
            "comment_product_join_rate": float(
                review_summary.get(
                    "product_join_rate",
                    0.0,
                )
            ),
            "products_with_reviews": int(
                (
                    self.repository
                    .review_stats()[
                        "review_count"
                    ]
                    > 0
                ).sum()
            ),
        }

        summary[
            "product_review_coverage"
        ] = (
            summary[
                "products_with_reviews"
            ]
            / summary[
                "canonical_product_count"
            ]
            if summary[
                "canonical_product_count"
            ]
            else 0.0
        )

        return AnalyticsAuditResult(
            summary=summary,
            column_quality=(
                column_quality
            ),
            numeric_quality=(
                numeric_quality
            ),
            review_quality=(
                review_quality
            ),
            metric_readiness=(
                metric_readiness
            ),
            top_category1=(
                top_category1
            ),
            top_category2=(
                top_category2
            ),
            top_brands=(
                top_brands
            ),
        )
