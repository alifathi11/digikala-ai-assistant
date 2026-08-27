
import numpy as np
import pandas as pd


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


def _clean_numeric(
    series,
):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_weighted_mean(
    values,
    weights,
):
    values = _clean_numeric(
        values
    )

    weights = (
        _clean_numeric(
            weights
        )
        .fillna(0)
        .clip(
            lower=0
        )
    )

    mask = (
        values.notna()
        & (
            weights
            > 0
        )
    )

    if not mask.any():
        return None

    denominator = float(
        weights[
            mask
        ].sum()
    )

    if denominator <= 0:
        return None

    return float(
        (
            values[
                mask
            ]
            * weights[
                mask
            ]
        ).sum()
        / denominator
    )



def _rated_product_mask(
    frame,
    rating_max=100.0,
):
    if (
        "Rate" not in frame.columns
        or "Rate_cnt" not in frame.columns
    ):
        return pd.Series(
            False,
            index=frame.index,
        )

    ratings = pd.to_numeric(
        frame["Rate"],
        errors="coerce",
    )

    rating_counts = (
        pd.to_numeric(
            frame["Rate_cnt"],
            errors="coerce",
        )
        .fillna(0)
    )

    return (
        ratings.between(
            0,
            float(rating_max),
            inclusive="both",
        )
        & (
            rating_counts
            > 0
        )
    )


def _product_rating_summary(
    frame,
    rating_max=100.0,
):
    mask = _rated_product_mask(
        frame,
        rating_max=rating_max,
    )

    ratings = pd.to_numeric(
        frame.get(
            "Rate",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    rated = ratings[
        mask
    ]

    summary = _numeric_summary(
        rated
    )

    total_products = max(
        len(frame),
        1,
    )

    summary[
        "rated_product_count"
    ] = int(
        mask.sum()
    )

    summary[
        "rated_product_coverage"
    ] = float(
        mask.sum()
        / total_products
    )

    summary[
        "scale_min"
    ] = 0.0

    summary[
        "scale_max"
    ] = float(
        rating_max
    )

    if summary[
        "mean"
    ] is not None:
        summary[
            "mean_5"
        ] = float(
            summary[
                "mean"
            ]
            / (
                float(
                    rating_max
                )
                / 5.0
            )
        )
    else:
        summary[
            "mean_5"
        ] = None

    return summary


def _numeric_summary(
    series,
):
    values = _clean_numeric(
        series
    )

    valid = values.dropna()

    if len(valid) == 0:
        return {
            "valid_count": 0,
            "coverage": 0.0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }

    return {
        "valid_count": int(
            len(valid)
        ),
        "coverage": float(
            len(valid)
            / max(
                len(values),
                1,
            )
        ),
        "mean": float(
            valid.mean()
        ),
        "median": float(
            valid.median()
        ),
        "p25": float(
            valid.quantile(
                0.25
            )
        ),
        "p75": float(
            valid.quantile(
                0.75
            )
        ),
        "min": float(
            valid.min()
        ),
        "max": float(
            valid.max()
        ),
    }


class AnalyticsService:
    """
    Deterministic manager/category analytics.

    All numbers are computed in Python from the canonical product table and
    per-product review aggregates. No LLM is involved in this layer.
    """

    FILTER_FIELDS = {
        "category1": "Category1",
        "category2": "Category2",
        "sub_category": "sub_category",
        "brand": "Brand",
    }


    def __init__(
        self,
        repository,
        generic_brand_values=None,
        unknown_category_values=None,
        min_rating_count_for_leaders=10,
        default_top_n=10,
        product_rating_max=100.0,
    ):
        self.repository = repository

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

        self.min_rating_count_for_leaders = int(
            min_rating_count_for_leaders
        )

        self.default_top_n = int(
            default_top_n
        )

        self.product_rating_max = float(
            product_rating_max
        )

        self._enriched = None


    @property
    def products(
        self,
    ):
        if self._enriched is None:
            self._enriched = (
                self.repository
                .enriched_products()
            )

        return self._enriched


    def distinct_values(
        self,
        field,
        filters=None,
        include_unknown=False,
    ):
        frame = self._filter(
            filters
        )

        if field not in frame.columns:
            raise ValueError(
                f"Unknown analytics field: {field}"
            )

        values = (
            frame[
                field
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        values = values[
            values
            != ""
        ]

        if (
            not include_unknown
            and field
            in {
                "Category1",
                "Category2",
                "sub_category",
            }
        ):
            values = values[
                ~self._is_unknown_category(
                    values
                )
            ]

        if (
            not include_unknown
            and field == "Brand"
        ):
            values = values[
                ~self._is_generic_brand(
                    values
                )
            ]

        return sorted(
            values.unique().tolist()
        )


    def _filter(
        self,
        filters=None,
    ):
        frame = self.products

        if not filters:
            return frame.copy()

        result = frame

        for key, value in (
            filters.items()
        ):
            if value is None:
                continue

            column = (
                self.FILTER_FIELDS.get(
                    key,
                    key,
                )
            )

            if column not in result.columns:
                raise ValueError(
                    "Unknown analytics filter "
                    f"column: {column}"
                )

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

            normalized_values = {
                str(
                    item
                )
                .strip()
                .casefold()
                for item
                in values
            }

            mask = (
                _normalized_text(
                    result[
                        column
                    ]
                )
                .isin(
                    normalized_values
                )
            )

            result = result[
                mask
            ]

        return (
            result
            .copy()
        )


    def _is_generic_brand(
        self,
        series,
    ):
        return (
            _normalized_text(
                series
            )
            .isin(
                self.generic_brand_values
            )
            |
            (
                _normalized_text(
                    series
                )
                == ""
            )
        )


    def _is_unknown_category(
        self,
        series,
    ):
        normalized = (
            _normalized_text(
                series
            )
        )

        return (
            normalized.isin(
                self.unknown_category_values
            )
            |
            (
                normalized
                == ""
            )
        )


    def overview(
        self,
        filters=None,
    ):
        frame = self._filter(
            filters
        )

        product_count = int(
            len(frame)
        )

        if product_count == 0:
            return {
                "filters": (
                    dict(
                        filters
                        or {}
                    )
                ),
                "product_count": 0,
                "brand_count": 0,
                "brand_count_all": 0,
                "generic_brand_product_share": 0.0,
                "review_count": 0,
                "products_with_reviews": 0,
                "review_coverage": 0.0,
                "price": _numeric_summary(
                    pd.Series(
                        dtype=float
                    )
                ),
                "product_rating": (
                    _numeric_summary(
                        pd.Series(
                            dtype=float
                        )
                    )
                ),
                "weighted_product_rating": None,
                "weighted_product_rating_5": None,
                "rating_count_total": 0,
                "review_rating": (
                    _numeric_summary(
                        pd.Series(
                            dtype=float
                        )
                    )
                ),
                "weighted_review_rating": None,
            }

        brand_series = (
            frame[
                "Brand"
            ]
            if "Brand"
            in frame.columns
            else pd.Series(
                "",
                index=frame.index,
            )
        )

        generic_brand_mask = (
            self._is_generic_brand(
                brand_series
            )
        )

        brand_count_all = int(
            _normalized_text(
                brand_series
            )[
                _normalized_text(
                    brand_series
                )
                != ""
            ].nunique()
        )

        brand_count = int(
            _normalized_text(
                brand_series[
                    ~generic_brand_mask
                ]
            )[
                _normalized_text(
                    brand_series[
                        ~generic_brand_mask
                    ]
                )
                != ""
            ].nunique()
        )

        review_count = int(
            pd.to_numeric(
                frame[
                    "review_count"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        products_with_reviews = int(
            (
                pd.to_numeric(
                    frame[
                        "review_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                > 0
            ).sum()
        )

        rating_count = (
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

        rated_mask = _rated_product_mask(
            frame,
            rating_max=(
                self.product_rating_max
            ),
        )

        rating_values = pd.to_numeric(
            frame.get(
                "Rate",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )

        weighted_product_rating = (
            _safe_weighted_mean(
                rating_values[
                    rated_mask
                ],
                rating_count[
                    rated_mask
                ],
            )
        )

        review_rate_count = (
            pd.to_numeric(
                frame[
                    "review_rate_count"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        review_rate_sum = (
            pd.to_numeric(
                frame[
                    "review_rate_sum"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        valid_review_rate_total = float(
            review_rate_count.sum()
        )

        weighted_review_rating = (
            float(
                review_rate_sum.sum()
                / valid_review_rate_total
            )
            if valid_review_rate_total
            > 0
            else None
        )

        return {
            "filters": (
                dict(
                    filters
                    or {}
                )
            ),
            "product_count": (
                product_count
            ),
            "brand_count": (
                brand_count
            ),
            "brand_count_all": (
                brand_count_all
            ),
            "generic_brand_product_share": (
                float(
                    generic_brand_mask.mean()
                )
            ),
            "review_count": (
                review_count
            ),
            "products_with_reviews": (
                products_with_reviews
            ),
            "review_coverage": (
                float(
                    products_with_reviews
                    / product_count
                )
            ),
            "price": _numeric_summary(
                frame.get(
                    "Price",
                    pd.Series(
                        dtype=float
                    ),
                )
            ),
            "product_rating": (
                _product_rating_summary(
                    frame,
                    rating_max=(
                        self.product_rating_max
                    ),
                )
            ),
            "weighted_product_rating": (
                weighted_product_rating
            ),
            "weighted_product_rating_5": (
                (
                    weighted_product_rating
                    / (
                        self.product_rating_max
                        / 5.0
                    )
                )
                if weighted_product_rating
                is not None
                else None
            ),
            "rating_count_total": int(
                rating_count.sum()
            ),
            "review_rating": (
                _numeric_summary(
                    frame[
                        "avg_review_rate"
                    ]
                )
            ),
            "weighted_review_rating": (
                weighted_review_rating
            ),
            "valid_review_rating_count": int(
                valid_review_rate_total
            ),
        }


    def top_brands(
        self,
        filters=None,
        top_n=None,
        include_generic=False,
    ):
        frame = self._filter(
            filters
        )

        if len(frame) == 0:
            return pd.DataFrame()

        if "Brand" not in frame.columns:
            return pd.DataFrame()

        working = frame.copy()

        working[
            "_brand"
        ] = (
            working[
                "Brand"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        working = working[
            working[
                "_brand"
            ]
            != ""
        ].copy()

        if not include_generic:
            working = working[
                ~self._is_generic_brand(
                    working[
                        "_brand"
                    ]
                )
            ].copy()

        if len(working) == 0:
            return pd.DataFrame()

        rows = []

        for brand, group in (
            working.groupby(
                "_brand",
                sort=False,
            )
        ):
            overview = self._overview_frame(
                group
            )

            rows.append(
                {
                    "Brand": brand,
                    **overview,
                }
            )

        result = pd.DataFrame(
            rows
        )

        total_products = max(
            len(frame),
            1,
        )

        result[
            "product_share"
        ] = (
            result[
                "product_count"
            ]
            / total_products
        )

        result = result.sort_values(
            [
                "product_count",
                "review_count",
                "weighted_product_rating",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )

        return (
            result
            .head(
                int(
                    top_n
                    or self.default_top_n
                )
            )
            .reset_index(drop=True)
        )


    def _overview_frame(
        self,
        frame,
    ):
        product_count = int(
            len(frame)
        )

        review_counts = (
            pd.to_numeric(
                frame[
                    "review_count"
                ],
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

        rating_count = (
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

        prices = _clean_numeric(
            frame.get(
                "Price",
                pd.Series(
                    dtype=float
                ),
            )
        )

        ratings = _clean_numeric(
            frame.get(
                "Rate",
                pd.Series(
                    dtype=float
                ),
            )
        )

        rated_mask = _rated_product_mask(
            frame,
            rating_max=(
                self.product_rating_max
            ),
        )

        rated_values = ratings[
            rated_mask
        ]

        return {
            "product_count": (
                product_count
            ),
            "review_count": int(
                review_counts.sum()
            ),
            "products_with_reviews": (
                products_with_reviews
            ),
            "review_coverage": (
                float(
                    products_with_reviews
                    / product_count
                )
                if product_count
                else 0.0
            ),
            "median_price": (
                float(
                    prices.median()
                )
                if prices.notna().any()
                else None
            ),
            "avg_rating": (
                float(
                    rated_values.mean()
                )
                if rated_values.notna().any()
                else None
            ),
            "rated_product_count": int(
                rated_mask.sum()
            ),
            "rated_product_coverage": float(
                rated_mask.sum()
                / max(
                    product_count,
                    1,
                )
            ),
            "weighted_product_rating": (
                _safe_weighted_mean(
                    rated_values,
                    rating_count[
                        rated_mask
                    ],
                )
            ),
            "rating_count_total": int(
                rating_count.sum()
            ),
        }


    def category_table(
        self,
        category_field="Category2",
        filters=None,
        top_n=None,
        include_unknown=False,
    ):
        frame = self._filter(
            filters
        )

        if category_field not in (
            frame.columns
        ):
            raise ValueError(
                f"Unknown category field: "
                f"{category_field}"
            )

        working = frame.copy()

        working[
            "_category"
        ] = (
            working[
                category_field
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        working = working[
            working[
                "_category"
            ]
            != ""
        ].copy()

        if not include_unknown:
            working = working[
                ~self._is_unknown_category(
                    working[
                        "_category"
                    ]
                )
            ].copy()

        rows = []

        for category, group in (
            working.groupby(
                "_category",
                sort=False,
            )
        ):
            brand_series = (
                group[
                    "Brand"
                ]
                if "Brand"
                in group.columns
                else pd.Series(
                    "",
                    index=group.index,
                )
            )

            generic_brand_mask = (
                self._is_generic_brand(
                    brand_series
                )
            )

            meaningful_brands = (
                _normalized_text(
                    brand_series[
                        ~generic_brand_mask
                    ]
                )
            )

            rows.append(
                {
                    category_field: (
                        category
                    ),
                    "brand_count": int(
                        meaningful_brands[
                            meaningful_brands
                            != ""
                        ].nunique()
                    ),
                    **self._overview_frame(
                        group
                    ),
                }
            )

        result = pd.DataFrame(
            rows
        )

        if len(result) == 0:
            return result

        result = result.sort_values(
            [
                "product_count",
                "review_count",
            ],
            ascending=[
                False,
                False,
            ],
        )

        if top_n is not None:
            result = result.head(
                int(
                    top_n
                )
            )

        return result.reset_index(
            drop=True
        )


    def compare_categories(
        self,
        category_values,
        category_field="Category2",
        base_filters=None,
    ):
        rows = []

        for category in (
            category_values
        ):
            filters = dict(
                base_filters
                or {}
            )

            filters[
                category_field
            ] = category

            overview = self.overview(
                filters=filters
            )

            rows.append(
                {
                    category_field: (
                        category
                    ),
                    "product_count": (
                        overview[
                            "product_count"
                        ]
                    ),
                    "brand_count": (
                        overview[
                            "brand_count"
                        ]
                    ),
                    "review_count": (
                        overview[
                            "review_count"
                        ]
                    ),
                    "review_coverage": (
                        overview[
                            "review_coverage"
                        ]
                    ),
                    "median_price": (
                        overview[
                            "price"
                        ][
                            "median"
                        ]
                    ),
                    "price_coverage": (
                        overview[
                            "price"
                        ][
                            "coverage"
                        ]
                    ),
                    "avg_product_rating": (
                        overview[
                            "product_rating"
                        ][
                            "mean"
                        ]
                    ),
                    "weighted_product_rating": (
                        overview[
                            "weighted_product_rating"
                        ]
                    ),
                    "rating_count_total": (
                        overview[
                            "rating_count_total"
                        ]
                    ),
                    "weighted_review_rating": (
                        overview[
                            "weighted_review_rating"
                        ]
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


    def top_products(
        self,
        filters=None,
        sort_by="review_count",
        top_n=None,
        min_rating_count=None,
    ):
        frame = self._filter(
            filters
        )

        if len(frame) == 0:
            return frame

        result = frame.copy()

        min_rating_count = int(
            (
                self.min_rating_count_for_leaders
                if min_rating_count
                is None
                else min_rating_count
            )
        )

        if sort_by == "review_count":
            result = result.sort_values(
                [
                    "review_count",
                    "Rate_cnt",
                    "Rate",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )

        elif sort_by == "rating":
            rate_count = (
                pd.to_numeric(
                    result.get(
                        "Rate_cnt",
                        0,
                    ),
                    errors="coerce",
                )
                .fillna(0)
            )

            result = result[
                rate_count
                >= min_rating_count
            ].copy()

            result = result.sort_values(
                [
                    "Rate",
                    "Rate_cnt",
                    "review_count",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )

        elif sort_by == "rating_count":
            rate_count = (
                pd.to_numeric(
                    result.get(
                        "Rate_cnt",
                        0,
                    ),
                    errors="coerce",
                )
                .fillna(0)
            )

            result = result[
                rate_count
                > 0
            ].copy()

            result = result.assign(
                _rating_count=(
                    rate_count[
                        rate_count
                        > 0
                    ]
                )
            )

            result = result.sort_values(
                [
                    "_rating_count",
                    "Rate",
                ],
                ascending=[
                    False,
                    False,
                ],
                na_position="last",
            )

        elif sort_by == "price_low":
            result = result[
                pd.to_numeric(
                    result[
                        "Price"
                    ],
                    errors="coerce",
                )
                > 0
            ].copy()

            result = result.sort_values(
                "Price",
                ascending=True,
            )

        elif sort_by == "price_high":
            result = result[
                pd.to_numeric(
                    result[
                        "Price"
                    ],
                    errors="coerce",
                )
                > 0
            ].copy()

            result = result.sort_values(
                "Price",
                ascending=False,
            )

        else:
            raise ValueError(
                "sort_by must be one of: "
                "review_count, rating, rating_count, "
                "price_low, price_high"
            )

        preferred_columns = [
            "id",
            "title_fa",
            "Brand",
            "Category1",
            "Category2",
            "sub_category",
            "Price",
            "Rate",
            "Rate_cnt",
            "review_count",
            "avg_review_rate",
        ]

        columns = [
            column
            for column
            in preferred_columns
            if column
            in result.columns
        ]

        return (
            result[
                columns
            ]
            .head(
                int(
                    top_n
                    or self.default_top_n
                )
            )
            .reset_index(drop=True)
        )
