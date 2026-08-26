from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_TEXT_FIELDS = (
    "title_fa",
    "Brand",
    "Category1",
    "Category2",
    "Seller",
)

DEFAULT_NUMERIC_FIELDS = (
    "Price",
    "min_price_last_month",
    "Rate",
    "Rate_cnt",
)


def _json_safe(value):
    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


def _series_top_values(
    series,
    n=20,
):
    cleaned = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned != ""
    ]

    counts = (
        cleaned
        .value_counts()
        .head(n)
    )

    return pd.DataFrame(
        {
            "value": counts.index,
            "count": counts.values,
            "share": (
                counts.values
                / max(
                    len(cleaned),
                    1,
                )
            ),
        }
    )


def _text_field_stats(
    products,
    fields,
):
    rows = []

    total = max(
        len(products),
        1,
    )

    for field in fields:
        if field not in products:
            continue

        series = products[
            field
        ]

        cleaned = (
            series
            .fillna("")
            .astype(str)
            .str.strip()
        )

        non_empty = (
            cleaned != ""
        )

        lengths = (
            cleaned[
                non_empty
            ]
            .str.len()
        )

        rows.append(
            {
                "field": field,
                "non_empty_count": int(
                    non_empty.sum()
                ),
                "coverage": float(
                    non_empty.sum()
                    / total
                ),
                "unique_values": int(
                    cleaned[
                        non_empty
                    ].nunique()
                ),
                "avg_length": (
                    float(
                        lengths.mean()
                    )
                    if len(lengths)
                    else 0.0
                ),
                "p95_length": (
                    float(
                        lengths.quantile(
                            0.95
                        )
                    )
                    if len(lengths)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _numeric_field_stats(
    products,
    fields,
):
    rows = []

    for field in fields:
        if field not in products:
            continue

        values = pd.to_numeric(
            products[field],
            errors="coerce",
        )

        valid = values.dropna()

        rows.append(
            {
                "field": field,
                "valid_count": int(
                    valid.shape[0]
                ),
                "coverage": float(
                    valid.shape[0]
                    / max(
                        len(products),
                        1,
                    )
                ),
                "min": (
                    float(
                        valid.min()
                    )
                    if len(valid)
                    else None
                ),
                "median": (
                    float(
                        valid.median()
                    )
                    if len(valid)
                    else None
                ),
                "mean": (
                    float(
                        valid.mean()
                    )
                    if len(valid)
                    else None
                ),
                "p95": (
                    float(
                        valid.quantile(
                            0.95
                        )
                    )
                    if len(valid)
                    else None
                ),
                "max": (
                    float(
                        valid.max()
                    )
                    if len(valid)
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _comment_counts_from_parquet(
    comments_path,
    batch_size=250_000,
):
    comments_path = Path(
        comments_path
    )

    parquet_file = pq.ParquetFile(
        comments_path
    )

    if (
        "product_id"
        not in parquet_file.schema.names
    ):
        raise ValueError(
            "comments parquet does not "
            "contain product_id"
        )

    counts = Counter()
    total_comments = 0

    for batch in (
        parquet_file
        .iter_batches(
            batch_size=int(
                batch_size
            ),
            columns=[
                "product_id",
            ],
        )
    ):
        frame = (
            batch
            .to_pandas()
        )

        product_ids = (
            pd.to_numeric(
                frame[
                    "product_id"
                ],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
        )

        batch_counts = (
            product_ids
            .value_counts()
        )

        counts.update(
            {
                int(product_id): int(count)
                for product_id, count
                in batch_counts.items()
            }
        )

        total_comments += int(
            len(frame)
        )

    return (
        counts,
        total_comments,
    )


@dataclass
class ProductAuditResult:
    summary: dict
    columns: pd.DataFrame
    text_fields: pd.DataFrame
    numeric_fields: pd.DataFrame
    top_brands: pd.DataFrame
    top_category1: pd.DataFrame
    top_category2: pd.DataFrame
    comment_count_quantiles: pd.DataFrame
    top_commented_products: pd.DataFrame
    search_field_recommendation: pd.DataFrame

    def to_dict(
        self,
    ):
        return {
            "summary": (
                self.summary
            ),
            "columns": (
                self.columns
                .to_dict(
                    orient="records"
                )
            ),
            "text_fields": (
                self.text_fields
                .to_dict(
                    orient="records"
                )
            ),
            "numeric_fields": (
                self.numeric_fields
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
            "comment_count_quantiles": (
                self.comment_count_quantiles
                .to_dict(
                    orient="records"
                )
            ),
            "top_commented_products": (
                self.top_commented_products
                .to_dict(
                    orient="records"
                )
            ),
            "search_field_recommendation": (
                self.search_field_recommendation
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

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _json_safe(
                    self.to_dict()
                ),
                file,
                ensure_ascii=False,
                indent=2,
            )


def audit_product_data(
    products_path,
    comments_path=None,
    comment_batch_size=250_000,
    top_n=20,
):
    products_path = Path(
        products_path
    )

    if not products_path.exists():
        raise FileNotFoundError(
            f"Products parquet not found: "
            f"{products_path}"
        )

    products = (
        pd.read_parquet(
            products_path
        )
        .reset_index(drop=True)
    )

    if "id" not in products:
        raise ValueError(
            "products parquet requires "
            "an id column"
        )

    column_rows = []

    for column in products.columns:
        series = products[
            column
        ]

        column_rows.append(
            {
                "column": column,
                "dtype": str(
                    series.dtype
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_percent": float(
                    series.isna().mean()
                    * 100
                ),
                "unique_values": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    columns = (
        pd.DataFrame(
            column_rows
        )
        .sort_values(
            [
                "missing_percent",
                "column",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    text_fields = (
        _text_field_stats(
            products,
            DEFAULT_TEXT_FIELDS,
        )
    )

    numeric_fields = (
        _numeric_field_stats(
            products,
            DEFAULT_NUMERIC_FIELDS,
        )
    )

    top_brands = (
        _series_top_values(
            products["Brand"],
            n=top_n,
        )
        if "Brand" in products
        else pd.DataFrame()
    )

    top_category1 = (
        _series_top_values(
            products[
                "Category1"
            ],
            n=top_n,
        )
        if "Category1" in products
        else pd.DataFrame()
    )

    top_category2 = (
        _series_top_values(
            products[
                "Category2"
            ],
            n=top_n,
        )
        if "Category2" in products
        else pd.DataFrame()
    )

    product_ids = (
        pd.to_numeric(
            products["id"],
            errors="coerce",
        )
    )

    valid_product_ids = set(
        product_ids
        .dropna()
        .astype("int64")
        .tolist()
    )

    duplicate_id_rows = int(
        product_ids
        .duplicated(
            keep=False
        )
        .sum()
    )

    duplicate_ids = int(
        product_ids[
            product_ids.duplicated(
                keep=False
            )
        ]
        .nunique()
    )

    duplicate_title_rows = None

    if "title_fa" in products:
        titles = (
            products[
                "title_fa"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        non_empty_titles = (
            titles != ""
        )

        duplicate_title_rows = int(
            titles[
                non_empty_titles
            ]
            .duplicated(
                keep=False
            )
            .sum()
        )

    comment_count_quantiles = (
        pd.DataFrame()
    )

    top_commented_products = (
        pd.DataFrame()
    )

    products_with_comments = None
    products_without_comments = None
    comment_product_coverage = None
    orphan_comment_products = None
    orphan_comments = None
    total_comments = None
    median_comments = None

    if comments_path is not None:
        comments_path = Path(
            comments_path
        )

        if not comments_path.exists():
            raise FileNotFoundError(
                f"Comments parquet not found: "
                f"{comments_path}"
            )

        counts, total_comments = (
            _comment_counts_from_parquet(
                comments_path,
                batch_size=(
                    comment_batch_size
                ),
            )
        )

        comment_product_ids = set(
            counts
        )

        matched_ids = (
            valid_product_ids
            & comment_product_ids
        )

        orphan_ids = (
            comment_product_ids
            - valid_product_ids
        )

        products_with_comments = int(
            len(matched_ids)
        )

        products_without_comments = int(
            len(valid_product_ids)
            - products_with_comments
        )

        comment_product_coverage = float(
            products_with_comments
            / max(
                len(valid_product_ids),
                1,
            )
        )

        orphan_comment_products = int(
            len(orphan_ids)
        )

        orphan_comments = int(
            sum(
                counts[
                    product_id
                ]
                for product_id
                in orphan_ids
            )
        )

        matched_counts = pd.Series(
            {
                product_id: counts.get(
                    product_id,
                    0,
                )
                for product_id
                in valid_product_ids
            },
            dtype="int64",
            name="comment_count",
        )

        quantiles = [
            0.00,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ]

        comment_count_quantiles = (
            matched_counts
            .quantile(
                quantiles
            )
            .rename_axis(
                "quantile"
            )
            .reset_index(
                name="comment_count"
            )
        )

        median_comments = float(
            matched_counts.median()
        )

        top_counts = (
            matched_counts
            .sort_values(
                ascending=False
            )
            .head(top_n)
            .rename_axis(
                "id"
            )
            .reset_index()
        )

        display_columns = [
            column
            for column in (
                "id",
                "title_fa",
                "Brand",
                "Category1",
                "Category2",
                "Rate",
                "Rate_cnt",
                "Price",
            )
            if column in products
        ]

        top_commented_products = (
            top_counts
            .merge(
                products[
                    display_columns
                ],
                on="id",
                how="left",
            )
        )

    recommendation_rows = []

    for field in DEFAULT_TEXT_FIELDS:
        if field not in products:
            continue

        row = text_fields[
            text_fields[
                "field"
            ]
            == field
        ]

        coverage = (
            float(
                row.iloc[0][
                    "coverage"
                ]
            )
            if len(row)
            else 0.0
        )

        if field == "title_fa":
            role = (
                "primary_retrieval"
            )
            reason = (
                "Primary identity and "
                "semantic product text."
            )
        elif field == "Brand":
            role = (
                "boost_and_filter"
            )
            reason = (
                "Useful for exact brand "
                "intent and filtering."
            )
        elif field in (
            "Category1",
            "Category2",
        ):
            role = (
                "filter_and_context"
            )
            reason = (
                "Useful for category-aware "
                "retrieval and filters."
            )
        else:
            role = (
                "optional_filter"
            )
            reason = (
                "Potential filter; avoid "
                "overweighting seller text "
                "in semantic ranking."
            )

        recommendation_rows.append(
            {
                "field": field,
                "coverage": coverage,
                "suggested_role": role,
                "reason": reason,
            }
        )

    search_field_recommendation = (
        pd.DataFrame(
            recommendation_rows
        )
    )

    summary = {
        "product_rows": int(
            len(products)
        ),
        "product_columns": int(
            len(products.columns)
        ),
        "unique_product_ids": int(
            len(valid_product_ids)
        ),
        "duplicate_product_id_rows": (
            duplicate_id_rows
        ),
        "duplicate_product_ids": (
            duplicate_ids
        ),
        "duplicate_title_rows": (
            duplicate_title_rows
        ),
        "total_comments": (
            total_comments
        ),
        "products_with_comments": (
            products_with_comments
        ),
        "products_without_comments": (
            products_without_comments
        ),
        "comment_product_coverage": (
            comment_product_coverage
        ),
        "orphan_comment_products": (
            orphan_comment_products
        ),
        "orphan_comments": (
            orphan_comments
        ),
        "median_comments_per_product": (
            median_comments
        ),
    }

    return ProductAuditResult(
        summary=summary,
        columns=columns,
        text_fields=text_fields,
        numeric_fields=numeric_fields,
        top_brands=top_brands,
        top_category1=top_category1,
        top_category2=top_category2,
        comment_count_quantiles=(
            comment_count_quantiles
        ),
        top_commented_products=(
            top_commented_products
        ),
        search_field_recommendation=(
            search_field_recommendation
        ),
    )
