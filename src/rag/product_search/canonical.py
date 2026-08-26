from pathlib import Path

import numpy as np
import pandas as pd


SEARCH_COLUMNS = (
    "id",
    "title_fa",
    "Brand",
    "Category1",
    "Category2",
    "sub_category",
    "Price",
    "min_price_last_month",
    "Rate",
    "Rate_cnt",
    "Is_Fake",
)


def _clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(value).strip()


def build_product_search_text(
    row,
):
    """
    Build one product-level retrieval document.

    Title is intentionally repeated to give it more lexical weight without
    relying on backend-specific field-boost APIs.
    """
    title = _clean_text(
        row.get(
            "title_fa"
        )
    )

    brand = _clean_text(
        row.get(
            "Brand"
        )
    )

    category1 = _clean_text(
        row.get(
            "Category1"
        )
    )

    category2 = _clean_text(
        row.get(
            "Category2"
        )
    )

    if category2.lower() == "unknown":
        category2 = ""

    sub_category = _clean_text(
        row.get(
            "sub_category"
        )
    )

    parts = [
        title,
        title,
        brand,
        category1,
        category2,
        sub_category,
    ]

    return " ".join(
        part
        for part in parts
        if part
    )


def canonicalize_products(
    products,
):
    """
    Collapse seller-level duplicate rows into one canonical row per product ID.

    Strategy:
    - representative metadata row: highest Rate_cnt, then lowest Price
    - Price: minimum observed current price
    - min_price_last_month: minimum observed non-null value
    - seller_count: number of distinct sellers
    - row_count: number of raw rows collapsed into this product

    This avoids duplicated products in retrieval results.
    """
    products = (
        products
        .copy()
        .reset_index(drop=True)
    )

    if "id" not in products:
        raise ValueError(
            "products requires an id column"
        )

    products["id"] = pd.to_numeric(
        products["id"],
        errors="coerce",
    )

    products = products[
        products["id"].notna()
    ].copy()

    products["id"] = (
        products["id"]
        .astype("int64")
    )

    rate_cnt = pd.to_numeric(
        products.get(
            "Rate_cnt",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    price = pd.to_numeric(
        products.get(
            "Price",
            np.nan,
        ),
        errors="coerce",
    )

    products[
        "_sort_rate_cnt"
    ] = rate_cnt

    products[
        "_sort_price"
    ] = price.fillna(
        np.inf
    )

    representative = (
        products
        .sort_values(
            [
                "id",
                "_sort_rate_cnt",
                "_sort_price",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "id",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_sort_rate_cnt",
                "_sort_price",
            ],
        )
        .set_index(
            "id"
        )
    )

    grouped = products.groupby(
        "id",
        sort=False,
    )

    row_count = grouped.size().rename(
        "source_row_count"
    )

    if "Seller" in products:
        seller_count = (
            grouped["Seller"]
            .nunique(
                dropna=True
            )
            .rename(
                "seller_count"
            )
        )
    else:
        seller_count = (
            row_count
            .copy()
            .rename(
                "seller_count"
            )
        )

    aggregates = [
        row_count,
        seller_count,
    ]

    if "Price" in products:
        min_price = (
            pd.to_numeric(
                products["Price"],
                errors="coerce",
            )
            .groupby(
                products["id"]
            )
            .min()
            .rename(
                "Price"
            )
        )

        representative[
            "Price"
        ] = min_price

    if (
        "min_price_last_month"
        in products
    ):
        min_last_month = (
            pd.to_numeric(
                products[
                    "min_price_last_month"
                ],
                errors="coerce",
            )
            .groupby(
                products["id"]
            )
            .min()
        )

        representative[
            "min_price_last_month"
        ] = min_last_month

    representative = (
        representative
        .join(
            aggregates,
            how="left",
        )
        .reset_index()
    )

    representative[
        "search_text"
    ] = (
        representative
        .apply(
            build_product_search_text,
            axis=1,
        )
    )

    return (
        representative
        .sort_values(
            "id"
        )
        .reset_index(drop=True)
    )


def build_canonical_products_file(
    input_path,
    output_path,
    overwrite=False,
):
    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} exists. "
                "Use overwrite=True."
            )

        output_path.unlink()

    products = pd.read_parquet(
        input_path
    )

    canonical = (
        canonicalize_products(
            products
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical.to_parquet(
        output_path,
        index=False,
    )

    return canonical
