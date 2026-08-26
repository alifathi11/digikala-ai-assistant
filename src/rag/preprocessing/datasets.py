from pathlib import Path

import jdatetime
import numpy as np
import pandas as pd

from .processor import TextProcessor


JALALI_MONTHS = {
    "فروردین": 1,
    "اردیبهشت": 2,
    "خرداد": 3,
    "تیر": 4,
    "مرداد": 5,
    "شهریور": 6,
    "مهر": 7,
    "آبان": 8,
    "آذر": 9,
    "دی": 10,
    "بهمن": 11,
    "اسفند": 12,
}


def jalali_to_gregorian(value):
    if value is None:
        return pd.NaT

    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass

    try:
        day, month_name, year = str(value).split()
        gregorian = jdatetime.date(
            int(year),
            JALALI_MONTHS[month_name],
            int(day),
        ).togregorian()
        return pd.Timestamp(gregorian)
    except (ValueError, KeyError):
        return pd.NaT


def _normalize_columns(frame, columns):
    processor = TextProcessor()

    for column in columns:
        if column not in frame.columns:
            continue

        frame[column] = frame[column].map(
            lambda value: (
                processor.process(value)
                if pd.notna(value)
                else value
            )
        )

    return frame


def clean_products(products):
    products = (
        products
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    if "Category2" in products:
        products["Category2"] = (
            products["Category2"]
            .fillna("Unknown")
        )

    if "Seller" in products:
        products["Seller"] = (
            products["Seller"]
            .fillna("Unknown")
        )

    for column in (
        "Price",
        "min_price_last_month",
    ):
        if column in products:
            products[column] = (
                products[column]
                .replace(0, np.nan)
            )

    products = _normalize_columns(
        products,
        [
            "title_fa",
            "Brand",
            "Seller",
        ],
    )

    return products


def clean_comments(comments):
    comments = (
        comments
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    if "id" in comments:
        likes = (
            comments["likes"].fillna(0)
            if "likes" in comments
            else 0
        )

        dislikes = (
            comments["dislikes"].fillna(0)
            if "dislikes" in comments
            else 0
        )

        comments["_engagement"] = (
            likes + dislikes
        )

        comments = (
            comments
            .sort_values(
                "_engagement",
                ascending=False,
            )
            .drop_duplicates(
                subset="id",
                keep="first",
            )
            .drop(
                columns="_engagement",
            )
            .reset_index(drop=True)
        )

    comments = _normalize_columns(
        comments,
        [
            "body",
            "title",
            "advantages",
            "disadvantages",
            "seller_title",
        ],
    )

    if "rate" in comments:
        comments.loc[
            comments["rate"] > 5,
            "rate",
        ] = np.nan

    if "created_at" in comments:
        unique_dates = (
            comments["created_at"]
            .drop_duplicates()
        )

        mapping = {
            value: jalali_to_gregorian(
                value
            )
            for value in unique_dates
        }

        comments[
            "created_at_gregorian"
        ] = pd.to_datetime(
            comments[
                "created_at"
            ].map(mapping),
            errors="coerce",
        )

    return comments


def preprocess_datasets(
    products_path,
    comments_path,
    output_dir,
):
    products_path = Path(
        products_path
    )

    comments_path = Path(
        comments_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    products = pd.read_csv(
        products_path,
        low_memory=False,
    )

    comments = pd.read_csv(
        comments_path,
        low_memory=False,
    )

    products = clean_products(
        products
    )

    comments = clean_comments(
        comments
    )

    products_output = (
        output_dir
        / "products_clean.parquet"
    )

    comments_output = (
        output_dir
        / "comments_clean.parquet"
    )

    products.to_parquet(
        products_output,
        index=False,
    )

    comments.to_parquet(
        comments_output,
        index=False,
    )

    return {
        "products": products,
        "comments": comments,
        "products_path": (
            products_output
        ),
        "comments_path": (
            comments_output
        ),
    }
