import pandas as pd


def low_recommendation_products(comments_df, products_df, min_comments=10, top_n=20):
    grouped = comments_df.groupby("product_id").agg(
        total_comments=("recommendation_status", "count"),
        rec_rate=("recommendation_status", lambda x: (x == "recommended").mean()),
    ).reset_index()

    grouped = grouped[grouped["total_comments"] >= min_comments]
    grouped = grouped.sort_values("rec_rate")

    merged = grouped.merge(
        products_df[["id", "title", "brand", "category", "price"]],
        left_on="product_id",
        right_on="id",
        how="left",
    )

    return merged[["product_id", "title", "brand", "category", "total_comments", "rec_rate"]].head(top_n)


def brand_comparison(comments_df, products_df, category):
    cat_products = products_df[products_df["category"] == category]

    merged = comments_df.merge(
        cat_products[["id", "brand"]],
        left_on="product_id",
        right_on="id",
        how="inner",
    )

    stats = merged.groupby("brand").agg(
        avg_rate=("rate", "mean"),
        total_comments=("rate", "count"),
        rec_rate=("recommendation_status", lambda x: (x == "recommended").mean()),
    ).reset_index()

    return stats.sort_values("total_comments", ascending=False)


def top_complaints(comments_df, products_df, category, min_length=15, n=20):
    cat_products = products_df[products_df["category"] == category]

    merged = comments_df.merge(
        cat_products[["id", "title", "brand"]],
        left_on="product_id",
        right_on="id",
        how="inner",
    )

    complaints = merged[
        (merged["recommendation_status"] == "not_recommended")
        & (merged["body"].str.len() >= min_length)
    ]

    return complaints[["product_id", "title", "brand", "body", "rate"]].sample(
        n=min(n, len(complaints))
    )
