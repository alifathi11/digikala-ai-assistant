import pandas as pd


def top_complaints(
    comments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    category: str,
    n_samples: int = 500,
) -> pd.DataFrame:
    cat_product_ids = products_df[
        products_df["category"].str.contains(category, case=False, na=False)
    ]["id"].tolist()

    cat_comments = comments_df[
        (comments_df["product_id"].isin(cat_product_ids)) &
        (comments_df["recommendation_status"] == "not_recommended") &
        (comments_df["body"].str.len() >= 15)
    ]
    return cat_comments.sample(min(n_samples, len(cat_comments)), random_state=42)


def low_recommendation_products(
    comments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    min_comment_count: int = 10,
) -> pd.DataFrame:
    stats = (
        comments_df.groupby("product_id")
        .agg(
            total=("id", "count"),
            not_recommended=("recommendation_status", lambda x: (x == "not_recommended").sum()),
        )
        .reset_index()
    )
    stats["not_rec_ratio"] = stats["not_recommended"] / stats["total"]
    stats = stats[stats["total"] >= min_comment_count]
    merged = stats.merge(products_df[["id", "title", "category", "brand"]], left_on="product_id", right_on="id")
    return merged.sort_values("not_rec_ratio", ascending=False)


def brand_comparison(
    comments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    cat_products = products_df[
        products_df["category"].str.contains(category, case=False, na=False)
    ][["id", "brand"]]

    merged = comments_df.merge(cat_products, left_on="product_id", right_on="id")
    result = (
        merged.groupby("brand")
        .agg(
            avg_rate=("rate", "mean"),
            total_comments=("id", "count"),
            recommended=("recommendation_status", lambda x: (x == "recommended").sum()),
        )
        .reset_index()
    )
    result["rec_ratio"] = result["recommended"] / result["total_comments"]
    return result.sort_values("avg_rate", ascending=False)
