import pandas as pd
from src.config import PRODUCTS_CSV, COMMENTS_CSV


def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV, low_memory=False)


def load_comments() -> pd.DataFrame:
    return pd.read_csv(COMMENTS_CSV, low_memory=False)


def load_comments_for_product(product_id: int) -> pd.DataFrame:
    comments = load_comments()
    return comments[comments["product_id"] == product_id].copy()
