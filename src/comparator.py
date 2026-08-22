import pandas as pd
from src.pipeline import QAPipeline


def compare_products(
    product_ids: list[int],
    query: str,
    products_df: pd.DataFrame,
) -> dict:
    results = {}
    for pid in product_ids:
        pipeline = QAPipeline(pid)
        answer, contexts = pipeline.run(query, return_contexts=True)
        product_info = (
            products_df[products_df["id"] == pid].iloc[0].to_dict()
            if pid in products_df["id"].values else {}
        )
        results[pid] = {
            "title": product_info.get("title_fa", str(pid)),
            "answer": answer,
            "evidence": [
                {"comment_id": c["id"], "body": c["body"], "rate": c["rate"]}
                for c in contexts
            ],
        }
    return results
