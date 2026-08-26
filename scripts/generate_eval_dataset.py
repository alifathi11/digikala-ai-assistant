import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)

import json
import os

import pandas as pd
from dotenv import load_dotenv

from src.rag.evaluation.openai_generator import OpenAIGenerator
from src.rag.evaluation.prompt import build_eval_prompt
from src.rag.evaluation.validator import validate_result
from src.rag.utils.text import build_comment_text


load_dotenv()


COMMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comments_clean.parquet"
)

PRODUCTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_clean.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_queries.json"
)


NUM_PRODUCTS = 50
QUERIES_PER_PRODUCT = 3
COMMENTS_PER_PRODUCT = 20
RANDOM_STATE = 42


def save_dataset(dataset):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    comments = pd.read_parquet(
        COMMENTS_PATH
    )

    products = pd.read_parquet(
        PRODUCTS_PATH
    )

    # Use the same text fields that are indexed:
    # title + body + advantages + disadvantages.
    comments = comments.copy()

    comments["eval_text"] = (
        comments
        .apply(
            build_comment_text,
            axis=1
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    comments = comments[
        comments["eval_text"].str.len() > 0
    ].copy()

    comment_counts = (
        comments
        .groupby("product_id")
        .size()
        .rename("comment_count")
    )

    eligible_products = (
        products[
            [
                "id",
                "title_fa",
            ]
        ]
        .rename(
            columns={
                "id": "product_id",
                "title_fa": "product_title",
            }
        )
        .merge(
            comment_counts,
            on="product_id",
            how="inner"
        )
    )

    eligible_products = eligible_products[
        eligible_products["comment_count"]
        >= COMMENTS_PER_PRODUCT
    ].copy()

    # Shuffle eligible products instead of taking the 50 most-commented
    # products, which can strongly bias the benchmark toward one category.
    eligible_products = (
        eligible_products
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    generator = OpenAIGenerator(
        api_key=os.getenv(
            "METIS_API_KEY"
        ),
        base_url=os.getenv(
            "METIS_BASE_URL"
        ),
        model="gpt-5.6-terra"
    )

    dataset = []
    successful_products = 0
    api_calls = 0

    for row in eligible_products.itertuples(
        index=False
    ):

        if successful_products >= NUM_PRODUCTS:
            break

        product_id = int(
            row.product_id
        )

        product_title = str(
            row.product_title
        )

        product_comments = (
            comments[
                comments["product_id"]
                == product_id
            ]
            .sample(
                n=COMMENTS_PER_PRODUCT,
                random_state=(
                    RANDOM_STATE
                    + product_id
                )
            )
            .copy()
        )

        comment_list = [
            {
                "id": int(r.id),
                "text": str(r.eval_text),
            }
            for r in product_comments.itertuples(
                index=False
            )
        ]

        candidate_ids = [
            c["id"]
            for c in comment_list
        ]

        prompt = build_eval_prompt(
            product_title,
            comment_list
        )

        try:
            api_calls += 1

            result = generator.generate(
                prompt
            )

        except Exception as exc:
            print(
                f"API failure for product "
                f"{product_id}: {exc}"
            )
            continue

        if result is None:
            print(
                f"Empty result for product "
                f"{product_id}"
            )
            continue

        is_valid = validate_result(
            result,
            candidate_ids,
            expected_queries=(
                QUERIES_PER_PRODUCT
            )
        )

        if not is_valid:
            print(
                f"Invalid result for product "
                f"{product_id}: {result}"
            )
            continue

        for query_item in result["queries"]:

            dataset.append(
                {
                    "product_id": product_id,
                    "product_title": (
                        product_title
                    ),
                    "query": (
                        query_item["query"]
                        .strip()
                    ),
                    "candidate_ids": (
                        candidate_ids
                    ),
                    "relevant_ids": (
                        query_item[
                            "relevant_ids"
                        ]
                    ),
                }
            )

        successful_products += 1

        # Checkpoint after every successful product so an interruption
        # does not lose previous API generations.
        save_dataset(
            dataset
        )

        print(
            f"Products: "
            f"{successful_products}"
            f"/{NUM_PRODUCTS} | "
            f"Samples: {len(dataset)} | "
            f"API calls: {api_calls}"
        )

    save_dataset(
        dataset
    )

    expected_samples = (
        NUM_PRODUCTS
        * QUERIES_PER_PRODUCT
    )

    print()
    print(
        f"Saved {len(dataset)} samples "
        f"from {successful_products} products"
    )
    print(
        f"Expected samples: "
        f"{expected_samples}"
    )
    print(
        f"API calls: {api_calls}"
    )

    if len(dataset) != expected_samples:
        print(
            "WARNING: benchmark is incomplete."
        )


if __name__ == "__main__":
    main()
