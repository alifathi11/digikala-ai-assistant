import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)

import json
import pandas as pd

import os
from dotenv import load_dotenv

from src.rag.evaluation.openai_generator import OpenAIGenerator

from src.rag.evaluation.prompt import build_eval_prompt
from src.rag.evaluation.validator import validate_result

load_dotenv()

COMMENTS_PATH = (
    "data/processed/comments_clean.parquet"
)

OUTPUT_PATH = (
    "data/evaluation/retrieval_queries.json"
)


NUM_PRODUCTS = 10



def main():

    comments = pd.read_parquet(
        COMMENTS_PATH
    )


    products = (
        comments
        .groupby("product_id")
        .size()
        .sort_values(
            ascending=False
        )
        .head(NUM_PRODUCTS)
        .index
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


    for product_id in products:


        product_comments = (
            comments[
                comments.product_id == product_id
            ]
            .head(20)
        )


        comment_list = []


        for _, row in product_comments.iterrows():

            comment_list.append(
                {
                    "id": int(row["id"]),
                    "text": str(row["body"])
                }
            )

        title = str(
            product_comments.iloc[0]["title"]
        )


        prompt = build_eval_prompt(
            title,
            comment_list
        )


        result = generator.generate(
            prompt
        )

        if result is None:
            print(
                f"Skipping product {product_id}"
            )
            continue

        valid_ids = [
            c["id"]
            for c in comment_list
        ]


        is_valid = validate_result(
            result,
            valid_ids
        )


        print("\nRESULT:")
        print(result)

        print("\nVALID IDS:")
        print(valid_ids)

        print("\nVALID:")
        print(is_valid)


        if is_valid:

            dataset.append(
                {
                    "product_id": int(product_id),
                    "query": result["query"],
                    "relevant_ids": result["relevant_ids"]
                }
            )


    Path(
        OUTPUT_PATH
    ).parent.mkdir(
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


    print(
        f"Saved {len(dataset)} samples"
    )



if __name__ == "__main__":
    main()