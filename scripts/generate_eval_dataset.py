import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )

from src.rag.config import (
    get_model_pricing,
    load_project_config,
)
from src.rag.evaluation.dataset_generation import (
    EVAL_DATASET_SYSTEM_PROMPT,
    build_eval_prompt,
    validate_generated_queries,
)
from src.rag.generation import (
    OpenAIJSONGenerator,
)
from src.rag.utils.text import (
    build_comment_text,
)


load_dotenv(
    PROJECT_ROOT
    / ".env"
)

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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_query_generation_summary.json"
)

NUM_PRODUCTS = 50
QUERIES_PER_PRODUCT = 3
COMMENTS_PER_PRODUCT = 20
RANDOM_STATE = 42


def save_dataset(
    dataset
):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_summary(
    summary
):
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_existing_dataset():
    if not OUTPUT_PATH.exists():
        return [], set()

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        existing = json.load(
            file
        )

    if not isinstance(
        existing,
        list,
    ):
        raise ValueError(
            "Existing retrieval benchmark must be a JSON list."
        )

    product_counts = Counter(
        int(
            item["product_id"]
        )
        for item in existing
        if isinstance(
            item,
            dict,
        )
        and "product_id" in item
    )

    completed_product_ids = {
        product_id
        for product_id, count
        in product_counts.items()
        if count
        == QUERIES_PER_PRODUCT
    }

    cleaned = [
        item
        for item in existing
        if int(
            item["product_id"]
        )
        in completed_product_ids
    ]

    if len(
        cleaned
    ) != len(
        existing
    ):
        save_dataset(
            cleaned
        )

    return (
        cleaned,
        completed_product_ids,
    )


def main():
    if not COMMENTS_PATH.exists():
        raise FileNotFoundError(
            "Missing processed comments. Run Notebook 02 first: "
            f"{COMMENTS_PATH}"
        )

    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(
            "Missing processed products. Run Notebook 02 first: "
            f"{PRODUCTS_PATH}"
        )

    api_key = os.getenv(
        "METIS_API_KEY"
    )

    base_url = os.getenv(
        "METIS_BASE_URL"
    )

    if not api_key:
        raise RuntimeError(
            "METIS_API_KEY is missing from the environment or .env file."
        )

    if not base_url:
        raise RuntimeError(
            "METIS_BASE_URL is missing from the environment or .env file."
        )

    comments = pd.read_parquet(
        COMMENTS_PATH
    )

    products = pd.read_parquet(
        PRODUCTS_PATH
    )

    comments = comments.copy()

    comments["eval_text"] = (
        comments
        .apply(
            build_comment_text,
            axis=1,
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    comments = comments[
        comments[
            "eval_text"
        ].str.len()
        > 0
    ].copy()

    comment_counts = (
        comments
        .groupby(
            "product_id"
        )
        .size()
        .rename(
            "comment_count"
        )
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
                "title_fa": (
                    "product_title"
                ),
            }
        )
        .merge(
            comment_counts,
            on="product_id",
            how="inner",
        )
    )

    eligible_products = (
        eligible_products[
            eligible_products[
                "comment_count"
            ]
            >= COMMENTS_PER_PRODUCT
        ]
        .sample(
            frac=1.0,
            random_state=(
                RANDOM_STATE
            ),
        )
        .reset_index(
            drop=True
        )
    )

    config = load_project_config(
        PROJECT_ROOT
    )

    generation_config = config[
        "generation"
    ]

    pricing = get_model_pricing(
        config,
        generation_config[
            "model"
        ],
    )

    generator = (
        OpenAIJSONGenerator(
            api_key=api_key,
            base_url=base_url,
            model=generation_config[
                "model"
            ],
            input_cost_per_million=(
                pricing[
                    "input_cost_per_million"
                ]
            ),
            output_cost_per_million=(
                pricing[
                    "output_cost_per_million"
                ]
            ),
        )
    )

    dataset, completed_product_ids = (
        load_existing_dataset()
    )

    successful_products = len(
        completed_product_ids
    )

    api_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0

    if successful_products:
        print(
            "Resuming retrieval benchmark: "
            f"{successful_products}/{NUM_PRODUCTS} products already complete."
        )

    for row in (
        eligible_products
        .itertuples(
            index=False
        )
    ):
        if (
            successful_products
            >= NUM_PRODUCTS
        ):
            break

        product_id = int(
            row.product_id
        )

        if product_id in (
            completed_product_ids
        ):
            continue

        product_title = str(
            row.product_title
        )

        product_comments = (
            comments[
                comments[
                    "product_id"
                ]
                == product_id
            ]
            .sample(
                n=(
                    COMMENTS_PER_PRODUCT
                ),
                random_state=(
                    RANDOM_STATE
                    + product_id
                ),
            )
            .copy()
        )

        comment_list = [
            {
                "id": int(
                    record.id
                ),
                "text": str(
                    record.eval_text
                ),
            }
            for record
            in product_comments
            .itertuples(
                index=False
            )
        ]

        candidate_ids = [
            comment["id"]
            for comment
            in comment_list
        ]

        user_prompt = (
            build_eval_prompt(
                product_title,
                comment_list,
            )
        )

        try:
            api_calls += 1

            generation = (
                generator.generate(
                    system_prompt=(
                        EVAL_DATASET_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        user_prompt
                    ),
                )
            )

            prompt_tokens += int(
                generation.get(
                    "prompt_tokens",
                    0,
                )
            )

            completion_tokens += int(
                generation.get(
                    "completion_tokens",
                    0,
                )
            )

            total_tokens += int(
                generation.get(
                    "total_tokens",
                    0,
                )
            )

            call_cost = generation.get(
                "estimated_cost_usd"
            )

            if call_cost is not None:
                estimated_cost_usd += float(
                    call_cost
                )

            result = generation[
                "payload"
            ]

        except Exception as exc:
            print(
                "API failure for "
                f"product {product_id}: "
                f"{exc}"
            )
            continue

        if not (
            validate_generated_queries(
                result,
                candidate_ids,
                expected_queries=(
                    QUERIES_PER_PRODUCT
                ),
            )
        ):
            print(
                "Invalid result for "
                f"product {product_id}: "
                f"{result}"
            )
            continue

        for query_item in (
            result[
                "queries"
            ]
        ):
            dataset.append(
                {
                    "product_id": (
                        product_id
                    ),
                    "product_title": (
                        product_title
                    ),
                    "query": (
                        query_item[
                            "query"
                        ].strip()
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

        completed_product_ids.add(
            product_id
        )

        successful_products += 1

        save_dataset(
            dataset
        )

        save_summary(
            {
                "model": generation_config[
                    "model"
                ],
                "successful_products": (
                    successful_products
                ),
                "target_products": (
                    NUM_PRODUCTS
                ),
                "samples": len(
                    dataset
                ),
                "api_calls_this_run": (
                    api_calls
                ),
                "prompt_tokens_this_run": (
                    prompt_tokens
                ),
                "completion_tokens_this_run": (
                    completion_tokens
                ),
                "total_tokens_this_run": (
                    total_tokens
                ),
                "estimated_cost_usd_this_run": (
                    estimated_cost_usd
                ),
            }
        )

        print(
            f"Products: "
            f"{successful_products}"
            f"/{NUM_PRODUCTS} | "
            f"Samples: {len(dataset)} | "
            f"API calls this run: {api_calls} | "
            f"Cost this run: ${estimated_cost_usd:.4f}"
        )

    save_dataset(
        dataset
    )

    expected_samples = (
        NUM_PRODUCTS
        * QUERIES_PER_PRODUCT
    )

    summary = {
        "model": generation_config[
            "model"
        ],
        "successful_products": (
            successful_products
        ),
        "target_products": (
            NUM_PRODUCTS
        ),
        "samples": len(
            dataset
        ),
        "expected_samples": (
            expected_samples
        ),
        "api_calls_this_run": (
            api_calls
        ),
        "prompt_tokens_this_run": (
            prompt_tokens
        ),
        "completion_tokens_this_run": (
            completion_tokens
        ),
        "total_tokens_this_run": (
            total_tokens
        ),
        "estimated_cost_usd_this_run": (
            estimated_cost_usd
        ),
        "complete": (
            len(
                dataset
            )
            == expected_samples
        ),
    }

    save_summary(
        summary
    )

    print()
    print(
        f"Saved {len(dataset)} "
        "samples from "
        f"{successful_products} "
        "products"
    )
    print(
        "Expected samples: "
        f"{expected_samples}"
    )
    print(
        f"API calls this run: "
        f"{api_calls}"
    )
    print(
        "Prompt tokens this run: "
        f"{prompt_tokens:,}"
    )
    print(
        "Completion tokens this run: "
        f"{completion_tokens:,}"
    )
    print(
        "Total tokens this run: "
        f"{total_tokens:,}"
    )
    print(
        "Estimated API cost this run: "
        f"${estimated_cost_usd:.4f}"
    )
    print(
        "Summary: "
        f"{SUMMARY_PATH}"
    )

    if (
        len(dataset)
        != expected_samples
    ):
        raise RuntimeError(
            "Retrieval benchmark is incomplete. "
            "Re-run the script to resume from completed products."
        )


if __name__ == "__main__":
    main()
