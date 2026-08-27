from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import get_model_pricing, load_project_config
from ..generation import OpenAIJSONGenerator
from ..pipeline import ProductComparisonPipeline
from ..runtime import load_retrieval_stack
from .comparison_judge import ProductComparisonJudge


@dataclass
class ComparisonEvaluationContext:
    comparison: object
    judge: object
    product_documents: object
    evaluation_config: dict


def load_comparison_evaluation_context(
    project_root,
    api_key,
    base_url,
):
    project_root = Path(
        project_root
    )

    config = load_project_config(
        project_root
    )

    comparison_config = config[
        "comparison"
    ]

    evaluation_config = comparison_config[
        "evaluation"
    ]

    retrieval = load_retrieval_stack(
        project_root=project_root,
        rag_config=config,
    )

    product_documents = pd.read_parquet(
        project_root
        / "data"
        / "processed"
        / "products_search.parquet"
    )

    generation_config = config[
        "generation"
    ]

    generation_pricing = get_model_pricing(
        config,
        generation_config[
            "model"
        ],
    )

    comparison_generator = OpenAIJSONGenerator(
        api_key=api_key,
        base_url=base_url,
        model=generation_config[
            "model"
        ],
        input_cost_per_million=(
            generation_pricing[
                "input_cost_per_million"
            ]
        ),
        output_cost_per_million=(
            generation_pricing[
                "output_cost_per_million"
            ]
        ),
    )

    judge_config = evaluation_config[
        "judge"
    ]

    judge_model = (
        judge_config.get(
            "model"
        )
        or generation_config[
            "model"
        ]
    )

    judge_pricing = get_model_pricing(
        config,
        judge_model,
    )

    judge_generator = OpenAIJSONGenerator(
        api_key=api_key,
        base_url=base_url,
        model=judge_model,
        input_cost_per_million=(
            judge_pricing[
                "input_cost_per_million"
            ]
        ),
        output_cost_per_million=(
            judge_pricing[
                "output_cost_per_million"
            ]
        ),
    )

    comparison = ProductComparisonPipeline(
        product_documents=product_documents,
        review_retriever=retrieval.hybrid,
        review_documents=retrieval.documents,
        generator=comparison_generator,
        reviews_per_product=(
            comparison_config[
                "reviews_per_product"
            ]
        ),
        min_products=(
            comparison_config[
                "min_products"
            ]
        ),
        max_products=(
            comparison_config[
                "max_products"
            ]
        ),
        max_context_chars=(
            comparison_config[
                "max_context_chars"
            ]
        ),
        max_chars_per_review=(
            comparison_config[
                "max_chars_per_review"
            ]
        ),
    )

    judge = ProductComparisonJudge(
        generator=judge_generator,
        max_context_chars=(
            judge_config.get(
                "max_context_chars",
                18_000,
            )
        ),
        max_chars_per_review=(
            judge_config.get(
                "max_chars_per_review",
                900,
            )
        ),
    )

    return ComparisonEvaluationContext(
        comparison=comparison,
        judge=judge,
        product_documents=(
            product_documents
        ),
        evaluation_config=(
            evaluation_config
        ),
    )
