from dataclasses import dataclass
from pathlib import Path

from src.rag.config import (
    load_config,
)
from src.rag.generation import (
    OpenAIJSONGenerator,
)
from src.rag.pipeline import (
    GroundedQAPipeline,
)
from src.rag.runtime import (
    RetrievalStack,
    load_retrieval_stack,
)

from .catalog import (
    ProductCatalog,
)


@dataclass
class AppServices:
    retrieval: RetrievalStack
    qa: GroundedQAPipeline
    catalog: ProductCatalog


def create_app_services(
    project_root,
    api_key,
    base_url,
):
    project_root = Path(
        project_root
    )

    rag_config = load_config(
        project_root
        / "configs"
        / "rag.yaml"
    )

    qa_config = load_config(
        project_root
        / "configs"
        / "qa.yaml"
    )

    retrieval = (
        load_retrieval_stack(
            project_root=(
                project_root
            ),
            rag_config=(
                rag_config
            ),
        )
    )

    generation_config = (
        qa_config[
            "generation"
        ]
    )

    generator = (
        OpenAIJSONGenerator(
            api_key=api_key,
            base_url=base_url,
            model=(
                generation_config[
                    "model"
                ]
            ),
            input_cost_per_million=(
                generation_config.get(
                    "input_cost_per_million"
                )
            ),
            output_cost_per_million=(
                generation_config.get(
                    "output_cost_per_million"
                )
            ),
        )
    )

    qa_config_values = (
        qa_config[
            "qa"
        ]
    )

    qa = GroundedQAPipeline(
        retriever=(
            retrieval.hybrid
        ),
        generator=generator,
        documents=(
            retrieval.documents
        ),
        top_k=(
            qa_config_values[
                "top_k"
            ]
        ),
        max_context_chars=(
            qa_config_values[
                "max_context_chars"
            ]
        ),
        max_chars_per_comment=(
            qa_config_values[
                "max_chars_per_comment"
            ]
        ),
    )

    catalog = (
        ProductCatalog
        .from_parquet(
            project_root
            / "data"
            / "processed"
            / "products_clean.parquet",
            allowed_product_ids=(
                retrieval.documents[
                    "product_id"
                ]
                .dropna()
                .astype(int)
                .unique()
            ),
        )
    )

    return AppServices(
        retrieval=retrieval,
        qa=qa,
        catalog=catalog,
    )
