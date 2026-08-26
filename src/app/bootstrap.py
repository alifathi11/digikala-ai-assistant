from dataclasses import dataclass
import inspect
from pathlib import Path

from src.rag.config import (
    load_config,
)
from src.rag.generation import (
    OpenAIJSONGenerator,
)
from src.rag.pipeline import (
    GroundedQAPipeline,
    ProductSearchPipeline,
)
from src.rag.product_search import (
    ProductBM25Index,
    ProductFAISSIndex,
    ProductMetadataRetriever,
    ProductSearchReranker,
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
    product_search: ProductSearchPipeline


def _load_product_search(
    project_root,
    retrieval,
    generator,
):
    search_config = load_config(
        project_root
        / "configs"
        / "product_search.yaml"
    )

    product_search_config = (
        search_config[
            "product_search"
        ]
    )

    metadata_config = (
        product_search_config[
            "metadata"
        ]
    )

    candidate_config = (
        product_search_config[
            "candidates"
        ]
    )

    reranker_config = (
        product_search_config[
            "reranker"
        ]
    )

    dense_index = (
        ProductFAISSIndex()
        .load(
            project_root
            / "data"
            / "indexes"
            / "products_embedding"
        )
    )

    sparse_index = (
        ProductBM25Index(
            processor=(
                retrieval.processor
            )
        )
        .load(
            project_root
            / "data"
            / "indexes"
            / "products_bm25_tantivy"
        )
    )

    metadata_retriever = (
        ProductMetadataRetriever(
            embedding_model=(
                retrieval.embedding_model
            ),
            dense_index=dense_index,
            sparse_index=sparse_index,
            processor=(
                retrieval.processor
            ),
            bm25_weight=(
                metadata_config[
                    "bm25_weight"
                ]
            ),
            embedding_weight=(
                metadata_config[
                    "embedding_weight"
                ]
            ),
            candidate_multiplier=(
                metadata_config[
                    "candidate_multiplier"
                ]
            ),
            brand_boost=(
                metadata_config[
                    "brand_boost"
                ]
            ),
            lexical_weight=(
                metadata_config.get(
                    "lexical_weight",
                    0.30,
                )
            ),
            rrf_k=(
                metadata_config.get(
                    "rrf_k",
                    60,
                )
            ),
            validate_index_alignment=(
                metadata_config.get(
                    "validate_index_alignment",
                    True,
                )
            ),
        )
    )

    reranker = None

    if reranker_config.get(
        "enabled",
        True,
    ):
        reranker = (
            ProductSearchReranker(
                generator=generator,
                max_reviews_per_product=(
                    reranker_config[
                        "max_reviews_per_product"
                    ]
                ),
                max_review_chars=(
                    reranker_config[
                        "max_review_chars"
                    ]
                ),
            )
        )

    pipeline_kwargs = {
        "metadata_retriever": (
            metadata_retriever
        ),
        "review_retriever": (
            retrieval.hybrid
        ),
        "reranker": reranker,
        "metadata_candidates": (
            candidate_config[
                "metadata_products"
            ]
        ),
        "review_comment_candidates": (
            candidate_config.get(
                "review_comments",
                250,
            )
        ),
        "reranker_candidates": (
            candidate_config[
                "reranker_products"
            ]
        ),
        "metadata_weight": (
            reranker_config[
                "metadata_weight"
            ]
        ),
        "reranker_weight": (
            reranker_config[
                "llm_weight"
            ]
        ),
    }

    # Supports both:
    # - exact pre-BERT LLM pipeline;
    # - the later candidate-scoped review-evidence improvement.
    pipeline_signature = (
        inspect.signature(
            ProductSearchPipeline.__init__
        )
    )

    if (
        "review_comments_per_product"
        in pipeline_signature.parameters
    ):
        pipeline_kwargs[
            "review_comments_per_product"
        ] = candidate_config.get(
            "review_comments_per_product",
            reranker_config.get(
                "max_reviews_per_product",
                2,
            ),
        )

    return ProductSearchPipeline(
        **pipeline_kwargs
    )


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

    product_search = (
        _load_product_search(
            project_root=(
                project_root
            ),
            retrieval=retrieval,
            generator=generator,
        )
    )

    return AppServices(
        retrieval=retrieval,
        qa=qa,
        catalog=catalog,
        product_search=(
            product_search
        ),
    )
