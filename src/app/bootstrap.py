from dataclasses import dataclass
from pathlib import Path

from src.rag.config import (
    get_model_pricing,
    load_project_config,
)
from src.rag.generation import (
    OpenAIJSONGenerator,
)
from src.rag.pipeline import (
    GroundedQAPipeline,
    ManagerAnalyticsPipeline,
    ProductComparisonPipeline,
    ProductSearchPipeline,
)
from src.rag.analytics import (
    AnalyticsRepository,
    AnalyticsService,
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
    comparison: ProductComparisonPipeline
    analytics: ManagerAnalyticsPipeline


def _load_product_search(
    project_root,
    retrieval,
    generator,
    config,
):
    product_search_config = (
        config[
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


def _load_comparison(
    project_root,
    retrieval,
    generator,
    product_search,
    config,
):
    comparison_config = (
        config[
            "comparison"
        ]
    )

    return ProductComparisonPipeline(
        product_documents=(
            product_search.products
        ),
        review_retriever=(
            retrieval.hybrid
        ),
        review_documents=(
            retrieval.documents
        ),
        generator=generator,
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


def _load_analytics(
    project_root,
    generator,
    config,
):
    analytics_config = (
        config[
            "analytics"
        ]
    )

    audit_config = analytics_config[
        "audit"
    ]

    aggregation_config = analytics_config[
        "aggregation"
    ]

    manager_config = analytics_config[
        "manager_qa"
    ]

    repository = (
        AnalyticsRepository
        .from_project_root(
            project_root
        )
    )

    service = AnalyticsService(
        repository=repository,
        generic_brand_values=(
            audit_config[
                "generic_brand_values"
            ]
        ),
        unknown_category_values=(
            audit_config[
                "unknown_category_values"
            ]
        ),
        min_rating_count_for_leaders=(
            aggregation_config[
                "min_rating_count_for_leaders"
            ]
        ),
        default_top_n=(
            aggregation_config[
                "default_top_n"
            ]
        ),
        product_rating_max=(
            aggregation_config[
                "product_rating_max"
            ]
        ),
    )

    return ManagerAnalyticsPipeline(
        analytics_service=service,
        generator=generator,
        brand_usable_coverage=(
            manager_config[
                "brand_usable_coverage"
            ]
        ),
        product_rating_coverage=(
            manager_config[
                "product_rating_coverage"
            ]
        ),
        historical_price_enabled=(
            manager_config[
                "historical_price_enabled"
            ]
        ),
        review_volume_ranking_enabled=(
            manager_config[
                "review_volume_ranking_enabled"
            ]
        ),
        top_n=(
            manager_config[
                "top_n"
            ]
        ),
    )


def create_app_services(
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

    retrieval = (
        load_retrieval_stack(
            project_root=(
                project_root
            ),
            rag_config=(
                config
            ),
        )
    )

    generation_config = (
        config[
            "generation"
        ]
    )

    generation_pricing = get_model_pricing(
        config,
        generation_config[
            "model"
        ],
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
    )

    qa_config_values = (
        config[
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
            config=config,
        )
    )

    comparison = (
        _load_comparison(
            project_root=(
                project_root
            ),
            retrieval=retrieval,
            generator=generator,
            product_search=(
                product_search
            ),
            config=config,
        )
    )

    analytics = (
        _load_analytics(
            project_root=(
                project_root
            ),
            generator=generator,
            config=config,
        )
    )

    return AppServices(
        retrieval=retrieval,
        qa=qa,
        catalog=catalog,
        product_search=(
            product_search
        ),
        comparison=comparison,
        analytics=analytics,
    )
