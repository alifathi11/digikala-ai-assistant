from dataclasses import dataclass
from pathlib import Path

from ..config import load_config
from ..generation import OpenAIJSONGenerator
from ..product_search import ProductBM25Index, ProductFAISSIndex, ProductMetadataRetriever, ProductSearchReranker
from ..product_search.review_evidence import CandidateReviewEvidenceRetriever
from ..runtime import load_retrieval_stack
from .product_search_judge import ProductSearchRelevanceJudge


@dataclass
class ProductSearchEvaluationContext:
    metadata: object
    review_evidence: object
    reranker: object
    judge: object
    evaluation_config: dict


def load_product_search_evaluation_context(project_root, api_key, base_url):
    project_root = Path(project_root)
    rag_config = load_config(project_root / "configs" / "rag.yaml")
    qa_config = load_config(project_root / "configs" / "qa.yaml")
    search_config = load_config(project_root / "configs" / "product_search.yaml")
    evaluation_config = load_config(project_root / "configs" / "product_search_evaluation.yaml")

    retrieval = load_retrieval_stack(project_root=project_root, rag_config=rag_config)
    dense_index = ProductFAISSIndex().load(project_root / "data" / "indexes" / "products_embedding")
    sparse_index = ProductBM25Index(processor=retrieval.processor).load(
        project_root / "data" / "indexes" / "products_bm25_tantivy"
    )

    metadata_cfg = search_config["product_search"]["metadata"]
    metadata = ProductMetadataRetriever(
        embedding_model=retrieval.embedding_model,
        dense_index=dense_index,
        sparse_index=sparse_index,
        processor=retrieval.processor,
        bm25_weight=metadata_cfg["bm25_weight"],
        embedding_weight=metadata_cfg["embedding_weight"],
        candidate_multiplier=metadata_cfg["candidate_multiplier"],
        brand_boost=metadata_cfg["brand_boost"],
        lexical_weight=metadata_cfg.get("lexical_weight", 0.40),
        rrf_k=metadata_cfg.get("rrf_k", 60),
        validate_index_alignment=metadata_cfg.get("validate_index_alignment", True),
    )

    generation_cfg = qa_config["generation"]
    reranker_generator = OpenAIJSONGenerator(
        api_key=api_key,
        base_url=base_url,
        model=generation_cfg["model"],
        input_cost_per_million=generation_cfg.get("input_cost_per_million"),
        output_cost_per_million=generation_cfg.get("output_cost_per_million"),
    )

    eval_cfg = evaluation_config["product_search_evaluation"]
    judge_cfg = eval_cfg["qrel_judge"]
    judge_generator = OpenAIJSONGenerator(
        api_key=api_key,
        base_url=base_url,
        model=judge_cfg.get("model") or generation_cfg["model"],
        input_cost_per_million=judge_cfg.get("input_cost_per_million"),
        output_cost_per_million=judge_cfg.get("output_cost_per_million"),
    )

    product_cfg = search_config["product_search"]
    candidate_cfg = product_cfg["candidates"]
    reranker_cfg = product_cfg["reranker"]
    reviews_per_product = int(
        candidate_cfg.get("review_comments_per_product", reranker_cfg.get("max_reviews_per_product", 2))
    )

    review_evidence = CandidateReviewEvidenceRetriever(
        retriever=retrieval.hybrid,
        reviews_per_product=reviews_per_product,
    )
    reranker = ProductSearchReranker(
        generator=reranker_generator,
        max_reviews_per_product=reranker_cfg["max_reviews_per_product"],
        max_review_chars=reranker_cfg["max_review_chars"],
    )
    judge = ProductSearchRelevanceJudge(
        generator=judge_generator,
        max_reviews_per_product=judge_cfg.get("max_reviews_per_product", 1),
        max_review_chars=judge_cfg.get("max_review_chars", 450),
    )

    return ProductSearchEvaluationContext(
        metadata=metadata,
        review_evidence=review_evidence,
        reranker=reranker,
        judge=judge,
        evaluation_config=evaluation_config,
    )
