__all__ = [
    "build_product_search_text",
    "canonicalize_products",
    "build_canonical_products_file",
    "ProductFAISSIndex",
    "ProductBM25Index",
    "ProductMetadataRetriever",
    "ProductSearchReranker",
    "CandidateReviewEvidenceRetriever",
]


def __getattr__(name):
    if name in {
        "build_product_search_text",
        "canonicalize_products",
        "build_canonical_products_file",
    }:
        from .canonical import (
            build_canonical_products_file,
            build_product_search_text,
            canonicalize_products,
        )

        return {
            "build_product_search_text": build_product_search_text,
            "canonicalize_products": canonicalize_products,
            "build_canonical_products_file": build_canonical_products_file,
        }[name]

    if name == "ProductFAISSIndex":
        from .dense import ProductFAISSIndex

        return ProductFAISSIndex

    if name == "ProductBM25Index":
        from .sparse import ProductBM25Index

        return ProductBM25Index

    if name == "ProductMetadataRetriever":
        from .retriever import ProductMetadataRetriever

        return ProductMetadataRetriever

    if name == "ProductSearchReranker":
        from .reranker import ProductSearchReranker

        return ProductSearchReranker

    if name == "CandidateReviewEvidenceRetriever":
        from .review_evidence import CandidateReviewEvidenceRetriever

        return CandidateReviewEvidenceRetriever

    raise AttributeError(name)
