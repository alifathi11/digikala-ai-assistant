import time

import pandas as pd


class CandidateReviewEvidenceRetriever:
    """
    Retrieve query-relevant review evidence separately for each shortlisted
    product.

    Why:
    A global top-N review search can starve many product candidates of evidence:
    reviews from globally stronger products consume the whole review budget,
    then ProductSearchReranker filters them by candidate product ID.

    This component first resolves comment IDs for all shortlisted products in
    one vectorized metadata scan, then performs candidate-scoped Hybrid
    retrieval per product. Each product therefore gets a fair opportunity to
    contribute its own strongest query-relevant reviews.
    """

    def __init__(
        self,
        retriever,
        documents=None,
        reviews_per_product=2,
    ):
        self.retriever = retriever

        self.reviews_per_product = int(
            reviews_per_product
        )

        if self.reviews_per_product <= 0:
            raise ValueError(
                "reviews_per_product must be positive."
            )

        self.documents = (
            documents
            if documents is not None
            else self._infer_documents(
                retriever
            )
        )

        self.documents = (
            self.documents
            .reset_index(drop=True)
        )

        required = {
            "id",
            "product_id",
        }

        missing = (
            required
            - set(
                self.documents.columns
            )
        )

        if missing:
            raise ValueError(
                "Review metadata is missing: "
                f"{sorted(missing)}"
            )


    @staticmethod
    def _infer_documents(
        retriever,
    ):
        documents = getattr(
            retriever,
            "documents",
            None,
        )

        if documents is not None:
            return documents

        embedding_retriever = getattr(
            retriever,
            "embedding_retriever",
            None,
        )

        if embedding_retriever is not None:
            documents = getattr(
                embedding_retriever,
                "documents",
                None,
            )

        if documents is None:
            raise ValueError(
                "documents could not be inferred "
                "from review retriever."
            )

        return documents


    def _comment_ids_by_product(
        self,
        product_ids,
    ):
        product_ids = [
            int(product_id)
            for product_id
            in product_ids
        ]

        if not product_ids:
            return {}

        product_id_set = set(
            product_ids
        )

        product_series = (
            self.documents[
                "product_id"
            ]
        )

        if not pd.api.types.is_integer_dtype(
            product_series.dtype
        ):
            product_series = pd.to_numeric(
                product_series,
                errors="coerce",
            )

        subset = (
            self.documents
            .loc[
                product_series.isin(
                    product_id_set
                ),
                [
                    "id",
                    "product_id",
                ],
            ]
            .copy()
        )

        if len(subset) == 0:
            return {
                product_id: []
                for product_id
                in product_ids
            }

        subset[
            "id"
        ] = pd.to_numeric(
            subset[
                "id"
            ],
            errors="coerce",
        )

        subset[
            "product_id"
        ] = pd.to_numeric(
            subset[
                "product_id"
            ],
            errors="coerce",
        )

        subset = subset[
            subset[
                "id"
            ].notna()
            &
            subset[
                "product_id"
            ].notna()
        ].copy()

        subset[
            "id"
        ] = (
            subset[
                "id"
            ]
            .astype(int)
        )

        subset[
            "product_id"
        ] = (
            subset[
                "product_id"
            ]
            .astype(int)
        )

        grouped = {
            int(product_id): (
                group[
                    "id"
                ]
                .astype(int)
                .tolist()
            )
            for product_id, group
            in subset.groupby(
                "product_id",
                sort=False,
            )
        }

        return {
            product_id: grouped.get(
                product_id,
                [],
            )
            for product_id
            in product_ids
        }


    def retrieve(
        self,
        query,
        product_ids,
    ):
        start = time.perf_counter()

        product_ids = [
            int(product_id)
            for product_id
            in product_ids
        ]

        ids_by_product = (
            self._comment_ids_by_product(
                product_ids
            )
        )

        frames = []
        calls = 0
        products_with_comments = 0

        candidate_comment_count = sum(
            len(comment_ids)
            for comment_ids
            in ids_by_product.values()
        )

        for product_id in (
            product_ids
        ):
            candidate_ids = (
                ids_by_product.get(
                    product_id,
                    [],
                )
            )

            if not candidate_ids:
                continue

            products_with_comments += 1

            effective_top_k = min(
                self.reviews_per_product,
                len(
                    candidate_ids
                ),
            )

            if effective_top_k <= 0:
                continue

            retrieved = (
                self.retriever
                .retrieve(
                    query,
                    top_k=(
                        effective_top_k
                    ),
                    candidate_ids=(
                        candidate_ids
                    ),
                )
                .copy()
            )

            calls += 1

            if len(retrieved) == 0:
                continue

            # Product ID should already be attached by HybridRetriever.
            # Keep a defensive value in case another retriever is used later.
            if "product_id" not in retrieved.columns:
                retrieved[
                    "product_id"
                ] = int(
                    product_id
                )

            frames.append(
                retrieved
            )

        if frames:
            evidence = (
                pd.concat(
                    frames,
                    ignore_index=True,
                )
            )
        else:
            evidence = (
                self.documents
                .iloc[0:0]
                .copy()
            )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        telemetry = {
            "review_scope": (
                "candidate_product"
            ),
            "review_retrieval_calls": int(
                calls
            ),
            "review_candidate_products": int(
                len(
                    product_ids
                )
            ),
            "products_with_comments": int(
                products_with_comments
            ),
            "candidate_comment_count": int(
                candidate_comment_count
            ),
            "retrieved_review_count": int(
                len(
                    evidence
                )
            ),
            "review_retrieval_latency_ms": float(
                latency_ms
            ),
        }

        return (
            evidence,
            telemetry,
        )
