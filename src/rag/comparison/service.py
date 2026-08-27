import time

import pandas as pd


class ComparisonContextService:
    """
    Resolve selected product metadata and retrieve review evidence separately
    inside each selected product.

    The service intentionally does NOT perform product discovery. Product IDs
    are already selected before comparison begins.
    """

    def __init__(
        self,
        product_documents,
        review_retriever,
        review_documents=None,
        reviews_per_product=3,
        product_id_to_row=None,
    ):
        self.product_documents = (
            product_documents
            .reset_index(drop=True)
        )

        if "id" not in (
            self.product_documents.columns
        ):
            raise ValueError(
                "Product metadata requires an id column."
            )

        self.review_retriever = (
            review_retriever
        )

        self.review_documents = (
            review_documents
            if review_documents is not None
            else self._infer_review_documents(
                review_retriever
            )
        )

        self.review_documents = (
            self.review_documents
            .reset_index(drop=True)
        )

        required_review_columns = {
            "id",
            "product_id",
        }

        missing = (
            required_review_columns
            - set(
                self.review_documents.columns
            )
        )

        if missing:
            raise ValueError(
                "Review metadata is missing: "
                f"{sorted(missing)}"
            )

        self.reviews_per_product = int(
            reviews_per_product
        )

        if self.reviews_per_product <= 0:
            raise ValueError(
                "reviews_per_product must be positive."
            )

        if product_id_to_row is None:
            self._product_id_to_row = {
                int(product_id): row_index
                for row_index, product_id
                in enumerate(
                    self.product_documents[
                        "id"
                    ]
                )
            }
        else:
            self._product_id_to_row = {
                int(product_id): int(
                    row_index
                )
                for product_id, row_index
                in product_id_to_row.items()
            }


    @staticmethod
    def _infer_review_documents(
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
                "Review documents could not be inferred from retriever."
            )

        return documents


    @staticmethod
    def normalize_product_ids(
        product_ids,
    ):
        normalized = []
        seen = set()

        for value in product_ids:
            product_id = int(
                value
            )

            if product_id in seen:
                continue

            seen.add(
                product_id
            )

            normalized.append(
                product_id
            )

        return normalized


    def get_products(
        self,
        product_ids,
    ):
        product_ids = (
            self.normalize_product_ids(
                product_ids
            )
        )

        missing = [
            product_id
            for product_id
            in product_ids
            if product_id
            not in self._product_id_to_row
        ]

        if missing:
            raise ValueError(
                "Unknown product IDs: "
                f"{missing}"
            )

        row_ids = [
            self._product_id_to_row[
                product_id
            ]
            for product_id
            in product_ids
        ]

        products = (
            self.product_documents
            .iloc[
                row_ids
            ]
            .copy()
            .reset_index(drop=True)
        )

        products[
            "id"
        ] = (
            products[
                "id"
            ]
            .astype(int)
        )

        return products


    def _comment_ids_by_product(
        self,
        product_ids,
    ):
        product_ids = (
            self.normalize_product_ids(
                product_ids
            )
        )

        if not product_ids:
            return {}

        product_series = pd.to_numeric(
            self.review_documents[
                "product_id"
            ],
            errors="coerce",
        )

        subset = (
            self.review_documents
            .loc[
                product_series.isin(
                    set(
                        product_ids
                    )
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
            & subset[
                "product_id"
            ].notna()
        ].copy()

        subset[
            "id"
        ] = subset[
            "id"
        ].astype(int)

        subset[
            "product_id"
        ] = subset[
            "product_id"
        ].astype(int)

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


    def retrieve_reviews(
        self,
        query,
        product_ids,
    ):
        start = time.perf_counter()

        product_ids = (
            self.normalize_product_ids(
                product_ids
            )
        )

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

        for product_id in product_ids:
            candidate_ids = (
                ids_by_product.get(
                    product_id,
                    [],
                )
            )

            if not candidate_ids:
                continue

            products_with_comments += 1

            top_k = min(
                self.reviews_per_product,
                len(
                    candidate_ids
                ),
            )

            if top_k <= 0:
                continue

            retrieved = (
                self.review_retriever
                .retrieve(
                    query,
                    top_k=top_k,
                    candidate_ids=(
                        candidate_ids
                    ),
                )
                .copy()
            )

            calls += 1

            if len(retrieved) == 0:
                continue

            if "product_id" not in (
                retrieved.columns
            ):
                retrieved[
                    "product_id"
                ] = product_id

            retrieved[
                "product_id"
            ] = (
                pd.to_numeric(
                    retrieved[
                        "product_id"
                    ],
                    errors="coerce",
                )
                .fillna(
                    product_id
                )
                .astype(int)
            )

            # Hard ownership guard: a scoped retrieval call may only return
            # comments owned by the currently selected product.
            retrieved = retrieved[
                retrieved[
                    "product_id"
                ]
                == int(
                    product_id
                )
            ].copy()

            frames.append(
                retrieved
            )

        if frames:
            evidence = pd.concat(
                frames,
                ignore_index=True,
            )
        else:
            evidence = (
                self.review_documents
                .iloc[0:0]
                .copy()
            )

        telemetry = {
            "review_scope": (
                "selected_product"
            ),
            "review_retrieval_calls": (
                int(calls)
            ),
            "selected_products": int(
                len(product_ids)
            ),
            "products_with_comments": int(
                products_with_comments
            ),
            "candidate_comment_count": int(
                candidate_comment_count
            ),
            "retrieved_review_count": int(
                len(evidence)
            ),
            "review_retrieval_latency_ms": float(
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            ),
        }

        return (
            evidence,
            telemetry,
        )


    @staticmethod
    def allowed_evidence_by_product(
        review_documents,
        product_ids,
    ):
        result = {
            int(product_id): set()
            for product_id
            in product_ids
        }

        if (
            review_documents is None
            or len(
                review_documents
            )
            == 0
        ):
            return result

        for row in review_documents.itertuples(
            index=False
        ):
            product_id = int(
                getattr(
                    row,
                    "product_id",
                )
            )

            if product_id not in result:
                continue

            result[
                product_id
            ].add(
                int(
                    getattr(
                        row,
                        "id",
                    )
                )
            )

        return result


    @staticmethod
    def select_evidence_rows(
        review_documents,
        evidence_ids_by_product,
    ):
        if (
            review_documents is None
            or len(
                review_documents
            )
            == 0
        ):
            return (
                review_documents
                .copy()
                if review_documents
                is not None
                else pd.DataFrame()
            )

        ordered_pairs = []

        for product_id, evidence_ids in (
            evidence_ids_by_product.items()
        ):
            for evidence_id in (
                evidence_ids
            ):
                ordered_pairs.append(
                    (
                        int(product_id),
                        int(evidence_id),
                    )
                )

        if not ordered_pairs:
            return (
                review_documents
                .iloc[0:0]
                .copy()
            )

        order = {
            pair: index
            for index, pair
            in enumerate(
                ordered_pairs
            )
        }

        frame = review_documents.copy()

        frame[
            "_comparison_key"
        ] = list(
            zip(
                frame[
                    "product_id"
                ].astype(int),
                frame[
                    "id"
                ].astype(int),
            )
        )

        frame = frame[
            frame[
                "_comparison_key"
            ].isin(
                order
            )
        ].copy()

        frame[
            "_comparison_order"
        ] = frame[
            "_comparison_key"
        ].map(
            order
        )

        return (
            frame
            .sort_values(
                "_comparison_order"
            )
            .drop(
                columns=[
                    "_comparison_key",
                    "_comparison_order",
                ]
            )
            .reset_index(drop=True)
        )
