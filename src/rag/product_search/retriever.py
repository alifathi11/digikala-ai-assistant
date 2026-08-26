import re

import numpy as np
import pandas as pd


class ProductMetadataRetriever:
    """
    Hybrid product metadata retriever.

    Important design choices:

    - FAISS and BM25 are fused with weighted Reciprocal Rank Fusion (RRF),
      rather than per-query Min-Max score normalization.
    - A lightweight lexical grounding score rewards actual overlap between the
      user's words and title/brand/category metadata.
    - Brand intent remains an explicit bounded boost.
    - Final metadata_score is always in [0, 1].
    - Raw and diagnostic scores are preserved for evaluation/debugging.
    """

    DEFAULT_STOPWORDS = {
        "و",
        "یا",
        "برای",
        "با",
        "از",
        "به",
        "در",
        "که",
        "را",
        "رو",
        "این",
        "آن",
        "اون",
        "یک",
        "مناسب",
        "مدل",
        "حجم",
        "محصول",
    }

    LEXICAL_FIELDS = (
        "title_fa",
        "Category1",
        "Category2",
        "sub_category",
    )

    CATEGORY_FIELDS = (
        "Category2",
        "sub_category",
    )


    def __init__(
        self,
        embedding_model,
        dense_index,
        sparse_index,
        processor=None,
        bm25_weight=0.50,
        embedding_weight=0.50,
        candidate_multiplier=4,
        brand_boost=0.20,
        lexical_weight=0.30,
        rrf_k=60,
        validate_index_alignment=True,
    ):
        self.embedding_model = (
            embedding_model
        )

        self.dense_index = (
            dense_index
        )

        self.sparse_index = (
            sparse_index
        )

        self.processor = processor

        self.bm25_weight = float(
            bm25_weight
        )

        self.embedding_weight = float(
            embedding_weight
        )

        self.candidate_multiplier = int(
            candidate_multiplier
        )

        self.brand_boost = float(
            brand_boost
        )

        self.lexical_weight = float(
            lexical_weight
        )

        self.rrf_k = int(
            rrf_k
        )

        self.validate_index_alignment = bool(
            validate_index_alignment
        )

        if (
            self.bm25_weight < 0
            or self.embedding_weight < 0
            or (
                self.bm25_weight
                + self.embedding_weight
                <= 0
            )
        ):
            raise ValueError(
                "BM25 and embedding weights "
                "must be non-negative and not "
                "both zero."
            )

        if not (
            0.0
            <= self.lexical_weight
            <= 1.0
        ):
            raise ValueError(
                "lexical_weight must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.brand_boost
            <= 1.0
        ):
            raise ValueError(
                "brand_boost must be "
                "between 0 and 1."
            )

        if self.rrf_k < 1:
            raise ValueError(
                "rrf_k must be >= 1."
            )

        self.documents = (
            dense_index.documents
        )

        if self.validate_index_alignment:
            self._validate_index_metadata()


    @staticmethod
    def _numeric_ids(
        frame,
    ):
        return (
            pd.to_numeric(
                frame[
                    "id"
                ],
                errors="raise",
            )
            .astype(
                "int64"
            )
            .to_numpy()
        )


    def _validate_index_metadata(
        self,
    ):
        """
        Dense and sparse product indexes must refer to the same canonical
        product snapshot.

        Canonical products are sorted by product ID before both indexes are
        built, so an exact ID-sequence check is a cheap and strong guard
        against stale/mixed index folders.
        """
        dense_documents = getattr(
            self.dense_index,
            "documents",
            None,
        )

        sparse_documents = getattr(
            self.sparse_index,
            "documents",
            None,
        )

        if (
            dense_documents is None
            or sparse_documents is None
        ):
            raise ValueError(
                "Product indexes must be loaded "
                "before creating metadata retriever."
            )

        dense_ids = self._numeric_ids(
            dense_documents
        )

        sparse_ids = self._numeric_ids(
            sparse_documents
        )

        if (
            len(
                np.unique(
                    dense_ids
                )
            )
            != len(
                dense_ids
            )
        ):
            raise ValueError(
                "Dense product metadata contains "
                "duplicate product IDs."
            )

        if (
            len(
                np.unique(
                    sparse_ids
                )
            )
            != len(
                sparse_ids
            )
        ):
            raise ValueError(
                "Sparse product metadata contains "
                "duplicate product IDs."
            )

        if (
            len(
                dense_ids
            )
            != len(
                sparse_ids
            )
            or not np.array_equal(
                dense_ids,
                sparse_ids,
            )
        ):
            raise ValueError(
                "Product FAISS and Tantivy metadata "
                "are not aligned. Rebuild BOTH product "
                "indexes from the same "
                "data/processed/products_search.parquet."
            )


    def _process(
        self,
        text,
    ):
        text = str(
            text
        ).strip()

        if self.processor is not None:
            text = (
                self.processor
                .process(
                    text
                )
            )

        return str(
            text
        ).strip()


    def _tokens(
        self,
        text,
    ):
        processed = self._process(
            text
        )

        tokens = [
            token
            for token
            in re.findall(
                r"[^\W_]+",
                processed,
                flags=re.UNICODE,
            )
            if (
                token
                and token
                not in self.DEFAULT_STOPWORDS
            )
        ]

        return tokens


    def _candidate_metadata(
        self,
        product_ids,
    ):
        product_to_row = (
            self.dense_index
            ._product_id_to_row
        )

        row_ids = [
            product_to_row[
                int(product_id)
            ]
            for product_id
            in product_ids
        ]

        return (
            self.documents
            .iloc[
                row_ids
            ]
            .copy()
            .reset_index(drop=True)
        )


    def _brand_match_scores(
        self,
        query,
        candidate_metadata,
    ):
        processed_query = self._process(
            query
        )

        scores = []

        for brand in (
            candidate_metadata
            .get(
                "Brand",
                pd.Series(
                    [""] * len(
                        candidate_metadata
                    )
                ),
            )
            .fillna("")
            .astype(str)
        ):
            processed_brand = (
                self._process(
                    brand
                )
            )

            ignored = (
                not processed_brand
                or processed_brand
                in {
                    "متفرقه",
                    "unknown",
                    "نامشخص",
                }
            )

            if ignored:
                scores.append(
                    0.0
                )
                continue

            scores.append(
                float(
                    processed_brand
                    in processed_query
                )
            )

        return np.asarray(
            scores,
            dtype=float,
        )


    @staticmethod
    def _dice_overlap(
        left_tokens,
        right_tokens,
    ):
        left = set(
            left_tokens
        )
        right = set(
            right_tokens
        )

        if (
            not left
            or not right
        ):
            return 0.0

        return float(
            2.0
            * len(
                left
                & right
            )
            / (
                len(
                    left
                )
                + len(
                    right
                )
            )
        )


    def _field_tokens(
        self,
        row,
        fields,
    ):
        tokens = []

        for field in fields:
            value = getattr(
                row,
                field,
                "",
            )

            try:
                missing = pd.isna(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                missing = False

            if (
                value is None
                or missing
            ):
                continue

            value = str(
                value
            ).strip()

            if (
                not value
                or value.lower()
                == "unknown"
            ):
                continue

            tokens.extend(
                self._tokens(
                    value
                )
            )

        return tokens


    def _lexical_match_scores(
        self,
        query,
        candidate_metadata,
    ):
        """
        Product-type-aware lexical grounding.

        lexical_score =
            45% query-token coverage in title/category metadata
            35% adjacent query bigrams matched inside title_fa
            20% category compatibility (Category2/sub_category)

        Brand is excluded here because exact-brand intent already has a
        separate bounded boost.
        """
        query_tokens = self._tokens(
            query
        )

        if not query_tokens:
            zeros = np.zeros(
                len(
                    candidate_metadata
                ),
                dtype=float,
            )
            return (
                zeros,
                zeros.copy(),
                zeros.copy(),
                zeros.copy(),
            )

        query_token_set = set(
            query_tokens
        )

        query_bigrams = set(
            zip(
                query_tokens,
                query_tokens[
                    1:
                ],
            )
        )

        token_scores = []
        title_bigram_scores = []
        category_scores = []
        lexical_scores = []

        for row in candidate_metadata.itertuples(
            index=False
        ):
            metadata_tokens = (
                self._field_tokens(
                    row,
                    self.LEXICAL_FIELDS,
                )
            )
            metadata_token_set = set(
                metadata_tokens
            )

            token_overlap = (
                len(
                    query_token_set
                    & metadata_token_set
                )
                / len(
                    query_token_set
                )
            )

            title_tokens = (
                self._field_tokens(
                    row,
                    (
                        "title_fa",
                    ),
                )
            )

            if query_bigrams:
                title_bigrams = set(
                    zip(
                        title_tokens,
                        title_tokens[
                            1:
                        ],
                    )
                )
                title_bigram_overlap = (
                    len(
                        query_bigrams
                        & title_bigrams
                    )
                    / len(
                        query_bigrams
                    )
                )
            else:
                title_bigram_overlap = (
                    token_overlap
                )

            category_tokens = (
                self._field_tokens(
                    row,
                    self.CATEGORY_FIELDS,
                )
            )

            category_score = (
                self._dice_overlap(
                    query_tokens,
                    category_tokens,
                )
            )

            lexical = (
                0.45
                * token_overlap
                +
                0.35
                * title_bigram_overlap
                +
                0.20
                * category_score
            )

            token_scores.append(
                float(
                    token_overlap
                )
            )
            title_bigram_scores.append(
                float(
                    title_bigram_overlap
                )
            )
            category_scores.append(
                float(
                    category_score
                )
            )
            lexical_scores.append(
                float(
                    lexical
                )
            )

        return (
            np.asarray(
                token_scores,
                dtype=float,
            ),
            np.asarray(
                title_bigram_scores,
                dtype=float,
            ),
            np.asarray(
                category_scores,
                dtype=float,
            ),
            np.asarray(
                lexical_scores,
                dtype=float,
            ),
        )


    def _source_rank_score(
        self,
        ranks,
    ):
        """
        Per-source reciprocal-rank strength:
        rank 1 -> 1.0
        missing -> 0.0
        """
        ranks = pd.to_numeric(
            ranks,
            errors="coerce",
        )

        present = ranks.notna()

        output = np.zeros(
            len(
                ranks
            ),
            dtype=float,
        )

        output[
            present.to_numpy()
        ] = (
            (
                self.rrf_k
                + 1.0
            )
            /
            (
                self.rrf_k
                + ranks[
                    present
                ].to_numpy(
                    dtype=float
                )
            )
        )

        return output


    def retrieve(
        self,
        query,
        top_k=10,
    ):
        candidate_k = max(
            int(
                top_k
            ),
            int(
                top_k
            )
            * self.candidate_multiplier,
        )

        processed_query = (
            self._process(
                query
            )
        )

        query_embedding = (
            self.embedding_model
            .encode(
                [
                    processed_query
                ]
            )
        )

        dense = (
            self.dense_index
            .search(
                query_embedding,
                top_k=(
                    candidate_k
                ),
            )[
                [
                    "id",
                    "score",
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

        sparse = (
            self.sparse_index
            .retrieve(
                query,
                top_k=(
                    candidate_k
                ),
            )[
                [
                    "id",
                    "score",
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

        dense[
            "embedding_raw_score"
        ] = (
            dense[
                "score"
            ]
            .astype(float)
        )

        dense[
            "embedding_rank"
        ] = np.arange(
            1,
            len(
                dense
            )
            + 1,
            dtype=int,
        )

        sparse[
            "bm25_raw_score"
        ] = (
            sparse[
                "score"
            ]
            .astype(float)
        )

        sparse[
            "bm25_rank"
        ] = np.arange(
            1,
            len(
                sparse
            )
            + 1,
            dtype=int,
        )

        merged = (
            dense[
                [
                    "id",
                    "embedding_raw_score",
                    "embedding_rank",
                ]
            ]
            .merge(
                sparse[
                    [
                        "id",
                        "bm25_raw_score",
                        "bm25_rank",
                    ]
                ],
                on="id",
                how="outer",
            )
        )

        merged[
            "embedding_score"
        ] = self._source_rank_score(
            merged[
                "embedding_rank"
            ]
        )

        merged[
            "bm25_score"
        ] = self._source_rank_score(
            merged[
                "bm25_rank"
            ]
        )

        source_weight_sum = (
            self.embedding_weight
            + self.bm25_weight
        )

        merged[
            "rrf_score"
        ] = (
            (
                self.embedding_weight
                * merged[
                    "embedding_score"
                ]
                +
                self.bm25_weight
                * merged[
                    "bm25_score"
                ]
            )
            / source_weight_sum
        )

        candidate_metadata = (
            self._candidate_metadata(
                merged[
                    "id"
                ]
                .astype(int)
                .tolist()
            )
        )

        (
            token_overlap,
            title_bigram_overlap,
            category_score,
            lexical_score,
        ) = (
            self._lexical_match_scores(
                query,
                candidate_metadata,
            )
        )

        merged[
            "token_overlap"
        ] = token_overlap

        merged[
            "title_bigram_overlap"
        ] = (
            title_bigram_overlap
        )

        # Backward-compatible alias.
        merged[
            "bigram_overlap"
        ] = (
            title_bigram_overlap
        )

        merged[
            "category_score"
        ] = (
            category_score
        )

        merged[
            "lexical_score"
        ] = lexical_score

        merged[
            "metadata_base_score"
        ] = (
            (
                1.0
                - self.lexical_weight
            )
            * merged[
                "rrf_score"
            ]
            +
            self.lexical_weight
            * merged[
                "lexical_score"
            ]
        )

        brand_match = (
            self._brand_match_scores(
                query,
                candidate_metadata,
            )
        )

        merged[
            "brand_match"
        ] = brand_match

        # Bounded boost: an exact brand can improve a candidate but can never
        # push the retrieval score outside [0, 1].
        merged[
            "metadata_score"
        ] = (
            merged[
                "metadata_base_score"
            ]
            +
            (
                self.brand_boost
                * merged[
                    "brand_match"
                ]
                * (
                    1.0
                    - merged[
                        "metadata_base_score"
                    ]
                )
            )
        ).clip(
            lower=0.0,
            upper=1.0,
        )

        ranked = (
            merged
            .sort_values(
                [
                    "metadata_score",
                    "lexical_score",
                    "rrf_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .head(
                int(
                    top_k
                )
            )
            .reset_index(drop=True)
        )

        metadata = (
            self._candidate_metadata(
                ranked[
                    "id"
                ]
                .astype(int)
                .tolist()
            )
        )

        score_columns = [
            "metadata_score",
            "metadata_base_score",
            "rrf_score",
            "lexical_score",
            "token_overlap",
            "title_bigram_overlap",
            "bigram_overlap",
            "category_score",
            "bm25_score",
            "embedding_score",
            "bm25_raw_score",
            "embedding_raw_score",
            "bm25_rank",
            "embedding_rank",
            "brand_match",
        ]

        scores = (
            ranked[
                score_columns
            ]
            .reset_index(drop=True)
        )

        result = pd.concat(
            [
                metadata,
                scores,
            ],
            axis=1,
        )

        result.attrs[
            "scoring"
        ] = {
            "fusion": (
                "weighted_rrf_plus_lexical"
            ),
            "bm25_weight": float(
                self.bm25_weight
            ),
            "embedding_weight": float(
                self.embedding_weight
            ),
            "lexical_weight": float(
                self.lexical_weight
            ),
            "rrf_k": int(
                self.rrf_k
            ),
            "brand_boost": float(
                self.brand_boost
            ),
        }

        return result
