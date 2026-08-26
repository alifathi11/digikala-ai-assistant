import pandas as pd

from ..product_search.review_evidence import (
    CandidateReviewEvidenceRetriever,
)


class ProductSearchPipeline:
    """
    Product Search / Discovery with grounded LLM reranking.

    Retrieval remains deterministic:
      metadata FAISS + metadata Tantivy -> candidate products

    Reviews are retrieval evidence, not automatically positive ranking signals.

    A single LLM reranker call scores only the small top candidate set and
    explicitly distinguishes supporting, mixed and contradictory review
    evidence.
    """

    def __init__(
        self,
        metadata_retriever,
        review_retriever,
        reranker=None,
        metadata_candidates=50,
        review_comment_candidates=250,
        review_comments_per_product=2,
        reranker_candidates=12,
        metadata_weight=0.30,
        reranker_weight=0.70,
    ):
        self.metadata_retriever = (
            metadata_retriever
        )

        self.review_retriever = (
            review_retriever
        )

        self.reranker = reranker

        self.metadata_candidates = int(
            metadata_candidates
        )

        self.review_comment_candidates = int(
            review_comment_candidates
        )

        self.review_comments_per_product = int(
            review_comments_per_product
        )

        if self.review_comments_per_product <= 0:
            raise ValueError(
                "review_comments_per_product "
                "must be positive."
            )

        self.review_evidence_retriever = (
            CandidateReviewEvidenceRetriever(
                retriever=(
                    review_retriever
                ),
                reviews_per_product=(
                    self.review_comments_per_product
                ),
            )
        )

        self.reranker_candidates = int(
            reranker_candidates
        )

        self.metadata_weight = float(
            metadata_weight
        )

        self.reranker_weight = float(
            reranker_weight
        )

        if (
            self.metadata_weight
            + self.reranker_weight
            <= 0
        ):
            raise ValueError(
                "Search weights must "
                "be positive."
            )

        self.products = (
            metadata_retriever
            .documents
        )


    @staticmethod
    def _normalize(
        series,
    ):
        series = (
            series
            .astype(float)
        )

        if len(series) == 0:
            return series

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                [1.0] * len(
                    series
                ),
                index=series.index,
            )

        return (
            series
            - minimum
        ) / (
            maximum
            - minimum
        )


    def search(
        self,
        query,
        top_k=10,
    ):
        metadata = (
            self.metadata_retriever
            .retrieve(
                query,
                top_k=(
                    self.metadata_candidates
                ),
            )
            .reset_index(drop=True)
        )

        shortlist = (
            metadata
            .head(
                min(
                    self.reranker_candidates,
                    len(
                        metadata
                    ),
                )
            )
            .copy()
            .reset_index(drop=True)
        )

        (
            review_comments,
            review_telemetry,
        ) = (
            self.review_evidence_retriever
            .retrieve(
                query=query,
                product_ids=(
                    shortlist[
                        "id"
                    ]
                    .astype(int)
                    .tolist()
                ),
            )
        )

        # ProductMetadataRetriever now returns a bounded relevance score in
        # [0, 1]. Do not Min-Max normalize the shortlist again: that old step
        # forced the best candidate to display 100% even when its absolute
        # retrieval evidence was weak.
        shortlist[
            "metadata_score_norm"
        ] = (
            shortlist[
                "metadata_score"
            ]
            .astype(float)
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

        telemetry = {
            "reranker_applied": False,
            **review_telemetry,
        }

        if (
            self.reranker is not None
            and len(
                shortlist
            )
            > 0
        ):
            rankings, reranker_telemetry = (
                self.reranker
                .rerank(
                    query=query,
                    candidates=shortlist,
                    review_comments=(
                        review_comments
                    ),
                )
            )

            shortlist = (
                shortlist
                .merge(
                    rankings,
                    on="id",
                    how="left",
                )
            )

            shortlist[
                "llm_match_score"
            ] = (
                shortlist[
                    "llm_match_score"
                ]
                .fillna(
                    0.0
                )
            )

            shortlist[
                "llm_match_score_norm"
            ] = (
                shortlist[
                    "llm_match_score"
                ]
                / 5.0
            )

            shortlist[
                "evidence_status"
            ] = (
                shortlist[
                    "evidence_status"
                ]
                .fillna(
                    "none"
                )
            )

            shortlist[
                "evidence_ids"
            ] = (
                shortlist[
                    "evidence_ids"
                ]
                .apply(
                    lambda value: (
                        value
                        if isinstance(
                            value,
                            list,
                        )
                        else []
                    )
                )
            )

            shortlist[
                "reason"
            ] = (
                shortlist[
                    "reason"
                ]
                .fillna("")
            )

            total_weight = (
                self.metadata_weight
                + self.reranker_weight
            )

            shortlist[
                "score"
            ] = (
                self.metadata_weight
                * shortlist[
                    "metadata_score_norm"
                ]
                +
                self.reranker_weight
                * shortlist[
                    "llm_match_score_norm"
                ]
            ) / total_weight

            # The LLM is a true reranker, not a minor additive boost.
            #
            # A clearly irrelevant candidate (0/5) should never remain above a
            # strong 4/5 or 5/5 match only because dense retrieval assigned a
            # high semantic score.  Keep weak candidates as fallbacks, but
            # deterministically demote them.
            shortlist[
                "reranker_tier"
            ] = (
                shortlist[
                    "llm_match_score"
                ]
                .clip(
                    lower=0,
                    upper=5,
                )
            )

            telemetry = {
                "reranker_applied": True,
                **review_telemetry,
                **reranker_telemetry,
            }

        else:
            shortlist[
                "llm_match_score"
            ] = 0.0

            shortlist[
                "llm_match_score_norm"
            ] = 0.0

            shortlist[
                "evidence_status"
            ] = "none"

            shortlist[
                "evidence_ids"
            ] = [
                []
                for _ in range(
                    len(
                        shortlist
                    )
                )
            ]

            shortlist[
                "reason"
            ] = ""

            shortlist[
                "score"
            ] = (
                shortlist[
                    "metadata_score_norm"
                ]
            )

            shortlist[
                "reranker_tier"
            ] = 0.0

        ranked = (
            shortlist
            .sort_values(
                [
                    "reranker_tier",
                    "score",
                    "metadata_score",
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

        ranked.attrs[
            "telemetry"
        ] = telemetry

        return ranked
