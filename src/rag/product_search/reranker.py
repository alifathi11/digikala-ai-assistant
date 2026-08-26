import pandas as pd

from .reranker_prompt import (
    RERANKER_SYSTEM_PROMPT,
    build_reranker_prompt,
)


ALLOWED_STATUSES = {
    "support",
    "mixed",
    "contradict",
    "none",
}


class ProductSearchReranker:

    def __init__(
        self,
        generator,
        max_reviews_per_product=2,
        max_review_chars=700,
    ):
        self.generator = generator

        self.max_reviews_per_product = int(
            max_reviews_per_product
        )

        self.max_review_chars = int(
            max_review_chars
        )


    @staticmethod
    def _review_map(
        review_comments,
        candidate_ids,
        max_reviews_per_product,
    ):
        candidate_ids = {
            int(x)
            for x in candidate_ids
        }

        result = {
            product_id: []
            for product_id
            in candidate_ids
        }

        if (
            review_comments is None
            or len(
                review_comments
            )
            == 0
        ):
            return result

        frame = (
            review_comments
            .copy()
        )

        frame[
            "product_id"
        ] = pd.to_numeric(
            frame[
                "product_id"
            ],
            errors="coerce",
        )

        frame = frame[
            frame[
                "product_id"
            ].notna()
        ].copy()

        frame[
            "product_id"
        ] = (
            frame[
                "product_id"
            ]
            .astype(int)
        )

        frame = frame[
            frame[
                "product_id"
            ].isin(
                candidate_ids
            )
        ].copy()

        if "score" in frame:
            frame = (
                frame
                .sort_values(
                    "score",
                    ascending=False,
                )
            )

        for row in frame.itertuples(
            index=False
        ):
            product_id = int(
                getattr(
                    row,
                    "product_id"
                )
            )

            if (
                len(
                    result[
                        product_id
                    ]
                )
                >= int(
                    max_reviews_per_product
                )
            ):
                continue

            text = (
                getattr(
                    row,
                    "body",
                    None,
                )
                or getattr(
                    row,
                    "search_text",
                    None,
                )
                or ""
            )

            result[
                product_id
            ].append(
                {
                    "id": int(
                        getattr(
                            row,
                            "id"
                        )
                    ),
                    "text": str(
                        text
                    ),
                }
            )

        return result


    @staticmethod
    def _validate(
        payload,
        candidate_ids,
        review_map,
    ):
        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Reranker response "
                "must be a JSON object."
            )

        rankings = payload.get(
            "rankings"
        )

        if not isinstance(
            rankings,
            list,
        ):
            raise ValueError(
                "Reranker response "
                "requires rankings list."
            )

        candidate_ids = {
            int(x)
            for x in candidate_ids
        }

        cleaned = []
        seen = set()

        for item in rankings:
            if not isinstance(
                item,
                dict,
            ):
                continue

            try:
                product_id = int(
                    item.get(
                        "product_id"
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                product_id
                not in candidate_ids
                or product_id
                in seen
            ):
                continue

            try:
                score = int(
                    item.get(
                        "match_score"
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            score = max(
                0,
                min(
                    5,
                    score,
                ),
            )

            status = str(
                item.get(
                    "evidence_status",
                    "none",
                )
            ).strip().lower()

            if status not in (
                ALLOWED_STATUSES
            ):
                status = "none"

            allowed_review_ids = {
                int(
                    review[
                        "id"
                    ]
                )
                for review
                in review_map.get(
                    product_id,
                    []
                )
            }

            evidence_ids = []

            for value in item.get(
                "evidence_ids",
                [],
            ):
                try:
                    review_id = int(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    review_id
                    in allowed_review_ids
                    and review_id
                    not in evidence_ids
                ):
                    evidence_ids.append(
                        review_id
                    )

            # `evidence_status` describes REVIEW evidence only.
            # Metadata can justify a high product match score, but it must not
            # produce a fake "support" badge when no valid review was cited.
            if not evidence_ids:
                status = "none"

            cleaned.append(
                {
                    "id": product_id,
                    "llm_match_score": (
                        float(score)
                    ),
                    "evidence_status": (
                        status
                    ),
                    "evidence_ids": (
                        evidence_ids
                    ),
                    "reason": str(
                        item.get(
                            "reason",
                            "",
                        )
                    ).strip(),
                }
            )

            seen.add(
                product_id
            )

        return pd.DataFrame(
            cleaned
        )


    def rerank(
        self,
        query,
        candidates,
        review_comments,
    ):
        candidate_ids = (
            candidates[
                "id"
            ]
            .astype(int)
            .tolist()
        )

        review_map = (
            self._review_map(
                review_comments,
                candidate_ids,
                self.max_reviews_per_product,
            )
        )

        user_prompt = (
            build_reranker_prompt(
                query=query,
                candidates=candidates,
                review_map=review_map,
                max_reviews_per_product=(
                    self.max_reviews_per_product
                ),
                max_review_chars=(
                    self.max_review_chars
                ),
            )
        )

        generation = (
            self.generator
            .generate(
                system_prompt=(
                    RERANKER_SYSTEM_PROMPT
                ),
                user_prompt=(
                    user_prompt
                ),
            )
        )

        rankings = self._validate(
            generation[
                "payload"
            ],
            candidate_ids,
            review_map,
        )

        telemetry = {
            key: value
            for key, value
            in generation.items()
            if key != "payload"
        }

        return (
            rankings,
            telemetry,
        )
