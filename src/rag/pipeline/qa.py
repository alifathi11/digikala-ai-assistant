import time

import pandas as pd

from ..generation.prompt import (
    SYSTEM_PROMPT,
    build_qa_prompt,
    build_citation_repair_prompt,
)
from ..generation.validation import (
    validate_grounded_response,
)


class GroundedQAPipeline:

    def __init__(
        self,
        retriever,
        generator,
        documents=None,
        top_k=8,
        max_context_chars=10_000,
        max_chars_per_comment=1_500,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = int(
            top_k
        )

        self.max_context_chars = int(
            max_context_chars
        )

        self.max_chars_per_comment = int(
            max_chars_per_comment
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
                "QA metadata is missing: "
                f"{sorted(missing)}"
            )

        self._product_candidate_cache = {}


    @staticmethod
    def _infer_documents(
        retriever
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
                "from retriever"
            )

        return documents


    def _product_candidate_ids(
        self,
        product_id,
    ):
        product_id = int(
            product_id
        )

        cached = (
            self._product_candidate_cache
            .get(product_id)
        )

        if cached is not None:
            return cached

        mask = (
            self.documents[
                "product_id"
            ]
            == product_id
        )

        candidate_ids = (
            self.documents
            .loc[
                mask,
                "id",
            ]
            .astype(int)
            .tolist()
        )

        if not candidate_ids:
            raise ValueError(
                "No comments found for "
                f"product_id={product_id}"
            )

        self._product_candidate_cache[
            product_id
        ] = candidate_ids

        return candidate_ids


    @staticmethod
    def _evidence_rows(
        retrieved_documents,
        evidence_ids,
    ):
        evidence_ids = [
            int(x)
            for x in evidence_ids
        ]

        if not evidence_ids:
            return (
                retrieved_documents
                .iloc[0:0]
                .copy()
            )

        rank = {
            int(comment_id): index
            for index, comment_id
            in enumerate(
                evidence_ids
            )
        }

        evidence = (
            retrieved_documents[
                retrieved_documents[
                    "id"
                ]
                .astype(int)
                .isin(
                    evidence_ids
                )
            ]
            .copy()
        )

        evidence[
            "_evidence_order"
        ] = (
            evidence["id"]
            .astype(int)
            .map(rank)
        )

        return (
            evidence
            .sort_values(
                "_evidence_order"
            )
            .drop(
                columns=[
                    "_evidence_order"
                ]
            )
            .reset_index(drop=True)
        )



    @staticmethod
    def _sanitize_payload(
        payload,
        retrieved_ids,
    ):
        """
        Final hard safety net for structured citations.

        Invalid IDs are removed. If no valid evidence remains for a
        supposedly supported answer, the result is downgraded to
        insufficient evidence with low confidence.
        """
        retrieved_set = {
            int(x)
            for x in retrieved_ids
        }

        valid_ids = []
        seen = set()

        for value in payload.get(
            "evidence_ids",
            [],
        ):
            try:
                comment_id = int(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                comment_id
                in retrieved_set
                and comment_id
                not in seen
            ):
                valid_ids.append(
                    comment_id
                )
                seen.add(
                    comment_id
                )

        sanitized = dict(
            payload
        )

        sanitized[
            "evidence_ids"
        ] = valid_ids

        insufficient = bool(
            sanitized.get(
                "insufficient_evidence",
                False,
            )
        )

        if (
            not insufficient
            and not valid_ids
        ):
            sanitized[
                "insufficient_evidence"
            ] = True

            sanitized[
                "confidence"
            ] = "low"

        return sanitized


    def answer(
        self,
        query,
        product_id=None,
        candidate_ids=None,
        top_k=None,
    ):
        total_start = (
            time.perf_counter()
        )

        if (
            product_id is None
            and candidate_ids is None
        ):
            raise ValueError(
                "product_id or candidate_ids "
                "is required for grounded "
                "product QA"
            )

        if candidate_ids is None:
            candidate_ids = (
                self._product_candidate_ids(
                    product_id
                )
            )

        candidate_ids = [
            int(comment_id)
            for comment_id in candidate_ids
        ]

        effective_top_k = (
            int(top_k)
            if top_k is not None
            else self.top_k
        )

        effective_top_k = min(
            effective_top_k,
            len(candidate_ids),
        )

        retrieval_start = (
            time.perf_counter()
        )

        retrieved = (
            self.retriever
            .retrieve(
                query,
                top_k=effective_top_k,
                candidate_ids=(
                    candidate_ids
                ),
            )
            .reset_index(drop=True)
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieved_ids = (
            retrieved["id"]
            .astype(int)
            .tolist()
        )

        user_prompt = (
            build_qa_prompt(
                query=query,
                retrieved_documents=(
                    retrieved
                ),
                max_context_chars=(
                    self.max_context_chars
                ),
                max_chars_per_comment=(
                    self.max_chars_per_comment
                ),
            )
        )

        generation = (
            self.generator
            .generate(
                system_prompt=(
                    SYSTEM_PROMPT
                ),
                user_prompt=(
                    user_prompt
                ),
            )
        )

        generation_attempts = [
            generation
        ]

        payload = generation[
            "payload"
        ]

        citation_valid, validation_errors = (
            validate_grounded_response(
                payload,
                retrieved_ids,
            )
        )

        citation_retry_count = 0
        citation_repaired = False
        repair_generation = None

        if not citation_valid:
            citation_retry_count = 1

            repair_prompt = (
                build_citation_repair_prompt(
                    original_user_prompt=(
                        user_prompt
                    ),
                    previous_payload=(
                        payload
                    ),
                    retrieved_ids=(
                        retrieved_ids
                    ),
                    validation_errors=(
                        validation_errors
                    ),
                )
            )

            repair_generation = (
                self.generator
                .generate(
                    system_prompt=(
                        SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        repair_prompt
                    ),
                )
            )

            generation_attempts.append(
                repair_generation
            )

            repaired_payload = (
                repair_generation[
                    "payload"
                ]
            )

            repaired_valid, repaired_errors = (
                validate_grounded_response(
                    repaired_payload,
                    retrieved_ids,
                )
            )

            if repaired_valid:
                payload = (
                    repaired_payload
                )

                citation_valid = True
                validation_errors = []
                citation_repaired = True

                generation = (
                    repair_generation
                )

            else:
                payload = (
                    self._sanitize_payload(
                        repaired_payload,
                        retrieved_ids,
                    )
                )

                citation_valid, validation_errors = (
                    validate_grounded_response(
                        payload,
                        retrieved_ids,
                    )
                )

                citation_repaired = True

                generation = (
                    repair_generation
                )

        evidence_ids = []

        for value in payload.get(
            "evidence_ids",
            [],
        ):
            try:
                evidence_ids.append(
                    int(value)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        evidence = (
            self._evidence_rows(
                retrieved,
                evidence_ids,
            )
        )

        generation_latency_ms = sum(
            float(
                attempt[
                    "latency_ms"
                ]
            )
            for attempt
            in generation_attempts
        )

        prompt_tokens = sum(
            int(
                attempt[
                    "prompt_tokens"
                ]
            )
            for attempt
            in generation_attempts
        )

        completion_tokens = sum(
            int(
                attempt[
                    "completion_tokens"
                ]
            )
            for attempt
            in generation_attempts
        )

        total_tokens = sum(
            int(
                attempt[
                    "total_tokens"
                ]
            )
            for attempt
            in generation_attempts
        )

        attempt_costs = [
            attempt[
                "estimated_cost_usd"
            ]
            for attempt
            in generation_attempts
        ]

        estimated_cost_usd = (
            sum(
                float(cost)
                for cost
                in attempt_costs
            )
            if all(
                cost is not None
                for cost
                in attempt_costs
            )
            else None
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "query": str(query),
            "product_id": (
                int(product_id)
                if product_id
                is not None
                else None
            ),
            "answer": payload.get(
                "answer",
                "",
            ),
            "evidence_ids": (
                evidence_ids
            ),
            "confidence": (
                payload.get(
                    "confidence"
                )
            ),
            "insufficient_evidence": (
                payload.get(
                    "insufficient_evidence"
                )
            ),
            "citation_valid": (
                citation_valid
            ),
            "citation_repaired": (
                citation_repaired
            ),
            "citation_retry_count": (
                citation_retry_count
            ),
            "validation_errors": (
                validation_errors
            ),
            "retrieved_ids": (
                retrieved_ids
            ),
            "candidate_count": len(
                candidate_ids
            ),
            "retrieved_documents": (
                retrieved
            ),
            "evidence_documents": (
                evidence
            ),
            "telemetry": {
                "retrieval_latency_ms": (
                    float(
                        retrieval_latency_ms
                    )
                ),
                "generation_latency_ms": (
                    float(
                        generation_latency_ms
                    )
                ),
                "total_latency_ms": (
                    float(
                        total_latency_ms
                    )
                ),
                "model": generation[
                    "model"
                ],
                "prompt_tokens": (
                    prompt_tokens
                ),
                "completion_tokens": (
                    completion_tokens
                ),
                "total_tokens": (
                    total_tokens
                ),
                "estimated_cost_usd": (
                    estimated_cost_usd
                ),
                "citation_retry_count": (
                    citation_retry_count
                ),
                "citation_repaired": (
                    citation_repaired
                ),
            },
        }
