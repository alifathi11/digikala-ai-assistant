import time

from ..comparison import (
    COMPARISON_SYSTEM_PROMPT,
    ComparisonContextService,
    build_comparison_prompt,
    build_comparison_repair_prompt,
    sanitize_comparison_response,
    validate_comparison_response,
)


class ProductComparisonPipeline:
    """
    Grounded comparison for 2-3 already-selected products.

    Product discovery is intentionally outside this pipeline. Comparison uses
    direct product metadata plus candidate-scoped review evidence for each
    selected product.
    """

    DEFAULT_QUERY = (
        "این محصولات را از نظر تفاوت‌های مهم، نقاط قوت و ضعف و تجربه‌ی "
        "کاربران مقایسه کن و اگر شواهد کافی است مناسب‌ترین گزینه را مشخص کن."
    )

    def __init__(
        self,
        product_documents,
        review_retriever,
        generator,
        review_documents=None,
        reviews_per_product=3,
        min_products=2,
        max_products=3,
        max_context_chars=18_000,
        max_chars_per_review=900,
        product_id_to_row=None,
    ):
        self.generator = generator

        self.min_products = int(
            min_products
        )

        self.max_products = int(
            max_products
        )

        if self.min_products < 2:
            raise ValueError(
                "min_products must be at least 2."
            )

        if (
            self.max_products
            < self.min_products
        ):
            raise ValueError(
                "max_products must be >= min_products."
            )

        self.max_context_chars = int(
            max_context_chars
        )

        self.max_chars_per_review = int(
            max_chars_per_review
        )

        self.context_service = (
            ComparisonContextService(
                product_documents=(
                    product_documents
                ),
                review_retriever=(
                    review_retriever
                ),
                review_documents=(
                    review_documents
                ),
                reviews_per_product=(
                    reviews_per_product
                ),
                product_id_to_row=(
                    product_id_to_row
                ),
            )
        )


    def _validate_product_count(
        self,
        product_ids,
    ):
        product_ids = (
            self.context_service
            .normalize_product_ids(
                product_ids
            )
        )

        if len(product_ids) < (
            self.min_products
        ):
            raise ValueError(
                "Comparison requires at least "
                f"{self.min_products} unique products."
            )

        if len(product_ids) > (
            self.max_products
        ):
            raise ValueError(
                "Comparison supports at most "
                f"{self.max_products} products."
            )

        return product_ids


    @staticmethod
    def _evidence_ids_by_product(
        payload,
        product_ids,
    ):
        result = {
            int(product_id): []
            for product_id
            in product_ids
        }

        for criterion in payload.get(
            "criteria",
            [],
        ):
            if not isinstance(
                criterion,
                dict,
            ):
                continue

            for assessment in criterion.get(
                "assessments",
                [],
            ):
                if not isinstance(
                    assessment,
                    dict,
                ):
                    continue

                try:
                    product_id = int(
                        assessment.get(
                            "product_id"
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if product_id not in result:
                    continue

                for value in assessment.get(
                    "evidence_ids",
                    [],
                ):
                    try:
                        evidence_id = int(
                            value
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if evidence_id not in (
                        result[
                            product_id
                        ]
                    ):
                        result[
                            product_id
                        ].append(
                            evidence_id
                        )

        return result


    def compare(
        self,
        product_ids,
        query=None,
    ):
        total_start = time.perf_counter()

        product_ids = (
            self._validate_product_count(
                product_ids
            )
        )

        query = str(
            query
            or self.DEFAULT_QUERY
        ).strip()

        products = (
            self.context_service
            .get_products(
                product_ids
            )
        )

        (
            reviews,
            retrieval_telemetry,
        ) = (
            self.context_service
            .retrieve_reviews(
                query=query,
                product_ids=(
                    product_ids
                ),
            )
        )

        allowed_evidence = (
            self.context_service
            .allowed_evidence_by_product(
                review_documents=reviews,
                product_ids=product_ids,
            )
        )

        user_prompt = (
            build_comparison_prompt(
                query=query,
                product_metadata=products,
                review_documents=reviews,
                max_context_chars=(
                    self.max_context_chars
                ),
                max_chars_per_review=(
                    self.max_chars_per_review
                ),
            )
        )

        first_generation = (
            self.generator
            .generate(
                system_prompt=(
                    COMPARISON_SYSTEM_PROMPT
                ),
                user_prompt=(
                    user_prompt
                ),
            )
        )

        attempts = [
            first_generation
        ]

        payload = first_generation[
            "payload"
        ]

        valid, validation_errors = (
            validate_comparison_response(
                payload=payload,
                product_ids=product_ids,
                allowed_evidence_by_product=(
                    allowed_evidence
                ),
            )
        )

        repair_count = 0
        repaired = False

        if not valid:
            repair_count = 1

            repair_prompt = (
                build_comparison_repair_prompt(
                    original_user_prompt=(
                        user_prompt
                    ),
                    previous_payload=payload,
                    product_ids=product_ids,
                    allowed_evidence_by_product=(
                        allowed_evidence
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
                        COMPARISON_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        repair_prompt
                    ),
                )
            )

            attempts.append(
                repair_generation
            )

            repaired_payload = (
                repair_generation[
                    "payload"
                ]
            )

            repaired_valid, repaired_errors = (
                validate_comparison_response(
                    payload=(
                        repaired_payload
                    ),
                    product_ids=product_ids,
                    allowed_evidence_by_product=(
                        allowed_evidence
                    ),
                )
            )

            if repaired_valid:
                payload = (
                    repaired_payload
                )
                valid = True
                validation_errors = []
                repaired = True
            else:
                payload = (
                    sanitize_comparison_response(
                        payload=(
                            repaired_payload
                        ),
                        product_ids=product_ids,
                        allowed_evidence_by_product=(
                            allowed_evidence
                        ),
                    )
                )

                valid, validation_errors = (
                    validate_comparison_response(
                        payload=payload,
                        product_ids=product_ids,
                        allowed_evidence_by_product=(
                            allowed_evidence
                        ),
                    )
                )

                repaired = True

        evidence_ids_by_product = (
            self._evidence_ids_by_product(
                payload=payload,
                product_ids=product_ids,
            )
        )

        evidence_documents = (
            self.context_service
            .select_evidence_rows(
                review_documents=reviews,
                evidence_ids_by_product=(
                    evidence_ids_by_product
                ),
            )
        )

        generation_latency_ms = sum(
            float(
                attempt[
                    "latency_ms"
                ]
            )
            for attempt
            in attempts
        )

        prompt_tokens = sum(
            int(
                attempt[
                    "prompt_tokens"
                ]
            )
            for attempt
            in attempts
        )

        completion_tokens = sum(
            int(
                attempt[
                    "completion_tokens"
                ]
            )
            for attempt
            in attempts
        )

        total_tokens = sum(
            int(
                attempt[
                    "total_tokens"
                ]
            )
            for attempt
            in attempts
        )

        costs = [
            attempt[
                "estimated_cost_usd"
            ]
            for attempt
            in attempts
        ]

        estimated_cost_usd = (
            sum(
                float(value)
                for value
                in costs
            )
            if all(
                value is not None
                for value
                in costs
            )
            else None
        )

        return {
            "query": query,
            "product_ids": (
                product_ids
            ),
            "summary": payload.get(
                "summary",
                "",
            ),
            "criteria": payload.get(
                "criteria",
                [],
            ),
            "overall_winner_product_id": (
                payload.get(
                    "overall_winner_product_id"
                )
            ),
            "overall_recommendation": (
                payload.get(
                    "overall_recommendation",
                    "",
                )
            ),
            "confidence": payload.get(
                "confidence"
            ),
            "insufficient_evidence": (
                payload.get(
                    "insufficient_evidence"
                )
            ),
            "citation_valid": valid,
            "citation_repaired": repaired,
            "citation_retry_count": (
                repair_count
            ),
            "validation_errors": (
                validation_errors
            ),
            "product_metadata": products,
            "retrieved_reviews": reviews,
            "retrieved_review_ids_by_product": {
                product_id: sorted(
                    int(value)
                    for value
                    in allowed_evidence.get(
                        product_id,
                        set(),
                    )
                )
                for product_id
                in product_ids
            },
            "evidence_ids_by_product": (
                evidence_ids_by_product
            ),
            "evidence_documents": (
                evidence_documents
            ),
            "telemetry": {
                **retrieval_telemetry,
                "generation_latency_ms": (
                    float(
                        generation_latency_ms
                    )
                ),
                "total_latency_ms": float(
                    (
                        time.perf_counter()
                        - total_start
                    )
                    * 1000
                ),
                "model": attempts[
                    -1
                ][
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
                    repair_count
                ),
                "citation_repaired": (
                    repaired
                ),
            },
        }
