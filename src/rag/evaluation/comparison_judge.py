from .comparison_judge_prompt import (
    COMPARISON_JUDGE_SYSTEM_PROMPT,
    DIMENSIONS,
    build_comparison_judge_prompt,
)


ALLOWED_FAILURE_TAGS = {
    "unsupported_claim",
    "contradicts_evidence",
    "missed_criterion",
    "ignores_conflict",
    "overconfident_winner",
    "wrong_winner",
    "insufficient_evidence_mishandled",
    "cross_product_evidence",
    "off_topic",
    "format_issue",
    "unsafe_claim",
}


class ProductComparisonJudge:

    def __init__(
        self,
        generator,
        max_context_chars=18_000,
        max_chars_per_review=900,
    ):
        self.generator = generator
        self.max_context_chars = int(
            max_context_chars
        )
        self.max_chars_per_review = int(
            max_chars_per_review
        )


    @staticmethod
    def _validate_payload(
        payload,
    ):
        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Comparison judge payload must be an object."
            )

        result = {}

        for dimension in DIMENSIONS:
            value = payload.get(
                dimension
            )

            if not isinstance(
                value,
                dict,
            ):
                raise ValueError(
                    "Missing judge dimension: "
                    f"{dimension}"
                )

            try:
                score = int(
                    value.get(
                        "score"
                    )
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Invalid judge score for "
                    f"{dimension}"
                ) from error

            if score < 1 or score > 5:
                raise ValueError(
                    "Judge score must be in [1, 5] for "
                    f"{dimension}."
                )

            reason = str(
                value.get(
                    "reason",
                    "",
                )
            ).strip()

            result[
                dimension
            ] = {
                "score": score,
                "reason": reason,
            }

        failure_tags = []

        for value in payload.get(
            "failure_tags",
            [],
        ):
            tag = str(
                value
            ).strip()

            if (
                tag in ALLOWED_FAILURE_TAGS
                and tag not in failure_tags
            ):
                failure_tags.append(
                    tag
                )

        result[
            "failure_tags"
        ] = failure_tags

        result[
            "summary_reason"
        ] = str(
            payload.get(
                "summary_reason",
                "",
            )
        ).strip()

        return result


    def judge(
        self,
        query,
        product_metadata,
        retrieved_reviews,
        generated_result,
        case_notes="",
    ):
        prompt = build_comparison_judge_prompt(
            query=query,
            product_metadata=product_metadata,
            retrieved_reviews=retrieved_reviews,
            generated_result=generated_result,
            case_notes=case_notes,
            max_context_chars=(
                self.max_context_chars
            ),
            max_chars_per_review=(
                self.max_chars_per_review
            ),
        )

        generation = self.generator.generate(
            system_prompt=(
                COMPARISON_JUDGE_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
        )

        payload = self._validate_payload(
            generation[
                "payload"
            ]
        )

        return {
            "payload": payload,
            "telemetry": {
                key: value
                for key, value
                in generation.items()
                if key != "payload"
            },
        }
