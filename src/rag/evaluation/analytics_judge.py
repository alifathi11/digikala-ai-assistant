
from .analytics_judge_prompt import (
    ANALYTICS_JUDGE_SYSTEM_PROMPT,
    DIMENSIONS,
    build_analytics_judge_prompt,
)


ALLOWED_FAILURE_TAGS = {
    "unsupported_claim",
    "semantic_numeric_misinterpretation",
    "missed_requested_metric",
    "historical_price_used",
    "review_volume_overclaim",
    "brand_market_share_overclaim",
    "brand_coverage_ignored",
    "rating_coverage_ignored",
    "zero_review_mishandled",
    "comparison_direction_error",
    "off_topic",
    "format_issue",
}


class ManagerAnalyticsJudge:

    def __init__(
        self,
        generator,
    ):
        self.generator = generator


    @staticmethod
    def _validate_payload(
        payload,
    ):
        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Analytics judge payload must be an object."
            )

        result = {}

        for dimension in (
            DIMENSIONS
        ):
            value = payload.get(
                dimension
            )

            if not isinstance(
                value,
                dict,
            ):
                raise ValueError(
                    f"Missing analytics judge dimension: {dimension}"
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
                    f"Invalid analytics judge score for {dimension}"
                ) from error

            if score < 1 or score > 5:
                raise ValueError(
                    f"Analytics judge score must be in [1,5] for {dimension}."
                )

            result[
                dimension
            ] = {
                "score": score,
                "reason": str(
                    value.get(
                        "reason",
                        "",
                    )
                ).strip(),
            }

        failure_tags = []

        for value in (
            payload.get(
                "failure_tags",
                [],
            )
            or []
        ):
            tag = str(
                value
            ).strip()

            if (
                tag in (
                    ALLOWED_FAILURE_TAGS
                )
                and tag not in (
                    failure_tags
                )
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
        case,
        generated_result,
    ):
        generation = (
            self.generator
            .generate(
                system_prompt=(
                    ANALYTICS_JUDGE_SYSTEM_PROMPT
                ),
                user_prompt=(
                    build_analytics_judge_prompt(
                        case=case,
                        generated_result=(
                            generated_result
                        ),
                    )
                ),
            )
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
