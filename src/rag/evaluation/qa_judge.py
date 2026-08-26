from .qa_judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)


JUDGE_DIMENSIONS = (
    "correctness",
    "relevance",
    "completeness",
    "groundedness",
    "instruction_following",
    "safety",
)


class QAJudge:

    def __init__(
        self,
        generator,
        max_context_chars=12_000,
        max_chars_per_comment=1_500,
    ):
        self.generator = generator
        self.max_context_chars = int(
            max_context_chars
        )
        self.max_chars_per_comment = int(
            max_chars_per_comment
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
                "Judge response must be a JSON object"
            )

        for dimension in JUDGE_DIMENSIONS:
            value = payload.get(
                dimension
            )

            if not isinstance(
                value,
                dict,
            ):
                raise ValueError(
                    f"Missing judge dimension: {dimension}"
                )

            score = value.get(
                "score"
            )

            try:
                score = float(
                    score
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Invalid score for {dimension}"
                ) from exc

            if not (
                1.0
                <= score
                <= 5.0
            ):
                raise ValueError(
                    f"{dimension} score must be 1..5"
                )

            value["score"] = score

            reason = value.get(
                "reason",
                "",
            )

            value["reason"] = str(
                reason
            ).strip()

        tags = payload.get(
            "failure_tags",
            [],
        )

        if not isinstance(
            tags,
            list,
        ):
            raise ValueError(
                "failure_tags must be a list"
            )

        payload["failure_tags"] = [
            str(tag).strip()
            for tag in tags
            if str(tag).strip()
        ]

        payload[
            "summary_reason"
        ] = str(
            payload.get(
                "summary_reason",
                "",
            )
        ).strip()

        return payload


    def judge(
        self,
        query,
        retrieved_documents,
        answer,
        evidence_ids,
    ):
        user_prompt = build_judge_prompt(
            query=query,
            retrieved_documents=(
                retrieved_documents
            ),
            answer=answer,
            evidence_ids=evidence_ids,
            max_context_chars=(
                self.max_context_chars
            ),
            max_chars_per_comment=(
                self.max_chars_per_comment
            ),
        )

        generation = (
            self.generator
            .generate(
                system_prompt=(
                    JUDGE_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
            )
        )

        payload = self._validate_payload(
            generation["payload"]
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
