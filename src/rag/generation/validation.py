ALLOWED_CONFIDENCE = {
    "high",
    "medium",
    "low",
}


def validate_grounded_response(
    payload,
    retrieved_ids,
):
    errors = []

    if not isinstance(
        payload,
        dict,
    ):
        return False, [
            "response_is_not_object"
        ]

    answer = payload.get(
        "answer"
    )

    evidence_ids = payload.get(
        "evidence_ids"
    )

    confidence = payload.get(
        "confidence"
    )

    insufficient = payload.get(
        "insufficient_evidence"
    )

    if (
        not isinstance(
            answer,
            str,
        )
        or not answer.strip()
    ):
        errors.append(
            "invalid_answer"
        )

    if not isinstance(
        evidence_ids,
        list,
    ):
        errors.append(
            "invalid_evidence_ids"
        )

        evidence_ids = []

    normalized_ids = []

    for value in evidence_ids:
        try:
            normalized_ids.append(
                int(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            errors.append(
                "non_integer_evidence_id"
            )

    if len(
        normalized_ids
    ) != len(
        set(normalized_ids)
    ):
        errors.append(
            "duplicate_evidence_ids"
        )

    invalid_ids = (
        set(normalized_ids)
        - set(
            int(x)
            for x in retrieved_ids
        )
    )

    if invalid_ids:
        errors.append(
            "evidence_not_retrieved:"
            + ",".join(
                str(x)
                for x in sorted(
                    invalid_ids
                )
            )
        )

    if confidence not in (
        ALLOWED_CONFIDENCE
    ):
        errors.append(
            "invalid_confidence"
        )

    if not isinstance(
        insufficient,
        bool,
    ):
        errors.append(
            "invalid_insufficient_evidence"
        )

    if (
        insufficient is False
        and len(normalized_ids) == 0
    ):
        errors.append(
            "supported_answer_without_evidence"
        )

    return (
        len(errors) == 0,
        errors,
    )
