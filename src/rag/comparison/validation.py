ALLOWED_STANCES = {
    "positive",
    "mixed",
    "negative",
    "unknown",
}

ALLOWED_CONFIDENCE = {
    "high",
    "medium",
    "low",
}


def _normalize_int(
    value,
):
    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def validate_comparison_response(
    payload,
    product_ids,
    allowed_evidence_by_product,
):
    errors = []

    selected = [
        int(value)
        for value
        in product_ids
    ]

    selected_set = set(
        selected
    )

    if not isinstance(
        payload,
        dict,
    ):
        return False, [
            "response_is_not_object"
        ]

    summary = payload.get(
        "summary"
    )

    if (
        not isinstance(
            summary,
            str,
        )
        or not summary.strip()
    ):
        errors.append(
            "invalid_summary"
        )

    criteria = payload.get(
        "criteria"
    )

    if (
        not isinstance(
            criteria,
            list,
        )
        or len(criteria) == 0
    ):
        errors.append(
            "invalid_criteria"
        )
        criteria = []

    criterion_names = set()

    for criterion_index, criterion in enumerate(
        criteria
    ):
        prefix = (
            f"criterion_{criterion_index}"
        )

        if not isinstance(
            criterion,
            dict,
        ):
            errors.append(
                f"{prefix}:not_object"
            )
            continue

        name = criterion.get(
            "name"
        )

        if (
            not isinstance(
                name,
                str,
            )
            or not name.strip()
        ):
            errors.append(
                f"{prefix}:invalid_name"
            )
        else:
            normalized_name = (
                name.strip()
            )

            if (
                normalized_name
                in criterion_names
            ):
                errors.append(
                    f"{prefix}:duplicate_name"
                )

            criterion_names.add(
                normalized_name
            )

        assessments = criterion.get(
            "assessments"
        )

        if not isinstance(
            assessments,
            list,
        ):
            errors.append(
                f"{prefix}:invalid_assessments"
            )
            assessments = []

        seen_products = set()

        for assessment_index, assessment in enumerate(
            assessments
        ):
            assessment_prefix = (
                f"{prefix}:assessment_"
                f"{assessment_index}"
            )

            if not isinstance(
                assessment,
                dict,
            ):
                errors.append(
                    f"{assessment_prefix}:not_object"
                )
                continue

            product_id = _normalize_int(
                assessment.get(
                    "product_id"
                )
            )

            if (
                product_id
                not in selected_set
            ):
                errors.append(
                    f"{assessment_prefix}:invalid_product_id"
                )
                continue

            if (
                product_id
                in seen_products
            ):
                errors.append(
                    f"{assessment_prefix}:duplicate_product"
                )

            seen_products.add(
                product_id
            )

            stance = assessment.get(
                "stance"
            )

            if stance not in (
                ALLOWED_STANCES
            ):
                errors.append(
                    f"{assessment_prefix}:invalid_stance"
                )

            text = assessment.get(
                "text"
            )

            if (
                not isinstance(
                    text,
                    str,
                )
                or not text.strip()
            ):
                errors.append(
                    f"{assessment_prefix}:invalid_text"
                )

            evidence_ids = assessment.get(
                "evidence_ids"
            )

            if not isinstance(
                evidence_ids,
                list,
            ):
                errors.append(
                    f"{assessment_prefix}:invalid_evidence_ids"
                )
                evidence_ids = []

            normalized_ids = []

            for evidence_id in (
                evidence_ids
            ):
                normalized = _normalize_int(
                    evidence_id
                )

                if normalized is None:
                    errors.append(
                        f"{assessment_prefix}:non_integer_evidence"
                    )
                    continue

                normalized_ids.append(
                    normalized
                )

            if len(
                normalized_ids
            ) != len(
                set(
                    normalized_ids
                )
            ):
                errors.append(
                    f"{assessment_prefix}:duplicate_evidence"
                )

            allowed = {
                int(value)
                for value
                in allowed_evidence_by_product.get(
                    product_id,
                    [],
                )
            }

            invalid = (
                set(
                    normalized_ids
                )
                - allowed
            )

            if invalid:
                errors.append(
                    f"{assessment_prefix}:evidence_not_owned:"
                    + ",".join(
                        str(value)
                        for value
                        in sorted(
                            invalid
                        )
                    )
                )

        if (
            seen_products
            != selected_set
        ):
            missing = sorted(
                selected_set
                - seen_products
            )

            extra = sorted(
                seen_products
                - selected_set
            )

            errors.append(
                f"{prefix}:assessment_product_set_mismatch:"
                f"missing={missing},extra={extra}"
            )

        winner = criterion.get(
            "winner_product_id"
        )

        if winner is not None:
            winner = _normalize_int(
                winner
            )

            if winner not in (
                selected_set
            ):
                errors.append(
                    f"{prefix}:invalid_winner"
                )

        winner_reason = criterion.get(
            "winner_reason"
        )

        if not isinstance(
            winner_reason,
            str,
        ):
            errors.append(
                f"{prefix}:invalid_winner_reason"
            )

    overall_winner = payload.get(
        "overall_winner_product_id"
    )

    if overall_winner is not None:
        overall_winner = _normalize_int(
            overall_winner
        )

        if overall_winner not in (
            selected_set
        ):
            errors.append(
                "invalid_overall_winner"
            )

    recommendation = payload.get(
        "overall_recommendation"
    )

    if (
        not isinstance(
            recommendation,
            str,
        )
        or not recommendation.strip()
    ):
        errors.append(
            "invalid_overall_recommendation"
        )

    confidence = payload.get(
        "confidence"
    )

    if confidence not in (
        ALLOWED_CONFIDENCE
    ):
        errors.append(
            "invalid_confidence"
        )

    insufficient = payload.get(
        "insufficient_evidence"
    )

    if not isinstance(
        insufficient,
        bool,
    ):
        errors.append(
            "invalid_insufficient_evidence"
        )

    return (
        len(errors) == 0,
        errors,
    )


def sanitize_comparison_response(
    payload,
    product_ids,
    allowed_evidence_by_product,
):
    selected = [
        int(value)
        for value
        in product_ids
    ]

    selected_set = set(
        selected
    )

    if not isinstance(
        payload,
        dict,
    ):
        payload = {}

    criteria = payload.get(
        "criteria"
    )

    if not isinstance(
        criteria,
        list,
    ):
        criteria = []

    sanitized_criteria = []
    used_names = set()

    for criterion_index, criterion in enumerate(
        criteria
    ):
        if not isinstance(
            criterion,
            dict,
        ):
            continue

        name = str(
            criterion.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            name = (
                f"معیار {criterion_index + 1}"
            )

        if name in used_names:
            name = (
                f"{name} {criterion_index + 1}"
            )

        used_names.add(
            name
        )

        assessment_lookup = {}

        assessments = criterion.get(
            "assessments"
        )

        if not isinstance(
            assessments,
            list,
        ):
            assessments = []

        for assessment in (
            assessments
        ):
            if not isinstance(
                assessment,
                dict,
            ):
                continue

            product_id = _normalize_int(
                assessment.get(
                    "product_id"
                )
            )

            if (
                product_id
                not in selected_set
                or product_id
                in assessment_lookup
            ):
                continue

            stance = assessment.get(
                "stance"
            )

            if stance not in (
                ALLOWED_STANCES
            ):
                stance = "unknown"

            text = str(
                assessment.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                text = (
                    "شواهد کافی برای جمع‌بندی این معیار وجود ندارد."
                )

            allowed = {
                int(value)
                for value
                in allowed_evidence_by_product.get(
                    product_id,
                    [],
                )
            }

            valid_ids = []

            for evidence_id in assessment.get(
                "evidence_ids",
                [],
            ):
                normalized = _normalize_int(
                    evidence_id
                )

                if (
                    normalized in allowed
                    and normalized
                    not in valid_ids
                ):
                    valid_ids.append(
                        normalized
                    )

            assessment_lookup[
                product_id
            ] = {
                "product_id": (
                    product_id
                ),
                "stance": stance,
                "text": text,
                "evidence_ids": (
                    valid_ids
                ),
            }

        sanitized_assessments = []

        for product_id in selected:
            sanitized_assessments.append(
                assessment_lookup.get(
                    product_id,
                    {
                        "product_id": (
                            product_id
                        ),
                        "stance": (
                            "unknown"
                        ),
                        "text": (
                            "شواهد کافی برای جمع‌بندی این معیار وجود ندارد."
                        ),
                        "evidence_ids": [],
                    },
                )
            )

        winner = _normalize_int(
            criterion.get(
                "winner_product_id"
            )
        )

        if winner not in selected_set:
            winner = None

        sanitized_criteria.append(
            {
                "name": name,
                "assessments": (
                    sanitized_assessments
                ),
                "winner_product_id": (
                    winner
                ),
                "winner_reason": str(
                    criterion.get(
                        "winner_reason",
                        "",
                    )
                ).strip(),
            }
        )

    if not sanitized_criteria:
        sanitized_criteria = [
            {
                "name": "جمع‌بندی شواهد",
                "assessments": [
                    {
                        "product_id": (
                            product_id
                        ),
                        "stance": (
                            "unknown"
                        ),
                        "text": (
                            "شواهد کافی برای مقایسه ساختاریافته وجود ندارد."
                        ),
                        "evidence_ids": [],
                    }
                    for product_id
                    in selected
                ],
                "winner_product_id": None,
                "winner_reason": "",
            }
        ]

    overall_winner = _normalize_int(
        payload.get(
            "overall_winner_product_id"
        )
    )

    if overall_winner not in (
        selected_set
    ):
        overall_winner = None

    confidence = payload.get(
        "confidence"
    )

    if confidence not in (
        ALLOWED_CONFIDENCE
    ):
        confidence = "low"

    insufficient = payload.get(
        "insufficient_evidence"
    )

    if not isinstance(
        insufficient,
        bool,
    ):
        insufficient = True

    summary = str(
        payload.get(
            "summary",
            "",
        )
    ).strip()

    if not summary:
        summary = (
            "شواهد کافی برای یک جمع‌بندی قطعی وجود ندارد."
        )

    recommendation = str(
        payload.get(
            "overall_recommendation",
            "",
        )
    ).strip()

    if not recommendation:
        recommendation = (
            "برنده‌ی قطعی از شواهد موجود قابل تعیین نیست."
        )

    return {
        "summary": summary,
        "criteria": sanitized_criteria,
        "overall_winner_product_id": (
            overall_winner
        ),
        "overall_recommendation": (
            recommendation
        ),
        "confidence": confidence,
        "insufficient_evidence": (
            insufficient
        ),
    }
