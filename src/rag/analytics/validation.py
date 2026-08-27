
import re


PLACEHOLDER_RE = re.compile(
    r"\{\{metric:([A-Za-z0-9_.-]+)\}\}"
)

DIGIT_RE = re.compile(
    r"[0-9۰-۹٠-٩]"
)

ALLOWED_CONFIDENCE = {
    "high",
    "medium",
    "low",
}


def _text_errors(
    text,
    field_name,
    allowed_metric_keys,
):
    errors = []

    if (
        not isinstance(
            text,
            str,
        )
        or not text.strip()
    ):
        return [
            f"{field_name}:invalid_text"
        ]

    placeholders = (
        PLACEHOLDER_RE.findall(
            text
        )
    )

    invalid = sorted(
        set(
            placeholders
        )
        - set(
            allowed_metric_keys
        )
    )

    if invalid:
        errors.append(
            f"{field_name}:invalid_metric_refs:"
            + ",".join(
                invalid
            )
        )

    without_placeholders = (
        PLACEHOLDER_RE.sub(
            "",
            text,
        )
    )

    if DIGIT_RE.search(
        without_placeholders
    ):
        errors.append(
            f"{field_name}:literal_numeric_digit"
        )

    return errors


def validate_manager_response(
    payload,
    allowed_metric_keys,
):
    if not isinstance(
        payload,
        dict,
    ):
        return False, [
            "response_is_not_object"
        ]

    errors = []

    errors.extend(
        _text_errors(
            payload.get(
                "answer_template"
            ),
            "answer_template",
            allowed_metric_keys,
        )
    )

    insights = payload.get(
        "insights"
    )

    if not isinstance(
        insights,
        list,
    ):
        errors.append(
            "insights:not_list"
        )
        insights = []

    for index, insight in enumerate(
        insights
    ):
        prefix = (
            f"insight_{index}"
        )

        if not isinstance(
            insight,
            dict,
        ):
            errors.append(
                f"{prefix}:not_object"
            )
            continue

        errors.extend(
            _text_errors(
                insight.get(
                    "title"
                ),
                f"{prefix}:title",
                allowed_metric_keys,
            )
        )

        errors.extend(
            _text_errors(
                insight.get(
                    "text_template"
                ),
                f"{prefix}:text_template",
                allowed_metric_keys,
            )
        )

        refs = insight.get(
            "metric_refs"
        )

        if not isinstance(
            refs,
            list,
        ):
            errors.append(
                f"{prefix}:metric_refs_not_list"
            )
            refs = []

        invalid_refs = sorted(
            set(
                str(
                    value
                )
                for value
                in refs
            )
            - set(
                allowed_metric_keys
            )
        )

        if invalid_refs:
            errors.append(
                f"{prefix}:invalid_metric_refs_field:"
                + ",".join(
                    invalid_refs
                )
            )

    caveats = payload.get(
        "caveats"
    )

    if not isinstance(
        caveats,
        list,
    ):
        errors.append(
            "caveats:not_list"
        )
        caveats = []

    for index, caveat in enumerate(
        caveats
    ):
        errors.extend(
            _text_errors(
                caveat,
                f"caveat_{index}",
                allowed_metric_keys,
            )
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

    return (
        len(errors) == 0,
        errors,
    )


def render_metric_template(
    text,
    facts,
):
    def replace(
        match,
    ):
        key = match.group(
            1
        )

        fact = facts.get(
            key
        )

        if fact is None:
            return "—"

        return str(
            fact[
                "display_value"
            ]
        )

    return PLACEHOLDER_RE.sub(
        replace,
        str(
            text
        ),
    )


def sanitize_manager_response(
    payload,
    facts,
):
    allowed = set(
        facts
    )

    answer_template = (
        payload.get(
            "answer_template"
        )
        if isinstance(
            payload,
            dict,
        )
        else None
    )

    valid_answer = (
        isinstance(
            answer_template,
            str,
        )
        and not DIGIT_RE.search(
            PLACEHOLDER_RE.sub(
                "",
                answer_template,
            )
        )
        and set(
            PLACEHOLDER_RE.findall(
                answer_template
            )
        ).issubset(
            allowed
        )
    )

    if not valid_answer:
        core = [
            key
            for key in (
                "overview.product_count",
                "overview.median_price",
                "overview.review_coverage_pct",
            )
            if key in facts
        ]

        if core:
            fragments = [
                f"{{{{metric:{key}}}}}"
                for key in core
            ]

            answer_template = (
                "خلاصه‌ی عددی معتبر این محدوده بر اساس "
                "شاخص‌های محاسبه‌شده در داشبورد ارائه شده است: "
                + "، ".join(
                    fragments
                )
                + "."
            )
        else:
            answer_template = (
                "برای این سؤال داده‌ی عددی معتبر کافی در محدوده‌ی "
                "انتخاب‌شده وجود ندارد."
            )

    return {
        "answer_template": (
            answer_template
        ),
        "insights": [],
        "caveats": [
            "پاسخ به نسخه‌ی محافظه‌کارانه‌ی deterministic کاهش یافت."
        ],
        "confidence": "low",
    }
