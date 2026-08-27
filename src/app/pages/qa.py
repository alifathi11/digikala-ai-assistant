import html

import pandas as pd
import streamlit as st

from src.app.safety import (
    render_ui_error,
    require_mapping,
)


def _escape(
    value,
):
    return html.escape(
        str(value),
        quote=True,
    )


def _safe_value(
    row,
    field,
    default="—",
):
    value = getattr(
        row,
        field,
        None,
    )

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def _render_product(
    row,
):
    title = _safe_value(
        row,
        "title_fa",
        "بدون عنوان",
    )

    product_id = int(
        _safe_value(
            row,
            "id",
            -1,
        )
    )

    brand = _safe_value(
        row,
        "Brand",
    )

    category = _safe_value(
        row,
        "Category2",
        _safe_value(
            row,
            "Category1",
        ),
    )

    rate = _safe_value(
        row,
        "Rate",
    )

    price = _safe_value(
        row,
        "Price",
    )

    price_text = "—"

    try:
        if price != "—":
            price_text = (
                f"{float(price):,.0f}"
            )
    except (
        TypeError,
        ValueError,
    ):
        price_text = str(
            price
        )

    st.markdown(
        f"""
        <div class="product-card">
            <div class="product-title-row">
                <div class="product-icon">▣</div>
                <div class="product-title">{_escape(title)}</div>
            </div>
            <div class="product-meta">
                <span class="meta-chip"><strong>شناسه</strong> {product_id}</span>
                <span class="meta-chip"><strong>برند</strong> {_escape(brand)}</span>
                <span class="meta-chip"><strong>دسته</strong> {_escape(category)}</span>
                <span class="meta-chip"><strong>امتیاز</strong> {_escape(rate)}</span>
                <span class="meta-chip"><strong>قیمت</strong> {_escape(price_text)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_evidence(
    evidence_documents,
):
    if (
        evidence_documents
        is None
        or len(
            evidence_documents
        )
        == 0
    ):
        st.caption(
            "برای این پاسخ شاهدی "
            "انتخاب نشده است."
        )
        return

    for row in (
        evidence_documents
        .itertuples(
            index=False
        )
    ):
        comment_id = int(
            getattr(
                row,
                "id"
            )
        )

        rate = getattr(
            row,
            "rate",
            "—",
        )

        body = getattr(
            row,
            "body",
            "",
        )

        if body is None:
            body = ""

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-meta">
                    <span>نظر #{comment_id}</span>
                    <span>·</span>
                    <span>امتیاز {_escape(rate)}</span>
                </div>
                <div class="evidence-body">{_escape(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_telemetry(
    telemetry,
):
    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "بازیابی",
        (
            f"{telemetry.get('retrieval_latency_ms', 0):.0f} ms"
        ),
    )

    col2.metric(
        "تولید پاسخ",
        (
            f"{telemetry.get('generation_latency_ms', 0) / 1000:.2f} s"
        ),
    )

    col3.metric(
        "زمان کل",
        (
            f"{telemetry.get('total_latency_ms', 0) / 1000:.2f} s"
        ),
    )

    st.caption(
        "توکن‌ها: "
        f"{telemetry.get('total_tokens', 0)}"
        " · مدل: "
        f"{telemetry.get('model', '—')}"
    )


def _history_key(
    product_id,
):
    return (
        f"qa_history_"
        f"{int(product_id)}"
    )


def render(
    services,
):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-copy">
                <div class="hero-kicker">● دستیار تحلیل نظرات</div>
                <h1>از تجربه‌ی واقعی کاربران بپرسید</h1>
                <p>
                    محصول را انتخاب کنید و سؤال‌تان را بپرسید؛ پاسخ هوشمند
                    فقط بر اساس نظرات بازیابی‌شده ساخته می‌شود و شواهد آن
                    همیشه در دسترس شماست.
                </p>
            </div>
            <div class="hero-icon">✦</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    product_id = st.selectbox(
        "محصول موردنظر",
        options=(
            services
            .catalog
            .product_ids
        ),
        index=None,
        format_func=(
            services
            .catalog
            .label
        ),
        placeholder=(
            "نام یا شناسه‌ی "
            "محصول را جست‌وجو کنید"
        ),
    )

    if product_id is None:
        st.info(
            "برای شروع یک محصول "
            "انتخاب کنید."
        )
        return

    product = (
        services
        .catalog
        .get(
            product_id
        )
    )

    _render_product(
        product
    )

    key = _history_key(
        product_id
    )

    if key not in (
        st.session_state
    ):
        st.session_state[
            key
        ] = []

    history = (
        st.session_state[
            key
        ]
    )

    if not history:
        st.markdown(
            """
            <div class="hint-card">
                <span class="hint-icon">✦</span>
                <span>
                    نمونه سؤال: «ایرادهای پرتکرار این محصول چیست؟» یا
                    «برای پوست چرب چه تجربه‌ای گزارش شده؟»
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in history:
        with st.chat_message(
            message[
                "role"
            ]
        ):
            st.markdown(
                message[
                    "content"
                ]
            )

            if (
                message[
                    "role"
                ]
                == "assistant"
            ):
                if message.get(
                    "insufficient_evidence"
                ):
                    st.warning(
                        "شواهد بازیابی‌شده "
                        "برای پاسخ قطعی کافی "
                        "نبوده است."
                    )

                confidence = (
                    message.get(
                        "confidence"
                    )
                )

                if confidence:
                    st.caption(
                        "اطمینان پاسخ: "
                        f"{confidence}"
                    )

                with st.expander(
                    "شواهد پاسخ"
                ):
                    _render_evidence(
                        message.get(
                            "evidence_documents"
                        )
                    )

                with st.expander(
                    "جزئیات فنی"
                ):
                    _render_telemetry(
                        message.get(
                            "telemetry",
                            {},
                        )
                    )

    prompt = st.chat_input(
        "سؤال‌تان درباره‌ی "
        "تجربه‌ی کاربران..."
    )

    if not prompt:
        return

    history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            prompt
        )

    with st.chat_message(
        "assistant"
    ):
        try:
            with st.spinner(
                "در حال بررسی نظرات..."
            ):
                result = (
                    services
                    .qa
                    .answer(
                        query=prompt,
                        product_id=(
                            product_id
                        ),
                    )
                )

            result = require_mapping(
                result,
                required_keys=(
                    "answer",
                ),
                label="پاسخ پرسش و پاسخ",
            )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            if not answer:
                raise ValueError(
                    "پاسخ مدل خالی است."
                )

            insufficient_evidence = bool(
                result.get(
                    "insufficient_evidence",
                    True,
                )
            )

            confidence = result.get(
                "confidence",
                "نامشخص",
            )

            evidence_documents = result.get(
                "evidence_documents"
            )

            telemetry = result.get(
                "telemetry",
                {},
            )

            st.markdown(
                answer
            )

            if insufficient_evidence:
                st.warning(
                    "شواهد بازیابی‌شده "
                    "برای پاسخ قطعی کافی "
                    "نبوده است."
                )

            st.caption(
                "اطمینان پاسخ: "
                f"{confidence}"
            )

            with st.expander(
                "شواهد پاسخ",
                expanded=True,
            ):
                _render_evidence(
                    evidence_documents
                )

            with st.expander(
                "جزئیات فنی"
            ):
                _render_telemetry(
                    telemetry
                )

        except Exception as exc:
            render_ui_error(
                "پاسخ مدل قابل پردازش نبود و چیزی به‌عنوان پاسخ نهایی ثبت نشد.",
                exc,
            )
            return

    history.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "insufficient_evidence": (
                insufficient_evidence
            ),
            "evidence_documents": (
                evidence_documents
            ),
            "telemetry": telemetry,
        }
    )
