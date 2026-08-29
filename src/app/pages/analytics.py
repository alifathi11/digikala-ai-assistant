
import html
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.app.safety import (
    render_ui_error,
    require_mapping,
)


CONFIDENCE_LABELS = {
    "high": "بالا",
    "medium": "متوسط",
    "low": "پایین",
}


def _escape(
    value,
):
    return html.escape(
        str(
            value
        ),
        quote=True,
    )


def _price(
    value,
):
    if value is None:
        return "—"

    try:
        if pd.isna(
            value
        ):
            return "—"
    except (
        TypeError,
        ValueError,
    ):
        pass

    return f"{float(value):,.0f}"


def _percent(
    value,
):
    if value is None:
        return "—"

    return f"{float(value) * 100:.1f}%"


def _score(
    value,
    scale=100,
):
    if value is None:
        return "—"

    try:
        if pd.isna(
            value
        ):
            return "—"
    except (
        TypeError,
        ValueError,
    ):
        pass

    return f"{float(value):.2f}/{scale}"


def _scope_label(
    filters,
):
    parts = []

    if filters.get(
        "Category1"
    ):
        parts.append(
            str(
                filters[
                    "Category1"
                ]
            )
        )

    if filters.get(
        "Category2"
    ):
        parts.append(
            str(
                filters[
                    "Category2"
                ]
            )
        )

    if not parts:
        return "کل کاتالوگ"

    return " ← ".join(
        parts
    )


def _render_hero():
    st.html(
        dedent(
            """
            <div class="hero analytics-hero">
                <div class="hero-copy">
                    <div class="hero-kicker">● تحلیل قطعی از داده‌ی واقعی</div>
                    <h1>نمای مدیریتی کاتالوگ، قیمت، امتیاز و پوشش بازخورد</h1>
                    <p>
                        همه‌ی KPIها در Python محاسبه می‌شوند. مدل زبانی فقط
                        توضیح مدیریتی تولید می‌کند و اجازه‌ی ساختن عدد ندارد.
                    </p>
                </div>
                <div class="hero-icon">📊</div>
            </div>
            """
        ).strip()
    )


def _render_quality_notice():
    st.html(
        dedent(
            """
            <div class="analytics-quality-note">
                <strong>محدودیت‌های داده</strong>
                <span>
                    Brand فقط برای حدود ۴۴٪ کاتالوگ قابل اتکاست؛
                    historical price به دلیل coverage پایین استفاده نمی‌شود؛
                    و review_count به‌عنوان حجم واقعی بازار رتبه‌بندی نمی‌شود.
                </span>
            </div>
            """
        ).strip()
    )


def _render_overview(
    overview,
):
    rating = overview[
        "product_rating"
    ]

    columns = st.columns(
        4
    )

    columns[
        0
    ].metric(
        "محصول",
        f"{overview['product_count']:,}",
    )

    columns[
        1
    ].metric(
        "میانه قیمت",
        (
            f"{_price(overview['price']['median'])} تومان"
        ),
    )

    columns[
        2
    ].metric(
        "پوشش review",
        _percent(
            overview[
                "review_coverage"
            ]
        ),
    )

    columns[
        3
    ].metric(
        "پوشش rating",
        _percent(
            rating[
                "rated_product_coverage"
            ]
        ),
    )

    second = st.columns(
        4
    )

    second[
        0
    ].metric(
        "امتیاز وزنی محصول",
        _score(
            overview[
                "weighted_product_rating"
            ],
            100,
        ),
    )

    second[
        1
    ].metric(
        "معادل پنج‌نمره‌ای",
        _score(
            overview[
                "weighted_product_rating_5"
            ],
            5,
        ),
    )

    second[
        2
    ].metric(
        "امتیاز reviewها",
        _score(
            overview[
                "weighted_review_rating"
            ],
            5,
        ),
    )

    second[
        3
    ].metric(
        "تعداد امتیاز ثبت‌شده",
        f"{overview['rating_count_total']:,}",
    )


def _clean_table(
    frame,
    columns,
):
    available = [
        column
        for column
        in columns
        if column in (
            frame.columns
        )
    ]

    return (
        frame[
            available
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )



def _safe_bar_chart(
    frame,
    label_column,
    value_column,
    title,
):
    """
    Render a deterministic Streamlit bar chart without affecting the page
    when chart data is empty or malformed.
    """
    try:
        if (
            not isinstance(
                frame,
                pd.DataFrame,
            )
            or len(frame) == 0
            or label_column
            not in frame.columns
            or value_column
            not in frame.columns
        ):
            st.info(
                "داده‌ی کافی برای این نمودار وجود ندارد."
            )
            return

        chart = frame[
            [
                label_column,
                value_column,
            ]
        ].copy()

        chart[
            label_column
        ] = (
            chart[
                label_column
            ]
            .fillna("—")
            .astype(str)
        )

        chart[
            value_column
        ] = pd.to_numeric(
            chart[
                value_column
            ],
            errors="coerce",
        )

        chart = chart.dropna(
            subset=[
                value_column
            ]
        )

        if len(chart) == 0:
            st.info(
                "داده‌ی کافی برای این نمودار وجود ندارد."
            )
            return

        st.markdown(
            f"**{title}**"
        )

        st.bar_chart(
            chart.set_index(
                label_column
            )[
                [
                    value_column
                ]
            ],
            use_container_width=True,
        )

    except Exception as exc:
        render_ui_error(
            "نمایش نمودار با خطا مواجه شد.",
            exc,
            retry_hint=False,
        )


def _render_overview_charts(
    analytics,
    filters,
    overview,
):
    st.markdown(
        "### نمودارها"
    )

    left, right = st.columns(
        2
    )

    with left:
        try:
            brands = (
                analytics
                .top_brands(
                    filters=filters,
                    top_n=8,
                    include_generic=False,
                )
            )

            _safe_bar_chart(
                frame=brands,
                label_column="Brand",
                value_column="product_count",
                title="تعداد محصول برندهای برتر",
            )

        except Exception as exc:
            render_ui_error(
                "نمودار برندها قابل محاسبه نبود.",
                exc,
                retry_hint=False,
            )

    with right:
        coverage = pd.DataFrame(
            {
                "شاخص": [
                    "قیمت",
                    "Review",
                    "Rating",
                ],
                "پوشش (%)": [
                    (
                        float(
                            overview[
                                "price"
                            ][
                                "coverage"
                            ]
                        )
                        * 100
                    ),
                    (
                        float(
                            overview[
                                "review_coverage"
                            ]
                        )
                        * 100
                    ),
                    (
                        float(
                            overview[
                                "product_rating"
                            ][
                                "rated_product_coverage"
                            ]
                        )
                        * 100
                    ),
                ],
            }
        )

        _safe_bar_chart(
            frame=coverage,
            label_column="شاخص",
            value_column="پوشش (%)",
            title="پوشش داده",
        )

    price_summary = pd.DataFrame(
        {
            "شاخص": [
                "چارک اول",
                "میانه",
                "چارک سوم",
            ],
            "قیمت": [
                overview[
                    "price"
                ][
                    "p25"
                ],
                overview[
                    "price"
                ][
                    "median"
                ],
                overview[
                    "price"
                ][
                    "p75"
                ],
            ],
        }
    )

    _safe_bar_chart(
        frame=price_summary,
        label_column="شاخص",
        value_column="قیمت",
        title="بازه میانی قیمت",
    )


def _render_comparison_charts(
    comparison,
):
    if (
        not isinstance(
            comparison,
            pd.DataFrame,
        )
        or len(
            comparison
        )
        == 0
    ):
        return

    st.markdown(
        "### نمودارهای مقایسه"
    )

    left, right = st.columns(
        2
    )

    with left:
        _safe_bar_chart(
            frame=comparison,
            label_column="Category2",
            value_column="product_count",
            title="تعداد محصول",
        )

    with right:
        _safe_bar_chart(
            frame=comparison,
            label_column="Category2",
            value_column="median_price",
            title="میانه قیمت",
        )

    coverage = comparison[
        [
            "Category2",
            "review_coverage",
        ]
    ].copy()

    coverage[
        "review_coverage"
    ] = (
        pd.to_numeric(
            coverage[
                "review_coverage"
            ],
            errors="coerce",
        )
        * 100
    )

    left, right = st.columns(
        2
    )

    with left:
        _safe_bar_chart(
            frame=coverage,
            label_column="Category2",
            value_column="review_coverage",
            title="پوشش Review (%)",
        )

    with right:
        _safe_bar_chart(
            frame=comparison,
            label_column="Category2",
            value_column="weighted_product_rating",
            title="امتیاز وزنی محصول /100",
        )


def _render_tables(
    analytics,
    filters,
):
    tabs = st.tabs(
        [
            "برندها",
            "بالاترین امتیاز",
            "بیشترین تعداد امتیاز",
            "قیمت",
        ]
    )

    with tabs[0]:
        brands = (
            analytics
            .top_brands(
                filters=filters,
                top_n=10,
                include_generic=False,
            )
        )

        if len(brands) == 0:
            st.info(
                "داده‌ی برند قابل نمایش نیست."
            )
        else:
            display = _clean_table(
                brands,
                [
                    "Brand",
                    "product_count",
                    "review_coverage",
                    "weighted_product_rating",
                    "rating_count_total",
                ],
            )

            display = display.rename(
                columns={
                    "Brand": "برند",
                    "product_count": "تعداد محصول",
                    "review_coverage": "پوشش review",
                    "weighted_product_rating": "امتیاز وزنی /100",
                    "rating_count_total": "تعداد امتیاز",
                }
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "رتبه‌ی برندها بر اساس تعداد محصول در کاتالوگ است؛ "
                "این جدول سهم فروش یا سهم بازار نیست."
            )

    with tabs[1]:
        rated = (
            analytics
            .top_products(
                filters=filters,
                sort_by="rating",
                top_n=10,
            )
        )

        st.dataframe(
            _clean_table(
                rated,
                [
                    "title_fa",
                    "Brand",
                    "Price",
                    "Rate",
                    "Rate_cnt",
                ],
            ).rename(
                columns={
                    "title_fa": "محصول",
                    "Brand": "برند",
                    "Price": "قیمت",
                    "Rate": "امتیاز /100",
                    "Rate_cnt": "تعداد امتیاز",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "محصولات با تعداد امتیاز بسیار کم از leaderboard حذف شده‌اند."
        )

    with tabs[2]:
        engagement = (
            analytics
            .top_products(
                filters=filters,
                sort_by=(
                    "rating_count"
                ),
                top_n=10,
            )
        )

        st.dataframe(
            _clean_table(
                engagement,
                [
                    "title_fa",
                    "Brand",
                    "Rate_cnt",
                    "Rate",
                    "Price",
                ],
            ).rename(
                columns={
                    "title_fa": "محصول",
                    "Brand": "برند",
                    "Rate_cnt": "تعداد امتیاز ثبت‌شده",
                    "Rate": "امتیاز /100",
                    "Price": "قیمت",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "این رتبه‌بندی بر اساس Rate_cnt است، نه تعداد reviewهای corpus."
        )

    with tabs[3]:
        low = (
            analytics
            .top_products(
                filters=filters,
                sort_by="price_low",
                top_n=5,
            )
        )

        high = (
            analytics
            .top_products(
                filters=filters,
                sort_by="price_high",
                top_n=5,
            )
        )

        left, right = st.columns(
            2
        )

        with left:
            st.markdown(
                "**کم‌قیمت‌ترین‌ها**"
            )

            st.dataframe(
                _clean_table(
                    low,
                    [
                        "title_fa",
                        "Brand",
                        "Price",
                    ],
                ).rename(
                    columns={
                        "title_fa": "محصول",
                        "Brand": "برند",
                        "Price": "قیمت",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown(
                "**پرقیمت‌ترین‌ها**"
            )

            st.dataframe(
                _clean_table(
                    high,
                    [
                        "title_fa",
                        "Brand",
                        "Price",
                    ],
                ).rename(
                    columns={
                        "title_fa": "محصول",
                        "Brand": "برند",
                        "Price": "قیمت",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_manager_answer(
    result,
):
    result = require_mapping(
        result,
        required_keys=(
            "answer",
        ),
        label="پاسخ تحلیل مدیریتی",
    )

    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:
        raise ValueError(
            "پاسخ تحلیل مدیریتی خالی است."
        )

    valid = bool(
        result.get(
            "numeric_faithfulness_valid",
            False,
        )
    )

    badge = (
        "Numeric grounding: معتبر"
        if valid
        else "Numeric grounding: نیازمند بررسی"
    )

    confidence_key = result.get(
        "confidence",
        "low",
    )

    confidence = (
        CONFIDENCE_LABELS.get(
            confidence_key,
            str(
                confidence_key
            ),
        )
    )

    st.html(
        dedent(
            f"""
            <div class="analytics-answer">
                <div class="analytics-answer-head">
                    <span>{_escape(badge)}</span>
                    <span>اعتماد: {_escape(confidence)}</span>
                </div>
                <div class="analytics-answer-text">
                    {_escape(answer)}
                </div>
            </div>
            """
        ).strip()
    )

    insights = result.get(
        "insights",
        [],
    )

    if not isinstance(
        insights,
        list,
    ):
        insights = []

    for insight in insights:
        if not isinstance(
            insight,
            dict,
        ):
            continue

        st.html(
            dedent(
                f"""
                <div class="analytics-insight">
                    <strong>{_escape(insight.get("title", ""))}</strong>
                    <span>{_escape(insight.get("text", ""))}</span>
                </div>
                """
            ).strip()
        )

    caveats = result.get(
        "caveats",
        [],
    )

    if not isinstance(
        caveats,
        list,
    ):
        caveats = []

    if caveats:
        with st.expander(
            "محدودیت‌ها و caveatها",
            expanded=False,
        ):
            for caveat in caveats:
                st.write(
                    "•",
                    str(
                        caveat
                    ),
                )

    telemetry = result.get(
        "telemetry",
        {},
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        telemetry = {}

    try:
        total_latency_seconds = (
            float(
                telemetry.get(
                    "total_latency_ms",
                    0,
                )
                or 0
            )
            / 1000
        )
    except (
        TypeError,
        ValueError,
    ):
        total_latency_seconds = 0.0

    with st.expander(
        "جزئیات فنی پاسخ مدیریتی",
        expanded=False,
    ):
        columns = st.columns(
            3
        )

        columns[
            0
        ].metric(
            "زمان کل",
            f"{total_latency_seconds:.2f}s",
        )

        columns[
            1
        ].metric(
            "توکن",
            telemetry.get(
                "total_tokens",
                "—",
            ),
        )

        columns[
            2
        ].metric(
            "Repair",
            (
                "بله"
                if result.get(
                    "repaired"
                )
                else "خیر"
            ),
        )


def render(
    services,
):
    _render_hero()
    _render_quality_notice()

    analytics = (
        services
        .analytics
        .analytics
    )

    product_frame = (
        analytics
        .repository
        .products
    )

    category1_values = (
        analytics.distinct_values(
            "Category1"
        )
    )

    mode = st.radio(
        "نوع تحلیل",
        [
            "نمای کلی دسته",
            "مقایسه دسته‌ها",
        ],
        horizontal=True,
    )

    if mode == "نمای کلی دسته":
        category1 = st.selectbox(
            "Category1",
            [
                "همه‌ی کاتالوگ",
                *category1_values,
            ],
        )

        filters = {}

        if category1 != (
            "همه‌ی کاتالوگ"
        ):
            filters[
                "Category1"
            ] = category1

        category2_values = (
            analytics
            .distinct_values(
                "Category2",
                filters=filters,
            )
        )

        category2 = st.selectbox(
            "Category2",
            [
                "همه",
                *category2_values,
            ],
        )

        if category2 != "همه":
            filters[
                "Category2"
            ] = category2

        st.html(
            dedent(
                f"""
                <div class="analytics-scope">
                    محدوده‌ی فعلی:
                    <strong>{_escape(_scope_label(filters))}</strong>
                </div>
                """
            ).strip()
        )

        with st.spinner(
            "در حال محاسبه‌ی KPIها..."
        ):
            overview = (
                analytics
                .overview(
                    filters=filters
                )
            )

        _render_overview(
            overview
        )

        _render_overview_charts(
            analytics=analytics,
            filters=filters,
            overview=overview,
        )

        _render_tables(
            analytics=analytics,
            filters=filters,
        )

        st.divider()

        st.markdown(
            "### سؤال مدیریتی"
        )

        question = st.text_area(
            "سؤال",
            placeholder=(
                "مثلاً وضعیت قیمت، پوشش بازخورد و کیفیت امتیازدهی "
                "در این دسته چطور است؟"
            ),
            key="analytics_manager_question",
        )

        if st.button(
            "تحلیل مدیریتی هوشمند",
            use_container_width=True,
            disabled=(
                not str(
                    question
                ).strip()
            ),
        ):
            try:
                with st.spinner(
                    "در حال ساخت پاسخ grounded..."
                ):
                    result = (
                        services
                        .analytics
                        .answer(
                            question=question,
                            filters=filters,
                        )
                    )

                require_mapping(
                    result,
                    required_keys=(
                        "answer",
                    ),
                    label="پاسخ تحلیل مدیریتی",
                )

                st.session_state[
                    "analytics_manager_result"
                ] = result

                st.session_state[
                    "analytics_manager_scope"
                ] = filters

            except Exception as exc:
                st.session_state.pop(
                    "analytics_manager_result",
                    None,
                )
                render_ui_error(
                    "مدل پاسخ مدیریتی قابل نمایش برنگرداند.",
                    exc,
                )

        result = st.session_state.get(
            "analytics_manager_result"
        )

        result_scope = (
            st.session_state.get(
                "analytics_manager_scope"
            )
        )

        if (
            result is not None
            and result_scope
            == filters
        ):
            try:
                _render_manager_answer(
                    result
                )
            except Exception as exc:
                st.session_state.pop(
                    "analytics_manager_result",
                    None,
                )
                render_ui_error(
                    "پاسخ مدیریتی تولید شد اما ساختار آن قابل نمایش نبود.",
                    exc,
                )

    else:
        category2_values = (
            analytics
            .distinct_values(
                "Category2"
            )
        )

        selected = st.multiselect(
            "دسته‌ها برای مقایسه",
            category2_values,
            max_selections=3,
            placeholder=(
                "دو یا سه Category2 انتخاب کنید"
            ),
        )

        if len(selected) < 2:
            st.info(
                "برای مقایسه حداقل دو دسته انتخاب کنید."
            )
            return

        comparison = (
            analytics
            .compare_categories(
                selected,
                category_field=(
                    "Category2"
                ),
            )
        )

        display = comparison.copy()

        display[
            "review_coverage"
        ] = (
            display[
                "review_coverage"
            ]
            * 100
        )

        display = display.rename(
            columns={
                "Category2": "دسته",
                "product_count": "تعداد محصول",
                "brand_count": "برند قابل شناسایی",
                "review_coverage": "پوشش review (%)",
                "median_price": "میانه قیمت",
                "weighted_product_rating": "امتیاز وزنی /100",
                "rating_count_total": "تعداد امتیاز",
                "weighted_review_rating": "امتیاز review /5",
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Brand coverage محدود است و review_count برای رتبه‌بندی "
            "حجم واقعی بازار استفاده نمی‌شود."
        )

        _render_comparison_charts(
            comparison
        )

        question = st.text_area(
            "سؤال درباره‌ی این مقایسه",
            placeholder=(
                "مثلاً کدام دسته از نظر قیمت، پوشش بازخورد "
                "و تعداد امتیاز فعال‌تر به نظر می‌رسد؟"
            ),
            key="analytics_compare_question",
        )

        if st.button(
            "توضیح مدیریتی مقایسه",
            use_container_width=True,
            disabled=(
                not str(
                    question
                ).strip()
            ),
        ):
            try:
                with st.spinner(
                    "در حال تحلیل مقایسه..."
                ):
                    result = (
                        services
                        .analytics
                        .answer(
                            question=question,
                            comparison_categories=(
                                selected
                            ),
                            category_field=(
                                "Category2"
                            ),
                        )
                    )

                require_mapping(
                    result,
                    required_keys=(
                        "answer",
                    ),
                    label="پاسخ مقایسه مدیریتی",
                )

                st.session_state[
                    "analytics_compare_result"
                ] = result

                st.session_state[
                    "analytics_compare_scope"
                ] = list(
                    selected
                )

            except Exception as exc:
                st.session_state.pop(
                    "analytics_compare_result",
                    None,
                )
                render_ui_error(
                    "مدل توضیح مقایسه‌ی قابل نمایش برنگرداند.",
                    exc,
                )

        result = st.session_state.get(
            "analytics_compare_result"
        )

        if (
            result is not None
            and st.session_state.get(
                "analytics_compare_scope"
            )
            == list(
                selected
            )
        ):
            try:
                _render_manager_answer(
                    result
                )
            except Exception as exc:
                st.session_state.pop(
                    "analytics_compare_result",
                    None,
                )
                render_ui_error(
                    "پاسخ مقایسه تولید شد اما ساختار آن قابل نمایش نبود.",
                    exc,
                )
