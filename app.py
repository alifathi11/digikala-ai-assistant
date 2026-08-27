import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.app.bootstrap import (
    create_app_services,
)
from src.app.navigation import Feature
from src.app.safety import render_ui_error
from src.app.pages.qa import (
    render as render_qa,
)
from src.app.pages.product_search import (
    render as render_product_search,
)
from src.app.pages.comparison import (
    render as render_comparison,
)
from src.app.pages.analytics import (
    render as render_analytics,
)
from src.app.theme import (
    APP_CSS,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

FEATURES = [
    Feature(
        key="qa",
        label="پرسش از نظرات",
        icon="💬",
        enabled=True,
        renderer=render_qa,
    ),
    Feature(
        key="product_search",
        label="جست‌وجوی محصول",
        icon="🔎",
        enabled=True,
        renderer=render_product_search,
    ),
    Feature(
        key="comparison",
        label="مقایسه‌ی محصولات",
        icon="⚖️",
        enabled=True,
        renderer=render_comparison,
    ),
    Feature(
        key="analytics",
        label="تحلیل مدیریتی",
        icon="📊",
        enabled=True,
        renderer=render_analytics,
    ),
]

load_dotenv(
    PROJECT_ROOT
    / ".env"
)

st.set_page_config(
    page_title=(
        "Digikala AI Assistant"
    ),
    page_icon="💬",
    layout="wide",
    initial_sidebar_state=(
        "expanded"
    ),
)

st.markdown(
    APP_CSS,
    unsafe_allow_html=True,
)


@st.cache_resource(
    show_spinner=False
)
def load_services():
    api_key = os.getenv(
        "METIS_API_KEY"
    )

    base_url = os.getenv(
        "METIS_BASE_URL"
    )

    if not api_key:
        raise RuntimeError(
            "METIS_API_KEY is missing."
        )

    if not base_url:
        raise RuntimeError(
            "METIS_BASE_URL is missing."
        )

    return create_app_services(
        project_root=(
            PROJECT_ROOT
        ),
        api_key=api_key,
        base_url=base_url,
    )


with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">D</div>
            <div class="brand-copy">
                <div class="brand-name">Digikala AI</div>
                <div class="brand-tagline">دستیار هوشمند تجربه‌ی خرید</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section-title">فضای کاری</div>',
        unsafe_allow_html=True,
    )

    enabled_features = [
        feature
        for feature in FEATURES
        if feature.enabled
    ]

    selected_label = st.radio(
        "قابلیت",
        options=[
            feature.display_label
            for feature
            in enabled_features
        ],
        label_visibility="collapsed",
    )

    selected_feature = next(
        feature
        for feature
        in enabled_features
        if feature.display_label
        == selected_label
    )

    future_features = [
        feature
        for feature in FEATURES
        if not feature.enabled
    ]

    if future_features:
        st.markdown(
            '<div class="sidebar-section-title">به‌زودی</div>',
            unsafe_allow_html=True,
        )

        for feature in (
            future_features
        ):
            st.button(
                feature.display_label,
                key=(
                    f"future_"
                    f"{feature.key}"
                ),
                use_container_width=True,
                disabled=True,
            )

    st.markdown(
        """
        <div class="sidebar-status">
            <span class="status-dot"></span>
            <span>جست‌وجو + پرسش + مقایسه + تحلیل مدیریتی</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    with st.spinner(
        "در حال آماده‌سازی "
        "مدل‌ها و ایندکس‌ها..."
    ):
        services = (
            load_services()
        )

except Exception as exc:
    st.error(
        "راه‌اندازی برنامه "
        "ناموفق بود."
    )

    st.code(
        str(exc)
    )

    st.info(
        "فایل‌های processed/index "
        "و متغیرهای METIS_API_KEY / "
        "METIS_BASE_URL را بررسی کنید."
    )

    st.stop()


try:
    selected_feature.renderer(
        services
    )
except Exception as exc:
    render_ui_error(
        "این بخش نتوانست نتیجه را پردازش یا نمایش دهد.",
        exc,
    )
