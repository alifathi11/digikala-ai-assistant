import streamlit as st

from dashboard_pages.data_foundation import (
    render as render_data_foundation,
)
from dashboard_pages.embedding_indexing import (
    render as render_embedding_indexing,
)
from dashboard_pages.grounded_review_qa import (
    render as render_grounded_review_qa,
)
from dashboard_pages.product_search import (
    render as render_product_search,
)
from dashboard_pages.product_comparison import (
    render as render_product_comparison,
)
from dashboard_pages.manager_analytics import (
    render as render_manager_analytics,
)
from theme import apply_theme


st.set_page_config(
    page_title=(
        "Digikala AI Assistant · Technical Dashboard"
    ),
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

with st.sidebar:
    st.markdown(
        "## ◈ Digikala AI Assistant"
    )

    st.caption(
        "Technical Presentation Dashboard"
    )

    st.divider()

    page = st.radio(
        "Project section",
        [
            "01 · Data Foundation",
            "02 · Embedding & Indexing",
            "03 · Grounded Review Q&A",
            "04 · Product Search",
            "05 · Product Comparison",
            "06 · Manager Analytics",
        ],
        index=0,
    )

    st.divider()

    st.markdown(
        "**Current sources**"
    )

    if page == "01 · Data Foundation":
        st.caption("Notebook 01 · Data Analysis")
        st.caption("Notebook 02 · Data Preprocessing")

    elif page == "02 · Embedding & Indexing":
        st.caption("Notebook 03 · Comment FAISS")
        st.caption("Notebook 04 · Comment BM25")
        st.caption("Notebook 07 · Product Indexes")

    elif page == "03 · Grounded Review Q&A":
        st.caption("Notebook 05 · Retrieval Evaluation")
        st.caption("Notebook 06 · Grounded QA Evaluation")

    elif page == "04 · Product Search":
        st.caption("Notebook 08 · Product Search Qrels")
        st.caption("Notebook 09 · Product Search Evaluation")

    elif page == "05 · Product Comparison":
        st.caption("Notebook 10 · Product Comparison Evaluation")

    else:
        st.caption("Notebook 11 · Analytics Data Audit")
        st.caption("Notebook 12 · Analytics Evaluation")

    st.markdown(
        """
        <div class="small-muted" style="margin-top: 1rem;">
            Presentation values are derived from executed notebook outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "01 · Data Foundation":
    render_data_foundation()

elif page == "02 · Embedding & Indexing":
    render_embedding_indexing()

elif page == "03 · Grounded Review Q&A":
    render_grounded_review_qa()

elif page == "04 · Product Search":
    render_product_search()

elif page == "05 · Product Comparison":
    render_product_comparison()

else:
    render_manager_analytics()
