import streamlit as st


APP_CSS = """
<style>
:root {
    --bg: #08111f;
    --panel: rgba(15, 27, 46, 0.88);
    --panel-soft: rgba(21, 36, 59, 0.72);
    --border: rgba(148, 163, 184, 0.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #66e3c4;
    --accent-soft: rgba(102, 227, 196, 0.12);
    --blue: #73a8ff;
    --success: #60d394;
}

html, body, [class*="css"] {
    font-size: 16.5px;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(54, 99, 255, 0.13), transparent 28%),
        radial-gradient(circle at 95% 5%, rgba(102, 227, 196, 0.10), transparent 24%),
        var(--bg);
    color: var(--text);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: rgba(7, 16, 30, 0.96);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
}

.hero-card {
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2rem 2.15rem;
    background:
        linear-gradient(135deg, rgba(35, 59, 94, 0.90), rgba(12, 24, 43, 0.92));
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.20);
    margin-bottom: 1.15rem;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-card h1 {
    color: var(--text);
    font-size: 2.15rem;
    line-height: 1.12;
    margin: 0.85rem 0 0.55rem;
    letter-spacing: -0.035em;
}

.hero-card p {
    color: #cbd5e1;
    max-width: 940px;
    line-height: 1.75;
    margin: 0;
}

.pipeline-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.85rem 0 1.25rem;
}

.pipeline-chip {
    border: 1px solid var(--border);
    background: rgba(15, 27, 46, 0.78);
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    color: #dbeafe;
    font-size: 0.82rem;
    font-weight: 650;
}

.section-card {
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--panel);
    padding: 1.1rem 1.2rem;
    margin: 0.45rem 0 0.9rem;
}

.section-card strong {
    color: var(--text);
}

.section-card p,
.section-card li {
    color: #cbd5e1;
    line-height: 1.7;
}

.pass-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.11);
    padding: 0.72rem 0;
}

.pass-row:last-child {
    border-bottom: 0;
}

.pass-badge {
    min-width: 62px;
    text-align: center;
    border-radius: 999px;
    padding: 0.24rem 0.55rem;
    background: rgba(96, 211, 148, 0.12);
    color: var(--success);
    border: 1px solid rgba(96, 211, 148, 0.22);
    font-size: 0.73rem;
    font-weight: 800;
}

.note-box {
    border-left: 3px solid var(--accent);
    background: rgba(102, 227, 196, 0.06);
    padding: 0.9rem 1rem;
    border-radius: 10px;
    color: #cbd5e1;
}

.trace-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    background: var(--panel-soft);
    min-height: 195px;
}

.trace-card .trace-id {
    color: var(--accent);
    font-size: 0.76rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.trace-card h4 {
    color: var(--text);
    margin: 0.35rem 0 0.55rem;
}

.trace-card p {
    color: #cbd5e1;
    line-height: 1.65;
    font-size: 0.91rem;
}

.small-muted {
    color: var(--muted);
    font-size: 0.82rem;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--border);
    background: rgba(15, 27, 46, 0.76);
    border-radius: 16px;
    padding: 0.85rem 0.95rem;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.35rem;
}

[data-testid="stTabs"] button {
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

h2, h3 {
    letter-spacing: -0.018em;
}

hr {
    border-color: var(--border);
}

#MainMenu,
footer {
    visibility: hidden;
}
</style>
"""


def apply_theme():
    st.markdown(
        APP_CSS,
        unsafe_allow_html=True,
    )
