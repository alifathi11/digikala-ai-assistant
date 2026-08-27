APP_CSS = """
<style>
:root {
    color-scheme: dark;
    --bg: #080c14;
    --bg-soft: #0d1320;
    --surface: #111827;
    --surface-raised: #172033;
    --surface-hover: #1d293b;
    --sidebar: #090e18;
    --brand: #ff4b63;
    --brand-strong: #ff667a;
    --brand-soft: rgba(255, 75, 99, 0.14);
    --text: #f8fafc;
    --text-soft: #cbd5e1;
    --text-muted: #9ca9bb;
    --line: #2a3548;
    --line-soft: #202a3b;
    --success: #42d6a4;
    --warning: #f7bd4b;
    --shadow-sm: 0 10px 28px rgba(0, 0, 0, 0.24);
    --shadow-md: 0 22px 58px rgba(0, 0, 0, 0.34);
}

html,
body,
[class*="css"],
[data-testid="stAppViewContainer"] {
    font-family: "Vazirmatn", "IRANSansX", "Tahoma", sans-serif;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    direction: rtl;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    color: var(--text) !important;
    background-color: var(--bg) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 91% 2%, rgba(255, 75, 99, 0.11), transparent 30rem),
        radial-gradient(circle at 8% 52%, rgba(75, 105, 165, 0.10), transparent 28rem),
        linear-gradient(145deg, #0a0f19 0%, #080c14 52%, #0b111d 100%) !important;
}

.stApp p,
.stApp li,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
[data-testid="stMarkdownContainer"] {
    color: var(--text);
}

.stApp a {
    color: #7dc4ff;
}

.stApp strong,
.stApp b {
    color: #ffffff;
}

::selection {
    color: #ffffff;
    background: rgba(255, 75, 99, 0.48);
}

[data-testid="stHeader"] {
    background: rgba(8, 12, 20, 0.76) !important;
    backdrop-filter: blur(12px);
}

[data-testid="stToolbar"] {
    direction: ltr;
}

[data-testid="stToolbar"] button,
[data-testid="stHeader"] button {
    color: var(--text-soft) !important;
}

[data-testid="stToolbar"] svg,
[data-testid="stHeader"] svg {
    fill: currentColor !important;
}

.block-container {
    max-width: 1160px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

/* Sidebar: fixed and always expanded on desktop. */
[data-testid="stSidebar"] {
    color: var(--text) !important;
    background:
        radial-gradient(circle at 48% -8%, rgba(255, 75, 99, 0.20), transparent 19rem),
        linear-gradient(180deg, #0e1523 0%, var(--sidebar) 100%) !important;
    border-left: 1px solid var(--line-soft);
    box-shadow: 14px 0 48px rgba(0, 0, 0, 0.24);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.35rem;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: inherit;
}

[data-testid="stSidebar"] .stDivider {
    margin: 1rem 0;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.10);
}

@media (min-width: 901px) {
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        transform: none !important;
        visibility: visible !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        width: 300px !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    direction: rtl;
    padding: 0.3rem 0 0.8rem;
}

.brand-mark {
    width: 46px;
    height: 46px;
    flex: 0 0 46px;
    display: grid;
    place-items: center;
    border-radius: 15px;
    color: #ffffff !important;
    font-size: 1.35rem;
    font-weight: 900;
    background: linear-gradient(145deg, var(--brand-strong), var(--brand));
    box-shadow: 0 12px 30px rgba(255, 75, 99, 0.32);
}

.brand-name {
    color: #ffffff !important;
    font-size: 1.12rem;
    font-weight: 850;
    letter-spacing: -0.02em;
}

.brand-tagline {
    color: var(--text-soft) !important;
    font-size: 0.76rem;
    margin-top: 0.18rem;
    line-height: 1.8;
}

.sidebar-section-title {
    color: var(--text-muted) !important;
    font-size: 0.70rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    margin: 0.25rem 0 0.5rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: var(--text-soft) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    min-height: 45px;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.35rem;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 13px;
    background: rgba(255, 255, 255, 0.045);
    transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: rgba(255, 255, 255, 0.09);
    border-color: rgba(255, 255, 255, 0.18);
    transform: translateX(-2px);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: var(--text) !important;
    font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(255, 75, 99, 0.25), rgba(255, 75, 99, 0.10));
    border-color: rgba(255, 102, 122, 0.48);
}

[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    border-color: var(--text-soft) !important;
    background-color: transparent !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
    border-color: var(--brand-strong) !important;
}

[data-testid="stSidebar"] .stButton > button:disabled {
    min-height: 43px;
    color: var(--text-muted) !important;
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px;
    opacity: 1 !important;
}

[data-testid="stSidebar"] .stButton > button:disabled p {
    color: var(--text-muted) !important;
}

.sidebar-status {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    direction: rtl;
    color: var(--text-soft) !important;
    font-size: 0.70rem;
    line-height: 1.8;
    margin-top: 1rem;
    padding: 0.75rem 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.045);
}

.sidebar-status span {
    color: var(--text-soft) !important;
}

.status-dot {
    width: 7px;
    height: 7px;
    flex: 0 0 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(66, 214, 164, 0.12);
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    padding: 1.75rem 1.85rem;
    border: 1px solid rgba(255, 102, 122, 0.24);
    border-radius: 24px;
    background:
        radial-gradient(circle at 5% 0%, rgba(255, 75, 99, 0.12), transparent 14rem),
        linear-gradient(125deg, #151d2d 0%, #111827 70%, #171925 100%);
    box-shadow: var(--shadow-md);
    margin-bottom: 1.4rem;
}

.hero::before {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    left: -75px;
    top: -115px;
    border-radius: 50%;
    background: rgba(255, 75, 99, 0.10);
}

.hero-copy {
    position: relative;
    z-index: 1;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #ff8a9a !important;
    font-size: 0.76rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(1.55rem, 3vw, 2.1rem);
    letter-spacing: -0.035em;
    margin: 0 0 0.45rem;
    line-height: 1.5;
}

.hero p {
    max-width: 710px;
    color: var(--text-soft) !important;
    font-size: 0.93rem;
    margin: 0;
    line-height: 2;
}

.hero-icon {
    position: relative;
    z-index: 1;
    width: 76px;
    height: 76px;
    flex: 0 0 76px;
    display: grid;
    place-items: center;
    border-radius: 23px;
    color: #ffffff !important;
    font-size: 2rem;
    background: linear-gradient(145deg, var(--brand-strong), var(--brand));
    box-shadow: 0 18px 38px rgba(255, 75, 99, 0.30);
    transform: rotate(-3deg);
}

/* Native text, captions and form labels */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption,
.stCaption p {
    color: var(--text-muted) !important;
}

.stSelectbox label,
.stTextInput label,
.stTextArea label {
    color: var(--text) !important;
    font-size: 0.84rem !important;
    font-weight: 750 !important;
    margin-bottom: 0.35rem;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
.stTextInput input,
.stTextArea textarea {
    min-height: 49px;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    background: var(--surface-raised) !important;
    border-color: var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 7px 22px rgba(0, 0, 0, 0.16);
    transition: border-color 160ms ease, box-shadow 160ms ease;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="input"] input,
.stTextInput input,
.stTextArea textarea {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

input::placeholder,
textarea::placeholder,
[data-baseweb="select"] input::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}

[data-baseweb="select"] svg,
[data-baseweb="input"] svg {
    color: var(--text-soft) !important;
    fill: currentColor !important;
}

[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
    border-color: rgba(255, 102, 122, 0.72) !important;
    box-shadow: 0 0 0 4px rgba(255, 75, 99, 0.12);
}

[data-baseweb="select"] input,
[data-baseweb="popover"],
[role="listbox"] {
    direction: rtl;
    text-align: right;
}

[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
    color: var(--text) !important;
    background: var(--surface-raised) !important;
    border-color: var(--line) !important;
}

[role="option"],
[role="listbox"] li,
[data-baseweb="menu"] li {
    color: var(--text) !important;
    background: var(--surface-raised) !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"],
[role="listbox"] li:hover,
[data-baseweb="menu"] li:hover {
    color: #ffffff !important;
    background: var(--surface-hover) !important;
}

/* Product */
.product-card {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 19px;
    padding: 1.05rem 1.15rem 1.15rem;
    margin: 0.8rem 0 1.15rem;
    background: linear-gradient(120deg, #141c2b, #101722);
    box-shadow: var(--shadow-sm);
}

.product-card::after {
    content: "";
    position: absolute;
    inset: 0 0 0 auto;
    width: 4px;
    background: linear-gradient(180deg, var(--brand-strong), var(--brand));
}

.product-title-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.75rem;
}

.product-icon {
    width: 36px;
    height: 36px;
    flex: 0 0 36px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    background: var(--brand-soft);
    color: #ff8a9a !important;
    font-size: 1rem;
}

.product-title {
    color: #ffffff !important;
    font-size: 0.98rem;
    font-weight: 800;
    line-height: 1.8;
}

.product-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 0.22rem 0.65rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--text-soft) !important;
    background: #1a2333;
    font-size: 0.76rem;
    line-height: 1.7;
}

.meta-chip strong {
    color: #ffffff !important;
    font-weight: 750;
    margin-left: 0.25rem;
}

.hint-card {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    color: var(--text-soft) !important;
    font-size: 0.79rem;
    line-height: 1.9;
    margin: 0.2rem 0 1rem;
    padding: 0.7rem 0.85rem;
    border: 1px dashed #344157;
    border-radius: 12px;
    background: rgba(20, 28, 43, 0.72);
}

.hint-card span {
    color: var(--text-soft) !important;
}

.hint-icon {
    color: var(--warning) !important;
    font-size: 0.95rem;
}

/* Chat */
[data-testid="stChatMessage"] {
    color: var(--text) !important;
    padding: 1rem 1.1rem;
    margin: 0.68rem 0;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(17, 24, 39, 0.94) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(255, 102, 122, 0.27);
    background: linear-gradient(110deg, #1b1c2a, #121a28) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--text) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    line-height: 2;
}

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    color: #ffffff !important;
    background: #283348 !important;
}

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    color: #ffffff !important;
    background: var(--brand) !important;
}

[data-testid="stChatInput"] {
    color: var(--text) !important;
    border: 1px solid #354157 !important;
    border-radius: 17px !important;
    background: #141c2b !important;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255, 102, 122, 0.75) !important;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38), 0 0 0 4px rgba(255, 75, 99, 0.12);
}

[data-testid="stChatInput"] textarea {
    direction: rtl;
    text-align: right;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    color: var(--brand-strong) !important;
}

[data-testid="stChatInput"] button:disabled {
    color: var(--text-muted) !important;
}

[data-testid="stChatInput"] button svg {
    fill: currentColor !important;
}

[data-testid="stBottom"] > div {
    background: linear-gradient(180deg, rgba(8, 12, 20, 0), rgba(8, 12, 20, 0.98) 38%) !important;
}

/* Evidence, expander and telemetry */
.evidence-card {
    position: relative;
    padding: 0.85rem 1rem;
    margin: 0.72rem 0;
    border: 1px solid var(--line);
    border-right: 4px solid var(--brand);
    border-radius: 13px;
    color: var(--text) !important;
    background: linear-gradient(110deg, #1a1b29, #141c29);
    line-height: 2;
}

.evidence-meta {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #ff91a0 !important;
    font-size: 0.73rem;
    font-weight: 750;
    margin-bottom: 0.25rem;
}

.evidence-meta span {
    color: #ff91a0 !important;
}

.evidence-body {
    color: var(--text-soft) !important;
    font-size: 0.88rem;
}

[data-testid="stExpander"],
details[data-testid="stExpander"] {
    overflow: hidden;
    margin-top: 0.55rem;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background: rgba(17, 24, 39, 0.88) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
    font-size: 0.84rem;
    font-weight: 700;
}

[data-testid="stExpander"] summary svg {
    color: var(--text-soft) !important;
    fill: currentColor !important;
}

[data-testid="stMetric"] {
    padding: 0.85rem;
    border: 1px solid var(--line);
    border-radius: 13px;
    background: #141c2a;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #ffffff !important;
    font-size: 1.25rem;
}

/* Alerts, spinners, code and generic buttons */
[data-testid="stAlert"],
div[data-baseweb="notification"] {
    color: var(--text) !important;
    border: 1px solid #344056 !important;
    border-radius: 14px !important;
    background: #172033 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: var(--text) !important;
}

[data-testid="stAlert"] svg {
    color: var(--text-soft) !important;
    fill: currentColor !important;
}

.stSpinner,
.stSpinner p {
    color: var(--text-soft) !important;
}

.stSpinner > div {
    border-top-color: var(--brand) !important;
}

.stButton > button:not(:disabled) {
    color: #ffffff !important;
    background: var(--surface-raised) !important;
    border-color: var(--line) !important;
}

.stButton > button:not(:disabled):hover {
    border-color: var(--brand-strong) !important;
    background: var(--surface-hover) !important;
}

.small-muted {
    color: var(--text-muted) !important;
    font-size: 0.78rem;
}

/* Persian content: enforce RTL flow and right alignment everywhere. */
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stText"],
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] button,
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"],
[data-testid="stExpander"] summary,
[data-testid="stMetric"] {
    direction: rtl !important;
    text-align: right !important;
}

.stApp [dir="auto"] {
    direction: rtl !important;
    text-align: right !important;
}

.hero-copy,
.hero-kicker,
.hero h1,
.hero p,
.brand-copy,
.brand-name,
.brand-tagline,
.sidebar-section-title,
.sidebar-status,
.product-title,
.product-meta,
.meta-chip,
.hint-card,
.evidence-card,
.evidence-meta,
.evidence-body,
.small-muted {
    direction: rtl !important;
    text-align: right !important;
}

.stApp ul,
.stApp ol {
    direction: rtl !important;
    text-align: right !important;
    padding-right: 1.5rem;
    padding-left: 0;
}

.stApp table,
.stApp th,
.stApp td {
    direction: rtl !important;
    text-align: right !important;
}

[data-baseweb="select"],
[data-baseweb="select"] > div,
[role="listbox"],
[role="option"],
.stTextInput input,
.stTextArea textarea,
[data-testid="stChatInput"] textarea {
    direction: rtl !important;
    text-align: right !important;
}

.stButton > button,
[data-testid="stSidebar"] [data-baseweb="radio"] {
    direction: rtl !important;
    text-align: right !important;
    justify-content: flex-start !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricValue"] > div {
    direction: rtl !important;
    text-align: right !important;
    justify-content: flex-start !important;
}

code,
pre,
[data-testid="stCode"] {
    direction: ltr !important;
    text-align: left !important;
    color: #e7edf7 !important;
    background: #0b111c !important;
}

@media (max-width: 900px) {
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        color: var(--text) !important;
    }

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: currentColor !important;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1.35rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 1.35rem;
        border-radius: 20px;
    }

    .hero-icon {
        width: 56px;
        height: 56px;
        flex-basis: 56px;
        border-radius: 17px;
        font-size: 1.5rem;
    }

    .hero h1 {
        font-size: 1.4rem;
    }

    .hero p {
        font-size: 0.84rem;
    }
}

@media (max-width: 520px) {
    .hero-icon {
        display: none;
    }

    .product-card {
        padding: 0.95rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        transition: none !important;
        animation: none !important;
    }
}

/* Product Search */
.search-hint-card {
    margin-top: -0.3rem;
    margin-bottom: 1.25rem;
}

.search-results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.5rem 0 0.85rem;
    padding: 0 0.15rem;
}

.search-results-kicker {
    color: var(--text-muted) !important;
    font-size: 0.72rem;
    font-weight: 750;
}

.search-results-header h2 {
    color: #ffffff !important;
    font-size: 1.12rem;
    line-height: 1.8;
    margin: 0.15rem 0 0;
}

.search-result-count {
    flex: 0 0 auto;
    padding: 0.4rem 0.75rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--text-soft) !important;
    background: var(--surface-raised);
    font-size: 0.76rem;
    font-weight: 750;
}

.search-result-card {
    position: relative;
    overflow: hidden;
    margin: 0.85rem 0 0.45rem;
    padding: 1.1rem 1.15rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    color: var(--text) !important;
    background:
        radial-gradient(circle at 100% 0%, rgba(255, 75, 99, 0.08), transparent 17rem),
        linear-gradient(125deg, #151d2c 0%, #111827 72%);
    box-shadow: var(--shadow-sm);
}

.search-result-card::after {
    content: "";
    position: absolute;
    top: 0.9rem;
    right: 0;
    bottom: 0.9rem;
    width: 4px;
    border-radius: 4px 0 0 4px;
    background: linear-gradient(180deg, var(--brand-strong), var(--brand));
}

.search-result-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    direction: rtl;
}

.search-rank {
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    color: #ffffff !important;
    background: var(--brand-soft);
    border: 1px solid rgba(255, 102, 122, 0.30);
    font-size: 0.88rem;
    font-weight: 850;
}

.search-result-title-wrap {
    min-width: 0;
    flex: 1 1 auto;
}

.search-result-title {
    color: #ffffff !important;
    font-size: 0.98rem;
    font-weight: 820;
    line-height: 1.9;
}

.search-result-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.70rem;
    margin-top: 0.05rem;
}

.search-final-score {
    min-width: 88px;
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.05rem;
    padding: 0.5rem 0.7rem;
    border: 1px solid rgba(66, 214, 164, 0.25);
    border-radius: 13px;
    background: rgba(66, 214, 164, 0.08);
}

.search-final-score span {
    color: var(--text-muted) !important;
    font-size: 0.64rem;
}

.search-final-score strong {
    color: var(--success) !important;
    font-size: 1.15rem;
    line-height: 1.2;
}

.search-result-meta {
    margin-top: 0.85rem;
}

.brand-match-chip {
    color: #ffd7dd !important;
    border-color: rgba(255, 102, 122, 0.32);
    background: rgba(255, 75, 99, 0.12);
}

.search-score-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.6rem;
    margin-top: 0.85rem;
}

.score-stat {
    min-width: 0;
    padding: 0.7rem 0.75rem;
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    background: rgba(9, 14, 24, 0.45);
}

.score-stat span {
    display: block;
    color: var(--text-muted) !important;
    font-size: 0.66rem;
    line-height: 1.7;
    margin-bottom: 0.12rem;
}

.score-stat strong {
    display: block;
    color: #ffffff !important;
    font-size: 0.86rem;
    line-height: 1.7;
}

.status-support {
    border-color: rgba(66, 214, 164, 0.22);
    background: rgba(66, 214, 164, 0.07);
}

.status-support strong {
    color: var(--success) !important;
}

.status-mixed {
    border-color: rgba(247, 189, 75, 0.25);
    background: rgba(247, 189, 75, 0.07);
}

.status-mixed strong {
    color: var(--warning) !important;
}

.status-contradict {
    border-color: rgba(255, 102, 122, 0.30);
    background: rgba(255, 75, 99, 0.08);
}

.status-contradict strong {
    color: #ff8a9a !important;
}

.status-none strong {
    color: var(--text-soft) !important;
}

.search-reason {
    margin-top: 0.75rem;
    padding: 0.7rem 0.8rem;
    border: 1px dashed #344157;
    border-radius: 11px;
    color: var(--text-soft) !important;
    background: rgba(8, 12, 20, 0.28);
    font-size: 0.80rem;
    line-height: 1.9;
}

.search-reason div {
    color: var(--text-soft) !important;
}

.search-reason-label {
    display: inline-block;
    color: #ff91a0 !important;
    font-size: 0.68rem;
    font-weight: 800;
    margin-bottom: 0.12rem;
}

.search-results-header,
.search-results-kicker,
.search-result-count,
.search-result-card,
.search-result-head,
.search-rank,
.search-result-title-wrap,
.search-result-title,
.search-result-subtitle,
.search-final-score,
.search-score-grid,
.score-stat,
.search-reason,
.search-reason-label {
    direction: rtl !important;
    text-align: right !important;
}

.search-final-score {
    text-align: center !important;
}

[data-testid="stForm"] {
    margin-bottom: 0.8rem;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 17px;
    background: rgba(17, 24, 39, 0.70);
}

[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    min-height: 49px;
    justify-content: center !important;
    color: #ffffff !important;
    font-weight: 800;
    border-color: rgba(255, 102, 122, 0.55) !important;
    background: linear-gradient(145deg, var(--brand-strong), var(--brand)) !important;
    box-shadow: 0 10px 28px rgba(255, 75, 99, 0.22);
}

[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--brand-strong) !important;
    background: linear-gradient(145deg, #ff7487, #ff536b) !important;
}

@media (max-width: 768px) {
    .search-result-head {
        align-items: flex-start;
        flex-wrap: wrap;
    }

    .search-result-title-wrap {
        flex-basis: calc(100% - 58px);
    }

    .search-final-score {
        margin-right: 57px;
    }

    .search-score-grid {
        grid-template-columns: 1fr;
    }

    .search-results-header {
        align-items: flex-start;
    }
}


/* Product Comparison */
.comparison-hero {
    margin-bottom: 1.15rem;
}

.comparison-section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.1rem 0 0.7rem;
    direction: rtl;
    text-align: right;
}

.comparison-section-head > div:first-child {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
}

.comparison-section-head span {
    color: var(--text-muted) !important;
    font-size: 0.72rem;
    font-weight: 750;
}

.comparison-section-head strong {
    color: #ffffff !important;
    font-size: 0.82rem;
}

.comparison-selection-hint {
    color: var(--text-muted) !important;
    font-size: 0.7rem;
}

.comparison-empty-state {
    padding: 1.15rem;
    border: 1px dashed #344157;
    border-radius: 16px;
    color: var(--text-muted) !important;
    background: rgba(9, 14, 24, 0.35);
    text-align: center;
    direction: rtl;
}

.comparison-product-card {
    position: relative;
    margin: 0.55rem 0 0.4rem;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 16px;
    background:
        radial-gradient(circle at 100% 0%, rgba(255, 75, 99, 0.06), transparent 13rem),
        linear-gradient(135deg, #151d2c, #111827);
    box-shadow: var(--shadow-sm);
    min-height: 155px;
    direction: rtl;
    text-align: right;
}

.comparison-product-selected {
    border-color: rgba(66, 214, 164, 0.32);
    background:
        radial-gradient(circle at 100% 0%, rgba(66, 214, 164, 0.07), transparent 13rem),
        linear-gradient(135deg, #151d2c, #111827);
}

.comparison-product-id {
    color: var(--text-muted) !important;
    font-size: 0.66rem;
    margin-bottom: 0.25rem;
}

.comparison-product-title {
    color: #ffffff !important;
    font-size: 0.9rem;
    font-weight: 820;
    line-height: 1.85;
    min-height: 3.3rem;
}

.comparison-product-meta {
    margin-top: 0.75rem;
}

.comparison-query-intro {
    margin: 1rem 0 0.65rem;
    color: var(--text-soft) !important;
    font-size: 0.82rem;
    direction: rtl;
    text-align: right;
}

.comparison-result-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 0.34fr);
    gap: 1rem;
    padding: 1.15rem;
    margin: 1rem 0;
    border: 1px solid var(--line);
    border-radius: 19px;
    background:
        radial-gradient(circle at 90% 10%, rgba(66, 214, 164, 0.09), transparent 18rem),
        linear-gradient(130deg, #151d2c, #111827);
    direction: rtl;
    text-align: right;
}

.comparison-result-kicker {
    color: #ff91a0 !important;
    font-size: 0.68rem;
    font-weight: 850;
}

.comparison-result-copy h2 {
    margin: 0.3rem 0 0.45rem;
    color: #ffffff !important;
    font-size: 1rem;
    line-height: 1.95;
}

.comparison-result-copy p {
    margin: 0;
    color: var(--text-soft) !important;
    font-size: 0.8rem;
    line-height: 2;
}

.comparison-result-winner {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.3rem;
    padding: 0.9rem;
    border: 1px solid rgba(66, 214, 164, 0.24);
    border-radius: 15px;
    background: rgba(66, 214, 164, 0.07);
}

.comparison-result-winner span,
.comparison-result-winner small {
    color: var(--text-muted) !important;
    font-size: 0.66rem;
}

.comparison-result-winner strong {
    color: var(--success) !important;
    font-size: 0.84rem;
    line-height: 1.85;
}

.comparison-criterion-head {
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr) minmax(180px, 0.34fr);
    gap: 0.75rem;
    align-items: center;
    margin: 1.3rem 0 0.55rem;
    padding: 0.8rem 0.9rem;
    border: 1px solid var(--line);
    border-radius: 15px;
    background: rgba(17, 24, 39, 0.72);
    direction: rtl;
    text-align: right;
}

.comparison-criterion-index {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    color: #ffffff !important;
    background: var(--brand-soft);
    border: 1px solid rgba(255, 102, 122, 0.28);
    font-weight: 850;
}

.comparison-criterion-head span {
    color: var(--text-muted) !important;
    font-size: 0.64rem;
}

.comparison-criterion-head h3 {
    margin: 0.1rem 0 0;
    color: #ffffff !important;
    font-size: 0.93rem;
    line-height: 1.8;
}

.comparison-criterion-winner {
    padding: 0.55rem 0.65rem;
    border: 1px solid rgba(66, 214, 164, 0.20);
    border-radius: 11px;
    background: rgba(66, 214, 164, 0.06);
}

.comparison-criterion-winner strong {
    display: block;
    margin-top: 0.12rem;
    color: var(--success) !important;
    font-size: 0.72rem;
    line-height: 1.75;
}

.comparison-winner-reason {
    margin: 0 0 0.65rem;
    padding: 0.65rem 0.75rem;
    border: 1px dashed #344157;
    border-radius: 11px;
    color: var(--text-soft) !important;
    background: rgba(8, 12, 20, 0.3);
    font-size: 0.77rem;
    line-height: 1.9;
    direction: rtl;
    text-align: right;
}

.comparison-assessment {
    margin: 0.55rem 0 0.3rem;
    padding: 0.85rem 0.9rem;
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    background: rgba(9, 14, 24, 0.45);
    direction: rtl;
    text-align: right;
}

.comparison-assessment-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.8rem;
}

.comparison-assessment-head > div {
    min-width: 0;
}

.comparison-assessment-id {
    display: block;
    color: var(--text-muted) !important;
    font-size: 0.63rem;
    margin-bottom: 0.1rem;
}

.comparison-assessment-head strong {
    color: #ffffff !important;
    font-size: 0.79rem;
    line-height: 1.75;
}

.comparison-stance {
    flex: 0 0 auto;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    color: var(--text-soft) !important;
    font-size: 0.64rem;
    font-weight: 800;
}

.comparison-assessment-text {
    margin-top: 0.5rem;
    color: var(--text-soft) !important;
    font-size: 0.78rem;
    line-height: 1.95;
}

.comparison-evidence-count {
    margin-top: 0.45rem;
    color: var(--text-muted) !important;
    font-size: 0.64rem;
}

.comparison-stance-positive {
    border-color: rgba(66, 214, 164, 0.23);
}

.comparison-stance-positive .comparison-stance {
    color: var(--success) !important;
    border-color: rgba(66, 214, 164, 0.27);
    background: rgba(66, 214, 164, 0.07);
}

.comparison-stance-mixed {
    border-color: rgba(247, 189, 75, 0.23);
}

.comparison-stance-mixed .comparison-stance {
    color: var(--warning) !important;
    border-color: rgba(247, 189, 75, 0.27);
    background: rgba(247, 189, 75, 0.07);
}

.comparison-stance-negative {
    border-color: rgba(255, 102, 122, 0.27);
}

.comparison-stance-negative .comparison-stance {
    color: #ff8a9a !important;
    border-color: rgba(255, 102, 122, 0.30);
    background: rgba(255, 75, 99, 0.08);
}

.comparison-stance-unknown .comparison-stance {
    color: var(--text-muted) !important;
}

.comparison-evidence-card {
    margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
    .comparison-section-head {
        align-items: flex-start;
        flex-direction: column;
    }

    .comparison-result-hero {
        grid-template-columns: 1fr;
    }

    .comparison-criterion-head {
        grid-template-columns: 42px minmax(0, 1fr);
    }

    .comparison-criterion-winner {
        grid-column: 1 / -1;
    }
}



/* Manager Analytics */
.analytics-hero {
    margin-bottom: 0.85rem;
}

.analytics-quality-note {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin: 0.4rem 0 1rem;
    padding: 0.85rem 0.95rem;
    border: 1px solid rgba(247, 189, 75, 0.22);
    border-radius: 14px;
    background: rgba(247, 189, 75, 0.06);
    direction: rtl;
    text-align: right;
}

.analytics-quality-note strong {
    flex: 0 0 auto;
    color: var(--warning) !important;
    font-size: 0.76rem;
}

.analytics-quality-note span {
    color: var(--text-soft) !important;
    font-size: 0.76rem;
    line-height: 1.9;
}

.analytics-scope {
    margin: 0.65rem 0;
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    color: var(--text-muted) !important;
    background: rgba(17, 24, 39, 0.55);
    direction: rtl;
    text-align: right;
    font-size: 0.75rem;
}

.analytics-scope strong {
    color: #ffffff !important;
}

.analytics-answer {
    margin: 1rem 0 0.65rem;
    padding: 1rem 1.05rem;
    border: 1px solid rgba(66, 214, 164, 0.24);
    border-radius: 17px;
    background:
        radial-gradient(circle at 100% 0%, rgba(66, 214, 164, 0.08), transparent 16rem),
        linear-gradient(135deg, #151d2c, #111827);
    direction: rtl;
    text-align: right;
}

.analytics-answer-head {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}

.analytics-answer-head span {
    color: var(--success) !important;
    font-size: 0.66rem;
    font-weight: 800;
}

.analytics-answer-text {
    color: #ffffff !important;
    font-size: 0.88rem;
    line-height: 2.05;
}

.analytics-insight {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin: 0.45rem 0;
    padding: 0.75rem 0.85rem;
    border: 1px solid var(--line-soft);
    border-radius: 13px;
    background: rgba(9, 14, 24, 0.45);
    direction: rtl;
    text-align: right;
}

.analytics-insight strong {
    color: #ffffff !important;
    font-size: 0.76rem;
}

.analytics-insight span {
    color: var(--text-soft) !important;
    font-size: 0.75rem;
    line-height: 1.9;
}

@media (max-width: 768px) {
    .analytics-quality-note {
        flex-direction: column;
    }

    .analytics-answer-head {
        flex-direction: column;
    }
}



/* Final UI robustness/readability overrides. */
html {
    font-size: 17px !important;
}

/* Hide Streamlit keyboard-submit helper text under text inputs/text areas. */
[data-testid="InputInstructions"],
.stTextInput [data-testid="InputInstructions"],
.stTextArea [data-testid="InputInstructions"],
[data-testid="stTextInput"] [data-testid="InputInstructions"],
[data-testid="stTextArea"] [data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Slightly larger native controls without making the layout bulky. */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] span,
[data-testid="stRadio"] p,
.stButton button p,
[data-testid="stFormSubmitButton"] button p {
    font-size: 0.94rem !important;
}

.stSelectbox label,
.stTextInput label,
.stTextArea label,
[data-testid="stWidgetLabel"] p {
    font-size: 0.90rem !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption,
.stCaption p {
    font-size: 0.79rem !important;
}

</style>

"""
