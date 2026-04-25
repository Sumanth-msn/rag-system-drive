"""
app.py
──────
Streamlit frontend for DriveMind Google Drive RAG system.

Communicates with the FastAPI backend via HTTP.
Midnight Royal luxury aesthetic — executive dashboard.

Tabs:
  💬 Chat     — Q&A over synced Drive documents
  🔄 Sync     — Trigger Drive sync, view status
  📁 Documents — Browse indexed documents
"""

import streamlit as st
import requests
import time
from src.chat_history import (
    create_new_session,
    add_message_to_session,
    get_all_sessions,
    delete_session,
    format_session_label,
    group_sessions_by_date,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "https://scoop-bust-jailbreak.ngrok-free.dev"  # FastAPI backend URL (update with your ngrok URL)

st.set_page_config(
    page_title="DriveMind — RAG Q&A",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Midnight Royal CSS ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #020617 !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Obsidian canvas ambient glow */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99,102,241,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(56,189,248,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Sidebar brand */
.sidebar-brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.sidebar-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #475569 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.95) !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px) !important;
}

/* ── Hero Header block ───────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #020617 60%, #0c1425 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 30%;
    width: 300px; height: 100px;
    background: radial-gradient(ellipse, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #6366f1 !important;
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(100deg, #f1f5f9 30%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 0.82rem;
    color: #64748b !important;
    margin: 0 0 1.2rem 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* Pill tags */
.tag {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    color: #818cf8 !important;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    margin: 3px 3px 0 0;
    letter-spacing: 0.04em;
}

.tag-blue {
    background: rgba(56,189,248,0.07);
    border-color: rgba(56,189,248,0.18);
    color: #7dd3fc !important;
}

/* ── Input ───────────────────────────────────────────────────────────────── */
.stTextInput input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 0.75rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 0 20px rgba(99,102,241,0.08) !important;
    outline: none !important;
}

.stTextInput input::placeholder { color: #334155 !important; }
.stTextInput label { color: #64748b !important; font-size: 0.8rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.3rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.25) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

.stButton > button:disabled {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #334155 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Ghost sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
    font-size: 0.8rem !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(56,189,248,0.1) 100%) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #e2e8f0 !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.15) !important;
    transform: none !important;
}

/* ── Tabs — Segmented Control style ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    backdrop-filter: blur(8px) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #64748b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}

.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
.msg-label-user {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366f1 !important;
    margin-bottom: 0.35rem;
    text-align: right;
    padding-right: 0.3rem;
}

.msg-label-ai {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8 !important;
    margin-bottom: 0.35rem;
    padding-left: 0.3rem;
}

/* User bubble */
.bubble-user {
    background: linear-gradient(135deg, #1e1b4b 0%, #172554 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-right: 3px solid #6366f1;
    color: #e2e8f0 !important;
    border-radius: 14px 4px 14px 14px;
    padding: 1rem 1.3rem;
    margin: 0.4rem 0 0.8rem 5rem;
    font-size: 0.9rem;
    line-height: 1.65;
    backdrop-filter: blur(8px);
}

/* AI bubble — glassmorphism with blue glow */
.bubble-ai {
    background: rgba(2, 6, 23, 0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 2px solid rgba(99,102,241,0.4);
    color: #f1f5f9 !important;
    border-radius: 4px 14px 14px 14px;
    padding: 1.1rem 1.4rem;
    margin: 0.4rem 5rem 0.4rem 0;
    font-size: 0.9rem;
    line-height: 1.8;
    box-shadow: 0 0 24px rgba(99,102,241,0.12), 0 4px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
}

/* ── Confidence score badge ──────────────────────────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0 10px 0.3rem;
    flex-wrap: wrap;
}

.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.conf-high {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399 !important;
}

.conf-mid {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24 !important;
}

.conf-low {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171 !important;
}

.meta-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: #334155 !important;
    letter-spacing: 0.04em;
}

/* ── Source cards ────────────────────────────────────────────────────────── */
.source-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 2px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    backdrop-filter: blur(6px);
}

.source-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #6366f1 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    color: #34d399 !important;
    border-radius: 8px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    margin-bottom: 0.6rem;
}

.status-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fbbf24 !important;
    border-radius: 8px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    margin-bottom: 0.6rem;
}

/* ── Chip ────────────────────────────────────────────────────────────────── */
.chip {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    color: #818cf8 !important;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
    letter-spacing: 0.03em;
}

/* ── Session date labels ─────────────────────────────────────────────────── */
.session-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #334155 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 0.8rem 0 0.3rem 0;
    padding-left: 0.2rem;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(8px) !important;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px) !important;
}

[data-testid="stExpander"] summary {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stExpander"] summary:hover { color: #94a3b8 !important; }

/* ── Info / Warning / Success / Error boxes ──────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(56,189,248,0.05) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stInfo"] * { color: #7dd3fc !important; }

[data-testid="stWarning"] {
    background: rgba(245,158,11,0.05) !important;
    border: 1px solid rgba(245,158,11,0.15) !important;
    border-radius: 10px !important;
}

[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.06) !important;
    border: 1px solid rgba(16,185,129,0.18) !important;
    border-radius: 10px !important;
}

[data-testid="stError"] {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.18) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 0.8rem 0 !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── Typography ──────────────────────────────────────────────────────────── */
p, span, div, li { color: #e2e8f0; }
h1, h2, h3, h4 { color: #f1f5f9 !important; font-weight: 700 !important; }
.stMarkdown p { color: #cbd5e1 !important; line-height: 1.7 !important; }
strong { color: #f1f5f9 !important; font-weight: 700 !important; }

/* ── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}

/* ── Checkbox ────────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #94a3b8 !important; }

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.footer-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #1e293b !important;
    line-height: 2;
    letter-spacing: 0.05em;
}

/* ── Turn separator ──────────────────────────────────────────────────────── */
.turn-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.12), transparent);
    margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state (unchanged) ─────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "current_session": None,
    "api_status": None,
    "prefill_question": "",
    "indexed_docs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers (unchanged) ───────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path: str, data: dict, timeout: int = 300):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


# ── Confidence score helper ───────────────────────────────────────────────────
_NO_SCORE_HTML = (
    "<span class='conf-badge' style='background:rgba(100,116,139,0.1);"
    "border:1px solid rgba(100,116,139,0.2);color:#64748b !important;'>"
    "— &nbsp;NO SCORE</span>"
)

_NOT_FOUND_ANSWER_PHRASES = [
    "couldn't find",
    "could not find",
    "not find this",
    "no information",
    "not mentioned",
    "does not contain",
    "not available in",
    "i don't have",
    "not provided in",
    "not in the uploaded",
]


def get_confidence_badge(confidence, answer: str = "") -> str:
    """
    Render a confidence badge.

    Distinguishes four cases:
      None  → key missing entirely (very old session) → NO SCORE (grey)
      0 + answer has "not found" phrases → NOT FOUND (red) — genuine miss
      0 + answer is a real answer → legacy zero saved before scoring existed
                                    → NO SCORE (grey, not misleading red)
      1-100 → real scored answer → VERIFIED / SOURCED / PARTIAL tier

    This correctly handles sessions saved at different points in time:
      - Sessions saved before confidence existed: key missing → None → NO SCORE
      - Sessions saved when confidence defaulted to 0: value is 0, but answer
        is real → detect via answer content → NO SCORE
      - Sessions saved with LLM-as-judge: real score 1-100 → show tier
      - Genuine not-found: score is 0 AND answer contains not-found phrases
    """
    # Case 1: key entirely missing (very old session)
    if confidence is None:
        return _NO_SCORE_HTML

    pct = max(0, min(100, int(confidence)))

    # Case 2: score is 0 — distinguish genuine not-found from legacy zero
    if pct == 0:
        answer_lower = answer.strip().lower()
        is_genuine_not_found = any(p in answer_lower for p in _NOT_FOUND_ANSWER_PHRASES)
        is_empty_answer = len(answer.strip()) < 10

        if is_genuine_not_found or is_empty_answer:
            # Real not-found response
            return "<span class='conf-badge conf-low'>○&nbsp; NOT FOUND</span>"
        else:
            # Has a real answer but confidence=0 → legacy session saved with default 0
            return _NO_SCORE_HTML

    # Case 3: real scored answer
    if pct >= 75:
        cls, icon, label = "conf-high", "●", f"VERIFIED · {pct}%"
    elif pct >= 35:
        cls, icon, label = "conf-high", "◉", f"SOURCED · {pct}%"
    elif pct >= 15:
        cls, icon, label = "conf-mid", "◐", f"PARTIAL · {pct}%"
    else:
        cls, icon, label = "conf-low", "○", f"LOW · {pct}%"

    return f"<span class='conf-badge {cls}'>{icon}&nbsp; {label}</span>"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div class='sidebar-brand'>✦ DriveMind</div>
    <div class='sidebar-sub'>RAG · Google Drive Intelligence</div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    # API Status
    status = api_get("/status")
    if status:
        st.session_state.api_status = status
        chunks = status.get("index", {}).get("total_chunks", 0)
        docs = status.get("index", {}).get("total_documents", 0)
        if chunks > 0:
            st.markdown(
                f"<div class='status-ok'>✓ &nbsp;{docs} docs &nbsp;·&nbsp; {chunks} chunks</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='status-warn'>⚠ &nbsp;No documents indexed</div>",
                unsafe_allow_html=True,
            )
    else:
        st.error("API offline — run: uvicorn api.main:app --reload")

    st.divider()

    st.markdown("**Quick Sync**")
    if st.button("↑ Sync Google Drive", use_container_width=True):
        with st.spinner("Connecting to Drive..."):
            result, code = api_post("/sync-drive", {})
            if code == 200:
                st.success(result.get("message", "Sync complete"))
                st.rerun()
            else:
                st.error(result.get("detail", "Sync failed"))

    if st.session_state.chat_history:
        st.divider()
        if st.button("✕ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_session = create_new_session()
            st.rerun()

    st.divider()
    st.markdown("**History**")
    all_sessions = get_all_sessions()
    if not all_sessions:
        st.markdown(
            "<p style='color:#334155; font-size:0.75rem; font-family: JetBrains Mono, monospace;'>No sessions yet.</p>",
            unsafe_allow_html=True,
        )
    else:
        grouped = group_sessions_by_date(all_sessions)
        for group_name, sessions in grouped.items():
            if not sessions:
                continue
            st.markdown(
                f"<div class='session-date'>{group_name}</div>", unsafe_allow_html=True
            )
            for session in sessions:
                label = format_session_label(session)
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"💬 {label}",
                        key=f"load_{session['session_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.chat_history = session.get("messages", [])
                        st.session_state.current_session = session
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"del_{session['session_id']}"):
                        delete_session(session["session_id"])
                        st.rerun()

    st.divider()
    st.markdown(
        "<div class='footer-text'>DRIVE · FAISS · BM25<br>GROQ · LLAMA3 · RERANK<br>FASTAPI · STREAMLIT</div>",
        unsafe_allow_html=True,
    )


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class='hero-header'>
    <div class='hero-eyebrow'>✦ &nbsp; Highwatch AI &nbsp;·&nbsp; Drive Intelligence Platform</div>
    <div class='hero-title'>DriveMind</div>
    <div class='hero-sub'>Your personal AI over Google Drive — hybrid search, re-ranked precision, grounded answers.</div>
    <div>
        <span class='tag'>Google Drive</span>
        <span class='tag'>Incremental Sync</span>
        <span class='tag tag-blue'>Hybrid MMR+BM25</span>
        <span class='tag tag-blue'>Cross-Encoder Re-rank</span>
        <span class='tag'>Conversational Memory</span>
        <span class='tag'>Source Attribution</span>
        <span class='tag tag-blue'>Metadata Filter</span>
        <span class='tag'>Answer Cache</span>
        <span class='tag'>FastAPI</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_sync, tab_docs = st.tabs(
    [
        "  💬  Chat  ",
        "  ↑  Drive Sync  ",
        "  📁  Documents  ",
    ]
)


# ═══════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════
with tab_chat:
    api_ready = (
        st.session_state.api_status is not None
        and st.session_state.api_status.get("index", {}).get("total_chunks", 0) > 0
    )

    if not api_ready:
        st.info(
            "Sync your Google Drive in the **Drive Sync** tab or click **↑ Sync Google Drive** in the sidebar to get started."
        )

    # Optional: filter by specific files
    if api_ready:
        with st.expander("🔍 Filter by specific documents (optional)"):
            docs_resp = api_get("/documents")
            if docs_resp:
                doc_names = [d["file_name"] for d in docs_resp.get("documents", [])]
                selected_files = st.multiselect(
                    "Only search within these files:",
                    options=doc_names,
                    placeholder="Leave empty to search all documents",
                )
            else:
                selected_files = []
    else:
        selected_files = []

    # Question input
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "question",
            value=st.session_state.prefill_question,
            placeholder="Ask anything about your Drive documents...",
            label_visibility="collapsed",
            disabled=not api_ready,
        )
    with col2:
        ask_btn = st.button("Ask →", disabled=not api_ready, use_container_width=True)

    if st.session_state.prefill_question:
        st.session_state.prefill_question = ""

    # Handle question (logic unchanged)
    if ask_btn and question.strip() and api_ready:
        with st.spinner("Searching knowledge base..."):
            start = time.time()

            api_history = [
                {"question": m["question"], "answer": m["answer"]}
                for m in st.session_state.chat_history[-5:]
            ]

            if selected_files:
                result, code = api_post(
                    "/ask/filtered",
                    {
                        "query": question,
                        "file_names": selected_files,
                        "chat_history": api_history,
                    },
                )
            else:
                result, code = api_post(
                    "/ask",
                    {
                        "query": question,
                        "chat_history": api_history,
                        "use_cache": True,
                    },
                )
            elapsed = round(time.time() - start, 2)

        if code == 200:
            new_message = {
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "source_details": result.get("source_details", []),
                "confidence": result.get("confidence", 0),
                "cached": result.get("cached", False),
                "elapsed": elapsed,
                "filtered_to": selected_files if selected_files else None,
            }
            st.session_state.chat_history.append(new_message)

            if st.session_state.current_session is None:
                st.session_state.current_session = create_new_session()

            saveable = {
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "source_details": result.get("source_details", []),
                "confidence": result.get("confidence", 0),
                "elapsed": elapsed,
                "cached": result.get("cached", False),
                "filtered_to": selected_files if selected_files else None,
            }
            st.session_state.current_session = add_message_to_session(
                st.session_state.current_session, saveable
            )
        else:
            st.error(f"Error: {result.get('detail', 'Unknown error')}")

    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        for turn in reversed(st.session_state.chat_history):
            source_details = turn.get("source_details", [])
            cached_txt = " · 💾 cached" if turn.get("cached") else ""
            filtered_txt = (
                f" · filtered: {', '.join(turn['filtered_to'])}"
                if turn.get("filtered_to")
                else ""
            )

            # ── User bubble ──
            st.markdown("<div class='msg-label-user'>You</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='bubble-user'>{turn['question']}</div>",
                unsafe_allow_html=True,
            )

            # ── AI label ──
            st.markdown(
                "<div class='msg-label-ai'>✦ &nbsp; DriveMind</div>",
                unsafe_allow_html=True,
            )

            # ── Confidence + meta row (NEW) ──
            conf_html = get_confidence_badge(
                turn.get("confidence"), turn.get("answer", "")
            )
            elapsed_html = (
                f"<span class='meta-chip'>{turn.get('elapsed', '')}s"
                f"{cached_txt}{filtered_txt}</span>"
            )
            st.markdown(
                f"<div class='conf-row'>{conf_html}{elapsed_html}</div>",
                unsafe_allow_html=True,
            )

            # ── AI bubble ──
            st.markdown(
                f"<div class='bubble-ai'>{turn['answer']}</div>",
                unsafe_allow_html=True,
            )

            # ── Sources ──
            sources = turn.get("sources", [])
            if sources:
                with st.expander(
                    f"📄 Sources — {len(sources)} document{'s' if len(sources) > 1 else ''}"
                ):
                    if source_details:
                        for detail in source_details:
                            st.markdown(
                                f"<div class='source-card'>"
                                f"<div class='source-label'>"
                                f"📄 &nbsp;{detail.get('file_name', '')} &nbsp;·&nbsp; "
                                f"pg {detail.get('page', 1)} &nbsp;·&nbsp; "
                                f"{detail.get('source', 'gdrive')}"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        for src in sources:
                            st.markdown(
                                f"<div class='source-card'>"
                                f"<div class='source-label'>📄 &nbsp;{src}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            st.markdown("<div class='turn-sep'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — DRIVE SYNC
# ═══════════════════════════════════════════
with tab_sync:
    st.markdown("### Drive Sync")
    st.markdown(
        "<p style='color:#475569; font-size:0.85rem;'>Connect to Google Drive and index documents. Incremental sync only fetches new or modified files.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        folder_id = st.text_input(
            "Folder ID (optional)",
            placeholder="Leave empty to sync all Drive files",
            help="Google Drive folder ID from the URL: drive.google.com/drive/folders/FOLDER_ID",
        )
        max_files = st.slider(
            "Max files to sync", min_value=10, max_value=500, value=200, step=10
        )

    with col2:
        force_full = st.checkbox(
            "Force full re-sync",
            value=False,
            help="Re-process ALL files, ignoring the incremental cache",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "First sync downloads all files. Subsequent syncs only fetch new or modified files."
        )

    st.divider()

    col_sync, col_cache = st.columns(2)
    with col_sync:
        if st.button("↑ Start Sync", type="primary", use_container_width=True):
            with st.spinner("Syncing Google Drive..."):
                payload = {
                    "folder_id": folder_id if folder_id.strip() else None,
                    "force_full": force_full,
                    "max_files": max_files,
                }
                result, code = api_post("/sync-drive", payload, timeout=600)

            if code == 200:
                st.success(f"✓ {result.get('message', 'Sync complete')}")
                stats = result.get("stats", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Found on Drive", stats.get("total_on_drive", 0))
                c2.metric("New / Updated", stats.get("fetched", 0))
                c3.metric("Unchanged", stats.get("skipped_unchanged", 0))
                c4.metric("Chunks Added", stats.get("chunks_added", 0))

                if stats.get("processed_files"):
                    with st.expander("Processed files"):
                        for f in stats["processed_files"]:
                            st.markdown(
                                f"<span class='chip'>✓ {f}</span>",
                                unsafe_allow_html=True,
                            )

                if stats.get("errors"):
                    with st.expander(f"⚠ {len(stats['errors'])} error(s)"):
                        for err in stats["errors"]:
                            st.error(err)

            elif code == 409:
                st.warning("Sync already in progress. Please wait.")
            elif code == 400:
                st.error(result.get("detail", "Setup error"))
                st.markdown("""
                **Credential setup required:**
                1. Place `credentials.json` (OAuth) OR `service_account.json` in the `credentials/` folder
                2. See README for step-by-step Google Cloud Console instructions
                """)
            else:
                st.error(f"Sync failed: {result.get('detail', 'Unknown error')}")

    with col_cache:
        if st.button("✕ Clear Answer Cache", use_container_width=True):
            r = requests.delete(f"{API_BASE}/cache", timeout=10)
            if r.ok:
                st.success(
                    f"Cleared {r.json().get('cleared_entries', 0)} cached answers"
                )
            else:
                st.error("Failed to clear cache")

    # Status
    st.divider()
    st.markdown("### Index Status")
    status = api_get("/status")
    if status:
        index = status.get("index", {})
        sync = status.get("sync", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Chunks", index.get("total_chunks", 0))
        c2.metric("Documents", index.get("total_documents", 0))
        c3.metric("Index", "Ready" if index.get("index_loaded") else "Empty")

        if sync.get("last_sync"):
            st.markdown(
                f"<span class='chip'>Last sync: {sync['last_sync'][:19]}</span>",
                unsafe_allow_html=True,
            )
        if sync.get("is_running"):
            st.warning("Sync in progress...")


# ═══════════════════════════════════════════
# TAB 3 — DOCUMENTS
# ═══════════════════════════════════════════
with tab_docs:
    st.markdown("### Indexed Documents")
    st.markdown(
        "<p style='color:#475569; font-size:0.85rem;'>All documents currently in the knowledge base, sourced from Google Drive.</p>",
        unsafe_allow_html=True,
    )

    docs_resp = api_get("/documents")
    if docs_resp and docs_resp.get("documents"):
        docs = docs_resp["documents"]
        c1, c2 = st.columns(2)
        c1.metric("Total Documents", docs_resp.get("total_documents", 0))
        c2.metric("Total Chunks", docs_resp.get("total_chunks", 0))

        st.divider()
        for doc in sorted(docs, key=lambda x: x["file_name"]):
            mime_icon = {
                "application/pdf": "📄",
                "text/plain": "📝",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📃",
            }.get(doc.get("mime_type", ""), "📎")

            with st.expander(
                f"{mime_icon}  {doc['file_name']} — {doc['chunk_count']} chunks"
            ):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Doc ID:** `{doc.get('doc_id', 'N/A')}`")
                col2.markdown(f"**Source:** `{doc.get('source', 'gdrive')}`")
                col1.markdown(f"**MIME:** `{doc.get('mime_type', 'unknown')}`")
                col2.markdown(f"**Chunks:** `{doc.get('chunk_count', 0)}`")

                if st.button("Ask about this document", key=f"ask_{doc['doc_id']}"):
                    st.session_state.prefill_question = (
                        f"What is {doc['file_name']} about?"
                    )
                    st.rerun()

    elif docs_resp and docs_resp.get("total_documents", 0) == 0:
        st.info("No documents indexed yet. Go to the Drive Sync tab to get started.")
    else:
        st.error("Could not connect to API. Make sure the FastAPI server is running.")
