"""
app.py
──────
Streamlit frontend for DriveMind Google Drive RAG system.

Communicates with the FastAPI backend via HTTP.
Midnight Royal luxury aesthetic — electric indigo + azure mist on obsidian.

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
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="DriveMind — RAG Q&A",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles — Midnight Royal Luxury Aesthetic ──────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --obsidian:        #020617;
    --midnight:        #0f172a;
    --slate-deep:      #1e293b;
    --slate-mid:       #334155;
    --slate-muted:     #475569;
    --indigo:          #6366f1;
    --indigo-dim:      rgba(99, 102, 241, 0.15);
    --indigo-glow:     rgba(99, 102, 241, 0.25);
    --indigo-border:   rgba(99, 102, 241, 0.35);
    --azure:           #38bdf8;
    --azure-dim:       rgba(56, 189, 248, 0.12);
    --azure-border:    rgba(56, 189, 248, 0.3);
    --white-high:      #f1f5f9;
    --white-mid:       #cbd5e1;
    --white-low:       #64748b;
    --white-ghost:     rgba(255, 255, 255, 0.06);
    --white-border:    rgba(255, 255, 255, 0.08);
    --glass-bg:        rgba(15, 23, 42, 0.75);
    --glass-border:    rgba(255, 255, 255, 0.07);
    --radius-sm:       8px;
    --radius-md:       12px;
    --radius-lg:       16px;
    --radius-xl:       20px;
}

/* ── Global Reset ── */
html, body, .stApp {
    background-color: var(--obsidian) !important;
    color: var(--white-high) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, var(--midnight) 60%, #080d1a 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}
[data-testid="stSidebar"] * {
    color: var(--white-mid) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--white-high) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--white-high) !important;
    margin-bottom: 0 !important;
}

/* ── Header bar ── */
[data-testid="stHeader"] {
    background: var(--obsidian) !important;
    border-bottom: 1px solid var(--glass-border) !important;
}

/* ── Text inputs ── */
.stTextInput input {
    background: var(--midnight) !important;
    border: 1.5px solid var(--slate-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--white-high) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px var(--indigo-dim), 0 0 16px var(--indigo-dim) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: var(--slate-muted) !important; }
.stTextInput label { color: var(--white-mid) !important; font-size: 0.82rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--indigo) 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 0 14px var(--indigo-dim) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 0 24px var(--indigo-glow) !important;
}
.stButton > button:disabled {
    background: var(--slate-deep) !important;
    color: var(--white-low) !important;
    box-shadow: none !important;
}

/* ── Tabs as Segmented Controls ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--midnight) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 4px !important;
    gap: 2px !important;
    display: inline-flex !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--white-low) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.025em !important;
    padding: 0.4rem 1.1rem !important;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--indigo) 0%, #4338ca 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px var(--indigo-glow) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    background: var(--white-ghost) !important;
    color: var(--white-mid) !important;
}
[data-testid="stTabsContent"] {
    border: none !important;
    background: transparent !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--midnight) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stMetricLabel"] { color: var(--white-low) !important; font-size: 0.76rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--indigo) !important; font-weight: 700 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--midnight) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    color: var(--white-mid) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div {
    background: var(--midnight) !important;
    border: 1.5px solid var(--slate-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--white-high) !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--indigo-dim) !important;
    border: 1px solid var(--indigo-border) !important;
    border-radius: 6px !important;
    color: #a5b4fc !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--indigo) !important;
    box-shadow: 0 0 10px var(--indigo-glow) !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, var(--indigo), var(--azure)) !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label {
    color: var(--white-mid) !important;
    font-size: 0.86rem !important;
}

/* ── Info / Warning / Error ── */
[data-testid="stAlert"] {
    background: var(--midnight) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--white-mid) !important;
}

/* ── HR / divider ── */
hr { border-color: var(--glass-border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--obsidian); }
::-webkit-scrollbar-thumb { background: var(--slate-mid); border-radius: 3px; }

/* ── General text ── */
p, span, div, li { color: var(--white-high); }
h1, h2, h3, h4 { color: var(--white-high) !important; }
.stMarkdown p { color: var(--white-high) !important; }

/* ────────────────────────────────────────────
   CUSTOM COMPONENT CLASSES
   ──────────────────────────────────────────── */

/* Hero Header */
.dm-hero {
    background: linear-gradient(135deg, #0d1340 0%, #070d24 40%, var(--obsidian) 100%);
    border: 1px solid var(--indigo-border);
    border-radius: var(--radius-xl);
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.dm-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.dm-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.dm-hero-wordmark {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--azure) !important;
    margin-bottom: 0.5rem;
}
.dm-hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.4rem 0 !important;
    background: linear-gradient(90deg, #e0e7ff 0%, var(--azure) 55%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dm-hero p {
    font-size: 0.83rem !important;
    color: var(--white-low) !important;
    margin: 0 0 1rem 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.01em;
}
.dm-pill {
    display: inline-block;
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.22);
    color: #a5b4fc !important;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    margin: 3px 3px 0 0;
    letter-spacing: 0.03em;
}

/* Drive badge in sidebar */
.dm-drive-badge {
    background: rgba(56,189,248,0.08);
    border: 1px solid var(--azure-border);
    color: var(--azure) !important;
    border-radius: var(--radius-md);
    padding: 5px 12px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
}
.dm-drive-badge-warn {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.28);
    color: #fbbf24 !important;
    border-radius: var(--radius-md);
    padding: 5px 12px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.4rem;
}

/* Chat bubbles */
.dm-label-you {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #818cf8 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
    text-align: right;
}
.dm-label-ai {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--azure) !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
}
.dm-user-bubble {
    background: linear-gradient(135deg, #1a1f4e 0%, #1e2d5a 100%);
    border: 1px solid var(--indigo-border);
    border-right: 3px solid var(--indigo);
    color: var(--white-high) !important;
    border-radius: 16px 4px 16px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0 0.4rem 5rem;
    font-size: 0.92rem;
    line-height: 1.65;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.dm-assistant-bubble {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-left: 3px solid var(--azure);
    color: var(--white-high) !important;
    border-radius: 4px 16px 16px 16px;
    padding: 1rem 1.3rem;
    margin: 0.4rem 5rem 0.4rem 0;
    font-size: 0.92rem;
    line-height: 1.8;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 0 24px rgba(99,102,241,0.13), 0 8px 32px rgba(0,0,0,0.4);
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.dm-meta {
    color: var(--slate-muted) !important;
    font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}

/* Source cards */
.dm-source-card {
    background: var(--midnight);
    border: 1px solid var(--glass-border);
    border-left: 3px solid var(--indigo);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
}
.dm-source-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #818cf8 !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Chips */
.dm-chip {
    display: inline-block;
    background: var(--indigo-dim);
    border: 1px solid var(--indigo-border);
    color: #a5b4fc !important;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
    letter-spacing: 0.03em;
}

/* Session date headers */
.dm-session-date {
    font-size: 0.62rem;
    color: var(--slate-muted) !important;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.7rem 0 0.25rem 0;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* Footer */
.dm-footer {
    font-size: 0.68rem;
    color: var(--slate-mid) !important;
    font-family: 'JetBrains Mono', monospace;
    line-height: 2;
    letter-spacing: 0.03em;
}

/* Sidebar title */
.dm-sidebar-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--white-high) !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.dm-sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--white-low) !important;
    margin-bottom: 0.5rem;
    margin-top: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "current_session": None,
    "api_status": None,
    "prefill_question": "",
    "indexed_docs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers ───────────────────────────────────────────────────────────────
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div class='dm-sidebar-title'>🌌 DriveMind</div>",
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
                f"<span class='dm-drive-badge'>✅ {docs} docs · {chunks} chunks indexed</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='dm-drive-badge-warn'>⚠️ No documents indexed</span>",
                unsafe_allow_html=True,
            )
    else:
        st.error("❌ API offline — run: uvicorn api.main:app --reload")

    st.divider()

    # Quick sync button
    st.markdown(
        "<div class='dm-sidebar-section'>🔄 Quick Sync</div>", unsafe_allow_html=True
    )
    if st.button("Sync Google Drive", use_container_width=True):
        with st.spinner("Syncing Drive..."):
            result, code = api_post("/sync-drive", {})
            if code == 200:
                st.success(result.get("message", "Sync complete"))
                st.rerun()
            else:
                st.error(result.get("detail", "Sync failed"))

    if st.session_state.chat_history:
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_session = create_new_session()
            st.rerun()

    st.divider()
    st.markdown(
        "<div class='dm-sidebar-section'>🕐 Chat History</div>", unsafe_allow_html=True
    )
    all_sessions = get_all_sessions()
    if not all_sessions:
        st.markdown(
            "<p style='color:#334155; font-size:0.76rem; font-family:JetBrains Mono,monospace;'>No saved sessions yet.</p>",
            unsafe_allow_html=True,
        )
    else:
        grouped = group_sessions_by_date(all_sessions)
        for group_name, sessions in grouped.items():
            if not sessions:
                continue
            st.markdown(
                f"<div class='dm-session-date'>{group_name}</div>",
                unsafe_allow_html=True,
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
        "<div class='dm-footer'>Google Drive · FastAPI<br>FAISS MMR+BM25 · Groq LLaMA3<br>Cross-Encoder · Streamlit</div>",
        unsafe_allow_html=True,
    )


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class='dm-hero'>
    <div class='dm-hero-wordmark'>DriveMind ·The Intelligence Layer for your Drive.</div>
    <h1>DriveMind — RAG Q&A</h1>
    <p>Your private AI over Google Drive. Powered by LLaMA3 + Hybrid Search + Re-ranking.</p>
    <div style='margin-top:0.8rem;'>
        <span class='dm-pill'>Google Drive</span>
        <span class='dm-pill'>Incremental Sync</span>
        <span class='dm-pill'>Hybrid MMR+BM25</span>
        <span class='dm-pill'>Cross-Encoder Re-ranking</span>
        <span class='dm-pill'>Conversational Memory</span>
        <span class='dm-pill'>Source Attribution</span>
        <span class='dm-pill'>Metadata Filtering</span>
        <span class='dm-pill'>Answer Caching</span>
        <span class='dm-pill'>FastAPI Backend</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_sync, tab_docs = st.tabs(["Chat", "🔄 Drive Sync", "📁 Documents"])


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
            "👈 Sync your Google Drive in the **Drive Sync** tab or click **Sync Google Drive** in the sidebar to get started."
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
            placeholder="Ask a question about your Google Drive documents...",
            label_visibility="collapsed",
            disabled=not api_ready,
        )
    with col2:
        ask_btn = st.button("Ask →", disabled=not api_ready, use_container_width=True)

    if st.session_state.prefill_question:
        st.session_state.prefill_question = ""

    # Handle question
    if ask_btn and question.strip() and api_ready:
        with st.spinner("Searching Drive knowledge base..."):
            start = time.time()

            # Serialize chat history for API (only serializable fields)
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
                "cached": result.get("cached", False),
                "elapsed": elapsed,
                "filtered_to": selected_files if selected_files else None,
            }
            st.session_state.chat_history.append(new_message)

            # Persist to session
            if st.session_state.current_session is None:
                st.session_state.current_session = create_new_session()

            saveable = {
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "elapsed": elapsed,
                "cached": result.get("cached", False),
            }
            st.session_state.current_session = add_message_to_session(
                st.session_state.current_session, saveable
            )
        else:
            st.error(f"Error: {result.get('detail', 'Unknown error')}")

    # Display chat
    if st.session_state.chat_history:
        st.divider()
        for turn in reversed(st.session_state.chat_history):
            cached_badge = " · 💾 cached" if turn.get("cached") else ""
            filtered_badge = (
                f" · 🔍 {', '.join(turn['filtered_to'])}"
                if turn.get("filtered_to")
                else ""
            )

            st.markdown("<div class='dm-label-you'>You</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='dm-user-bubble'>{turn['question']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='dm-meta'>{turn.get('elapsed', '')}s{cached_badge}{filtered_badge}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='dm-label-ai'>DriveMind</div>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='dm-assistant-bubble'>{turn['answer']}</div>",
                unsafe_allow_html=True,
            )

            sources = turn.get("sources", [])
            source_details = turn.get("source_details", [])
            if sources:
                with st.expander(
                    f"📄 Sources ({len(sources)} document{'s' if len(sources) > 1 else ''})"
                ):
                    if source_details:
                        for detail in source_details:
                            st.markdown(
                                f"<div class='dm-source-card'>"
                                f"<div class='dm-source-label'>📄 {detail.get('file_name', '')} · Page {detail.get('page', 1)} · {detail.get('source', 'gdrive')}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        for src in sources:
                            st.markdown(
                                f"<div class='dm-source-card'><div class='dm-source-label'>📄 {src}</div></div>",
                                unsafe_allow_html=True,
                            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — DRIVE SYNC
# ═══════════════════════════════════════════
with tab_sync:
    st.markdown("### 🔄 Google Drive Sync")
    st.markdown(
        "<p style='color:#64748b; font-size:0.84rem;'>Connect to Google Drive and sync documents into the knowledge base. Incremental sync only fetches new or modified files.</p>",
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
            "💡 First sync downloads all files. Subsequent syncs only fetch new/modified files."
        )

    st.divider()

    col_sync, col_cache = st.columns(2)
    with col_sync:
        if st.button("▶ Start Sync", type="primary", use_container_width=True):
            with st.spinner("🔄 Syncing Google Drive... (this may take a minute)"):
                payload = {
                    "folder_id": folder_id if folder_id.strip() else None,
                    "force_full": force_full,
                    "max_files": max_files,
                }
                result, code = api_post("/sync-drive", payload, timeout=600)

            if code == 200:
                st.success(f"✅ {result.get('message', 'Sync complete')}")
                stats = result.get("stats", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Found on Drive", stats.get("total_on_drive", 0))
                c2.metric("New/Updated", stats.get("fetched", 0))
                c3.metric("Skipped", stats.get("skipped_unchanged", 0))
                c4.metric("Chunks Added", stats.get("chunks_added", 0))

                if stats.get("processed_files"):
                    with st.expander("📄 Processed files"):
                        for f in stats["processed_files"]:
                            st.markdown(
                                f"<span class='dm-chip'>✅ {f}</span>",
                                unsafe_allow_html=True,
                            )

                if stats.get("errors"):
                    with st.expander(f"⚠️ {len(stats['errors'])} error(s)"):
                        for err in stats["errors"]:
                            st.error(err)
            elif code == 409:
                st.warning("⏳ Sync already in progress. Please wait.")
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
        if st.button("🗑️ Clear Answer Cache", use_container_width=True):
            r = requests.delete(f"{API_BASE}/cache", timeout=10)
            if r.ok:
                st.success(
                    f"Cleared {r.json().get('cleared_entries', 0)} cached answers"
                )
            else:
                st.error("Failed to clear cache")

    # Current status
    st.divider()
    st.markdown("### Index Status")
    status = api_get("/status")
    if status:
        index = status.get("index", {})
        sync = status.get("sync", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Chunks", index.get("total_chunks", 0))
        c2.metric("Documents", index.get("total_documents", 0))
        c3.metric("Status", "✅ Ready" if index.get("index_loaded") else "⚠️ Empty")

        if sync.get("last_sync"):
            st.markdown(
                f"<span class='dm-chip'>Last sync: {sync['last_sync'][:19]}</span>",
                unsafe_allow_html=True,
            )

        if sync.get("is_running"):
            st.warning("🔄 Sync currently in progress...")


# ═══════════════════════════════════════════
# TAB 3 — DOCUMENTS
# ═══════════════════════════════════════════
with tab_docs:
    st.markdown("### 📁 Indexed Documents")
    st.markdown(
        "<p style='color:#64748b; font-size:0.84rem;'>All documents currently in the knowledge base, sourced from Google Drive.</p>",
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
                f"{mime_icon} {doc['file_name']} — {doc['chunk_count']} chunks"
            ):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Doc ID:** `{doc.get('doc_id', 'N/A')}`")
                col2.markdown(f"**Source:** `{doc.get('source', 'gdrive')}`")
                col1.markdown(f"**MIME:** `{doc.get('mime_type', 'unknown')}`")
                col2.markdown(f"**Chunks:** `{doc.get('chunk_count', 0)}`")

                if st.button(
                    f"Ask about {doc['file_name']}", key=f"ask_{doc['doc_id']}"
                ):
                    st.session_state.prefill_question = (
                        f"What is {doc['file_name']} about?"
                    )
                    st.rerun()
    elif docs_resp and docs_resp.get("total_documents", 0) == 0:
        st.info(
            "No documents indexed yet. Go to **Drive Sync** tab to sync your Google Drive."
        )
    else:
        st.error("Could not connect to API. Make sure the FastAPI server is running.")
