"""
app.py
──────
Streamlit frontend for DocMind Google Drive RAG system.

Communicates with the FastAPI backend via HTTP.
Keeps all the visual quality of the original DocMind UI.

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
    page_title="DocMind Drive — RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles (preserved from original DocMind) ──────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp { background-color: #0a0a0f !important; color: #e8e8f0 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stSidebar"] { background-color: #0f0f18 !important; border-right: 1px solid #1e1e2e !important; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stHeader"] { background: #0a0a0f !important; border-bottom: 1px solid #1e1e2e; }

.app-header { background: linear-gradient(135deg, #1a0533 0%, #0d1f4e 50%, #001a2e 100%); border: 1px solid #2a1a4e; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; }
.app-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem 0; background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.app-header p { font-size: 0.82rem; color: #7c7c9a !important; margin: 0; font-family: 'JetBrains Mono', monospace; }
.pill { display: inline-block; background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25); color: #a78bfa !important; border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin: 3px 3px 0 0; }

.stTextInput input { background: #0f0f18 !important; border: 1.5px solid #2a2a3e !important; border-radius: 12px !important; color: #e8e8f0 !important; font-size: 0.95rem !important; padding: 0.7rem 1rem !important; }
.stTextInput input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important; }
.stTextInput input::placeholder { color: #4a4a6a !important; }
.stTextInput label { color: #e8e8f0 !important; }

.stButton > button { background: linear-gradient(135deg, #7c3aed, #4f46e5) !important; color: #ffffff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.5rem 1.2rem !important; }
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { background: #1e1e2e !important; color: #4a4a6a !important; }

.user-bubble { background: linear-gradient(135deg, #2d1b69, #1e3a8a); border: 1px solid #3730a3; color: #e8e8f0 !important; border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.2rem; margin: 0.6rem 0 0.6rem 4rem; font-size: 0.93rem; line-height: 1.6; }
.assistant-bubble { background: #0f0f18; border: 1px solid #1e1e2e; color: #e8e8f0 !important; border-radius: 16px 16px 16px 4px; padding: 1rem 1.2rem; margin: 0.6rem 4rem 0.6rem 0; font-size: 0.93rem; line-height: 1.8; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
.label-you { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #7c3aed !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; text-align: right; }
.label-ai { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #34d399 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }

.source-card { background: #0a0a0f; border: 1px solid #1e1e2e; border-left: 3px solid #7c3aed; border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.source-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #7c3aed !important; font-weight: 600; margin-bottom: 0.4rem; text-transform: uppercase; }
.source-text { color: #9898b8 !important; font-size: 0.82rem; line-height: 1.6; }

.chip { display: inline-block; background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.25); color: #a78bfa !important; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin: 2px; }
.drive-badge { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34d399 !important; border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; display: inline-block; margin-bottom: 0.5rem; }
.session-date { font-size: 0.68rem; color: #4a4a6a !important; font-family: 'JetBrains Mono', monospace; margin: 0.5rem 0 0.2rem 0; text-transform: uppercase; letter-spacing: 0.5px; }

[data-testid="stMetric"] { background: #0f0f18 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; padding: 0.8rem !important; }
[data-testid="stMetricLabel"] { color: #7c7c9a !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; }
[data-testid="stExpander"] { background: #0f0f18 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; }
hr { border-color: #1e1e2e !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
p, span, div, li { color: #e8e8f0; }
h1, h2, h3, h4 { color: #e8e8f0 !important; }
.stMarkdown p { color: #e8e8f0 !important; }
.footer-text { font-size: 0.72rem; color: #3a3a5a !important; font-family: 'JetBrains Mono', monospace; line-height: 1.8; }
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
    st.markdown("## 📚 DocMind Drive")
    st.divider()

    # API Status
    status = api_get("/status")
    if status:
        st.session_state.api_status = status
        chunks = status.get("index", {}).get("total_chunks", 0)
        docs = status.get("index", {}).get("total_documents", 0)
        if chunks > 0:
            st.markdown(
                f"<span class='drive-badge'>✅ {docs} docs · {chunks} chunks indexed</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='drive-badge' style='color:#fbbf24 !important; border-color:rgba(251,191,36,0.3) !important;'>⚠️ No documents indexed</span>",
                unsafe_allow_html=True,
            )
    else:
        st.error("❌ API offline — run: uvicorn api.main:app --reload")

    st.divider()

    # Quick sync button
    st.markdown("### 🔄 Quick Sync")
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
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_session = create_new_session()
            st.rerun()

    st.divider()
    st.markdown("### 🕐 Chat History")
    all_sessions = get_all_sessions()
    if not all_sessions:
        st.markdown(
            "<p style='color:#4a4a6a; font-size:0.78rem;'>No saved sessions yet.</p>",
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
        "<div class='footer-text'>Google Drive · FastAPI<br>FAISS MMR+BM25 · Groq LLaMA3<br>Cross-Encoder · Streamlit</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class='app-header'>
    <h1>📚 DocMind Drive — RAG Q&A</h1>
    <p>Your personal ChatGPT over Google Drive. Powered by LLaMA3 + Hybrid Search + Re-ranking.</p>
    <div style='margin-top:0.8rem;'>
        <span class='pill'>Google Drive</span>
        <span class='pill'>Incremental Sync</span>
        <span class='pill'>Hybrid MMR+BM25</span>
        <span class='pill'>Cross-Encoder Re-ranking</span>
        <span class='pill'>Conversational Memory</span>
        <span class='pill'>Source Attribution</span>
        <span class='pill'>Metadata Filtering</span>
        <span class='pill'>Answer Caching</span>
        <span class='pill'>FastAPI Backend</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_sync, tab_docs = st.tabs(["💬 Chat", "🔄 Drive Sync", "📁 Documents"])


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
                f" · 🔍 filtered: {', '.join(turn['filtered_to'])}"
                if turn.get("filtered_to")
                else ""
            )

            st.markdown("<div class='label-you'>You</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='user-bubble'>{turn['question']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='color:#3a3a5a; font-size:0.7rem; font-family:monospace;'>"
                f"{turn.get('elapsed', '')}s{cached_badge}{filtered_badge}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='label-ai'>DocMind Drive</div>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='assistant-bubble'>{turn['answer']}</div>",
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
                                f"<div class='source-card'>"
                                f"<div class='source-label'>📄 {detail.get('file_name', '')} · Page {detail.get('page', 1)} · {detail.get('source', 'gdrive')}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        for src in sources:
                            st.markdown(
                                f"<div class='source-card'><div class='source-label'>📄 {src}</div></div>",
                                unsafe_allow_html=True,
                            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — DRIVE SYNC
# ═══════════════════════════════════════════
with tab_sync:
    st.markdown("### 🔄 Google Drive Sync")
    st.markdown(
        "<p style='color:#7c7c9a; font-size:0.85rem;'>Connect to Google Drive and sync documents into the knowledge base. Incremental sync only fetches new or modified files.</p>",
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
                                f"<span class='chip'>✅ {f}</span>",
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
    st.markdown("### 📊 Index Status")
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
                f"<span class='chip'>Last sync: {sync['last_sync'][:19]}</span>",
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
        "<p style='color:#7c7c9a; font-size:0.85rem;'>All documents currently in the knowledge base, sourced from Google Drive.</p>",
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
