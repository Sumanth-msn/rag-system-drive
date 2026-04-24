"""
src/rag_chain.py
────────────────
RAG answer generation adapted from existing DocMind codebase.

Uses the same proven approach:
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- Groq LLaMA3 for generation
- Context-grounded answer prompt (no hallucination)
- Source attribution — returns file names for every answer

Changes from original:
- Works with Drive metadata (doc_id, file_name, source, page)
- Returns structured response with sources list for API
- Caching layer for repeated queries
"""

import os
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ── Cache config ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
CACHE_TTL_SECONDS = 3600  # 1 hour

# ── Re-ranker config ──────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Load cross-encoder model (singleton)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_documents(
    query: str, documents: List[Document], top_n: int = 5
) -> List[Document]:
    """
    Re-rank documents using Cross Encoder for precision.

    Reads question + chunk TOGETHER → much more accurate than
    vector distance alone.

    Args:
        query: User's question
        documents: Candidate documents from hybrid retriever
        top_n: How many to return after re-ranking

    Returns:
        Top-n documents sorted by relevance
    """
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_n]]


# ── LLM ───────────────────────────────────────────────────────────────────────


def get_llm() -> ChatGroq:
    """Initialize Groq LLaMA3 (same model as existing codebase)."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )


# ── Answer prompt ─────────────────────────────────────────────────────────────

ANSWER_PROMPT = PromptTemplate.from_template("""
You are a precise document assistant. Answer using ONLY the context below.

Instructions:
- Be specific — use exact names, numbers, tools, examples from the document
- Write naturally in paragraphs or bullet points as appropriate
- Do NOT use bold headers or labels like "Direct Answer:" or "Supporting Details:"
- Do NOT add generic closing lines
- Do NOT repeat the same point in different words
- Stop your answer when the document information ends — no padding
- If the answer is not in context, say: "I couldn't find this in the uploaded documents."

Recent conversation:
{chat_history}

Context from documents (re-ranked by relevance):
{context}

Question: {question}

Answer:
""")


# ── Caching ───────────────────────────────────────────────────────────────────


def _cache_key(query: str) -> str:
    """Generate a cache key for a query."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _load_from_cache(query: str) -> Optional[Dict]:
    """Load cached answer if still valid."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(query)}.json"

    if not cache_path.exists():
        return None

    with open(cache_path, "r") as f:
        cached = json.load(f)

    age = time.time() - cached.get("timestamp", 0)
    if age > CACHE_TTL_SECONDS:
        cache_path.unlink()  # expired
        return None

    return cached


def _save_to_cache(query: str, result: Dict):
    """Save answer to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(query)}.json"

    cacheable = {
        "answer": result["answer"],
        "sources": result["sources"],
        "timestamp": time.time(),
        "cached": True,
    }
    with open(cache_path, "w") as f:
        json.dump(cacheable, f, indent=2)


# ── Main answer function ──────────────────────────────────────────────────────


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[Dict]] = None,
    use_cache: bool = True,
) -> Dict:
    """
    Generate an answer from retrieved documents with source attribution.

    Pipeline:
    1. Check cache for repeated query
    2. Re-rank documents with cross-encoder
    3. Format context string
    4. Call Groq LLaMA3 with grounded prompt
    5. Extract unique source filenames
    6. Cache + return result

    Args:
        query: User's question
        retrieved_docs: Candidate chunks from hybrid retriever
        chat_history: Recent Q&A turns for conversational context
        use_cache: Whether to use/populate the cache

    Returns:
        Dict with "answer" (str) and "sources" (List[str])
    """
    # 1. Cache check
    if use_cache:
        cached = _load_from_cache(query)
        if cached:
            return cached

    # 2. Re-rank
    reranked_docs = rerank_documents(query, retrieved_docs, top_n=5)

    if not reranked_docs:
        return {
            "answer": "I couldn't find relevant information in the documents. Please sync your Drive or check if documents were processed correctly.",
            "sources": [],
            "cached": False,
        }

    # 3. Format context with source labels
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('file_name', 'unknown')} "
            f"| Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in reranked_docs
        ]
    )

    # 4. Format chat history
    history_text = "No previous conversation."
    if chat_history:
        recent = chat_history[-3:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn.get('question', '')}")
            lines.append(f"Assistant: {turn.get('answer', '')[:200]}...")
        history_text = "\n".join(lines)

    # 5. Generate answer
    llm = get_llm()
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "question": query,
            "context": context,
            "chat_history": history_text,
        }
    )

    # 6. Extract unique source filenames for attribution
    sources = list(
        dict.fromkeys(
            [doc.metadata.get("file_name", "unknown") for doc in reranked_docs]
        )
    )

    result = {
        "answer": answer.strip(),
        "sources": sources,
        "source_details": [
            {
                "file_name": doc.metadata.get("file_name"),
                "doc_id": doc.metadata.get("doc_id"),
                "page": doc.metadata.get("page", 0) + 1,
                "source": doc.metadata.get("source", "gdrive"),
            }
            for doc in reranked_docs
        ],
        "cached": False,
    }

    # 7. Cache result
    if use_cache:
        _save_to_cache(query, result)

    return result
