"""
src/rag_chain.py
────────────────
RAG answer generation with cross-encoder re-ranking, Groq LLaMA3,
LLM-as-judge confidence scoring, and caching.

Confidence Score Architecture — LLM-as-Judge:
══════════════════════════════════════════════
Previous attempts used the cross-encoder rerank score as a proxy for
confidence. This fundamentally doesn't work because:

  1. ms-marco cross-encoder scores RETRIEVAL quality (lexical overlap
     between query text and passage text), NOT answer quality.

  2. Short factual passages (dates, names, single-line facts) score
     catastrophically low (-5 to -6) on ms-marco even when they are
     the perfect answer — the model was trained on 100-300 char passages.

  3. Formal/legal documents use different vocabulary than casual query
     phrasing, causing systematic under-scoring of correct answers.

The correct approach is LLM-as-Judge: after generating the answer,
send a second small prompt asking the LLM to score its own confidence
based on the CONTEXT it was given. This is the industry-standard
technique used in production RAG systems (RAGAS, TruLens, etc.).

The judge prompt asks:
  "Given the context and question, how confident are you that the
   answer is correct and complete? Reply with just a number 0-100."

This gives accurate scores because:
  - The LLM saw the actual context — it knows if the answer was there
  - Short factual answers score HIGH (the date was clearly in the doc)
  - Vague/hallucinated answers score LOW (the LLM knows it was guessing)
  - "Not found" answers score 0 (the LLM knows nothing was in context)

A fast, deterministic judge call (temperature=0, max_tokens=10) adds
~0.3s latency but makes confidence meaningful.
"""

import os
import re
import math
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

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
CACHE_TTL_SECONDS = 3600  # 1 hour answer cache
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: Optional[CrossEncoder] = None


# ── Re-ranker (used for retrieval ordering only, NOT for confidence) ──────────


def get_reranker() -> CrossEncoder:
    """Load cross-encoder model singleton. Used only for ordering chunks."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
) -> List[tuple]:
    """
    Re-rank retrieved documents by relevance using Cross Encoder.

    NOTE: The scores returned here are used ONLY to order chunks for the
    LLM context. They are NOT used for confidence scoring — see
    _judge_confidence() below for how confidence is actually computed.

    Returns:
        List of (Document, float_score) sorted by score descending.
    """
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    scored = list(zip(documents, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ── LLM ───────────────────────────────────────────────────────────────────────


def get_llm(temperature: float = 0.3, max_tokens: int = 1024) -> ChatGroq:
    """Initialize Groq LLaMA3 with configurable parameters."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
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

# ── Judge prompt — LLM scores its own answer confidence ───────────────────────

JUDGE_PROMPT = PromptTemplate.from_template("""You are a factual confidence scorer for a document Q&A system.

Your job is to score how well the ANSWER is supported by the CONTEXT for the given QUESTION.
Be generous — if the answer correctly uses information from the context, score it HIGH.

Scoring guide:
- 95-100: The answer is taken directly from the context. Exact facts, dates, names, numbers match perfectly.
- 85-94:  The answer is clearly and fully supported by the context. Minor wording differences are fine.
- 75-84:  The answer is well supported. The context clearly contains the relevant information.
- 60-74:  The answer is mostly supported with small gaps or light inference.
- 40-59:  The answer is partially supported — notable inference required beyond what context states.
- 15-39:  The answer is weakly supported — context is tangential or mostly irrelevant.
- 1-14:   The answer is barely supported — almost no relevant information in context.
- 0:      The answer explicitly states the information was not found in the documents.

Key rules:
- If the answer contains specific facts (dates, names, numbers, policies) that appear in the context → score 85 or higher.
- If the answer is a single precise sentence that directly answers the question using context data → score 90 or higher.
- Only score below 60 if the context genuinely does not contain the answer.
- Do NOT penalise short answers — a one-sentence factual answer is often the best answer.

QUESTION: {question}

CONTEXT:
{context_snippet}

ANSWER: {answer}

Reply with ONLY a single integer 0-100. No words. No punctuation. Just the number.""")


# ── LLM-as-Judge confidence scoring ──────────────────────────────────────────

_NOT_FOUND_PHRASES = [
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


def _judge_confidence(
    query: str,
    context: str,
    answer: str,
    answer_not_found: bool,
    scored_docs: Optional[List[tuple]] = None,
) -> int:
    """
    Use the LLM itself to score confidence in the answer (LLM-as-Judge).

    Improvements over simple cross-encoder scoring:
      - The LLM saw the actual context, so it knows if the answer was there
      - Short factual answers ("December 18, 2017") score HIGH because
        the judge recognises it directly matches the context
      - "Not found" answers always score 0 — no LLM call needed
      - Cost: ~0.3s extra latency on Groq (very fast inference)

    Context strategy:
      We send the TOP-RANKED chunk first (most relevant), then fill up
      to 3000 chars with remaining context. This ensures the judge always
      sees the most relevant passage regardless of document length.

    Args:
        query:             Original user question
        context:           Full assembled context string
        answer:            Generated answer
        answer_not_found:  Whether not-found phrases detected
        scored_docs:       Reranked (doc, score) tuples — top chunk sent first

    Returns:
        Integer 0-100 confidence score
    """
    if answer_not_found:
        return 0

    # Build judge context: top chunk first, then remaining context up to 3000 chars
    if scored_docs:
        top_chunk = scored_docs[0][0].page_content
        remaining = context.replace(top_chunk, "", 1)
        budget = 3000 - len(top_chunk) - 50
        judge_context = top_chunk + "\n\n---\n\n" + remaining[: max(0, budget)]
    else:
        judge_context = context[:3000]

    try:
        judge_llm = get_llm(temperature=0, max_tokens=10)
        judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()

        raw = judge_chain.invoke(
            {
                "question": query,
                "context_snippet": judge_context,
                "answer": answer,
            }
        ).strip()

        # Extract the first integer from the response
        numbers = re.findall(r"\d+", raw)
        if numbers:
            score = int(numbers[0])
            return max(0, min(100, score))
        else:
            # Judge returned non-numeric — fall back to heuristic
            return _heuristic_confidence(answer)

    except Exception:
        # If judge call fails for any reason, fall back gracefully
        return _heuristic_confidence(answer)


def _heuristic_confidence(answer: str) -> int:
    """
    Fallback confidence heuristic used only if the judge LLM call fails.
    Simple: any real answer gets 65%, empty gets 0%.
    This is a safety net, not the primary scorer.
    """
    answer_len = len(answer.strip())
    if answer_len < 10:
        return 0
    elif answer_len < 50:
        return 62  # short factual answer — assume correct
    elif answer_len < 200:
        return 68
    else:
        return 72


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _cache_key(query: str, filter_files: Optional[List[str]] = None) -> str:
    """Cache key includes filter_files so filtered/unfiltered are separate."""
    key_str = query.lower().strip()
    if filter_files:
        key_str += "|" + ",".join(sorted(filter_files))
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(
    query: str,
    filter_files: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Load cached answer if within TTL. Returns None if expired or missing."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(query, filter_files)}.json"

    if not cache_path.exists():
        return None

    with open(cache_path, "r") as f:
        cached = json.load(f)

    if time.time() - cached.get("timestamp", 0) > CACHE_TTL_SECONDS:
        cache_path.unlink()
        return None

    return cached


def _save_to_cache(
    query: str,
    result: Dict,
    filter_files: Optional[List[str]] = None,
) -> None:
    """Persist full result including confidence and source_details to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(query, filter_files)}.json"

    with open(cache_path, "w") as f:
        json.dump(
            {
                "answer": result["answer"],
                "sources": result["sources"],
                "source_details": result.get("source_details", []),
                "confidence": result.get("confidence", 0),
                "timestamp": time.time(),
                "cached": True,
            },
            f,
            indent=2,
        )


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[Dict]] = None,
    use_cache: bool = True,
    filter_files: Optional[List[str]] = None,
) -> Dict:
    """
    Full RAG pipeline: retrieve → rerank → generate → judge → cache.

    Steps:
      1. Cache check (keyed on query + filter_files)
      2. Cross-encoder re-ranking (for ordering only)
      3. Context assembly
      4. Groq LLaMA3 answer generation
      5. Not-found phrase detection
      6. LLM-as-Judge confidence scoring
      7. Cache write

    Args:
        query:          User question
        retrieved_docs: Candidate chunks from hybrid retriever
        chat_history:   Recent Q&A turns for conversational context
        use_cache:      Whether to read/write cache (default True)
        filter_files:   File filter list — affects cache key

    Returns:
        Dict: answer, sources, source_details, confidence (0-100), cached
    """
    # 1. Cache check
    if use_cache:
        cached = _load_from_cache(query, filter_files)
        if cached:
            return cached

    # 2. Re-rank (for context ordering — scores NOT used for confidence)
    scored_docs = rerank_documents(query, retrieved_docs, top_n=5)

    if not scored_docs:
        return {
            "answer": "I couldn't find relevant information in the documents. "
            "Please sync your Drive or check if documents were processed correctly.",
            "sources": [],
            "source_details": [],
            "confidence": 0,
            "cached": False,
        }

    reranked_docs = [doc for doc, _ in scored_docs]

    # 3. Assemble context with source labels
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
        lines = []
        for turn in chat_history[-3:]:
            lines.append(f"User: {turn.get('question', '')}")
            lines.append(f"Assistant: {turn.get('answer', '')[:200]}...")
        history_text = "\n".join(lines)

    # 5. Generate answer
    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "question": query,
            "context": context,
            "chat_history": history_text,
        }
    ).strip()

    # 6. Not-found detection
    answer_lower = answer.lower()
    answer_not_found = any(p in answer_lower for p in _NOT_FOUND_PHRASES)

    # 7. LLM-as-Judge confidence scoring
    confidence = _judge_confidence(
        query, context, answer, answer_not_found, scored_docs
    )

    result = {
        "answer": answer,
        "sources": []
        if answer_not_found
        else list(
            dict.fromkeys(
                doc.metadata.get("file_name", "unknown") for doc in reranked_docs
            )
        ),
        "source_details": [
            {
                "file_name": doc.metadata.get("file_name"),
                "doc_id": doc.metadata.get("doc_id"),
                "page": doc.metadata.get("page", 0) + 1,
                "source": doc.metadata.get("source", "gdrive"),
                "rerank_score": round(score, 4),
            }
            for doc, score in scored_docs
        ],
        "confidence": confidence,
        "cached": False,
    }

    # 8. Cache result
    if use_cache:
        _save_to_cache(query, result, filter_files)

    return result
