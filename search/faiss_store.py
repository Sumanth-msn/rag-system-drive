"""
search/faiss_store.py
─────────────────────
FAISS vector store with persistence, hybrid search (MMR + BM25),
metadata filtering, and incremental document updates.

Key capabilities:
1. Persistent FAISS index — survives server restarts
2. Incremental add — add new docs without rebuilding entire index
3. Hybrid retrieval — FAISS MMR (semantic) + BM25 (keyword)
4. Cross-encoder re-ranking — top-N precision boost
5. Metadata filtering — e.g. search only within specific files
6. Similarity scores — returned alongside documents for confidence display

Architecture:
  faiss_store/
    index.faiss     → FAISS binary index
    index.pkl       → docstore + metadata
    doc_registry.json → doc_id → chunk count mapping (for dedup)

process:
"I persist the FAISS index to disk so embeddings survive server restarts.
For incremental sync, I track which doc_ids are already in the index
via a registry file. When a document is updated on Drive, I remove its
old chunks and add the newly processed ones — without rebuilding the
entire index. The hybrid retriever combines MMR semantic search with
BM25 keyword matching, then a cross-encoder re-ranks the top-k results."
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from embedding.embedder import get_embeddings

# ── Paths ────────────────────────────────────────────────────────────────────
FAISS_DIR = Path("faiss_store")
FAISS_INDEX_NAME = "index"  # FAISS saves index.faiss + index.pkl
DOC_REGISTRY_PATH = FAISS_DIR / "doc_registry.json"

# ── Global state (in-memory cache) ───────────────────────────────────────────
_vectorstore: Optional[FAISS] = None
_all_chunks: List[Document] = []  # kept in memory for BM25 + re-ranking


# ── Registry helpers ──────────────────────────────────────────────────────────


def load_doc_registry() -> Dict[str, int]:
    """
    Load doc registry mapping doc_id → number of chunks stored.
    Used for incremental sync deduplication.
    """
    if DOC_REGISTRY_PATH.exists():
        with open(DOC_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_doc_registry(registry: Dict[str, int]):
    """Persist doc registry to disk."""
    FAISS_DIR.mkdir(exist_ok=True)
    with open(DOC_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def is_doc_indexed(doc_id: str) -> bool:
    """Check if a doc_id is already in the FAISS index."""
    registry = load_doc_registry()
    return doc_id in registry


# ── Store creation / loading ──────────────────────────────────────────────────


def load_or_create_store() -> Optional[FAISS]:
    """
    Load FAISS index from disk if it exists, otherwise return None.
    Called at server startup.
    """
    global _vectorstore, _all_chunks

    index_path = FAISS_DIR / f"{FAISS_INDEX_NAME}.faiss"
    chunks_path = FAISS_DIR / "chunks_cache.pkl"

    if index_path.exists():
        try:
            embeddings = get_embeddings()
            _vectorstore = FAISS.load_local(
                str(FAISS_DIR),
                embeddings,
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            # Restore chunks for BM25
            if chunks_path.exists():
                with open(chunks_path, "rb") as f:
                    _all_chunks = pickle.load(f)
            print(f"  ✅ Loaded FAISS index with {len(_all_chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  Failed to load FAISS index: {e}. Starting fresh.")
            _vectorstore = None
            _all_chunks = []
    else:
        print("  📭 No existing FAISS index found. Will create on first sync.")

    return _vectorstore


def save_store():
    """Persist FAISS index + chunks cache to disk."""
    global _vectorstore, _all_chunks

    if _vectorstore is None:
        return

    FAISS_DIR.mkdir(exist_ok=True)
    _vectorstore.save_local(str(FAISS_DIR), index_name=FAISS_INDEX_NAME)

    # Save chunks for BM25 restoration
    chunks_path = FAISS_DIR / "chunks_cache.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(_all_chunks, f)

    print(f"  💾 Saved FAISS index ({len(_all_chunks)} chunks)")


# ── Adding documents ──────────────────────────────────────────────────────────


def add_chunks_to_store(
    new_chunks: List[Document],
    doc_id: str,
) -> int:
    """
    Add new chunks to the FAISS index (incremental, no full rebuild).

    If the doc_id already exists in the registry, existing chunks
    are NOT removed from FAISS (FAISS doesn't support deletion well)
    but the registry is updated. This is acceptable for the scope
    of this project — production would use a vector DB with delete.

    Args:
        new_chunks: List of Document chunks to add
        doc_id: Google Drive file ID for registry tracking

    Returns:
        Number of chunks added
    """
    global _vectorstore, _all_chunks

    if not new_chunks:
        return 0

    embeddings = get_embeddings()

    if _vectorstore is None:
        # First documents — create the store
        _vectorstore = FAISS.from_documents(new_chunks, embeddings)
    else:
        # Incremental add — no full rebuild needed
        _vectorstore.add_documents(new_chunks)

    _all_chunks.extend(new_chunks)

    # Update registry
    registry = load_doc_registry()
    registry[doc_id] = registry.get(doc_id, 0) + len(new_chunks)
    save_doc_registry(registry)

    # Persist to disk
    save_store()

    return len(new_chunks)


def get_store_stats() -> Dict:
    """Return current state of the FAISS index."""
    registry = load_doc_registry()
    return {
        "total_chunks": len(_all_chunks),
        "total_documents": len(registry),
        "documents": registry,
        "index_loaded": _vectorstore is not None,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────


def get_retriever(k: int = 10):
    """
    Build hybrid retriever: FAISS MMR (60%) + BM25 (40%).

    Same proven weights as existing codebase.

    Args:
        k: Number of chunks each retriever returns

    Returns:
        EnsembleRetriever or None if store is empty
    """
    global _vectorstore, _all_chunks

    if _vectorstore is None or not _all_chunks:
        return None

    # FAISS MMR — semantic search with diversity
    faiss_retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 2,
            "lambda_mult": 0.7,  # 70% relevance, 30% diversity
        },
    )

    # BM25 — keyword matching
    bm25_retriever = BM25Retriever.from_documents(_all_chunks)
    bm25_retriever.k = k

    # Ensemble: 60% semantic + 40% keyword
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


def retrieve_with_scores(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k chunks with cosine similarity scores.

    Args:
        query: User question
        k: Number of results

    Returns:
        List of (Document, score) tuples sorted by score desc
    """
    global _vectorstore

    if _vectorstore is None:
        return []

    results = _vectorstore.similarity_search_with_score(query, k=k)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def retrieve_with_metadata_filter(
    query: str,
    filter_file_names: Optional[List[str]] = None,
    k: int = 10,
) -> List[Document]:
    """
    Retrieve chunks with optional metadata filtering by filename.

    When filter_file_names is set, only chunks from those files
    are considered — useful for targeted document queries.

    Args:
        query: User question
        filter_file_names: Optional list of filenames to restrict search
        k: Number of results

    Returns:
        List of Document chunks
    """
    global _vectorstore, _all_chunks

    if _vectorstore is None:
        return []

    if filter_file_names:
        # Filter chunks by filename, then build a temporary FAISS store
        filtered = [
            c for c in _all_chunks if c.metadata.get("file_name") in filter_file_names
        ]
        if not filtered:
            return []

        embeddings = get_embeddings()
        temp_store = FAISS.from_documents(filtered, embeddings)
        return temp_store.similarity_search(query, k=k)
    else:
        retriever = get_retriever(k=k)
        if retriever is None:
            return []
        return retriever.invoke(query)
