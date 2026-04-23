"""
embedding/embedder.py
─────────────────────
Generates embeddings using HuggingFace SentenceTransformers.

Model: sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast on CPU (~2000 chunks/min)
- No API key needed — completely free
- ~80MB, cached after first download

Features:
- Singleton pattern — model loaded once, reused
- Batch processing — embed many chunks in one forward pass
- Normalize embeddings for cosine similarity with FAISS

process:
"I use all-MiniLM-L6-v2 for embeddings — it's the industry standard
for RAG systems needing CPU-friendly, cost-free embeddings. I batch
all chunks together (rather than embedding one-by-one) which is ~10x
faster. Normalized embeddings mean FAISS inner product = cosine similarity."
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ── Model config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton — initialized once
_embeddings_model: HuggingFaceEmbeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get (or initialize) the HuggingFace embeddings model.

    Singleton pattern: model is loaded once and reused across calls.
    First call takes ~3-5s to load. Subsequent calls are instant.

    Returns:
        HuggingFaceEmbeddings instance
    """
    global _embeddings_model

    if _embeddings_model is None:
        print("  🧠 Loading embedding model (first time, ~5s)...")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,  # enables cosine similarity
                "batch_size": 64,  # batch processing
            },
        )
        print("  ✅ Embedding model loaded")

    return _embeddings_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings in batch.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (List[float] per text)
    """
    model = get_embeddings()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for retrieval.

    Uses embed_query (not embed_documents) — some models have
    different query vs passage prefixes for asymmetric retrieval.

    Args:
        query: User's question string

    Returns:
        Single embedding vector
    """
    model = get_embeddings()
    return model.embed_query(query)


def embed_chunks_batch(
    chunks: List[Document],
    batch_size: int = 64,
) -> List[Document]:
    """
    Embed all chunks in batches (for progress reporting on large sets).

    Does NOT modify chunk objects — just validates they can be embedded.
    The actual embedding storage happens in the FAISS store.

    Args:
        chunks: List of Document chunks
        batch_size: Process this many chunks per batch

    Returns:
        Same list of chunks (embedding happens at FAISS store level)
    """
    total = len(chunks)
    print(f"  📊 Preparing to embed {total} chunks in batches of {batch_size}")

    # Validate all chunks have non-empty content
    valid_chunks = [c for c in chunks if c.page_content.strip()]
    if len(valid_chunks) < len(chunks):
        print(f"  ⚠️  Filtered {len(chunks) - len(valid_chunks)} empty chunks")

    return valid_chunks
