"""
api/main.py
───────────
FastAPI backend for the DocMind Google Drive RAG system.

Endpoints:
  POST /sync-drive          → Connect to Drive, fetch + process new docs
  POST /ask                 → RAG Q&A over synced Drive documents
  GET  /status              → System status, index stats
  GET  /documents           → List all synced documents
  POST /ask/filtered        → Q&A filtered to specific files
  DELETE /cache             → Clear answer cache

Architecture:
  Request → FastAPI → connectors/gdrive.py → processing/ → embedding/ → search/
                     ↑ sync-drive only
  Request → FastAPI → search/ → src/rag_chain.py → Groq LLaMA3
                     ↑ ask only

process:
"I used FastAPI for the backend with async endpoints. The /sync-drive
endpoint runs the full pipeline: Drive OAuth → file download → parsing
→ chunking → embedding → FAISS storage. The /ask endpoint retrieves
relevant chunks via hybrid search, re-ranks them with a cross-encoder,
then generates a grounded answer via Groq. I also added caching for
repeated queries and metadata filtering for targeted search."
"""

import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from search.faiss_store import (
    load_or_create_store,
    add_chunks_to_store,
    get_store_stats,
    retrieve_with_metadata_filter,
    retrieve_with_scores,
)
from connectors.gdrive import sync_drive
from processing.chunker import chunk_drive_file
from src.rag_chain import generate_answer

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Lifespan: load FAISS on startup ──────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the persisted FAISS index on server startup."""
    print("🚀 DocMind Drive RAG — Starting up...")
    load_or_create_store()
    print("✅ Server ready")
    yield
    print("👋 Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocMind — Google Drive RAG API",
    description=(
        "Production-quality RAG system over Google Drive. "
        "Hybrid FAISS+BM25 retrieval, Cross-Encoder re-ranking, "
        "Groq LLaMA3 generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory sync status tracker ────────────────────────────────────────────
_sync_status = {
    "is_running": False,
    "last_sync": None,
    "last_stats": None,
}


# ── Request/Response Models ───────────────────────────────────────────────────


class SyncDriveRequest(BaseModel):
    folder_id: Optional[str] = None
    force_full: bool = False
    max_files: int = 200


class SyncDriveResponse(BaseModel):
    status: str
    message: str
    stats: dict


class AskRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    source_details: Optional[List[dict]] = []
    cached: bool = False
    query: str


class FilteredAskRequest(BaseModel):
    query: str
    file_names: List[str]
    chat_history: Optional[List[dict]] = []


# ── Helper: full sync pipeline ────────────────────────────────────────────────


async def run_sync_pipeline(
    folder_id: Optional[str],
    force_full: bool,
    max_files: int,
) -> dict:
    """
    Run the complete Drive sync pipeline in the background.

    Steps:
    1. Fetch files from Google Drive (with incremental sync)
    2. Parse + chunk each file
    3. Add chunks to FAISS store
    4. Return stats

    This is async-friendly: heavy work runs in a thread pool
    so the event loop isn't blocked.
    """
    loop = asyncio.get_event_loop()

    # Step 1: Sync Drive (network I/O + download — run in executor)
    def _do_sync():
        return sync_drive(
            folder_id=folder_id,
            force_full=force_full,
            max_files=max_files,
        )

    sync_result = await loop.run_in_executor(None, _do_sync)

    drive_files = sync_result["files"]
    stats = sync_result["stats"]

    # Step 2+3: Process each file
    total_chunks = 0
    processed_files = []
    failed_files = []

    for drive_file in drive_files:
        try:

            def _process(df=drive_file):
                chunks = chunk_drive_file(
                    content=df.content,
                    mime_type=df.mime_type,
                    doc_id=df.file_id,
                    file_name=df.file_name,
                )
                if chunks:
                    added = add_chunks_to_store(chunks, df.file_id)
                    return added
                return 0

            added = await loop.run_in_executor(None, _process)
            total_chunks += added
            processed_files.append(drive_file.file_name)

        except Exception as e:
            failed_files.append({"file": drive_file.file_name, "error": str(e)})
            print(f"  ❌ Failed to process {drive_file.file_name}: {e}")

    stats["chunks_added"] = total_chunks
    stats["processed_files"] = processed_files
    stats["failed_files"] = failed_files

    return stats


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "DocMind Drive RAG",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/sync-drive", "/ask", "/status", "/documents"],
    }


@app.get("/status", tags=["Health"])
async def get_status():
    """
    Get current system status including FAISS index stats.
    """
    store_stats = get_store_stats()
    return {
        "status": "ready" if store_stats["index_loaded"] else "empty",
        "index": store_stats,
        "sync": {
            "is_running": _sync_status["is_running"],
            "last_sync": _sync_status["last_sync"],
            "last_stats": _sync_status["last_stats"],
        },
        "cache_dir": str(Path("cache").absolute()),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/sync-drive", response_model=SyncDriveResponse, tags=["Sync"])
async def sync_drive_endpoint(
    request: SyncDriveRequest,
    background_tasks: BackgroundTasks,
):
    """
    Connect to Google Drive, fetch documents, and update the FAISS index.

    - Supports incremental sync (only new/modified files)
    - Handles PDF, Google Docs, TXT, DOCX
    - Processes files: parse → chunk → embed → store

    Request body:
        folder_id: Optional Google Drive folder ID to limit scope
        force_full: If true, re-process ALL files (ignore incremental cache)
        max_files: Maximum number of files to fetch (default 200)
    """
    if _sync_status["is_running"]:
        raise HTTPException(
            status_code=409, detail="Sync already in progress. Please wait."
        )

    _sync_status["is_running"] = True

    try:
        print(f"🔄 Starting Drive sync (force_full={request.force_full})")
        stats = await run_sync_pipeline(
            folder_id=request.folder_id,
            force_full=request.force_full,
            max_files=request.max_files,
        )

        _sync_status["last_sync"] = datetime.now().isoformat()
        _sync_status["last_stats"] = stats

        fetched = stats.get("fetched", 0)
        skipped = stats.get("skipped_unchanged", 0)
        chunks = stats.get("chunks_added", 0)
        errors = stats.get("errors", [])

        return SyncDriveResponse(
            status="success",
            message=(
                f"Synced {fetched} new/updated file(s), "
                f"skipped {skipped} unchanged, "
                f"added {chunks} chunks to knowledge base."
                + (f" {len(errors)} error(s)." if errors else "")
            ),
            stats=stats,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e) + " — See README for credential setup instructions.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
    finally:
        _sync_status["is_running"] = False


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask(request: AskRequest):
    """
    Answer a question using the synced Google Drive knowledge base.

    Pipeline:
    1. Hybrid retrieval (FAISS MMR + BM25)
    2. Cross-encoder re-ranking
    3. Groq LLaMA3 grounded generation
    4. Source attribution

    Request body:
        query: The question to answer
        chat_history: Optional list of previous Q&A dicts for memory
        use_cache: Whether to use cached answers (default true)

    Returns:
        answer: Generated answer grounded in your documents
        sources: List of source filenames used
        source_details: Page-level detail for each source chunk
        cached: Whether this answer came from cache
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    store_stats = get_store_stats()
    if not store_stats["index_loaded"] or store_stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please run POST /sync-drive first.",
        )

    try:
        loop = asyncio.get_event_loop()

        def _retrieve_and_answer():
            docs = retrieve_with_metadata_filter(query, k=10)
            return generate_answer(
                query=query,
                retrieved_docs=docs,
                chat_history=request.chat_history or [],
                use_cache=request.use_cache,
            )

        result = await loop.run_in_executor(None, _retrieve_and_answer)

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            source_details=result.get("source_details", []),
            cached=result.get("cached", False),
            query=query,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate answer: {str(e)}"
        )


@app.post("/ask/filtered", tags=["Q&A"])
async def ask_filtered(request: FilteredAskRequest):
    """
    Answer a question restricted to specific document files.

    Useful for targeted queries: "What does policy.pdf say about X?"

    Request body:
        query: The question to answer
        file_names: List of filenames to search within
        chat_history: Optional previous Q&A turns
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.file_names:
        raise HTTPException(status_code=400, detail="Provide at least one file_name.")

    try:
        loop = asyncio.get_event_loop()

        def _filtered_retrieve_and_answer():
            docs = retrieve_with_metadata_filter(
                query,
                filter_file_names=request.file_names,
                k=10,
            )
            return generate_answer(
                query=query,
                retrieved_docs=docs,
                chat_history=request.chat_history or [],
                use_cache=False,  # don't cache filtered queries
            )

        result = await loop.run_in_executor(None, _filtered_retrieve_and_answer)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "source_details": result.get("source_details", []),
            "filtered_to": request.file_names,
            "query": query,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate answer: {str(e)}"
        )


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all documents currently indexed in the knowledge base.
    Returns document names, chunk counts, and Drive file IDs.
    """
    from search.faiss_store import load_doc_registry, _all_chunks

    registry = load_doc_registry()

    # Build per-file stats from chunk metadata
    file_info = {}
    for chunk in _all_chunks:
        fname = chunk.metadata.get("file_name", "unknown")
        doc_id = chunk.metadata.get("doc_id", "")
        mime = chunk.metadata.get("mime_type", "")
        if fname not in file_info:
            file_info[fname] = {
                "file_name": fname,
                "doc_id": doc_id,
                "mime_type": mime,
                "chunk_count": 0,
                "source": "gdrive",
            }
        file_info[fname]["chunk_count"] += 1

    return {
        "total_documents": len(file_info),
        "total_chunks": sum(v["chunk_count"] for v in file_info.values()),
        "documents": list(file_info.values()),
    }


@app.delete("/cache", tags=["Admin"])
async def clear_cache():
    """Clear all cached answers (useful after re-syncing Drive)."""
    cache_dir = Path("cache")
    cleared = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()
            cleared += 1
    return {"status": "success", "cleared_entries": cleared}


@app.get("/docs-info", tags=["Health"])
async def api_docs_info():
    """Redirect hint to Swagger UI."""
    return {"swagger_ui": "/docs", "redoc": "/redoc"}
