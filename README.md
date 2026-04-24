# DriveMind — RAG over Google Drive

> **Your personal ChatGPT over Google Drive.**
> Ask questions in plain English. Get answers grounded in your documents.

Built for the Highwatch AI Platform Engineer Trial Assignment.

---

## Archirecture

![DriveMind Architecture](DriveMind_Architecture.jpeg)


### Project Structure

```
docmind-drive-rag/
├── api/
│   ├── __init__.py
│   └── main.py              ← FastAPI app: /sync-drive, /ask, /status
├── connectors/
│   ├── __init__.py
│   └── gdrive.py            ← Google Drive OAuth2, incremental sync
├── processing/
│   ├── __init__.py
│   ├── parser.py            ← PDF/TXT/DOCX text extraction + cleaning
│   └── chunker.py           ← RecursiveCharacterTextSplitter + metadata
├── embedding/
│   ├── __init__.py
│   └── embedder.py          ← HuggingFace MiniLM-L6-v2, batch embedding
├── search/
│   ├── __init__.py
│   └── faiss_store.py       ← Persistent FAISS, hybrid retrieval, filtering
├── src/
│   ├── __init__.py
│   ├── rag_chain.py         ← Cross-Encoder rerank, Groq LLaMA3, caching
│   └── chat_history.py      ← Session persistence
├── credentials/             ← Google credentials (gitignored)
│   └── .gitkeep
├── faiss_store/             ← Persisted FAISS index (gitignored)
├── cache/                   ← Answer cache (gitignored)
├── chat_sessions/           ← Chat history (gitignored)
├── app.py                   ← Streamlit frontend
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/docmind-drive-rag
cd docmind-drive-rag

# Using uv (recommended)
uv sync

# OR using pip
pip install -e .
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com

### 3. Google Drive Credentials

**Option A: OAuth2 (Recommended for local dev)**

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or select existing)
3. Enable **Google Drive API**: APIs & Services → Library → Search "Drive API" → Enable
4. Create OAuth credentials: APIs & Services → Credentials → Create Credentials → OAuth client ID
5. Select **Desktop app**, download the JSON
6. Rename to `credentials.json`, place in `credentials/` folder
7. First sync will open a browser window for consent — authenticate once, token is cached

**Option B: Service Account (For production / server deployment)**

1. APIs & Services → Credentials → Create Credentials → Service Account
2. Download the JSON key
3. Rename to `service_account.json`, place in `credentials/` folder
4. Share your Drive folder with the service account email (found in the JSON)

### 4. Run the System

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Sync and Query

1. Open the **Drive Sync** tab
2. Click **Start Sync** (optionally provide a folder ID)
3. Wait for sync to complete (first time downloads all files)
4. Switch to **Chat** tab and ask questions!

---
# API Reference & Live Examples

---

## `POST /sync-drive`

Fetch and index documents from Google Drive using MD5 hashing for incremental updates.

**Request Body**

```json
{
  "folder_id": null,
  "force_full": false,
  "max_files": 200
}
```

**Live Response — First Sync**

Downloads all files and builds the full index from scratch.

```json
"Synced 7 new files, skipped 0 unchanged, added 378 chunks to knowledge base"
```

**Live Response — Incremental Sync** *(next day, 2 files updated on Drive)*

```json
{
  "status": "success",
  "message": "Synced 2 new/updated file(s), skipped 7 unchanged, added 550 chunks to knowledge base.",
  "stats": {
    "total_on_drive": 9,
    "fetched": 2,
    "skipped_unchanged": 7,
    "errors": [],
    "sync_time": "2026-04-25T00:41:55.327588",
    "incremental": true,
    "chunks_added": 0,
    "processed_files": [],
    "failed_files": []
  }
}
```

---

## `POST /ask`

Answer a question using the hybrid retrieval pipeline (FAISS + BM25) with Groq LLaMA3.

**Request Body**

```json
{
  "query": "what is google's policy?",
  "chat_history": [],
  "use_cache": true
}
```

**Live Response**

```json
{
  "answer": "Google's policy applies to all of the services offered by Google LLC and its affiliates, including YouTube... The policy restricts access to personal information to Google employees, contractors and agents who need to know that information in order to process it for Google...",
  "sources": ["google_privacy_policy_en.pdf"],
  "source_details": [
    {
      "file_name": "google_privacy_policy_en.pdf",
      "doc_id": "1jiOlm-Q_rGeM0Bnq05prmYvFZWJc2I3e",
      "page": 4,
      "source": "gdrive"
    },
    {
      "file_name": "google_privacy_policy_en.pdf",
      "doc_id": "1jiOlm-Q_rGeM0Bnq05prmYvFZWJc2I3e",
      "page": 3,
      "source": "gdrive"
    }
  ],
  "cached": false,
  "query": "what is google's policy?"
}
```

---

## `POST /ask/filtered`

Search restricted to specific document files to eliminate retrieval noise.

**Request Body**

```json
{
  "query": "How do I update my information?",
  "file_names": ["google_privacy_policy_en.pdf"],
  "chat_history": []
}
```

**Live Response**

```json
{
  "answer": "You can update your personal information by using the Google activity controls to decide what types of data you would like saved... You can also review and control certain types of information tied to your Google Account by using Google Dashboard.",
  "sources": ["google_privacy_policy_en.pdf"],
  "filtered_to": ["google_privacy_policy_en.pdf"],
  "query": "How do I update my information?"
}
```

---

## Sample Showcases

### Showcase 1 — Corporate Strategy & Terms (Netflix)

**Query:** `"what are the Netflix company policies?"`

```json
{
  "answer": "The Netflix company policies include disclosing user information to the Netflix family of companies for purposes such as data processing... Users agree not to archive, download, reproduce, or modify content without express written permission.",
  "sources": ["theatrical-terms-and-privacy.pdf"],
  "cached": true
}
```

---

### Showcase 2 — Technical SOP Extraction (Amazon)

**Query:** `"tell me about SOP of Amazon"`

```json
{
  "answer": "The SOP for Amazon is related to the testing and enrollment of fragile ASINs at the Amazon Packaging Lab... allowing them to test, enroll, and certify their fragile ASINs under 50 lbs. to help vendors transition to the Ships in Product Packaging (SIPP) program.",
  "sources": [
    "SOP-for-Quality-Improvement.pdf",
    "sop-for-testing-and-enrollment-of-fragile-asins-at-amazon-lab-2024.pdf"
  ]
}
```

---

### Showcase 3 — Global Policy & Human Rights (Meta)

**Query:** `"what is meta's policy?"`

```json
{
  "answer": "Meta's policy is guided by its mission to give people the power to build community... Principles include giving people a voice, serving everyone, promoting economic opportunity, and protecting privacy. Meta has Community Standards highlighting voice, authenticity, safety, privacy, and dignity.",
  "sources": [
    "Facebooks-Corporate-Human-Rights-Policy.pdf",
    "Privacy-Within-Metas-Integrity-Systems.pdf"
  ]
}
```
---

## Design Decisions

### Why Hybrid Search (FAISS MMR + BM25)?

| Problem | Solution |
|---|---|
| Synonyms: "staff" vs "employees" | FAISS vector search understands meaning |
| Exact terms: "Section 4.2.1" | BM25 keyword search catches exact matches |
| Redundant results | MMR (Maximal Marginal Relevance) picks diverse chunks |

### Why Cross-Encoder Re-ranking?

FAISS embeds query and document **separately** — loses context between them. The cross-encoder reads both **together** in one pass, dramatically improving precision. We use `ms-marco-MiniLM-L-6-v2`, trained specifically for passage re-ranking.

### Why Incremental Sync?

On each `/sync-drive` call, we compare each Drive file's `modifiedTime` against a local manifest. Only changed files are re-downloaded and re-embedded. After initial sync, subsequent syncs are typically 10-20x faster.

### Why Caching?

Repeated queries (e.g. "what is the refund policy?" asked multiple times) hit the cache instantly without re-running retrieval + LLM. Cache TTL is 1 hour. Clear with `DELETE /cache` after re-syncing.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Google Drive | google-api-python-client + OAuth2 |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (persistent) |
| Keyword Search | BM25 (rank-bm25) |
| Re-ranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Groq LLaMA3 70B (llama-3.3-70b-versatile) |
| PDF parsing | PyPDF |
| DOCX parsing | python-docx |
| Session storage | JSON files |
| Answer cache | JSON files (1hr TTL) |

---

## Evaluation Criteria Coverage

| Requirement | Implementation |
|---|---|
| ✅ Google Drive integration | OAuth2 + Service Account, `connectors/gdrive.py` |
| ✅ PDF/Docs/TXT support | `processing/parser.py` handles all 3 |
| ✅ POST /sync-drive | `api/main.py`, incremental + full sync |
| ✅ POST /ask with sources | Returns answer + sources list + page details |
| ✅ Good chunking strategy | 800 char / 150 overlap, recursive splitting |
| ✅ Relevant answers | Hybrid retrieval + cross-encoder re-ranking |
| ✅ Clean API design | FastAPI with Pydantic models, proper HTTP codes |
| ✅ Incremental sync | Manifest-based modifiedTime comparison |
| ✅ Caching | 1hr TTL JSON cache, DELETE /cache to clear |
| ✅ Metadata filtering | POST /ask/filtered by filename |
| ✅ Async pipeline | FastAPI async endpoints + executor for CPU work |
| ✅ Architecture folders | connectors/ processing/ embedding/ search/ api/ |
