"""
processing/chunker.py
─────────────────────
Converts RawPages into LangChain Document chunks with rich metadata.

Chunking strategy:
- RecursiveCharacterTextSplitter (same as existing codebase)
- chunk_size=800, chunk_overlap=150 (proven settings from existing RAG)
- Separators: paragraphs → sentences → words (graceful degradation)

Metadata attached to every chunk:
  doc_id      → Google Drive file ID (for deduplication + filtering)
  file_name   → human-readable filename
  source      → always "gdrive"
  page        → page/section number within source document
  chunk_index → index of this chunk within its page
  mime_type   → original file type

process:
"I use RecursiveCharacterTextSplitter at 800 chars with 150-char overlap.
Each chunk carries doc_id for Drive deduplication, page number for source
attribution, and chunk_index for ordering. The metadata enables metadata
filtering — e.g. 'only search within policy.pdf'."
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from processing.parser import RawPage


# ── Splitter config — proven settings from existing codebase ─────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ".", "!", "?", " "]


def chunk_pages(raw_pages: List[RawPage]) -> List[Document]:
    """
    Split a list of RawPages into overlapping Document chunks.

    Each RawPage → one or more Document objects.
    All original metadata is preserved and extended with chunk_index.

    Args:
        raw_pages: List of RawPage objects from parser

    Returns:
        List of Document chunks ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    all_chunks: List[Document] = []

    for raw_page in raw_pages:
        if not raw_page.text.strip():
            continue

        # Split the page text into chunks
        texts = splitter.split_text(raw_page.text)

        for chunk_idx, text in enumerate(texts):
            if len(text.strip()) < 30:  # skip trivially short chunks
                continue

            chunk = Document(
                page_content=text,
                metadata={
                    "doc_id": raw_page.doc_id,
                    "file_name": raw_page.file_name,
                    "source": raw_page.source,  # always "gdrive"
                    "page": raw_page.page_number,
                    "chunk_index": chunk_idx,
                    "mime_type": raw_page.mime_type,
                    # Composite key for deduplication
                    "chunk_key": f"{raw_page.doc_id}_p{raw_page.page_number}_c{chunk_idx}",
                },
            )
            all_chunks.append(chunk)

    return all_chunks


def chunk_drive_file(
    content: bytes,
    mime_type: str,
    doc_id: str,
    file_name: str,
) -> List[Document]:
    """
    Full pipeline: raw bytes → parsed pages → chunks.
    Convenience wrapper used by the sync pipeline.

    Args:
        content: Raw file bytes from Drive
        mime_type: Effective MIME type
        doc_id: Google Drive file ID
        file_name: Original filename

    Returns:
        List of Document chunks with full metadata
    """
    from processing.parser import parse_file

    raw_pages = parse_file(content, mime_type, doc_id, file_name)

    if not raw_pages:
        print(f"  ⚠️  No content extracted from {file_name}")
        return []

    chunks = chunk_pages(raw_pages)
    print(f"  ✅ {file_name}: {len(raw_pages)} pages → {len(chunks)} chunks")
    return chunks
