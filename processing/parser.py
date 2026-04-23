"""
processing/parser.py
────────────────────
Extracts clean text from PDF, TXT (exported Google Docs), and DOCX files.

Each file type needs a different extraction strategy:
- PDF      → PyPDF page-by-page with page number tracking
- TXT      → direct decode (used for exported Google Docs)
- DOCX     → python-docx paragraph extraction

Every chunk gets rich metadata:
  doc_id, file_name, source="gdrive", page, chunk_index, mime_type

process:
"I built a unified parser that dispatches to the right extractor
based on MIME type. PDFs use PyPDF for page-level extraction,
Google Docs are exported as TXT by the connector so no special
handling is needed, and DOCX files use python-docx. Each extracted
page/section becomes a RawPage object that feeds into the chunker."
"""

import os
import io
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import pypdf
from langchain_core.documents import Document


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class RawPage:
    """A single page/section of extracted text before chunking."""

    text: str
    doc_id: str
    file_name: str
    page_number: int  # 0-indexed
    mime_type: str
    source: str = "gdrive"


# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Normalize extracted text:
    - Collapse excessive whitespace/newlines
    - Remove null bytes and control characters
    - Strip leading/trailing whitespace per line
    - Keep meaningful paragraph breaks
    """
    if not text:
        return ""

    # Remove null bytes and non-printable control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize unicode dashes, quotes
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse 3+ consecutive newlines → 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ── PDF Extractor ─────────────────────────────────────────────────────────────


def extract_pdf(content: bytes, doc_id: str, file_name: str) -> List[RawPage]:
    """
    Extract text from PDF bytes, one RawPage per page.

    Args:
        content: Raw PDF bytes
        doc_id: Google Drive file ID
        file_name: Original filename

    Returns:
        List of RawPage objects
    """
    pages = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            if len(text.strip()) < 20:  # skip nearly empty pages
                continue
            pages.append(
                RawPage(
                    text=text,
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=page_num,
                    mime_type="application/pdf",
                )
            )
    except Exception as e:
        print(f"  ⚠️  PDF extraction error for {file_name}: {e}")

    return pages


# ── TXT Extractor (Google Docs exported as plain text) ────────────────────────


def extract_txt(content: bytes, doc_id: str, file_name: str) -> List[RawPage]:
    """
    Extract text from plain text content (incl. exported Google Docs).

    Splits into logical "pages" by double-newline paragraphs grouped
    into ~2000 char sections to keep page granularity reasonable.

    Args:
        content: Raw text bytes
        doc_id: Google Drive file ID
        file_name: Original filename

    Returns:
        List of RawPage objects (one per ~2000 char section)
    """
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    text = clean_text(text)

    if not text.strip():
        return []

    # Split into paragraphs, then group into ~2000 char sections
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pages = []
    current_section = []
    current_len = 0
    section_idx = 0
    section_limit = 2000

    for para in paragraphs:
        current_section.append(para)
        current_len += len(para)
        if current_len >= section_limit:
            pages.append(
                RawPage(
                    text="\n\n".join(current_section),
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=section_idx,
                    mime_type="text/plain",
                )
            )
            section_idx += 1
            current_section = []
            current_len = 0

    # Remainder
    if current_section:
        pages.append(
            RawPage(
                text="\n\n".join(current_section),
                doc_id=doc_id,
                file_name=file_name,
                page_number=section_idx,
                mime_type="text/plain",
            )
        )

    return pages


# ── DOCX Extractor ────────────────────────────────────────────────────────────


def extract_docx(content: bytes, doc_id: str, file_name: str) -> List[RawPage]:
    """
    Extract text from DOCX bytes using python-docx.

    Groups paragraphs into ~2000 char sections, same strategy as TXT.

    Args:
        content: Raw DOCX bytes
        doc_id: Google Drive file ID
        file_name: Original filename

    Returns:
        List of RawPage objects
    """
    try:
        import docx  # python-docx
    except ImportError:
        print("  ⚠️  python-docx not installed. Skipping DOCX extraction.")
        return []

    try:
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print(f"  ⚠️  DOCX extraction error for {file_name}: {e}")
        return []

    pages = []
    current_section: List[str] = []
    current_len = 0
    section_idx = 0
    section_limit = 2000

    for para in paragraphs:
        current_section.append(para)
        current_len += len(para)
        if current_len >= section_limit:
            text = clean_text("\n\n".join(current_section))
            pages.append(
                RawPage(
                    text=text,
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=section_idx,
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            )
            section_idx += 1
            current_section = []
            current_len = 0

    if current_section:
        text = clean_text("\n\n".join(current_section))
        pages.append(
            RawPage(
                text=text,
                doc_id=doc_id,
                file_name=file_name,
                page_number=section_idx,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )

    return pages


# ── Dispatcher ────────────────────────────────────────────────────────────────


def parse_file(
    content: bytes,
    mime_type: str,
    doc_id: str,
    file_name: str,
) -> List[RawPage]:
    """
    Dispatch to the correct extractor based on MIME type.

    Args:
        content: Raw file bytes
        mime_type: MIME type string
        doc_id: Google Drive file ID
        file_name: Original filename

    Returns:
        List of RawPage objects ready for chunking
    """
    if mime_type == "application/pdf":
        return extract_pdf(content, doc_id, file_name)

    elif mime_type == "text/plain":
        return extract_txt(content, doc_id, file_name)

    elif "wordprocessingml" in mime_type:
        return extract_docx(content, doc_id, file_name)

    else:
        print(f"  ⚠️  Unsupported MIME type: {mime_type} for {file_name}")
        return []
