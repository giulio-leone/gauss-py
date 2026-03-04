"""Document loaders for RAG pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gauss.text_splitter import TextSplitter
from gauss.vector_store import Chunk

_FRONTMATTER_RE = re.compile(r"^---[\s\S]*?---\n?")


@dataclass
class LoadedDocument:
    """A loaded and chunked document."""

    document_id: str
    content: str
    chunks: list[Chunk]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_text(
    path_or_content: str,
    *,
    document_id: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict[str, Any] | None = None,
) -> LoadedDocument:
    """Load a plain text file or string and split into chunks.

    Example::

        doc = load_text("path/to/file.txt")
        store.upsert(doc.chunks)
    """
    content, doc_id = _resolve_source(path_or_content, document_id, "text-document")
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split(content)

    chunks = [
        Chunk(
            id=f"{doc_id}-{tc.index}",
            document_id=doc_id,
            content=tc.content,
            index=tc.index,
            metadata=metadata or {},
        )
        for tc in text_chunks
    ]
    return LoadedDocument(document_id=doc_id, content=content, chunks=chunks, metadata=metadata or {})


def load_markdown(
    path_or_content: str,
    *,
    document_id: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict[str, Any] | None = None,
) -> LoadedDocument:
    """Load a Markdown file — strips frontmatter, splits on headings."""
    content, doc_id = _resolve_source(path_or_content, document_id, "markdown-document")
    content = _FRONTMATTER_RE.sub("", content)

    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    text_chunks = splitter.split(content)

    chunks = [
        Chunk(
            id=f"{doc_id}-{tc.index}",
            document_id=doc_id,
            content=tc.content,
            index=tc.index,
            metadata=metadata or {},
        )
        for tc in text_chunks
    ]
    return LoadedDocument(document_id=doc_id, content=content, chunks=chunks, metadata=metadata or {})


def load_json(
    path_or_content: str,
    *,
    document_id: str | None = None,
    text_field: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LoadedDocument:
    """Load a JSON file — each key/item becomes a chunk."""
    raw, doc_id = _resolve_source(path_or_content, document_id, "json-document")
    parsed = json.loads(raw)
    items: list[tuple[str, str]] = []

    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            text = str(item.get(text_field, "")) if text_field and isinstance(item, dict) else json.dumps(item)
            items.append((str(i), text))
    elif isinstance(parsed, dict):
        for key, value in parsed.items():
            text = value if isinstance(value, str) else json.dumps(value)
            items.append((key, text))

    chunks = [
        Chunk(
            id=f"{doc_id}-{key}",
            document_id=doc_id,
            content=text,
            index=i,
            metadata={**(metadata or {}), "key": key},
        )
        for i, (key, text) in enumerate(items)
    ]
    return LoadedDocument(document_id=doc_id, content=raw, chunks=chunks, metadata=metadata or {})


def _resolve_source(path_or_content: str, document_id: str | None, default_id: str) -> tuple[str, str]:
    """Resolve content from file path or direct string."""
    if "\n" not in path_or_content and len(path_or_content) < 500:
        p = Path(path_or_content)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8"), document_id or p.name
    return path_or_content, document_id or default_id
