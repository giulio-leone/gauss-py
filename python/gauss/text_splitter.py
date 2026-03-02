"""Text splitting utilities for RAG document chunking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """A chunk of text produced by the splitter."""

    content: str
    index: int
    metadata: dict[str, object] | None = None


@dataclass
class TextSplitterOptions:
    """Configuration for text splitting."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None


_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class TextSplitter:
    """Recursive character text splitter for RAG chunking.

    Example::

        splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split("Long document text...")
        for chunk in chunks:
            print(f"[{chunk.index}] {chunk.content[:50]}...")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or list(_DEFAULT_SEPARATORS)

    def split(self, text: str) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        raw = self._split_recursive(text, self.separators)
        return [TextChunk(content=c, index=i) for i, c in enumerate(raw)]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        remaining = separators[1:] if separators else []

        parts = text.split(separator) if separator else list(text)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) > self.chunk_size and current:
                result.append(current)
                overlap_start = max(0, len(current) - self.chunk_overlap)
                current = current[overlap_start:] + separator + part
                if len(current) > self.chunk_size and remaining:
                    sub = self._split_recursive(current, remaining)
                    current = sub.pop() if sub else ""
                    result.extend(sub)
            else:
                current = candidate

        if current:
            result.append(current)
        return result


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[TextChunk]:
    """Convenience function to split text into chunks."""
    return TextSplitter(chunk_size, chunk_overlap, separators).split(text)
