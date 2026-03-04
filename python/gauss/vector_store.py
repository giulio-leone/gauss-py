"""Vector store for RAG (Retrieval-Augmented Generation)."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gauss._types import SearchResult
from gauss.base import StatefulResource

__all__ = ["Chunk", "VectorStore"]

@dataclass
class Chunk:
    """A document chunk for vector upsert."""

    id: str
    document_id: str
    content: str
    index: int
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class VectorStore(StatefulResource):
    """In-memory vector store powered by Rust.

    Example::

        store = VectorStore()
        store.upsert([Chunk(id="c1", text="hello", embedding=[0.1, 0.2])])
        results = store.search([0.1, 0.2], top_k=5)
        for r in results:
            print(f"{r.id}: {r.text} (score={r.score:.2f})")
    """

    def __init__(self) -> None:
        super().__init__()
        from gauss._native import create_vector_store

        self._handle: int = create_vector_store()

    @functools.cached_property
    def _resource_name(self) -> str:
        return "VectorStore"

    def upsert(self, chunks: list[Chunk | dict[str, Any]]) -> None:
        """Upsert document chunks into the store."""
        from gauss._native import vector_store_upsert

        self._check_alive()
        items = []
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                items.append(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "index": chunk.index,
                        "metadata": chunk.metadata or {},
                        **({"embedding": chunk.embedding} if chunk.embedding else {}),
                    }
                )
            else:
                items.append(chunk)
        vector_store_upsert(self._handle, json.dumps(items))

    def search(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks.

        Returns:
            List of SearchResult ordered by similarity score (descending).
        """
        from gauss._native import vector_store_search

        self._check_alive()
        result_json: str = vector_store_search(self._handle, json.dumps(embedding), top_k)
        data = json.loads(result_json)
        return [
            SearchResult(
                id=r["id"],
                text=r.get("content", r.get("text", "")),
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in data
        ]

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> list[SearchResult]:
        """Search by text query with auto-embedding.

        Args:
            query: Text query to search for.
            top_k: Number of results to return.
            embed_fn: Function that converts text to an embedding vector.
        """
        if embed_fn is None:
            raise ValueError("embed_fn is required for text-based search")
        embedding = embed_fn(query)
        return self.search(embedding, top_k)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        from gauss._native import cosine_similarity

        return cosine_similarity(json.dumps(a), json.dumps(b))  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_vector_store

            destroy_vector_store(self._handle)
        super().destroy()
