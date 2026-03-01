"""Vector store for RAG (Retrieval-Augmented Generation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gauss._types import SearchResult


@dataclass
class Chunk:
    """A document chunk for vector upsert."""

    id: str
    document_id: str
    content: str
    index: int
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class VectorStore:
    """In-memory vector store powered by Rust.

    Example::

        store = VectorStore()
        store.upsert([Chunk(id="c1", text="hello", embedding=[0.1, 0.2])])
        results = store.search([0.1, 0.2], top_k=5)
        for r in results:
            print(f"{r.id}: {r.text} (score={r.score:.2f})")
    """

    def __init__(self) -> None:
        from gauss._native import create_vector_store  # type: ignore[import-not-found]

        self._handle: int = create_vector_store()
        self._destroyed = False

    def upsert(self, chunks: list[Chunk | dict[str, Any]]) -> None:
        """Upsert document chunks into the store."""
        from gauss._native import vector_store_upsert  # type: ignore[import-not-found]

        self._check_alive()
        items = []
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                items.append({
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "index": chunk.index,
                    "metadata": chunk.metadata or {},
                    **({"embedding": chunk.embedding} if chunk.embedding else {}),
                })
            else:
                items.append(chunk)
        vector_store_upsert(self._handle, json.dumps(items))

    def search(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks.

        Returns:
            List of SearchResult ordered by similarity score (descending).
        """
        from gauss._native import vector_store_search  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = vector_store_search(
            self._handle, json.dumps(embedding), top_k
        )
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

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        from gauss._native import cosine_similarity  # type: ignore[import-not-found]

        return cosine_similarity(json.dumps(a), json.dumps(b))  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_vector_store  # type: ignore[import-not-found]

            destroy_vector_store(self._handle)
            self._destroyed = True

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("VectorStore has been destroyed")
