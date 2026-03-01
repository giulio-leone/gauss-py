"""
05 — Memory + VectorStore.

Demonstrates:
  • Storing and recalling conversation memory
  • Session-based memory isolation
  • VectorStore for RAG (upsert, search, cosine similarity)
"""

from gauss import Memory, VectorStore
from gauss.vector_store import Chunk


def main() -> None:
    # ── 1. Conversation Memory ───────────────────────────────────────
    print("=== Conversation Memory ===\n")

    with Memory() as mem:
        # Store entries in a session
        mem.store("conversation", "Hello! I'm looking for a restaurant.", session_id="session-1")
        mem.store("conversation", "I recommend the Italian place on Main St.", session_id="session-1")
        mem.store("fact", "User prefers Italian food.", session_id="session-1")
        mem.store("preference", "Vegetarian options preferred.", session_id="session-1")

        # Store in a different session
        mem.store("conversation", "What's the weather?", session_id="session-2")

        # Recall by session
        entries = mem.recall(session_id="session-1")
        print(f"Session 1 ({len(entries)} entries):")
        for entry in entries:
            print(f"  [{entry.get('entry_type')}] {entry.get('content')}")

        entries = mem.recall(session_id="session-2")
        print(f"\nSession 2 ({len(entries)} entries):")
        for entry in entries:
            print(f"  [{entry.get('entry_type')}] {entry.get('content')}")

        # Recall with limit
        recent = mem.recall(session_id="session-1", limit=2)
        print(f"\nLast 2 from session 1: {len(recent)} entries")

        # Clear a session
        mem.clear(session_id="session-2")
        entries = mem.recall(session_id="session-2")
        print(f"Session 2 after clear: {len(entries)} entries")

    # ── 2. Vector Store (RAG) ────────────────────────────────────────
    print("\n=== Vector Store ===\n")

    with VectorStore() as store:
        # Upsert document chunks with embeddings
        store.upsert([
            Chunk(
                id="c1",
                document_id="doc-1",
                content="Python is a versatile programming language.",
                index=0,
                embedding=[0.1, 0.3, 0.5, 0.7],
            ),
            Chunk(
                id="c2",
                document_id="doc-1",
                content="Rust provides memory safety without garbage collection.",
                index=1,
                embedding=[0.2, 0.4, 0.6, 0.8],
            ),
            Chunk(
                id="c3",
                document_id="doc-2",
                content="Machine learning uses statistical methods.",
                index=0,
                embedding=[0.9, 0.1, 0.3, 0.5],
            ),
        ])

        # Search by embedding similarity
        results = store.search(embedding=[0.15, 0.35, 0.55, 0.75], top_k=2)
        print("Top 2 similar chunks:")
        for r in results:
            print(f"  [{r.id}] score={r.score:.4f}: {r.text}")

        # Cosine similarity between two vectors
        sim = VectorStore.cosine_similarity(
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
        )
        print(f"\nCosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
