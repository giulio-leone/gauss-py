"""Unit tests for RAG features: TextSplitter, document loaders, VectorStore.search_by_text."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# ─── Mock native module before any gauss imports ────────────────────

_mock_native = MagicMock()
_mock_native.create_vector_store.return_value = 1
_mock_native.destroy_vector_store.return_value = None
_mock_native.vector_store_upsert.return_value = None
_mock_native.vector_store_search.return_value = json.dumps(
    [{"id": "c1", "text": "hello", "score": 0.95}]
)
_mock_native.cosine_similarity.return_value = 0.95


@pytest.fixture(autouse=True)
def _patch_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    _mock_native.reset_mock()
    _mock_native.create_vector_store.return_value = 1
    _mock_native.vector_store_search.return_value = json.dumps(
        [{"id": "c1", "text": "hello", "score": 0.95}]
    )


# ─── Imports (after mock setup) ────────────────────────────────────

from gauss.document_loader import load_json, load_markdown, load_text
from gauss.text_splitter import TextSplitter, split_text
from gauss.vector_store import VectorStore

# ─── TextSplitter ──────────────────────────────────────────────────


class TestTextSplitter:
    def test_splits_long_text(self) -> None:
        para = "Hello world. " * 20
        text = "\n\n".join([para, para, para, para])
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.index == i
            assert len(c.content) > 0

    def test_single_chunk_for_small_text(self) -> None:
        chunks = TextSplitter(chunk_size=1000).split("short text")
        assert len(chunks) == 1
        assert chunks[0].content == "short text"
        assert chunks[0].index == 0

    def test_overlap_between_chunks(self) -> None:
        para = "Hello world. " * 20
        text = "\n\n".join([para, para, para, para])
        splitter = TextSplitter(chunk_size=300, chunk_overlap=100)
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        tail = chunks[0].content[-50:]
        assert tail in chunks[1].content

    def test_custom_separators(self) -> None:
        text = "|".join("abcdefghijklmnopqrst")
        splitter = TextSplitter(chunk_size=10, chunk_overlap=0, separators=["|"])
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_markdown_separators(self) -> None:
        md = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
        splitter = TextSplitter(
            chunk_size=30, chunk_overlap=0, separators=["\n\n", "\n", " "]
        )
        chunks = splitter.split(md)
        assert len(chunks) > 1


# ─── split_text convenience ────────────────────────────────────────


class TestSplitText:
    def test_same_as_class(self) -> None:
        text = "hello world this is a test of the splitter function"
        from_class = TextSplitter(chunk_size=20, chunk_overlap=5).split(text)
        from_fn = split_text(text, chunk_size=20, chunk_overlap=5)
        assert len(from_fn) == len(from_class)
        for a, b in zip(from_fn, from_class):
            assert a.content == b.content
            assert a.index == b.index


# ─── load_text ─────────────────────────────────────────────────────


class TestLoadText:
    def test_loads_string_content(self) -> None:
        content = "This is a test document.\nIt has multiple lines."
        doc = load_text(content, document_id="doc1")
        assert doc.document_id == "doc1"
        assert doc.content == content
        assert len(doc.chunks) > 0
        assert doc.chunks[0].id == "doc1-0"
        assert doc.chunks[0].document_id == "doc1"
        assert doc.chunks[0].content == content

    def test_sequential_chunk_ids(self) -> None:
        long = "paragraph one\n\n" + " ".join(["word"] * 300)
        doc = load_text(long, document_id="d2", chunk_size=200, chunk_overlap=20)
        for i, c in enumerate(doc.chunks):
            assert c.id == f"d2-{i}"
            assert c.index == i


# ─── load_markdown ─────────────────────────────────────────────────


class TestLoadMarkdown:
    def test_strips_frontmatter(self) -> None:
        md = "---\ntitle: Test\n---\n# Heading\n\nBody text here."
        doc = load_markdown(md, document_id="md1")
        assert "---" not in doc.content
        assert "Heading" in doc.content

    def test_splits_on_headings(self) -> None:
        md = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        doc = load_markdown(md, document_id="md2", chunk_size=30, chunk_overlap=0)
        assert len(doc.chunks) > 1


# ─── load_json ─────────────────────────────────────────────────────


class TestLoadJson:
    def test_array_input(self) -> None:
        raw = json.dumps([{"name": "a"}, {"name": "b"}])
        doc = load_json(raw, document_id="j1")
        assert len(doc.chunks) == 2
        assert doc.chunks[0].id == "j1-0"
        assert doc.chunks[1].id == "j1-1"

    def test_object_input(self) -> None:
        raw = json.dumps({"intro": "Hello", "body": "World"})
        doc = load_json(raw, document_id="j2")
        assert len(doc.chunks) == 2
        assert doc.chunks[0].id == "j2-intro"
        assert doc.chunks[0].content == "Hello"
        assert doc.chunks[1].id == "j2-body"
        assert doc.chunks[1].content == "World"

    def test_text_field_extraction(self) -> None:
        raw = json.dumps([
            {"title": "One", "body": "First body"},
            {"title": "Two", "body": "Second body"},
        ])
        doc = load_json(raw, document_id="j3", text_field="body")
        assert doc.chunks[0].content == "First body"
        assert doc.chunks[1].content == "Second body"


# ─── VectorStore.search_by_text ────────────────────────────────────


class TestSearchByText:
    def test_calls_embed_fn_then_search(self) -> None:
        embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])
        store = VectorStore()
        results = store.search_by_text("hello world", top_k=5, embed_fn=embed_fn)

        embed_fn.assert_called_once_with("hello world")
        _mock_native.vector_store_search.assert_called_once_with(
            1, json.dumps([0.1, 0.2, 0.3]), 5
        )
        assert len(results) == 1
        assert results[0].score == 0.95
        store.destroy()

    def test_raises_without_embed_fn(self) -> None:
        store = VectorStore()
        with pytest.raises(ValueError, match="embed_fn is required"):
            store.search_by_text("query")
        store.destroy()
