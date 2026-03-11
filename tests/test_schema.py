"""Tests for schema models."""
import pytest
from rag_layer.schema import (
    Chunk,
    ChunkMetadata,
    Document,
    IndexInput,
    SearchQuery,
    SearchResult,
)


def make_chunk(doc_id: str = "doc1", content: str = "hello") -> Chunk:
    return Chunk(
        document_id=doc_id,
        content=content,
        content_type="text",
        metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="test"),
    )


def make_doc() -> Document:
    return Document(source="test.txt", modality="text")


class TestDocument:
    def test_default_id(self) -> None:
        doc = make_doc()
        assert len(doc.id) == 36  # UUID format

    def test_created_at_set(self) -> None:
        doc = make_doc()
        assert doc.created_at is not None

    def test_metadata_default_empty(self) -> None:
        doc = make_doc()
        assert doc.metadata == {}


class TestChunk:
    def test_default_id(self) -> None:
        chunk = make_chunk()
        assert len(chunk.id) == 36

    def test_embedding_none_by_default(self) -> None:
        chunk = make_chunk()
        assert chunk.embedding is None

    def test_bytes_content(self) -> None:
        chunk = Chunk(
            document_id="doc1",
            content=b"\xff\xd8\xff",
            content_type="image_bytes",
            metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="img.jpg"),
        )
        assert isinstance(chunk.content, bytes)


class TestSearchQuery:
    def test_defaults(self) -> None:
        q = SearchQuery(text="hello")
        assert q.limit == 10
        assert q.min_score == 0.0
        assert q.search_mode == "semantic"
        assert q.use_reranking is False

    def test_full_query(self) -> None:
        q = SearchQuery(
            text="test",
            limit=5,
            search_mode="hybrid",
            filters={"source": "pdf"},
        )
        assert q.filters == {"source": "pdf"}


class TestSearchResult:
    def test_creation(self) -> None:
        chunk = make_chunk()
        doc = make_doc()
        result = SearchResult(chunk=chunk, score=0.95, document=doc, rank=0)
        assert result.score == 0.95
        assert result.rank == 0


class TestIndexInput:
    def test_source_required(self) -> None:
        inp = IndexInput(source="report.pdf")
        assert inp.source == "report.pdf"
        assert inp.modality is None
