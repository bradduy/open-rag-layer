"""Tests for embedders (using MockEmbedder — no API calls)."""
import pytest
from rag_layer.embeddings.mock import MockEmbedder
from rag_layer.schema import Chunk, ChunkMetadata


def make_text_chunk(content: str = "hello world") -> Chunk:
    return Chunk(
        document_id="doc1",
        content=content,
        content_type="text",
        metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="test"),
    )


@pytest.mark.asyncio
class TestMockEmbedder:
    async def test_embed_chunks_fills_embedding(self) -> None:
        embedder = MockEmbedder(dims=64)
        chunks = [make_text_chunk("hello"), make_text_chunk("world")]
        result = await embedder.embed_chunks(chunks)
        for chunk in result:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 64

    async def test_embed_query_returns_vector(self) -> None:
        embedder = MockEmbedder(dims=64)
        vec = await embedder.embed_query("test query")
        assert len(vec) == 64

    async def test_embedding_is_unit_vector(self) -> None:
        embedder = MockEmbedder(dims=128)
        vec = await embedder.embed_query("test")
        norm = sum(x**2 for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_embed_empty_chunks(self) -> None:
        embedder = MockEmbedder()
        result = await embedder.embed_chunks([])
        assert result == []

    async def test_embed_image_query(self) -> None:
        embedder = MockEmbedder(dims=32)
        vec = await embedder.embed_query("dog", image=b"\xff\xd8\xff")
        assert len(vec) == 32
