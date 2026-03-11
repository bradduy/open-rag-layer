"""Tests for ingestion pipeline using MockEmbedder + InMemoryIndex."""
import pytest
from rag_layer.chunking.text_chunker import TextChunker
from rag_layer.config import ChunkingConfig, RAGConfig
from rag_layer.embeddings.mock import MockEmbedder
from rag_layer.indexes.memory import InMemoryIndex
from rag_layer.ingestion.ingestor import Ingestor
from rag_layer.schema import Chunk, ChunkMetadata


def make_ingestor(chunk_size: int = 512) -> Ingestor:
    config = RAGConfig(
        embedder="mock",
        index="memory",
        chunking=ChunkingConfig(chunk_size=chunk_size, chunk_overlap=0),
    )
    embedder = MockEmbedder(dims=32)
    index = InMemoryIndex()
    return Ingestor(config, embedder, index)


@pytest.mark.asyncio
class TestIngestor:
    async def test_ingest_raw_text(self) -> None:
        ingestor = make_ingestor()
        docs = await ingestor.ingest("Hello, this is a test document.")
        assert len(docs) == 1
        assert docs[0].modality == "text"

    async def test_ingest_list_of_texts(self) -> None:
        ingestor = make_ingestor()
        docs = await ingestor.ingest(["First doc.", "Second doc."])
        assert len(docs) == 2

    async def test_ingest_with_metadata(self) -> None:
        ingestor = make_ingestor()
        docs = await ingestor.ingest("content", metadata={"author": "alice"})
        assert docs[0].metadata["author"] == "alice"

    async def test_chunks_are_embedded(self) -> None:
        ingestor = make_ingestor()
        await ingestor.ingest("Some text to embed and store.")
        all_chunks = await ingestor.index.get_all_chunks()
        assert len(all_chunks) > 0
        for chunk in all_chunks:
            assert chunk.embedding is not None

    async def test_long_text_is_chunked(self) -> None:
        ingestor = make_ingestor(chunk_size=50)
        long_text = "word " * 200  # 1000 chars
        await ingestor.ingest(long_text)
        all_chunks = await ingestor.index.get_all_chunks()
        # Should have more than 1 chunk
        assert len(all_chunks) > 1


class TestTextChunker:
    def test_short_text_unchanged(self) -> None:
        chunker = TextChunker(ChunkingConfig(chunk_size=512, chunk_overlap=0))
        chunk = Chunk(
            document_id="d1",
            content="Short text.",
            content_type="text",
            metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="x"),
        )
        result = chunker.chunk([chunk])
        assert len(result) == 1
        assert result[0].content == "Short text."

    def test_long_text_split(self) -> None:
        chunker = TextChunker(
            ChunkingConfig(chunk_size=20, chunk_overlap=0, sentence_boundary=False)
        )
        chunk = Chunk(
            document_id="d1",
            content="A" * 100,
            content_type="text",
            metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="x"),
        )
        result = chunker.chunk([chunk])
        assert len(result) > 1

    def test_non_text_chunks_passthrough(self) -> None:
        chunker = TextChunker()
        chunk = Chunk(
            document_id="d1",
            content=b"\xff\xd8\xff",
            content_type="image_bytes",
            metadata=ChunkMetadata(chunk_index=0, total_chunks=1, source_ref="img"),
        )
        result = chunker.chunk([chunk])
        assert len(result) == 1
        assert result[0].content_type == "image_bytes"
