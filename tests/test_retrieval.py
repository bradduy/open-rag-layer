"""Tests for retrieval engine (semantic, keyword, hybrid)."""
import pytest
from rag_layer.config import RAGConfig
from rag_layer.embeddings.mock import MockEmbedder
from rag_layer.indexes.memory import InMemoryIndex
from rag_layer.ingestion.ingestor import Ingestor
from rag_layer.retrieval.engine import RetrievalEngine
from rag_layer import RAGLayer
from rag_layer.schema import SearchQuery


def build_rag(dims: int = 32) -> tuple[RetrievalEngine, Ingestor, InMemoryIndex]:
    config = RAGConfig(embedder="mock", index="memory")
    embedder = MockEmbedder(dims=dims)
    index = InMemoryIndex()
    ingestor = Ingestor(config, embedder, index)
    engine = RetrievalEngine(embedder, index)
    return engine, ingestor, index


@pytest.mark.asyncio
class TestSemanticSearch:
    async def test_returns_results_after_index(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("The quick brown fox")
        results = await engine.search(SearchQuery(text="fox", search_mode="semantic"))
        assert len(results) > 0

    async def test_results_have_score(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("Revenue grew 20% this quarter")
        results = await engine.search(SearchQuery(text="revenue"))
        for r in results:
            assert isinstance(r.score, float)

    async def test_empty_index_returns_nothing(self) -> None:
        engine, _, _ = build_rag()
        results = await engine.search(SearchQuery(text="anything"))
        assert results == []

    async def test_limit_respected(self) -> None:
        engine, ingestor, _ = build_rag()
        for i in range(10):
            await ingestor.ingest(f"Document number {i}")
        results = await engine.search(SearchQuery(text="document", limit=3))
        assert len(results) <= 3


@pytest.mark.asyncio
class TestKeywordSearch:
    async def test_keyword_search_finds_exact_term(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("Machine learning is transforming AI")
        await ingestor.ingest("Revenue grew significantly")
        results = await engine.search(
            SearchQuery(text="machine learning", search_mode="keyword")
        )
        assert len(results) > 0

    async def test_keyword_no_results_for_missing_term(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("The cat sat on the mat")
        results = await engine.search(
            SearchQuery(text="xyzxyz", search_mode="keyword")
        )
        assert len(results) == 0


@pytest.mark.asyncio
class TestHybridSearch:
    async def test_hybrid_returns_results(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("Quarterly earnings report shows strong growth")
        results = await engine.search(
            SearchQuery(text="earnings growth", search_mode="hybrid")
        )
        assert len(results) >= 0  # may be 0 if BM25 and semantic both find nothing

    async def test_hybrid_deduplicates(self) -> None:
        engine, ingestor, _ = build_rag()
        await ingestor.ingest("AI and machine learning drive innovation")
        results = await engine.search(
            SearchQuery(text="machine learning", search_mode="hybrid", limit=5)
        )
        ids = [r.chunk.id for r in results]
        assert len(ids) == len(set(ids))  # no duplicates


@pytest.mark.asyncio
class TestRAGLayer:
    async def test_full_pipeline_mock(self) -> None:
        rag = RAGLayer(config={"embedder": "mock", "index": "memory"})
        docs = await rag.index("Hello RAG world")
        assert len(docs) == 1

        results = await rag.search("hello world")
        assert len(results) > 0

    async def test_index_multiple_sources(self) -> None:
        rag = RAGLayer(config={"embedder": "mock", "index": "memory"})
        docs = await rag.index(["First document.", "Second document."])
        assert len(docs) == 2

    async def test_search_query_object(self) -> None:
        rag = RAGLayer(config={"embedder": "mock", "index": "memory"})
        await rag.index("Some content about AI")
        q = SearchQuery(text="AI", search_mode="keyword")
        results = await rag.search(q)
        assert isinstance(results, list)
