"""
open-rag-layer: Provider-agnostic multimodal retrieval layer.

Quick start:
    from rag_layer import RAGLayer

    rag = RAGLayer()                   # in-memory + mock embedder (no API keys)
    await rag.index("report.pdf")
    results = await rag.search("quarterly revenue")
"""
from __future__ import annotations

from typing import Any

from rag_layer.config import RAGConfig
from rag_layer.ingestion.ingestor import Ingestor
from rag_layer.retrieval.engine import RetrievalEngine
from rag_layer.schema import (
    Chunk,
    Document,
    IndexInput,
    Modality,
    SearchQuery,
    SearchResult,
)

__all__ = [
    "RAGLayer",
    "RAGConfig",
    "SearchQuery",
    "SearchResult",
    "Document",
    "Chunk",
    "IndexInput",
    "Modality",
]


class RAGLayer:
    """
    High-level retrieval layer.

    Usage::

        rag = RAGLayer(config={"index": "qdrant", "embedder": "gemini"})
        await rag.index("report.pdf")
        results = await rag.search("quarterly trends")
    """

    def __init__(
        self,
        config: RAGConfig | dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = RAGConfig()
        elif isinstance(config, dict):
            config = RAGConfig.from_dict(config)
        self.config: RAGConfig = config

        self._embedder = self._build_embedder()
        self._index = self._build_index()
        self._ingestor = Ingestor(self.config, self._embedder, self._index)
        self._engine = RetrievalEngine(self._embedder, self._index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index(
        self,
        source: str | list[str],
        modality: Modality | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Ingest one or more sources into the retrieval layer.

        Args:
            source: File path, URL, or raw text string (or a list thereof).
            modality: Override auto-detection.
            metadata: Arbitrary key-value metadata attached to the Document.

        Returns:
            List of indexed Document objects.
        """
        return await self._ingestor.ingest(source, modality=modality, metadata=metadata)

    async def search(
        self,
        query: str | SearchQuery,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        search_mode: str = "semantic",
        filters: dict[str, Any] | None = None,
        use_reranking: bool = False,
    ) -> list[SearchResult]:
        """Search the retrieval layer.

        Args:
            query: Plain text query string, or a full SearchQuery object.
            limit: Maximum number of results to return.
            min_score: Minimum cosine similarity threshold.
            search_mode: 'semantic' | 'keyword' | 'hybrid'.
            filters: Metadata filters to apply.
            use_reranking: Apply cross-encoder reranking (requires sentence-transformers).

        Returns:
            Ranked list of SearchResult objects.
        """
        if isinstance(query, str):
            query = SearchQuery(
                text=query,
                limit=limit,
                min_score=min_score,
                search_mode=search_mode,  # type: ignore[arg-type]
                filters=filters,
                use_reranking=use_reranking,
            )
        return await self._engine.search(query)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_embedder(self) -> Any:
        if self.config.embedder == "gemini":
            from rag_layer.embeddings.gemini import GeminiEmbedder
            return GeminiEmbedder(self.config.gemini)
        elif self.config.embedder == "mock":
            from rag_layer.embeddings.mock import MockEmbedder
            return MockEmbedder()
        else:
            raise ValueError(f"Unknown embedder: {self.config.embedder!r}")

    def _build_index(self) -> Any:
        if self.config.index == "memory":
            from rag_layer.indexes.memory import InMemoryIndex
            return InMemoryIndex()
        elif self.config.index == "qdrant":
            from rag_layer.indexes.qdrant import QdrantIndex
            return QdrantIndex(self.config.qdrant)
        else:
            raise ValueError(f"Unknown index: {self.config.index!r}")
