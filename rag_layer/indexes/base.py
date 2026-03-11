from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rag_layer.schema import Chunk, Document, SearchResult


@runtime_checkable
class IIndexAdapter(Protocol):
    async def upsert(self, chunks: list[Chunk], documents: list[Document]) -> None:
        ...

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        ...

    async def get_all_chunks(self) -> list[Chunk]:
        """Return all stored chunks (for BM25 keyword indexing)."""
        ...
