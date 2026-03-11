from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from rag_layer.schema import Chunk


@runtime_checkable
class IEmbedder(Protocol):
    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed chunks in-place (fills chunk.embedding) and return them."""
        ...

    async def embed_query(
        self, text: str, image: bytes | None = None
    ) -> list[float]:
        """Return embedding vector for a search query."""
        ...
