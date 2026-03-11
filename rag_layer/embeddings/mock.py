from __future__ import annotations

import random

from rag_layer.schema import Chunk

DIMS = 128  # small for tests


class MockEmbedder:
    """Random-vector embedder for testing (no API calls)."""

    def __init__(self, dims: int = DIMS) -> None:
        self.dims = dims

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            chunk.embedding = self._random_unit()
        return chunks

    async def embed_query(
        self, text: str, image: bytes | None = None
    ) -> list[float]:
        return self._random_unit()

    def _random_unit(self) -> list[float]:
        vec = [random.gauss(0, 1) for _ in range(self.dims)]
        norm = sum(x**2 for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]
