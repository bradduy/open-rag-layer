from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_layer.schema import Chunk


@runtime_checkable
class Chunker(Protocol):
    """Protocol for all chunkers. Splits a list of raw chunks into smaller pieces."""

    def chunk(self, chunks: list[Chunk]) -> list[Chunk]:
        ...
