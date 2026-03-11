from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_layer.schema import Chunk, Document


@runtime_checkable
class Extractor(Protocol):
    """Protocol for all extractors. Returns a list of Chunks from a Document."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        ...
