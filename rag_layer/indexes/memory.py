from __future__ import annotations

from typing import Any

import numpy as np

from rag_layer.schema import Chunk, Document, SearchResult


class InMemoryIndex:
    """numpy-backed cosine similarity index (dev / test / no-infra use)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._documents: dict[str, Document] = {}

    async def upsert(self, chunks: list[Chunk], documents: list[Document]) -> None:
        doc_map = {d.id: d for d in documents}
        self._documents.update(doc_map)

        existing_ids = {c.id for c in self._chunks}
        for chunk in chunks:
            if chunk.id in existing_ids:
                # Update in-place
                for i, c in enumerate(self._chunks):
                    if c.id == chunk.id:
                        self._chunks[i] = chunk
                        break
            else:
                self._chunks.append(chunk)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        candidates = [c for c in self._chunks if c.embedding is not None]

        if filters:
            candidates = self._apply_filters(candidates, filters)

        if not candidates:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        embeddings = np.array([c.embedding for c in candidates], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        scores = (embeddings @ q).tolist()

        ranked = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        # Only apply score threshold when explicitly set above 0
        if min_score > 0.0:
            ranked = [(s, c) for s, c in ranked if s >= min_score]

        results: list[SearchResult] = []
        for rank, (score, chunk) in enumerate(ranked[:limit]):
            doc = self._documents.get(chunk.document_id)
            if doc is None:
                continue
            results.append(
                SearchResult(chunk=chunk, score=score, document=doc, rank=rank)
            )
        return results

    async def get_all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def _apply_filters(
        self, chunks: list[Chunk], filters: dict[str, Any]
    ) -> list[Chunk]:
        result: list[Chunk] = []
        for chunk in chunks:
            doc = self._documents.get(chunk.document_id)
            match = True
            for key, value in filters.items():
                # Check chunk metadata.extra first, then document metadata
                meta_val = chunk.metadata.extra.get(key)
                if meta_val is None and doc:
                    meta_val = doc.metadata.get(key)
                if meta_val != value:
                    match = False
                    break
            if match:
                result.append(chunk)
        return result
