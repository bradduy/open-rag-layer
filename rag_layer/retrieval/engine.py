from __future__ import annotations

from typing import Any

from rag_layer.observability.logger import get_logger, timed
from rag_layer.schema import Chunk, SearchQuery, SearchResult

logger = get_logger(__name__)


class RetrievalEngine:
    """Semantic, keyword (BM25), and hybrid (RRF) retrieval."""

    def __init__(self, embedder: Any, index: Any) -> None:
        self.embedder = embedder
        self.index = index
        self._bm25: Any = None
        self._bm25_chunks: list[Chunk] = []

    @timed
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        mode = query.search_mode

        if mode == "semantic":
            return await self._semantic_search(query)
        elif mode == "keyword":
            return await self._keyword_search(query)
        elif mode == "hybrid":
            return await self._hybrid_search(query)
        else:
            raise ValueError(f"Unknown search_mode: {mode}")

    async def _semantic_search(self, query: SearchQuery) -> list[SearchResult]:
        query_vec = await self.embedder.embed_query(
            text=query.text, image=query.image
        )
        results = await self.index.search(
            query_vector=query_vec,
            limit=query.limit,
            min_score=query.min_score,
            filters=query.filters,
        )
        if query.use_reranking:
            results = self._rerank(results, query)
        return results

    async def _keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        bm25, indexed_chunks = await self._get_bm25()
        if bm25 is None or not indexed_chunks:
            logger.warning("BM25 index is empty; returning no results")
            return []

        tokenized_query = query.text.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        # Keep positively-scored results
        positive = [(i, s) for i, s in ranked if s > 0]
        if positive:
            ranked = positive
        else:
            # BM25 IDF can be 0 with small corpora even when terms match.
            # Fall back to a simple term-presence check to distinguish
            # "term found" from "term genuinely absent".
            terms = tokenized_query
            ranked = [
                (i, s) for i, s in ranked
                if any(t in indexed_chunks[i].content.lower().split()  # type: ignore[union-attr]
                       for t in terms)
            ]

        results: list[SearchResult] = []
        for rank, (idx, score) in enumerate(ranked[: query.limit]):
            chunk = indexed_chunks[idx]
            doc = await self._get_document(chunk.document_id)
            if doc is None:
                continue
            results.append(
                SearchResult(chunk=chunk, score=float(score), document=doc, rank=rank)
            )
        return results

    async def _hybrid_search(self, query: SearchQuery) -> list[SearchResult]:
        semantic = await self._semantic_search(
            SearchQuery(**{**query.model_dump(), "limit": query.limit * 2, "search_mode": "semantic"})
        )
        keyword = await self._keyword_search(
            SearchQuery(**{**query.model_dump(), "limit": query.limit * 2, "search_mode": "keyword"})
        )
        fused = self._reciprocal_rank_fusion([semantic, keyword], k=60)
        return fused[: query.limit]

    def _reciprocal_rank_fusion(
        self, result_lists: list[list[SearchResult]], k: int = 60
    ) -> list[SearchResult]:
        """Fuse ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                cid = result.chunk.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                chunk_map[cid] = result

        fused = sorted(chunk_map.values(), key=lambda r: scores[r.chunk.id], reverse=True)
        for rank, result in enumerate(fused):
            result.rank = rank
            result.score = scores[result.chunk.id]
        return fused

    def _rerank(
        self, results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Post-process with cross-encoder reranking if available."""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [
                (query.text, r.chunk.content if isinstance(r.chunk.content, str) else "")
                for r in results
            ]
            rerank_scores = model.predict(pairs)
            for result, score in zip(results, rerank_scores):
                result.score = float(score)
            results.sort(key=lambda r: r.score, reverse=True)
            for rank, result in enumerate(results):
                result.rank = rank
        except ImportError:
            logger.debug("sentence-transformers not installed; skipping reranking")
        return results

    async def _get_bm25(self) -> tuple[Any, list[Chunk]]:
        chunks = await self.index.get_all_chunks()
        text_chunks = [
            c for c in chunks
            if c.content_type == "text" and isinstance(c.content, str)
        ]

        if not text_chunks:
            return None, []

        # Rebuild BM25 if corpus changed
        if len(text_chunks) != len(self._bm25_chunks):
            from rank_bm25 import BM25Okapi
            corpus = [c.content.lower().split() for c in text_chunks]  # type: ignore[union-attr]
            self._bm25 = BM25Okapi(corpus)
            self._bm25_chunks = text_chunks

        return self._bm25, self._bm25_chunks

    async def _get_document(self, document_id: str) -> Any:
        # Documents are reconstructed from index payload; no separate store needed
        # In memory index has documents tracked internally
        chunks = await self.index.get_all_chunks()
        for chunk in chunks:
            if chunk.document_id == document_id:
                # Use index's document store if available
                if hasattr(self.index, "_documents"):
                    return self.index._documents.get(document_id)
        return None
