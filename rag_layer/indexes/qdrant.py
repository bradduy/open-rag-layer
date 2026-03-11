from __future__ import annotations

import json
from typing import Any

from rag_layer.config import QdrantIndexConfig
from rag_layer.observability.logger import get_logger
from rag_layer.schema import Chunk, ChunkMetadata, Document, SearchResult

logger = get_logger(__name__)


class QdrantIndex:
    """Async Qdrant-backed index."""

    def __init__(self, config: QdrantIndexConfig | None = None) -> None:
        self.config = config or QdrantIndexConfig()
        self._client: Any = None
        self._collection_created = False

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
            )
        return self._client

    async def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_created:
            return
        from qdrant_client.models import Distance, VectorParams
        client = self._get_client()
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        if self.config.collection_name not in names:
            await client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{self.config.collection_name}'")
        self._collection_created = True

    async def upsert(self, chunks: list[Chunk], documents: list[Document]) -> None:
        from qdrant_client.models import PointStruct

        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return

        await self._ensure_collection(len(embedded[0].embedding))  # type: ignore[arg-type]
        client = self._get_client()

        doc_map = {d.id: d for d in documents}
        points: list[PointStruct] = []

        for chunk in embedded:
            doc = doc_map.get(chunk.document_id)
            payload = {
                "chunk_json": chunk.model_dump_json(exclude={"embedding"}),
                "document_json": doc.model_dump_json() if doc else None,
                "document_id": chunk.document_id,
                "content_type": chunk.content_type,
            }
            if isinstance(chunk.content, str):
                payload["text"] = chunk.content

            # Add doc metadata for filtering
            if doc:
                payload.update(doc.metadata)

            points.append(
                PointStruct(
                    id=self._chunk_id_to_int(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        await client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )
        logger.debug(f"Upserted {len(points)} points to Qdrant")

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        client = self._get_client()
        qdrant_filter = self._build_filter(filters) if filters else None

        hits = await client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for rank, hit in enumerate(hits):
            payload = hit.payload or {}
            chunk = self._deserialize_chunk(payload)
            doc = self._deserialize_document(payload)
            if chunk and doc:
                results.append(
                    SearchResult(chunk=chunk, score=hit.score, document=doc, rank=rank)
                )
        return results

    async def get_all_chunks(self) -> list[Chunk]:
        client = self._get_client()
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, next_offset = await client.scroll(
                collection_name=self.config.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for record in records:
                chunk = self._deserialize_chunk(record.payload or {})
                if chunk:
                    chunks.append(chunk)
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    def _chunk_id_to_int(self, chunk_id: str) -> str:
        # Qdrant supports UUID strings directly
        return chunk_id

    def _build_filter(self, filters: dict[str, Any]) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    def _deserialize_chunk(self, payload: dict[str, Any]) -> Chunk | None:
        try:
            chunk_json = payload.get("chunk_json")
            if not chunk_json:
                return None
            data = json.loads(chunk_json)
            # content might be base64 encoded if bytes — handle gracefully
            return Chunk.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to deserialize chunk: {e}")
            return None

    def _deserialize_document(self, payload: dict[str, Any]) -> Document | None:
        try:
            doc_json = payload.get("document_json")
            if not doc_json:
                return None
            return Document.model_validate_json(doc_json)
        except Exception as e:
            logger.warning(f"Failed to deserialize document: {e}")
            return None
