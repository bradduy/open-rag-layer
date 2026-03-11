from __future__ import annotations

from rag_layer.schema import Chunk, ChunkMetadata, Document

MAX_CHUNK_SECONDS = 120.0


class VideoExtractor:
    """Extracts video as bytes for Gemini embedding (≤120s chunks)."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        if isinstance(source, str):
            source = source.encode("utf-8")

        # Video files are stored as-is; Gemini handles upload via Files API
        chunk = Chunk(
            document_id=document.id,
            content=source,
            content_type="video_bytes",
            metadata=ChunkMetadata(
                chunk_index=0,
                total_chunks=1,
                source_ref=document.source,
                timestamp_start=0.0,
                timestamp_end=None,
                extra={"note": "video content; embed via Gemini Files API"},
            ),
        )
        return [chunk]
