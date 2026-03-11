from __future__ import annotations

from rag_layer.schema import Chunk, ChunkMetadata, Document


class TextExtractor:
    """Extracts plain text from .txt / .md / .html sources."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            text = source

        # Strip basic HTML tags if present
        if "<html" in text.lower() or "<body" in text.lower():
            try:
                import re
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
            except Exception:
                pass

        chunk = Chunk(
            document_id=document.id,
            content=text,
            content_type="text",
            metadata=ChunkMetadata(
                chunk_index=0,
                total_chunks=1,
                source_ref=document.source,
            ),
        )
        return [chunk]
