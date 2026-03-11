from __future__ import annotations

from rag_layer.schema import Chunk, ChunkMetadata, Document


class ImageExtractor:
    """Extracts image bytes for Gemini multimodal embedding."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        if isinstance(source, str):
            source = source.encode("utf-8")

        # Validate it's a real image
        try:
            from PIL import Image
            import io
            Image.open(io.BytesIO(source)).verify()
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}") from e

        chunk = Chunk(
            document_id=document.id,
            content=source,
            content_type="image_bytes",
            metadata=ChunkMetadata(
                chunk_index=0,
                total_chunks=1,
                source_ref=document.source,
            ),
        )
        return [chunk]
