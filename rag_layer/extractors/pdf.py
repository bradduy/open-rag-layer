from __future__ import annotations

import io

from rag_layer.schema import Chunk, ChunkMetadata, Document


class PDFExtractor:
    """Extracts text per page from PDF files using pypdf."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is required for PDF extraction: pip install pypdf") from e

        if isinstance(source, str):
            source = source.encode("utf-8")

        reader = PdfReader(io.BytesIO(source))
        chunks: list[Chunk] = []
        total = len(reader.pages)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=text,
                    content_type="text",
                    metadata=ChunkMetadata(
                        chunk_index=i,
                        total_chunks=total,
                        source_ref=document.source,
                        page_number=i + 1,
                    ),
                )
            )

        # Also keep raw bytes as a single chunk for multimodal embedding
        chunks.append(
            Chunk(
                document_id=document.id,
                content=source,
                content_type="image_bytes",  # treated as image-like bytes by Gemini
                metadata=ChunkMetadata(
                    chunk_index=total,
                    total_chunks=total + 1,
                    source_ref=document.source,
                    extra={"raw_pdf": True},
                ),
            )
        )

        return chunks
