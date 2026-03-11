from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag_layer.chunking.text_chunker import TextChunker
from rag_layer.config import RAGConfig
from rag_layer.extractors.audio import AudioExtractor
from rag_layer.extractors.image import ImageExtractor
from rag_layer.extractors.pdf import PDFExtractor
from rag_layer.extractors.text import TextExtractor
from rag_layer.extractors.video import VideoExtractor
from rag_layer.observability.logger import get_logger, timed
from rag_layer.schema import Chunk, Document, Modality

logger = get_logger(__name__)

_MODALITY_MAP: dict[str, Modality] = {
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".htm": "text",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
}


class Ingestor:
    """Pipeline: detect → load → extract → chunk → embed → upsert."""

    def __init__(self, config: RAGConfig, embedder: Any, index: Any) -> None:
        self.config = config
        self.embedder = embedder
        self.index = index
        self.chunker = TextChunker(config.chunking)

        self._extractors = {
            "text": TextExtractor(),
            "pdf": PDFExtractor(),
            "image": ImageExtractor(),
            "audio": AudioExtractor(),
            "video": VideoExtractor(),
        }

    @timed
    async def ingest(
        self,
        source: str | list[str],
        modality: Modality | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        sources = [source] if isinstance(source, str) else source
        docs: list[Document] = []

        for src in sources:
            doc = await self._ingest_one(src, modality, metadata or {})
            docs.append(doc)

        return docs

    async def _ingest_one(
        self,
        source: str,
        modality: Modality | None,
        metadata: dict[str, Any],
    ) -> Document:
        detected_modality, raw_bytes = await self._load(source, modality)

        doc = Document(
            source=source,
            modality=detected_modality,
            metadata=metadata,
        )
        logger.info(f"Ingesting '{source}' as {detected_modality}")

        extractor = self._extractors[detected_modality]
        chunks = await extractor.extract(doc, raw_bytes)
        chunks = self.chunker.chunk(chunks)

        chunks = await self.embedder.embed_chunks(chunks)

        embedded = [c for c in chunks if c.embedding is not None]
        await self.index.upsert(embedded, [doc])

        logger.info(
            f"Indexed {len(embedded)}/{len(chunks)} chunks for '{source}'"
        )
        return doc

    async def _load(
        self, source: str, modality: Modality | None
    ) -> tuple[Modality, bytes]:
        if source.startswith("http://") or source.startswith("https://"):
            return await self._load_url(source, modality)

        path = Path(source)
        try:
            exists = path.exists()
        except (OSError, ValueError):
            exists = False

        if exists:
            raw = path.read_bytes()
            if modality is None:
                modality = self._detect_modality(source, raw)
            return modality, raw

        # Treat as raw text
        return "text", source.encode("utf-8")

    async def _load_url(
        self, url: str, modality: Modality | None
    ) -> tuple[Modality, bytes]:
        try:
            import httpx
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.content
                if modality is None:
                    content_type = resp.headers.get("content-type", "")
                    modality = self._detect_from_content_type(content_type)
                return modality or "text", raw
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL {url}: {e}") from e

    def _detect_modality(self, source: str, raw: bytes) -> Modality:
        ext = Path(source).suffix.lower()
        if ext in _MODALITY_MAP:
            return _MODALITY_MAP[ext]

        # Fallback: use filetype library
        try:
            import filetype
            kind = filetype.guess(raw)
            if kind:
                mime = kind.mime
                if mime.startswith("image/"):
                    return "image"
                if mime.startswith("audio/"):
                    return "audio"
                if mime.startswith("video/"):
                    return "video"
                if mime == "application/pdf":
                    return "pdf"
        except Exception:
            pass

        return "text"

    def _detect_from_content_type(self, content_type: str) -> Modality:
        ct = content_type.lower().split(";")[0].strip()
        if ct.startswith("image/"):
            return "image"
        if ct.startswith("audio/"):
            return "audio"
        if ct.startswith("video/"):
            return "video"
        if ct == "application/pdf":
            return "pdf"
        return "text"
