from __future__ import annotations

import io
from typing import Any

from rag_layer.config import GeminiEmbedderConfig
from rag_layer.observability.logger import get_logger, timed
from rag_layer.schema import Chunk

logger = get_logger(__name__)


class GeminiEmbedder:
    """Embeds chunks using Gemini Embedding 2 via google-genai SDK."""

    def __init__(self, config: GeminiEmbedderConfig | None = None) -> None:
        self.config = config or GeminiEmbedderConfig()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    @timed
    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        from google.genai import types

        client = self._get_client()
        batch_size = self.config.batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            contents = [self._chunk_to_content(c) for c in batch]

            result = client.models.embed_content(
                model=self.config.model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.config.output_dimensionality,
                ),
            )

            for chunk, emb in zip(batch, result.embeddings):
                chunk.embedding = list(emb.values)

            logger.debug(f"Embedded batch {i // batch_size + 1}, size={len(batch)}")

        return chunks

    @timed
    async def embed_query(
        self, text: str, image: bytes | None = None
    ) -> list[float]:
        from google.genai import types

        client = self._get_client()

        if image is not None:
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image))
            content = pil_image
        else:
            content = text

        result = client.models.embed_content(
            model=self.config.model,
            contents=content,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.config.output_dimensionality,
            ),
        )
        return list(result.embeddings[0].values)

    def _chunk_to_content(self, chunk: Chunk) -> Any:
        """Convert a Chunk to a Gemini-compatible content object."""
        if chunk.content_type == "text":
            return chunk.content

        if chunk.content_type == "image_bytes":
            try:
                from PIL import Image
                return Image.open(io.BytesIO(chunk.content))  # type: ignore[arg-type]
            except Exception:
                # Fall back to raw bytes
                return chunk.content

        # audio_bytes / video_bytes — upload via Files API
        if chunk.content_type in ("audio_bytes", "video_bytes"):
            return self._upload_file(chunk)

        return chunk.content

    def _upload_file(self, chunk: Chunk) -> Any:
        """Upload audio/video bytes to Gemini Files API and return the file handle."""
        client = self._get_client()
        content_type_map = {
            "audio_bytes": "audio/mpeg",
            "video_bytes": "video/mp4",
        }
        mime = content_type_map.get(chunk.content_type, "application/octet-stream")
        data = chunk.content if isinstance(chunk.content, bytes) else chunk.content.encode()
        uploaded = client.files.upload(
            file=io.BytesIO(data),
            config={"mime_type": mime},
        )
        return uploaded
