from __future__ import annotations

import io

from rag_layer.schema import Chunk, ChunkMetadata, Document

MAX_CHUNK_SECONDS = 80.0


class AudioExtractor:
    """Extracts audio chunks (≤80s) for Gemini embedding."""

    async def extract(self, document: Document, source: bytes | str) -> list[Chunk]:
        if isinstance(source, str):
            source = source.encode("utf-8")

        duration = self._get_duration(source)
        chunks: list[Chunk] = []

        if duration is None or duration <= MAX_CHUNK_SECONDS:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=source,
                    content_type="audio_bytes",
                    metadata=ChunkMetadata(
                        chunk_index=0,
                        total_chunks=1,
                        source_ref=document.source,
                        timestamp_start=0.0,
                        timestamp_end=duration,
                    ),
                )
            )
        else:
            # Split into segments — for now store as a single chunk with metadata
            # Full splitting would require an audio processing library like pydub
            num_segments = int(duration / MAX_CHUNK_SECONDS) + 1
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=source,
                    content_type="audio_bytes",
                    metadata=ChunkMetadata(
                        chunk_index=0,
                        total_chunks=num_segments,
                        source_ref=document.source,
                        timestamp_start=0.0,
                        timestamp_end=duration,
                        extra={"note": "audio exceeds 80s; consider splitting"},
                    ),
                )
            )

        return chunks

    def _get_duration(self, data: bytes) -> float | None:
        try:
            import soundfile as sf
            f = sf.SoundFile(io.BytesIO(data))
            return len(f) / f.samplerate
        except Exception:
            return None
