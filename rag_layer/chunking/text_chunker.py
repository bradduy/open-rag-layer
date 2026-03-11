from __future__ import annotations

import re

from rag_layer.config import ChunkingConfig
from rag_layer.schema import Chunk, ChunkMetadata


class TextChunker:
    """Sliding-window chunker with optional sentence-boundary snapping."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, chunks: list[Chunk]) -> list[Chunk]:
        result: list[Chunk] = []
        for raw in chunks:
            if raw.content_type != "text" or not isinstance(raw.content, str):
                result.append(raw)
                continue
            result.extend(self._split(raw))
        return result

    def _split(self, chunk: Chunk) -> list[Chunk]:
        text = chunk.content
        assert isinstance(text, str)

        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(text) <= size:
            return [chunk]

        windows = self._sliding_windows(text, size, overlap)
        total = len(windows)
        out: list[Chunk] = []

        for i, window_text in enumerate(windows):
            meta = chunk.metadata.model_copy(
                update={
                    "chunk_index": i,
                    "total_chunks": total,
                }
            )
            out.append(
                Chunk(
                    document_id=chunk.document_id,
                    content=window_text,
                    content_type="text",
                    metadata=meta,
                )
            )
        return out

    def _sliding_windows(self, text: str, size: int, overlap: int) -> list[str]:
        if self.config.sentence_boundary:
            sentences = self._split_sentences(text)
            return self._sentences_to_windows(sentences, size, overlap)
        return self._char_windows(text, size, overlap)

    def _char_windows(self, text: str, size: int, overlap: int) -> list[str]:
        """Pure character-level sliding window (no sentence detection)."""
        windows: list[str] = []
        step = max(1, size - overlap)
        for start in range(0, len(text), step):
            windows.append(text[start : start + size])
            if start + size >= len(text):
                break
        return windows

    def _split_sentences(self, text: str) -> list[str]:
        # Simple regex sentence splitter
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, text.strip())
        return [p for p in parts if p.strip()]

    def _sentences_to_windows(
        self, sentences: list[str], size: int, overlap: int
    ) -> list[str]:
        windows: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If a single sentence exceeds size, fall back to char-level splitting
            if sentence_len > size:
                if current:
                    windows.append(" ".join(current))
                    current = []
                    current_len = 0
                windows.extend(self._char_windows(sentence, size, overlap))
                continue

            if current_len + sentence_len > size and current:
                windows.append(" ".join(current))
                # Keep overlap worth of sentences
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len

            current.append(sentence)
            current_len += sentence_len

        if current:
            windows.append(" ".join(current))

        return windows
