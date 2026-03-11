from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Modality = Literal["text", "pdf", "image", "audio", "video"]
ContentType = Literal["text", "image_bytes", "audio_bytes", "video_bytes"]
SearchMode = Literal["semantic", "keyword", "hybrid"]


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # file path, URL, or "raw"
    modality: Modality
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkMetadata(BaseModel):
    chunk_index: int
    total_chunks: int
    source_ref: str
    page_number: int | None = None
    timestamp_start: float | None = None  # audio/video
    timestamp_end: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str | bytes
    content_type: ContentType
    embedding: list[float] | None = None
    metadata: ChunkMetadata

    model_config = {"arbitrary_types_allowed": True}


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    document: Document
    rank: int | None = None


class SearchQuery(BaseModel):
    text: str = ""
    image: bytes | None = None
    modalities: list[Modality] | None = None
    filters: dict[str, Any] | None = None
    limit: int = 10
    min_score: float = 0.0
    use_reranking: bool = False
    search_mode: SearchMode = "semantic"


class IndexInput(BaseModel):
    """Flexible input for indexing — path, URL, or raw text."""

    source: str
    modality: Modality | None = None  # auto-detected if None
    metadata: dict[str, Any] = Field(default_factory=dict)
