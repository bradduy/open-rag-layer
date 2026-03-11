from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class GeminiEmbedderConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    model: str = "gemini-embedding-2-preview"
    output_dimensionality: int = 3072  # supports 768, 1536, 3072
    batch_size: int = 100


class QdrantIndexConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "rag_layer"
    prefer_grpc: bool = False


class ChunkingConfig(BaseModel):
    chunk_size: int = 512          # tokens / chars
    chunk_overlap: int = 64
    sentence_boundary: bool = True  # snap to sentence boundaries


class RAGConfig(BaseModel):
    embedder: Literal["gemini", "mock"] = "gemini"
    index: Literal["memory", "qdrant"] = "memory"
    gemini: GeminiEmbedderConfig = Field(default_factory=GeminiEmbedderConfig)
    qdrant: QdrantIndexConfig = Field(default_factory=QdrantIndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGConfig":
        return cls(**data)
