"""
Multimodal search example — index an image and search with text.

Requirements:
    GOOGLE_API_KEY set in environment.
    An image file at ./examples/sample.png (or update the path below).

Run:
    uv run python examples/multimodal_search.py
"""
import asyncio
from pathlib import Path

from rag_layer import RAGLayer


async def main() -> None:
    rag = RAGLayer(config={"embedder": "gemini", "index": "memory"})

    # Index text
    await rag.index("A diagram showing the architecture of a distributed system.")
    await rag.index("Photo of a golden retriever playing fetch in the park.")

    # Index image if it exists
    sample_img = Path(__file__).parent / "sample.png"
    if sample_img.exists():
        print(f"Indexing image: {sample_img}")
        await rag.index(str(sample_img), metadata={"type": "photo"})
    else:
        print(f"No sample image found at {sample_img}; skipping image ingestion.")

    print("\nSearching: 'dog playing outside'")
    results = await rag.search("dog playing outside", limit=3)
    for r in results:
        content_preview = (
            r.chunk.content[:80]
            if isinstance(r.chunk.content, str)
            else f"<{r.chunk.content_type}: {len(r.chunk.content)} bytes>"
        )
        print(f"  [{r.score:.3f}] ({r.chunk.content_type}) {content_preview}")


if __name__ == "__main__":
    asyncio.run(main())
