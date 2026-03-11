"""
Basic RAG example — text ingestion + semantic search.

Requirements:
    GOOGLE_API_KEY set in environment.

Run:
    uv run python examples/basic_rag.py
"""
import asyncio

from rag_layer import RAGLayer


async def main() -> None:
    # Uses Gemini Embedding 2 + in-memory index (no Qdrant needed)
    rag = RAGLayer(config={"embedder": "gemini", "index": "memory"})

    print("Indexing documents...")
    await rag.index(
        "The quarterly revenue for Q3 2024 reached $2.5B, up 18% YoY.",
        metadata={"source": "earnings_report"},
    )
    await rag.index(
        "Our machine learning platform now supports multimodal embeddings including text, images, and audio.",
        metadata={"source": "tech_blog"},
    )
    await rag.index(
        "Employee headcount grew by 12% this quarter, with most hires in engineering.",
        metadata={"source": "hr_report"},
    )

    print("\nSemantic search: 'quarterly revenue trends'")
    results = await rag.search("quarterly revenue trends", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")

    print("\nKeyword search: 'machine learning'")
    results = await rag.search("machine learning", search_mode="keyword", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")

    print("\nHybrid search: 'headcount engineering'")
    results = await rag.search("headcount engineering", search_mode="hybrid", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
