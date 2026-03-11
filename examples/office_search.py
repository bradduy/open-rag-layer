"""
Office document search — index PDFs and search across them.

Requirements:
    GOOGLE_API_KEY set in environment.
    PDF files in the current directory or specify paths below.

Run:
    uv run python examples/office_search.py
"""
import asyncio
import sys
from pathlib import Path

from rag_layer import RAGLayer


async def main() -> None:
    pdf_paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not pdf_paths:
        print("Usage: uv run python examples/office_search.py file1.pdf file2.pdf ...")
        print("No PDFs provided; indexing a sample text instead.\n")
        pdf_paths = None

    rag = RAGLayer(config={"embedder": "gemini", "index": "memory"})

    if pdf_paths:
        print(f"Indexing {len(pdf_paths)} PDF(s)...")
        for path in pdf_paths:
            p = Path(path)
            if not p.exists():
                print(f"  Skipping {path}: file not found")
                continue
            await rag.index(str(p), metadata={"filename": p.name})
            print(f"  Indexed: {p.name}")
    else:
        # Fallback to text snippets
        snippets = [
            "Section 1: Executive Summary. Revenue grew 20% in FY2024.",
            "Section 2: Product Updates. We launched three new AI features.",
            "Section 3: Market Analysis. Competitor share decreased by 5%.",
        ]
        for i, text in enumerate(snippets):
            await rag.index(text, metadata={"section": i + 1})

    query = input("\nEnter search query (or press Enter for default): ").strip()
    if not query:
        query = "revenue growth"

    print(f"\nSearching for: '{query}'")
    results = await rag.search(query, limit=5, search_mode="hybrid")
    for r in results:
        content = r.chunk.content
        preview = content[:120] if isinstance(content, str) else f"<binary: {len(content)} bytes>"
        page = r.chunk.metadata.page_number
        page_str = f" (page {page})" if page else ""
        print(f"  [{r.score:.3f}]{page_str} {preview}...")


if __name__ == "__main__":
    asyncio.run(main())
