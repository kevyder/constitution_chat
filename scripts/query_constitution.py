"""Semantic search over the constitution in the vector database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_db import create_vector_db_client  # noqa: E402
from vectorize.config import Config  # noqa: E402
from vectorize.embed import Embedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the constitution in the vector database")
    parser.add_argument("query", help="Search query in Spanish")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_env()

    embedder = Embedder(
        url=config.openai_url,
        api_key=config.openai_key,
        model=config.embedding_model,
    )
    query_vector = embedder.embed_query(args.query)

    client = create_vector_db_client(config)
    try:
        results = client.search(
            config.collection_name,
            query_vector,
            limit=args.limit,
        )
    finally:
        client.close()

    print(f"\nQuery: {args.query}\n")
    for i, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        article_type = payload.get("article_type", "permanent")
        prefix = "transitory" if article_type == "transitory" else "article"
        article_id = f"{prefix}-{payload.get('article_number', '?')}"
        title_label = payload.get("title") or ""
        title_name = payload.get("title_name") or ""
        chapter_label = payload.get("chapter") or ""
        chapter_name = payload.get("chapter_name") or ""
        print(f"[{i}] {article_id} (score={hit.score:.4f})")
        if title_label or title_name:
            parts = [p for p in (title_label, title_name) if p]
            print(f"    {' | '.join(parts)}")
        if chapter_label or chapter_name:
            parts = [p for p in (chapter_label, chapter_name) if p]
            print(f"    {' | '.join(parts)}")
        text = payload.get("text", "")
        preview = text if len(text) <= 400 else text[:400] + "..."
        print(f"    {preview}\n")


if __name__ == "__main__":
    main()
