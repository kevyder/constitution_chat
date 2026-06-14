"""Extract the constitution, embed each article, and upsert into the vector DB."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config  # noqa: E402
from vector_db import create_vector_db_client  # noqa: E402
from vectorize.chunk import parse_articles  # noqa: E402
from vectorize.embed import Embedder  # noqa: E402
from vectorize.extract import extract_text  # noqa: E402
from vectorize.store import article_to_record  # noqa: E402


def _first_batch_with_dim(embedder: Embedder, texts: list[str], batch_size: int) -> tuple[list[str], list[list[float]]]:
    """Embed just the first batch so we can discover the model's vector dimension."""
    first_batch_texts = texts[:batch_size]
    first_batch_vectors = embedder.embed_texts(first_batch_texts, batch_size=batch_size)
    return first_batch_texts, first_batch_vectors


def main() -> None:
    config = Config.from_env()

    print(f"Extracting text from {config.pdf_path}...")
    raw_text = extract_text(config.pdf_path)
    print(f"Extracted {len(raw_text):,} characters")

    print("Parsing articles...")
    articles = parse_articles(raw_text, source=config.pdf_path)
    permanent = sum(1 for a in articles if a.article_type == "permanent")
    transitory = sum(1 for a in articles if a.article_type == "transitory")
    print(f"Found {permanent} permanent + {transitory} transitory = {len(articles)} articles")

    print(f"Embedding with {config.embedding_model}...")
    embedder = Embedder(
        url=config.openai_url,
        api_key=config.openai_key,
        model=config.embedding_model,
    )
    texts = [article.text for article in articles]

    first_texts, first_vectors = _first_batch_with_dim(embedder, texts, config.embedding_batch_size)
    vector_size = len(first_vectors[0])
    print(f"Detected vector dimension: {vector_size}")

    client = create_vector_db_client(config)
    try:
        existing_size = client.collection_vector_size(config.collection_name)
        if existing_size is None:
            print(
                f"Creating collection '{config.collection_name}' in {config.vector_db_provider} (dim={vector_size})..."
            )
            client.create_collection(config.collection_name, vector_size)
        elif existing_size != vector_size:
            print(
                f"Recreating collection '{config.collection_name}': "
                f"existing dim={existing_size} != model dim={vector_size}"
            )
            client.recreate_collection(config.collection_name, vector_size)
        else:
            print(f"Collection '{config.collection_name}' already has matching dim={vector_size}")

        print("Embedding remaining articles...")
        remaining_texts = texts[config.embedding_batch_size :]
        remaining_vectors = (
            embedder.embed_texts(remaining_texts, batch_size=config.embedding_batch_size) if remaining_texts else []
        )

        all_records = [
            article_to_record(article, vector)
            for article, vector in zip(articles, [*first_vectors, *remaining_vectors], strict=True)
        ]
        print(f"Upserting {len(all_records)} records...")
        client.upsert(config.collection_name, all_records)

        total = client.count(config.collection_name)
        print(f"Done. Collection now has {total} points.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
