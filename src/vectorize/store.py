"""Map :class:`Article` chunks to :class:`VectorRecord` objects."""

from __future__ import annotations

from vector_db.base import VectorRecord
from vectorize.chunk import Article


def build_record_id(article: Article) -> str:
    """Return a stable, human-readable id for an article chunk."""
    prefix = "transitory" if article.article_type == "transitory" else "article"
    return f"{prefix}-{article.article_number}"


def article_to_record(article: Article, vector: list[float]) -> VectorRecord:
    """Project an :class:`Article` plus its embedding into a :class:`VectorRecord`."""
    return VectorRecord(
        id=build_record_id(article),
        vector=vector,
        payload={
            "type": "article",
            "article_type": article.article_type,
            "title": article.title,
            "title_name": article.title_name,
            "chapter": article.chapter,
            "chapter_name": article.chapter_name,
            "article_number": article.article_number,
            "text": article.text,
            "source": article.source,
        },
    )


__all__ = ["build_record_id", "article_to_record"]
