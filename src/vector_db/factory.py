"""Factory for instantiating :class:`VectorDBClient` implementations."""

from __future__ import annotations

from config import Config
from vector_db.base import VectorDBClient


def create_vector_db_client(config: Config) -> VectorDBClient:
    """Build a vector DB client based on ``config.vector_db_provider``."""
    provider = config.vector_db_provider.lower()
    if provider == "qdrant":
        from vector_db.qdrant import QdrantVectorDBClient

        return QdrantVectorDBClient(
            url=config.vector_db_url,
            api_key=config.vector_db_api_key,
        )
    msg = f"Unsupported vector DB provider: {config.vector_db_provider!r}"
    raise ValueError(msg)


__all__ = ["create_vector_db_client"]
