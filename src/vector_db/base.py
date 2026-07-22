"""Reusable vector database client abstraction.

The :class:`VectorDBClient` abstract base class lets the constitution
ingestion pipeline stay decoupled from any specific vector database
implementation. A concrete subclass (e.g. Qdrant) handles vendor details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorRecord:
    """A single upsertable vector with its identifier and metadata payload."""

    id: str
    vector: list[float]
    payload: dict


@dataclass(frozen=True)
class SearchResult:
    """A single result returned from a similarity search."""

    id: str
    score: float
    payload: dict


class VectorDBClient(ABC):
    """Abstract base class for vector database clients."""

    @abstractmethod
    def collection_vector_size(self, collection_name: str) -> int | None:
        """Return the configured vector size of ``collection_name``, or ``None`` if it does not exist."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection with the given vector size. Raise if it already exists."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete the collection if it exists; otherwise do nothing."""

    @abstractmethod
    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Drop the collection (if it exists) and create it fresh with the given vector size."""

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
    ) -> None:
        """Insert or update the given records in the collection."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        vector: list[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Return the top-``limit`` records most similar to ``vector``."""

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of points currently stored in the collection."""

    @abstractmethod
    def get_by_payload_filter(
        self,
        collection_name: str,
        filters: dict[str, str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Return records matching exact payload field values."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources held by the client."""


__all__ = ["VectorRecord", "SearchResult", "VectorDBClient"]
