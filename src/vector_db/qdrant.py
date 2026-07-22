"""Qdrant implementation of :class:`VectorDBClient`."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from vector_db.base import SearchResult, VectorDBClient, VectorRecord


def _to_uuid(raw_id: str) -> str:
    """Map a caller-supplied string id to a stable UUID5 for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


class QdrantVectorDBClient(VectorDBClient):
    """Thin :class:`VectorDBClient` adapter around ``qdrant_client.QdrantClient``."""

    def __init__(self, url: str, api_key: str | None = None, port: int | None = None) -> None:
        kwargs: dict[str, int | str | None] = {"url": url, "api_key": api_key}
        if port is not None:
            kwargs["port"] = port
        self._client = QdrantClient(**kwargs)

    def collection_vector_size(self, collection_name: str) -> int | None:
        if not self._client.collection_exists(collection_name=collection_name):
            return None
        info = self._client.get_collection(collection_name=collection_name)
        params = info.config.params.vectors
        if isinstance(params, models.VectorParams):
            return int(params.size)
        if isinstance(params, dict) and params:
            first = next(iter(params.values()))
            return int(first.size)
        return None

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def delete_collection(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name=collection_name):
            self._client.delete_collection(collection_name=collection_name)

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        self.delete_collection(collection_name)
        self.create_collection(collection_name, vector_size)

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
    ) -> None:
        if not records:
            return
        for start in range(0, len(records), 64):
            batch = records[start : start + 64]
            points = [
                models.PointStruct(
                    id=_to_uuid(record.id),
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in batch
            ]
            self._client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(
        self,
        collection_name: str,
        vector: list[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        response = self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(id=str(point.id), score=point.score, payload=point.payload or {}) for point in response.points
        ]

    def count(self, collection_name: str) -> int:
        info = self._client.get_collection(collection_name=collection_name)
        return int(info.points_count or 0)

    def get_by_payload_filter(
        self,
        collection_name: str,
        filters: dict[str, str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        results, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )
        return [SearchResult(id=str(r.id), score=1.0, payload=r.payload or {}) for r in results]

    def close(self) -> None:
        self._client.close()


__all__ = ["QdrantVectorDBClient"]
