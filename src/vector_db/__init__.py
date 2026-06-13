"""Vector database client abstraction.

Defines a vendor-neutral :class:`VectorDBClient` interface along with
concrete implementations (currently Qdrant).
"""

from vector_db.base import SearchResult, VectorDBClient, VectorRecord
from vector_db.factory import create_vector_db_client

__all__ = [
    "SearchResult",
    "VectorDBClient",
    "VectorRecord",
    "create_vector_db_client",
]
