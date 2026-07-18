"""Build a vector-DB-backed LangChain retriever from the existing collection."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from config import Config
from retrieval.embeddings_adapter import LangChainEmbedder
from vector_db import create_vector_db_client
from vectorize.embed import Embedder


class _PayloadRetriever(BaseRetriever):
    """A retriever that uses a VectorDBClient to expose the full payload as document metadata."""

    client: Any
    collection_name: str
    embeddings: Embeddings
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        results = self.client.search(self.collection_name, vector, limit=self.k)
        documents: list[Document] = []
        for hit in results:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            documents.append(Document(page_content=text, metadata=payload))
        return documents


def build_retriever(config: Config) -> BaseRetriever:
    """Build a LangChain retriever that reads from the existing vector-DB collection."""
    embedder = Embedder(
        url=config.embedding_openai_url,
        api_key=config.embedding_openai_key,
        model=config.embedding_model,
    )
    vector_db = create_vector_db_client(config)
    return _PayloadRetriever(
        client=vector_db,
        collection_name=config.collection_name,
        embeddings=LangChainEmbedder(embedder),
        k=config.retriever_top_k,
    )
