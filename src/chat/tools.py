"""Tools for the Constitution RAG agent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from retrieval.prompts import format_context
from vector_db.base import VectorDBClient


def _search_result_to_document(payload: dict) -> Document:
    """Convert a Qdrant payload dict to a LangChain Document."""
    text = payload.pop("text", "")
    return Document(page_content=text, metadata=payload)


def make_search_tool(retriever: BaseRetriever):
    """Create a ``search_constitution`` tool bound to ``retriever``."""

    @tool
    def search_constitution(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Busca artículos relevantes de la Constitución Política de Colombia."""
        docs = retriever.invoke(query)
        return Command(
            update={
                "documents": docs,
                "messages": [ToolMessage(content=format_context(docs), tool_call_id=tool_call_id)],
            }
        )

    return search_constitution


def make_get_article_tool(vector_db: VectorDBClient, collection_name: str):
    """Create a ``get_article_by_number`` tool for exact article lookups."""

    @tool
    def get_article_by_number(
        article_number: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Obtiene un artículo específico por número. Úsalo cuando el usuario pregunte por un artículo concreto."""
        filters = {"article_number": article_number, "article_type": "permanent"}
        results = vector_db.get_by_payload_filter(collection_name, filters, limit=1)
        if not results:
            results = vector_db.get_by_payload_filter(collection_name, {"article_number": article_number}, limit=1)
        docs = [_search_result_to_document(dict(r.payload)) for r in results]
        content = format_context(docs) if docs else "Artículo no encontrado."
        return Command(
            update={
                "documents": docs,
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    return get_article_by_number
