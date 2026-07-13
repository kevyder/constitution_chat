"""Tools for the Constitution RAG agent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from retrieval.prompts import format_context


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
