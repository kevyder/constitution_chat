"""Prompt template and document formatters for the constitution QA chain."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_PROMPT_ES = (
    "Eres un asistente experto en la Constitución Política de Colombia, tu lenguaje es no técnico. "
    "Responde la pregunta usando ÚNICAMENTE el contexto proporcionado. "
    "Cita el artículo exacto cuando sea posible (por ejemplo: 'Artículo 11'). "
    """Si la respuesta no se encuentra en el contexto, di claramente que no lo sabes
    o que no hay menciones en la constitución sobre el tema, no menciones la palabra 'contexto' en tu respuesta."""
    """Cuando solamente te saluden, da una pequeña respuesta de saludo y
    da una breve introducción de tu función como asistente experto en la Constitución Política de Colombia."""
)

_USER_TEMPLATE_ES = "Contexto:\n{context}\n\nPregunta: {question}"


def format_article(doc: Document) -> str:
    """Format a single retrieved article document for the prompt context."""
    payload = doc.metadata or {}
    article_type = payload.get("article_type", "permanent")
    prefix = "Artículo transitorio" if article_type == "transitory" else "Artículo"
    article_number = payload.get("article_number", "?")
    title = payload.get("title_name") or payload.get("title") or ""
    chapter = payload.get("chapter_name") or payload.get("chapter") or ""

    header_parts = [f"{prefix} {article_number}"]
    if title:
        header_parts.append(title)
    if chapter:
        header_parts.append(chapter)
    header = " | ".join(header_parts)
    return f"{header}\n{doc.page_content}"


def format_context(docs: list[Document]) -> str:
    """Format a list of retrieved documents into a single context block."""
    return "\n\n---\n\n".join(format_article(doc) for doc in docs)


def build_prompt() -> ChatPromptTemplate:
    """Build the Spanish chat prompt template used by the RAG chain."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT_ES),
            ("human", _USER_TEMPLATE_ES),
        ]
    )
