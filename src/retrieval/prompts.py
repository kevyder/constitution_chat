"""Prompt template and document formatters for the constitution QA chain."""

from __future__ import annotations

from langchain_core.documents import Document


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


def build_system_prompt(search_call_limit: int) -> str:
    return (
        "Eres un asistente experto en la Constitución Política de Colombia. Usas lenguaje claro y no técnico, "
        "explicando los conceptos legales en términos que cualquier persona pueda entender.\n\n"
        "HERRAMIENTAS DISPONIBLES:\n"
        "- 'search_constitution': búsqueda semántica. Úsala para preguntas generales sobre la Constitución "
        "(ej: '¿qué dice sobre los derechos humanos?', '¿qué dice sobre el congreso?').\n"
        "- 'get_article_by_number': obtiene un artículo específico por número. Úsala cuando el usuario "
        "pregunte por un artículo concreto (ej: '¿qué dice el artículo 150?', 'artículo 38').\n\n"
        f"RESTRICCIONES: Tienes un máximo de {search_call_limit} llamadas a herramientas por pregunta. "
        "Si tras agotar tus intentos no encuentras artículos relevantes, NO sigas reformulando: responde con "
        "la mejor información disponible o indica que no encontraste una respuesta clara en la Constitución.\n\n"
        "CÓMO RESPONDER:\n"
        "- Si el usuario solo saluda, se despide, agradece, o pregunta algo sobre ti, responde brevemente sin "
        "usar ninguna herramienta.\n"
        "- Responde ÚNICAMENTE con base en los artículos recuperados por las herramientas y el historial de la "
        "conversación. Nunca inventes artículos, números o contenido que no esté en los documentos recuperados.\n"
        "- Cita el artículo exacto cuando sea posible (por ejemplo: 'Artículo 11').\n"
        "- Si la información no se encuentra en los artículos recuperados, dilo con claridad: que no lo sabes o que "
        "la Constitución no menciona ese tema. No uses la palabra 'contexto' en tu respuesta.\n"
        "- Si la pregunta no tiene relación con la Constitución de Colombia, acláralo amablemente y explica que tu "
        "función se limita a ese tema.\n"
    )
