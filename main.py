"""Streamlit RAG chat over the Colombian Constitution."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from chat.agent import build_rag  # noqa: E402
from config import Config  # noqa: E402

st.set_page_config(page_title="Chat Constitución de Colombia de 1991", page_icon="📜🇨🇴", layout="centered")
st.title("Chat de la constitución de Colombia de 1991", text_alignment="center")
st.caption(
    """Pregunta cualquier cosa sobre la Constitución de 1991.
    Las respuestas se basan únicamente en los artículos indexados de la Constitución Colombiana de 1991."""
)


@st.cache_resource(show_spinner="Cargando modelo y base de datos...")
def get_rag() -> object:
    config = Config.from_env()
    return build_rag(config)


def main() -> None:
    rag = get_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Artículos consultados", expanded=False):
                    for i, src in enumerate(message["sources"], start=1):
                        st.markdown(f"**[{i}] {src['id']}**")
                        st.markdown(f"_{src['preview']}_")

    question = st.chat_input("Escribe tu pregunta sobre la Constitución...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Generando respuesta...", expanded=True) as status:
            st.write("✍️ Generando respuesta con el modelo...")
            try:
                answer = st.write_stream(
                    rag.stream(question, st.session_state.session_id),
                )
                status.update(
                    label="Respuesta lista",
                    state="complete",
                    expanded=True,
                )
            except Exception as exc:
                status.update(label="Error", state="error")
                st.error(f"Error al generar la respuesta: {exc}")
                return

        docs = rag.get_documents(st.session_state.session_id)
        sources_payload: list[dict] = []
        for doc in docs:
            article_type = doc.metadata.get("article_type", "permanent")
            prefix = "Artículo transitorio" if article_type == "transitory" else "Artículo"
            article_id = f"{prefix} {doc.metadata.get('article_number', '?')}"
            text = doc.page_content or ""
            preview = text if len(text) <= 220 else text[:220] + "..."
            sources_payload.append(
                {
                    "id": article_id,
                    "preview": preview,
                }
            )

        if sources_payload:
            with st.expander("📚 Artículos consultados", expanded=False):
                for i, src in enumerate(sources_payload, start=1):
                    st.markdown(f"**[{i}] {src['id']}**")
                    st.markdown(f"_{src['preview']}_")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
        }
    )


if __name__ == "__main__":
    main()
