"""LangChain adapter that wraps :class:`vectorize.embed.Embedder`."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from vectorize.embed import Embedder


class LangChainEmbedder(Embeddings):
    """Expose :class:`Embedder` through LangChain's :class:`Embeddings` interface."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_texts(texts, batch_size=len(texts) or 1)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)
