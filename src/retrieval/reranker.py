"""Rerank retrieved documents against a query using an OpenRouter-compatible rerank API."""

from __future__ import annotations

import logging
from typing import Any

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGReranker:
    """Rerank documents with an external ``/rerank`` endpoint (e.g. OpenRouter)."""

    def __init__(self, url: str, api_key: str, model: str, top_k: int) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._top_k = top_k

    @property
    def enabled(self) -> bool:
        """Whether the reranker has enough configuration to run."""
        return bool(self._url and self._api_key and self._model)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Return ``documents`` reordered by relevance to ``query``.

        Fails open: if the reranker is unavailable or errors, the original
        ``documents`` are returned unchanged.
        """
        if not self.enabled or not documents:
            return documents

        try:
            response = requests.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": [doc.page_content for doc in documents],
                    "top_n": self._top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except requests.RequestException as exc:
            logger.warning("Rerank request failed; returning unranked documents: %s", exc)
            return documents
        except ValueError as exc:
            logger.warning("Rerank returned invalid JSON; returning unranked documents: %s", exc)
            return documents

        return self._apply_indices(documents, results)

    def _apply_indices(self, documents: list[Document], results: list[Any]) -> list[Document]:
        """Reorder ``documents`` following the reranker result indices."""
        by_idx: dict[int, Document] = {i: doc for i, doc in enumerate(documents)}

        reranked: list[Document] = []
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or index not in by_idx:
                continue
            doc = by_idx[index]
            score = result.get("relevance_score")
            if score is not None:
                doc = Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score})
            reranked.append(doc)

        return reranked if reranked else documents[: self._top_k]
