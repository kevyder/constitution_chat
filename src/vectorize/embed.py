"""OpenRouter embeddings client (OpenAI-compatible)."""

from __future__ import annotations

from openai import OpenAI
from tqdm import tqdm


class Embedder:
    def __init__(self, url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=url, api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                encoding_format="float",
            )
            batch_vectors = [item.embedding for item in response.data]
            vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=[text],
            encoding_format="float",
        )
        return response.data[0].embedding
