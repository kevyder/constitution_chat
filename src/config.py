"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    openai_url: str
    openai_key: str
    vector_db_provider: str
    vector_db_url: str
    vector_db_api_key: str | None
    collection_name: str
    embedding_model: str
    embedding_batch_size: int
    pdf_path: str
    chat_model: str
    retriever_top_k: int

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            openai_url=os.environ.get("OPENAI_URL", "https://api.openai.com/v1/"),
            openai_key=os.environ.get("OPENAI_KEY", ""),
            vector_db_provider=os.environ.get("VECTOR_DB_PROVIDER", "qdrant"),
            vector_db_url=os.environ.get("VECTOR_DB_URL", "http://localhost:6333"),
            vector_db_api_key=os.environ.get("VECTOR_DB_API_KEY"),
            collection_name=os.environ.get("VECTOR_DB_COLLECTION_NAME", "colombia_constitution"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-large"),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")),
            pdf_path=os.environ.get("PDF_PATH", "assets/constitucion-politica-colombia-1991.pdf"),
            chat_model=os.environ.get("CHAT_MODEL", "openai/gpt-4o-mini"),
            retriever_top_k=int(os.environ.get("RETRIEVER_TOP_K", "5")),
        )
