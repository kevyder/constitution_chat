"""Assemble the streaming RAG chain for the Constitution chat."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import Config
from retrieval.prompts import build_prompt, format_context
from retrieval.retriever import build_retriever


@dataclass(frozen=True)
class ConstitutionRAG:
    """A streaming RAG chain over the constitution, plus an explicit retriever."""

    chain: Runnable
    retriever: BaseRetriever

    def stream(self, question: str):
        """Yield answer tokens for ``question``."""
        return self.chain.stream(question)

    def retrieve(self, question: str) -> list[Document]:
        """Return the source documents used to answer ``question``."""
        return self.retriever.invoke(question)


def build_rag(config: Config) -> ConstitutionRAG:
    """Build a streaming RAG chain from application config."""
    retriever = build_retriever(config)

    llm = ChatOpenAI(
        base_url=config.openai_url,
        api_key=config.openai_key,
        model=config.chat_model,
        streaming=True,
        temperature=0.2,
    )

    prompt = build_prompt()

    def _format(docs: list[Document]) -> str:
        return format_context(docs)

    chain: Runnable = (
        RunnableMap(
            {
                "context": retriever | RunnableLambda(_format),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return ConstitutionRAG(chain=chain, retriever=retriever)
