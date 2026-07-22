"""Streaming RAG agent for the Constitution chat."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from chat.tools import make_get_article_tool, make_search_tool
from config import Config
from retrieval.prompts import build_system_prompt
from retrieval.retriever import build_retriever


class ConstitutionAgentState(AgentState):
    """Agent state that tracks retrieved source documents."""

    documents: list[Document]


@dataclass
class ConstitutionRAG:
    """A streaming RAG agent over the constitution, plus source retrieval."""

    agent: object
    checkpointer: InMemorySaver
    langfuse_handler: CallbackHandler = field(default_factory=CallbackHandler)

    def stream(self, question: str, thread_id: str):
        """Yield answer tokens for ``question`` within the given session."""
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [self.langfuse_handler],
            "metadata": {"langfuse_session_id": thread_id},
        }
        self.agent.update_state(config, {"documents": []})
        for chunk, _meta in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    def get_documents(self, thread_id: str) -> list[Document]:
        """Return source documents stored in state by the search tool."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.agent.get_state(config).values.get("documents", [])

    def clear_history(self, thread_id: str) -> None:
        """Delete conversation history for ``thread_id``."""
        self.checkpointer.delete_thread(thread_id)


def build_rag(config: Config) -> ConstitutionRAG:
    """Build a streaming RAG agent from application config."""
    retriever, vector_db = build_retriever(config)

    llm = ChatOpenAI(
        base_url=config.openai_url,
        api_key=config.openai_key,
        model=config.chat_model,
        streaming=True,
        temperature=0.2,
        extra_body={
            "reasoning_effort": "medium",
            "reasoning_format": "hidden",
        },
    )

    search_tool = make_search_tool(retriever)
    get_article_tool = make_get_article_tool(vector_db, config.collection_name)

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[search_tool, get_article_tool],
        system_prompt=build_system_prompt(config.search_call_limit),
        state_schema=ConstitutionAgentState,
        checkpointer=checkpointer,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_constitution",
                run_limit=3,  # max calls per user turn
                exit_behavior="end",  # gracefully stop instead of spinning
            ),
        ],
    )

    return ConstitutionRAG(
        agent=agent,
        checkpointer=checkpointer,
        langfuse_handler=CallbackHandler(),
    )
