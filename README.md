# Constitución Colombia Chat

A Streamlit RAG chat over the **Constitución Política de Colombia de 1991**.

The app indexes the constitution by article, stores the embeddings in a vector database, and answers questions using only the retrieved articles.

## Quickstart

```bash
cp .env.example .env
uv sync
uv run python scripts/vectorize_constitution.py
uv run streamlit run main.py
```

Open the app at the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Requirements

- Python 3.14
- uv
- A running vector database (default: Qdrant at `http://localhost:6333`)
- An OpenAI-compatible API endpoint that supports both embeddings and chat

## Configuration

Edit `.env`:

```bash
OPENAI_URL=...
OPENAI_KEY=...

VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_COLLECTION_NAME=colombia_constitution

EMBEDDING_MODEL=...
CHAT_MODEL=...
RETRIEVER_TOP_K=5
```

## How it works

1. `scripts/vectorize_constitution.py` extracts the PDF text.
2. The text is split into one chunk per article.
3. Each chunk is embedded and upserted into the vector database.
4. `main.py` streams the Streamlit chat.
5. Each question retrieves relevant articles and passes them to the chat model.

## Development

- `uv sync` — install dependencies
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `codegraph sync` — refresh local CodeGraph index

See `AGENTS.md` for repo-specific instructions for future coding sessions.
