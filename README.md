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

## Docker Compose quickstart

```bash
cp .env.example .env
# Edit .env with your OpenAI-compatible endpoint and Qdrant settings.
docker compose build
docker compose up
```

Open the app at:

```text
http://localhost:8501
```

The Docker entrypoint runs `scripts/vectorize_constitution.py` before starting Streamlit. Ingest is idempotent: it creates the collection if missing, recreates it only if the embedding dimension changes, and upserts all articles otherwise.

### Docker/Qdrant settings

For a remote Qdrant instance:

```bash
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=https://qdrantinstance.com/
VECTOR_DB_PORT=443
VECTOR_DB_API_KEY=...
VECTOR_DB_COLLECTION_NAME=colombia_constitution
```

For the included local Qdrant service:

```bash
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_PORT=
VECTOR_DB_API_KEY=
```

The compose file maps:

- Streamlit: `http://localhost:8501`
- Qdrant REST UI/API: `http://localhost:6333`

Do not commit `.env`; it contains API keys.

## Requirements

- Python 3.14
- uv
- Docker Compose, for containerized runs
- A running vector database (default: Qdrant at `http://localhost:6333`)
- An OpenAI-compatible API endpoint that supports both embeddings and chat

## Configuration

Edit `.env`:

```bash
OPENAI_URL=...
OPENAI_KEY=...

VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_PORT=
VECTOR_DB_API_KEY=
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
