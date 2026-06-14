# Agents

This project is a **Streamlit RAG chat** over the Colombian Constitution (1991), using OpenAI-compatible APIs and a vector database.

## Key gotchas

- **Do not treat this as an installable package.** `pyproject.toml` has `package = false`.
  - Entry-point scripts must add `src/` to `sys.path` before importing project modules.
  - Keep the `# noqa: E402` comments on imports that follow that `sys.path` setup.
- **The app entry point is `main.py`.**
  - Run it with `uv run streamlit run main.py`, not `python main.py`.
- **The project is not installed as a package.** Do not add `package = true` unless you also remove the `sys.path` setup from scripts.
- **Qdrant is external.** There is no `docker-compose.yml` or Dockerfile in the repo.
  - The app expects a vector DB reachable at `VECTOR_DB_URL` (or `QDRANT_URL`) and `QDRANT_API_KEY`.
  - Default collection: `colombia_constitution`.
- **Embeddings and chat both use `OPENAI_URL` / `OPENAI_KEY`.**
  - These are generic OpenAI-compatible env vars, not hardcoded to OpenAI.
- **Vector dimension is auto-detected on ingest.**
  - If `EMBEDDING_MODEL` changes, the collection is recreated automatically when the dimension changes.
  - This is intentional, but it means changing the embedding model clears the current collection.
- **Use the vector DB factory.**
  - Always prefer `vector_db.factory.create_vector_db_client(config)`.
  - Do not instantiate `QdrantVectorDBClient` directly in app code.
- **Retriever contract.**
  - `retrieval.retriever.build_retriever(config)` returns a LangChain retriever.
  - Retrieved documents carry the raw Qdrant payload as metadata, with `text` moved to `Document.page_content`.
  - Use metadata keys such as `article_number`, `title_name`, `article_type`, `title`, `chapter`, `chapter_name`.

## Commands

```bash
# Install / sync dependencies
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# One-time ingest into the vector DB
uv run python scripts/vectorize_constitution.py

# Run the Streamlit chat
uv run streamlit run main.py
```

## Environment

Copy `.env.example` to `.env` and fill in the real values.

```bash
cp .env.example .env
```

Important env vars:

```bash
OPENAI_URL=https://your-openai-compatible-endpoint/v1/
OPENAI_KEY=your-key

VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_COLLECTION_NAME=colombia_constitution

EMBEDDING_MODEL=...
CHAT_MODEL=...
RETRIEVER_TOP_K=5
```

## CodeGraph

This repo uses CodeGraph for local code intelligence.

```bash
codegraph sync
```

Use CodeGraph search/explore before editing unfamiliar code. It is the preferred way to find symbols and call sites in this project.
