# Agents

Streamlit RAG chat over the Constitución Política de Colombia (1991) using OpenAI-compatible APIs and Qdrant.

## Key gotchas

- **Not an installable package.** `pyproject.toml` sets `package = false` (`[tool.uv]`). `main.py` and `scripts/vectorize_constitution.py` prepend `src/` to `sys.path` and mark the imports that follow `# noqa: E402`. Do not flip `package = true` unless you also remove that `sys.path` setup.
- **Run with `uv`, not `python`.** Entry point is `uv run streamlit run main.py`. Python 3.14 is pinned in `.python-version`; `uv` reads it automatically.
- **Qdrant is external.** There is no bundled vector DB; `docker compose up -d qdrant` provides one at `http://localhost:6333`. There is no `QDRANT_URL` alias: `Config.from_env()` reads `VECTOR_DB_URL` (+ optional `VECTOR_DB_API_KEY`) only.
- **Port precedence.** qdrant-client uses the port embedded in `VECTOR_DB_URL` when present; `VECTOR_DB_PORT` only applies to portless URLs (e.g. set `VECTOR_DB_PORT=443` for `https://host/`). `.env.example` ships `VECTOR_DB_PORT=6663` while Compose exposes 6333 — fix the port when copying for local runs.
- **The Docker app image does not ingest.** `Dockerfile` copies only `src/`, `main.py`, `assets/` and starts Streamlit; the README claim that the entrypoint runs `scripts/vectorize_constitution.py` is stale. Run ingest once from the host against the mapped port before `docker compose up` with an empty collection.
- **Blank env values don't fall back.** `EMBEDDING_OPENAI_URL` / `EMBEDDING_OPENAI_KEY` fall back to `OPENAI_URL` / `OPENAI_KEY` only when unset; `.env.example` ships them blank, which loads as empty strings and bypasses the fallback. Set real values or delete the lines.
- **Changing `EMBEDDING_MODEL` can erase the collection.** Ingest detects the vector dimension on the first batch; on mismatch it recreates the collection and drops existing points.
- **Env is cached per process.** `main.py` builds the RAG behind `@st.cache_resource`; `.env` changes require a Streamlit restart.
- **Search-call limit is half config, half hard-coded.** `SEARCH_CALL_LIMIT` only shapes the system prompt text; the enforced cap is hard-coded (`ToolCallLimitMiddleware(run_limit=3)`) in `src/chat/agent.py`.
- **Reranker fails open.** `retrieval/reranker.py` (`RAGReranker`, `requests`-based `/rerank`) returns documents unranked if unconfigured or on error. Rerank models use the `nvidia/` prefix, not `openrouter/` (see `.env.example`).

## Architecture

- `scripts/vectorize_constitution.py` — one-shot ingest: extract PDF text → parse articles (`vectorize/chunk.py`, regex-driven and brittle to PDF layout changes) → embed → upsert. Idempotent: creates the collection, recreates only on dimension change.
- `main.py` — Streamlit entry; builds the RAG agent via `chat.agent.build_rag(config)`.
- `src/vector_db/` — `VectorDBClient` ABC + Qdrant impl. Always go through `vector_db.factory.create_vector_db_client(config)`; do not import `QdrantVectorDBClient` from app code. Add providers by subclassing and wiring them in `factory.py`.
- `src/retrieval/` — `build_retriever(config)` returns `(retriever, vector_db_client)`. Payload `text` moves to `Document.page_content`; the rest of the Qdrant payload stays in `Document.metadata` (`type`, `article_type`, `article_number`, `title`, `title_name`, `chapter`, `chapter_name`, `source`). Fetches `RETRIEVER_TOP_K` candidates, then reranks to `RERANKING_TOP_K`. `get_article_by_number` looks up by payload filter.
- `src/chat/` — `ConstitutionRAG` wraps a LangGraph `create_agent` (tools: `search_constitution`, `get_article_by_number`) with an `InMemorySaver` checkpointer and Langfuse callback tracing. Exposes `stream(question, thread_id)` and `get_documents(thread_id)`.

## Commands

```bash
uv sync                                              # install deps
uv run ruff check .                                  # lint
uv run ruff format --check .                         # format check (drop --check to write)
uv run python scripts/vectorize_constitution.py      # one-time ingest
uv run streamlit run main.py                         # run the chat
pre-commit install                                   # one-time
pre-commit run --all-files                           # manual hook run (auto-applies ruff fixes)
codegraph sync                                       # refresh local CodeGraph index
```

There are no tests: no `tests/` directory and no test runner in `pyproject.toml`.

## Environment

Copy `.env.example` to `.env`; it lists every variable (`OPENAI_URL`/`KEY`, `EMBEDDING_*`, `VECTOR_DB_*`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `RETRIEVER_TOP_K`, `SEARCH_CALL_LIMIT`, `RERANKING_*`, `LANGFUSE_*`, `PDF_PATH`). Watch for the blank-embedding and port gotchas above.

## Conventions

- **Commits** follow Conventional Commits; enforced by the `commit-msg` hook (`conventional-pre-commit`).
- **Ruff** (`pyproject.toml`): line length 120, google docstring convention; rules `E, F, UP, B, SIM, I` (ignore `E203`, `E305`).
- **Pre-commit ruff hooks auto-fix**, so manual hook runs may rewrite files.
- **Do not commit** `.env`, `.cavemem/`, or local Qdrant data (`.qdrant/`).

## CodeGraph

`opencode.jsonc` enables a local CodeGraph MCP server (`.codegraph/`). Prefer CodeGraph over grep/find when investigating unfamiliar code; run `codegraph sync` after refactors.

## Caveman mode

Project convention: caveman **lite** is the default communication style.

- Activate at session start: `/caveman lite` (other levels: `full|ultra|wenyan`; off: `/caveman off` or "stop caveman")
- Terse replies (~75% fewer tokens); code, commits, and security warnings always in normal English.
