# Agents

Streamlit RAG chat over the Constitución Política de Colombia (1991) using OpenAI-compatible APIs and Qdrant.

## Key gotchas

- **Not an installable package.** `pyproject.toml` has `package = false` (`[tool.uv]`). Entry-point scripts (`main.py`, `scripts/vectorize_constitution.py`) prepend `src/` to `sys.path` and use `# noqa: E402` on imports that follow. Do not flip `package = true` unless you also remove that `sys.path` setup.
- **Run with `uv`, not `python`.** App entry point is `main.py`; use `uv run streamlit run main.py`, not `python main.py`.
- **Qdrant is external.** There is no `docker-compose.yml` or `Dockerfile`. The app expects a reachable vector DB at `VECTOR_DB_URL` with optional `VECTOR_DB_API_KEY`. Default provider: `qdrant`. Default collection: `colombia_constitution`.
- **No `QDRANT_URL` alias.** `Config.from_env()` reads `VECTOR_DB_URL` and `VECTOR_DB_API_KEY` only.
- **OpenAI env vars are generic.** `OPENAI_URL` / `OPENAI_KEY` are used for both embeddings and chat, and are not pinned to OpenAI.
- **Changing embedding model can erase the collection.** Ingest detects vector dimension on first batch; if `EMBEDDING_MODEL` changes dimension, `scripts/vectorize_constitution.py` recreates the collection and clears existing points.
- **Use the vector DB factory.** Always go through `vector_db.factory.create_vector_db_client(config)`. Do not import `QdrantVectorDBClient` from app code.
- **Retriever contract.** `retrieval.retriever.build_retriever(config)` returns a LangChain retriever. `text` moves to `Document.page_content`; Qdrant payload stays in `Document.metadata`. Expected keys: `type`, `article_type`, `article_number`, `title`, `title_name`, `chapter`, `chapter_name`, `source`.
- **Python 3.14 required.** Pinned in `.python-version`; `uv` reads it automatically.
- **No tests configured.** No `tests/` directory and no test runner in `pyproject.toml`.

## Architecture

- `scripts/vectorize_constitution.py` — one-shot ingest: extract PDF text, parse articles, embed, upsert.
- `main.py` — Streamlit entry. Builds the RAG chain via `chat.chain.build_rag(config)`.
- `src/vectorize/` — `extract` → `chunk` → `embed` → `store`. The chunker (`chunk.py`) is regex-driven and brittle if the source PDF layout changes.
- `src/vector_db/` — `VectorDBClient` ABC plus a Qdrant implementation. Add new providers by subclassing and wiring them in `factory.py`.
- `src/retrieval/` — LangChain retriever adapter + Spanish prompt (`prompts.py`).
- `src/chat/chain.py` — `ConstitutionRAG` exposes `stream(question)` and `retrieve(question)`.

## Commands

```bash
uv sync                                              # install deps
uv run ruff check .                                  # lint
uv run ruff format .                                 # format
uv run python scripts/vectorize_constitution.py      # one-time ingest
uv run streamlit run main.py                         # run the chat
pre-commit install                                   # one-time, enables hooks
pre-commit run --all-files                           # manual hook run
codegraph sync                                       # refresh local CodeGraph index
```

## Environment

Copy `.env.example` to `.env`. Key vars:

```bash
OPENAI_URL=...
OPENAI_KEY=...

VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_API_KEY=
VECTOR_DB_COLLECTION_NAME=colombia_constitution

EMBEDDING_MODEL=...        # changing this can recreate the collection
EMBEDDING_BATCH_SIZE=64
CHAT_MODEL=...
RETRIEVER_TOP_K=5
PDF_PATH=assets/constitucion-politica-colombia-1991.pdf
```

## Conventions

- **Commits** must follow Conventional Commits; enforced by the `commit-msg` pre-commit hook (`conventional-pre-commit`).
- **Ruff** (`pyproject.toml`): `line-length = 120`, pydocstyle `google` convention. Rules: `E, F, UP, B, SIM, I`. Ignored: `E203`, `E305`.
- **Pre-commit ruff hook auto-fixes.** Manual hook runs also apply `ruff --fix` and `ruff-format`.
- **Do not commit** `.env`, `.cavemem/`, `~/.opencode/config.json`, or local Qdrant data (`.qdrant/`).

## CodeGraph

`opencode.jsonc` enables a local CodeGraph MCP server. In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for CodeGraph before grep/find when investigating unfamiliar code. Run `codegraph sync` to refresh the local index.

## Caveman mode

Project convention: caveman **lite** is the default communication style.

- Activate at session start: `/caveman lite`
- Other levels: `/caveman full|ultra|wenyan`
- Deactivate: `/caveman off` or "stop caveman" / "normal mode"
- While active, agent replies in terse caveman style (≈75% fewer tokens).
- Code, commits, security warnings: always normal English.
