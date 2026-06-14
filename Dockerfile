FROM python:3.14.3-slim

ENV UV_VERSION=0.11.21

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv==$UV_VERSION

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser assets/ ./assets/

USER appuser

EXPOSE 8501

ENTRYPOINT ["sh", "-c", "uv run streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]
