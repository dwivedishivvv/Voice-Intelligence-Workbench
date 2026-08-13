FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

COPY api/pyproject.toml api/pyproject.toml
RUN cd api && uv pip install --system -r pyproject.toml

COPY common/ common/
COPY api/app/ api/app/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
