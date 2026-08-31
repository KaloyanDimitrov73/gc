FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

# Upgrade pip first to avoid deprecation warnings during subsequent installs
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

# 1. Install dependencies (cached unless pyproject.toml changes)

# Copy only dependency metadata first
COPY backend/HublinkImplementation/pyproject.toml backend/HublinkImplementation/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml

# Create minimal package stubs so "pip install -e" can resolve them.
# NOTE: keep in sync with the top-level packages in HublinkImplementation —
# the editable install only registers packages that exist at build time.
RUN install -d backend/HublinkImplementation/core \
               backend/HublinkImplementation/knowledge_base \
               backend/HublinkImplementation/language_model \
               backend/HublinkImplementation/retrieval \
               backend/app && \
    touch backend/HublinkImplementation/core/__init__.py \
          backend/HublinkImplementation/knowledge_base/__init__.py \
          backend/HublinkImplementation/language_model/__init__.py \
          backend/HublinkImplementation/retrieval/__init__.py \
          backend/__init__.py backend/app/__init__.py __init__.py

# Install dependencies (HublinkImplementation first for correct resolution)
RUN --mount=type=cache,target=/root/.cache/pip pip install -e backend/HublinkImplementation/
RUN --mount=type=cache,target=/root/.cache/pip pip install -e backend/

# Download the local models shipped by the public PyPI validator packages.
# The packages themselves are installed through backend/pyproject.toml above.
RUN python -m guardrails_ai.toxic_language.post_install && \
    python -m guardrails_ai.detect_jailbreak.post_install && \
    python -m guardrails_ai.gibberish_text.post_install

# 2. Copy source code (only this re-runs on code changes)
COPY . .

EXPOSE 8000
# The entrypoint is served from the bind-mounted source tree at runtime.
ENTRYPOINT ["/bin/sh", "/app/backend/docker-entrypoint.sh"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
