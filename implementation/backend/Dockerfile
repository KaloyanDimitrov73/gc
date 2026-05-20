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

# Create minimal package stubs so "pip install -e" can resolve them
RUN install -d backend/HublinkImplementation/sqa_system backend/app && \
    touch backend/HublinkImplementation/sqa_system/__init__.py \
          backend/__init__.py backend/app/__init__.py __init__.py

# Install dependencies (HublinkImplementation first for correct resolution)
RUN --mount=type=cache,target=/root/.cache/pip pip install -e backend/HublinkImplementation/
RUN --mount=type=cache,target=/root/.cache/pip pip install -e backend/

# Install Guardrails hub validators using a BuildKit secret so the token is
# never baked into an image layer. --no-install-local-models skips downloading
# ML models since remote inference is used.
# The pip cache mount avoids re-downloading packages when this layer is invalidated.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=guardrails_token \
    GUARDRAILS_TOKEN="$(cat /run/secrets/guardrails_token)" && \
    guardrails configure --token "$GUARDRAILS_TOKEN" --disable-metrics --enable-remote-inferencing && \
    guardrails hub install hub://guardrails/toxic_language --no-install-local-models --quiet && \
    guardrails hub install hub://guardrails/detect_jailbreak --no-install-local-models --quiet && \
    guardrails hub install hub://guardrails/gibberish_text --no-install-local-models --quiet && \
    rm -f /root/.guardrailsrc

# 2. Copy source code (only this re-runs on code changes)
COPY . .

EXPOSE 8000

# The entrypoint is served from the bind-mounted source tree at runtime.
ENTRYPOINT ["/bin/sh", "/app/backend/docker-entrypoint.sh"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
