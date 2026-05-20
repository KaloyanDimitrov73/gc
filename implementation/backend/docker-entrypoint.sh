#!/bin/sh
# Docker entrypoint script for the backend container.
#
# Writes ~/.guardrailsrc so guardrails can authenticate for remote inference.
# The file was deleted at image build time to avoid storing the token in an
# image layer; we recreate it here from the runtime GUARDRAILS_TOKEN env var.
#
# Hub validators are pip-installed at image build time and their registry is
# pre-committed at .guardrails/hub_registry.json, so no runtime hub install
# is needed here.
#
# Usage: Automatically invoked by Docker as the container entrypoint.
#        The main application command (e.g. uvicorn ...) is passed as "$@".

set -eu

if [ -n "${GUARDRAILS_TOKEN:-}" ]; then
  # Write ~/.guardrailsrc directly — guardrails has no env var fallback and
  # only reads this file. use_remote_inferencing defaults to true so it is
  # omitted. enable_metrics defaults to true so we disable it explicitly.
  printf 'token=%s\nenable_metrics=false\n' "$GUARDRAILS_TOKEN" > ~/.guardrailsrc
else
  echo "GUARDRAILS_TOKEN not set; Guardrails remote inference may fail."
fi

exec "$@"
