#!/bin/sh
# Docker entrypoint script for the backend container.
#
# Configure Guardrails to use the local validator models baked into the image.
# Hosted remote inference has been retired, and anonymous metrics are disabled
# so validation does not make outbound Guardrails requests.

set -eu

printf 'enable_metrics=false\nuse_remote_inferencing=false\n' > ~/.guardrailsrc

exec "$@"
