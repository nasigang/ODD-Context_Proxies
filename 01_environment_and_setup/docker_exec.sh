#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
"${SCRIPT_DIR}/host_preflight.sh"
ENV_FILE="${PROJECT_ROOT}/runtime/docker.env"

exec docker compose --env-file "${ENV_FILE}" \
  -f "${PROJECT_ROOT}/docker/compose.yaml" \
  run --rm phase2 "$@"
