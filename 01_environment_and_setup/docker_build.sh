#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
"${SCRIPT_DIR}/host_preflight.sh"
ENV_FILE="${PROJECT_ROOT}/runtime/docker.env"

docker compose --env-file "${ENV_FILE}" -f "${PROJECT_ROOT}/docker/compose.yaml" build phase2
