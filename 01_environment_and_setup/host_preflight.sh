#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the currently mounted WOMD SSD on the host and create a stable
# project-local link. Docker and the Python code never need to know the
# changing /media/... path.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DATASET_NAME="waymo_open_dataset_v1_3_1"
DATASET_LINK="${PROJECT_ROOT}/.mounts/womd"
ENV_DIR="${PROJECT_ROOT}/runtime"
ENV_FILE="${ENV_DIR}/docker.env"
REPORT_DIR="${PROJECT_ROOT}/reports"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

dataset_root_ok() {
  local root="$1"
  [[ -d "${root}" ]] || return 1
  local split first_file
  for split in testing training validation; do
    [[ -d "${root}/${split}" ]] || return 1
    first_file="$(find "${root}/${split}" -type f \( -name '*.tfrecord' -o -name '*.tfrecord-*' -o -name '*.tfrecord*' \) -print -quit 2>/dev/null || true)"
    [[ -n "${first_file}" ]] || return 1
  done
  return 0
}

command -v docker >/dev/null 2>&1 || fail "docker 명령을 찾을 수 없습니다. Docker Engine을 먼저 설치하세요."
docker compose version >/dev/null 2>&1 || fail "docker compose 플러그인을 찾을 수 없습니다. 'docker compose version'이 동작해야 합니다."
docker info >/dev/null 2>&1 || fail "현재 사용자가 Docker daemon에 접근할 수 없습니다. Docker 서비스와 사용자 권한을 확인하세요."

mkdir -p "${PROJECT_ROOT}/.mounts" "${ENV_DIR}" "${REPORT_DIR}"

declare -a CANDIDATES=()
if [[ -n "${WOMD_HOST_ROOT_CANDIDATE:-}" ]]; then
  CANDIDATES+=("${WOMD_HOST_ROOT_CANDIDATE}")
fi
if [[ -n "${WOMD_HOST_ROOT:-}" ]]; then
  CANDIDATES+=("${WOMD_HOST_ROOT}")
fi
CANDIDATES+=(
  "/media/kiapi/28648BA9648B787810/${DATASET_NAME}"
  "/mnt/waymo_womd/${DATASET_NAME}"
  "/mnt/waymo_womd"
  "${HOME}/${DATASET_NAME}"
)

# udisks2 commonly uses one of these per-user roots. The scan is restricted to
# the dataset directory name and never scans the whole filesystem.
for media_base in "/media/${USER}" "/run/media/${USER}" "/media/kiapi"; do
  if [[ -d "${media_base}" ]]; then
    discovered="$(find "${media_base}" -maxdepth 4 -type d -name "${DATASET_NAME}" -print -quit 2>/dev/null || true)"
    [[ -n "${discovered}" ]] && CANDIDATES+=("${discovered}")
  fi
done

DATASET_ROOT=""
for candidate in "${CANDIDATES[@]}"; do
  [[ -n "${candidate}" ]] || continue
  if dataset_root_ok "${candidate}"; then
    DATASET_ROOT="$(realpath -e -- "${candidate}")"
    break
  fi
done

if [[ -z "${DATASET_ROOT}" ]]; then
  cat >&2 <<'EOF'
[FAIL] WOMD root를 찾지 못했습니다.
       testing/, training/, validation/ 디렉터리와 *.tfrecord* 파일이 모두 존재해야 합니다.
       현재 SSD를 마운트한 뒤 다음처럼 재실행하세요.

       WOMD_HOST_ROOT_CANDIDATE=/실제/waymo_open_dataset_v1_3_1 \
         ./scripts/host_preflight.sh

       원본 TFRecord를 프로젝트로 복사하지 마세요.
EOF
  exit 1
fi

[[ -r "${DATASET_ROOT}/training" ]] || fail "training 디렉터리를 읽을 수 없습니다: ${DATASET_ROOT}/training"
[[ -r "${DATASET_ROOT}/validation" ]] || fail "validation 디렉터리를 읽을 수 없습니다: ${DATASET_ROOT}/validation"
[[ -r "${DATASET_ROOT}/testing" ]] || fail "testing 디렉터리를 읽을 수 없습니다: ${DATASET_ROOT}/testing"

if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  fail "${DATASET_LINK}가 디렉터리/파일로 이미 존재합니다. 데이터를 삭제하지 않고 수동 확인이 필요합니다."
fi
ln -sfn -- "${DATASET_ROOT}" "${DATASET_LINK}"

OUTPUT_ROOT="${WOMD_OUTPUT_HOST_ROOT_CANDIDATE:-${WOMD_OUTPUT_HOST_ROOT:-${PROJECT_ROOT}/runtime/outputs}}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(realpath -m -- "${OUTPUT_ROOT}")"

HOST_UID_VALUE="$(id -u)"
HOST_GID_VALUE="$(id -g)"
INSTALL_MODEL_VALUE="${INSTALL_MODEL:-0}"

umask 077
tmp_env="${ENV_FILE}.tmp.$$"
{
  printf 'COMPOSE_PROJECT_NAME=waymo_motion_phase2\n'
  printf 'PROJECT_HOST_ROOT=%s\n' "${PROJECT_ROOT}"
  printf 'WOMD_HOST_ROOT=%s\n' "${DATASET_LINK}"
  printf 'WOMD_RESOLVED_HOST_ROOT=%s\n' "${DATASET_ROOT}"
  printf 'WOMD_OUTPUT_HOST_ROOT=%s\n' "${OUTPUT_ROOT}"
  printf 'HOST_UID=%s\n' "${HOST_UID_VALUE}"
  printf 'HOST_GID=%s\n' "${HOST_GID_VALUE}"
  printf 'INSTALL_MODEL=%s\n' "${INSTALL_MODEL_VALUE}"
} > "${tmp_env}"
mv -f -- "${tmp_env}" "${ENV_FILE}"

MOUNT_INFO="$(findmnt -T "${DATASET_ROOT}" -no SOURCE,FSTYPE,UUID,TARGET 2>/dev/null || true)"
DOCKER_GPU="false"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DOCKER_GPU="true"
fi

cat > "${REPORT_DIR}/host_preflight.txt" <<EOF
PASS: host preflight
project_root=${PROJECT_ROOT}
dataset_link=${DATASET_LINK}
dataset_resolved=${DATASET_ROOT}
mount_info=${MOUNT_INFO}
output_root=${OUTPUT_ROOT}
host_uid=${HOST_UID_VALUE}
host_gid=${HOST_GID_VALUE}
nvidia_smi_available=${DOCKER_GPU}
raw_data_policy=container_read_only
record_parse_performed=false
EOF

echo "[PASS] WOMD root: ${DATASET_ROOT}"
echo "[PASS] Stable container source: ${DATASET_LINK}"
echo "[PASS] Output root: ${OUTPUT_ROOT}"
echo "[PASS] Runtime env: ${ENV_FILE}"
echo "[PASS] Record parsing was not performed."
