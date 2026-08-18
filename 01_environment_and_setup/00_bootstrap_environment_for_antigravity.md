# Antigravity Bootstrap Prompt — run before Prompt 0

당신은 Ubuntu 20.04 호스트에서 실행되는 재현 가능한 Docker 기반 WOMD Motion v1.3.1 연구환경을 구축하는 엔지니어다.

## 목표

기존 Prompt 0을 실행하기 전에 다음을 완료하라.

1. 프로젝트를 내부 SSD의 `~/waymo_motion_project`에 둔다.
2. 외장 SSD의 WOMD root를 현재 mount 상태에서 찾는다.
3. 원본 `testing/`, `training/`, `validation/` 폴더를 Docker 안의 `/mnt/womd`에 read-only로 연결한다.
4. Docker 이미지에서 Python 3.10, TensorFlow 2.12, Waymo Open Dataset proto, pandas, PyArrow를 import할 수 있게 한다.
5. 컨테이너에서 실제 TFRecord record를 parse하지 않고, split 디렉터리와 파일 접근성만 확인한다.
6. 모든 후속 코드는 호스트의 `/media/...` 경로가 아니라 컨테이너의 `/mnt/womd` 경로를 사용하게 한다.

## 강제 제약

- 사용자의 Ubuntu 호스트에 이미 존재하는 파일이나 원본 데이터는 삭제·이동·복사하지 마라.
- `/media/kiapi/28648BA9648B787810/waymo_open_dataset_v1_3_1/`를 프로젝트로 복사하지 마라.
- 기존 Phase 1 Perception 코드를 `phase2_womd/`로 복사하거나 import하지 마라.
- `dataset_pb2.Frame`을 이용한 데이터 parse, `scenario_pb2.Scenario.FromString`, `tf.train.Example.ParseFromString`은 이 단계에서 실행하지 마라. 그것은 Prompt 0의 책임이다.
- `/etc/fstab`을 자동으로 수정하지 마라. 현재 mount를 찾지 못하면 중단하고 사용자에게 실제 mount 경로를 요청하라.
- Docker, NVIDIA Toolkit, filesystem 권한 문제를 임의의 Python 코드 수정으로 우회하지 마라.
- `sudo`가 필요하면 실행하지 말고 정확한 명령과 이유를 보고한 뒤 중단하라.

## 작업 디렉터리

반드시 다음 디렉터리에서 작업하라.

```text
~/waymo_motion_project
```

이 템플릿의 파일이 아직 없다면, 현재 템플릿 내용을 이 디렉터리에 배치한 뒤 그 디렉터리를 Antigravity workspace로 연다. `/media/...`를 workspace로 열지 마라.

## 실행 순서

### 1. 호스트 사전점검

다음 정보를 먼저 확인하라.

```bash
uname -a
lsb_release -a || true
docker version
docker compose version
findmnt -T /media/kiapi/28648BA9648B787810/waymo_open_dataset_v1_3_1 || true
lsblk -f
nvidia-smi || true
```

Docker daemon 또는 Compose plugin이 없으면 설치를 시작하지 말고 실패 원인과 공식 설치 문서를 보고하라.

### 2. 데이터 root 탐색

다음 스크립트를 실행하라.

```bash
cd ~/waymo_motion_project
chmod +x scripts/*.sh
./scripts/host_preflight.sh
```

스크립트가 수행해야 하는 검사는 다음과 같다.

- `testing/`, `training/`, `validation/` 디렉터리가 모두 존재하는가
- 각 split 아래 `*.tfrecord*` 파일이 하나 이상 존재하는가
- 현재 사용자가 파일을 읽을 수 있는가
- `.mounts/womd`가 현재 실제 dataset root를 가리키는가
- `runtime/docker.env`가 생성되었는가
- 원본 root가 Docker에서 read-only로 연결될 준비가 되었는가

찾지 못하면 중단하라. 사용자가 다음처럼 직접 후보 경로를 지정할 수 있다는 안내만 하라.

```bash
WOMD_HOST_ROOT_CANDIDATE=/실제/waymo_open_dataset_v1_3_1 \
  ./scripts/host_preflight.sh
```

### 3. 이미지 빌드

Prompt 0의 parser 환경만 먼저 빌드하라.

```bash
cd ~/waymo_motion_project
INSTALL_MODEL=0 ./scripts/docker_build.sh
```

이 단계에서 `INSTALL_MODEL=1`을 기본값으로 바꾸지 마라. 모델 패키지는 parser identity audit이 통과한 뒤 별도로 추가한다.

### 4. 컨테이너 import 및 mount 검증

```bash
cd ~/waymo_motion_project
./scripts/docker_exec.sh python scripts/verify_container.py \
  --output reports/container_environment.json
```

다음이 모두 만족되어야 PASS다.

- Python 3.10
- TensorFlow import 성공
- `waymo_open_dataset.dataset_pb2` import 성공
- `waymo_open_dataset.protos.scenario_pb2` import 성공
- pandas와 PyArrow import 성공
- 세 split이 컨테이너의 `/mnt/womd`에서 보임
- 각 split의 TFRecord 파일 수가 0보다 큼
- 결과 파일을 `/mnt/phase2_outputs`에 쓸 수 있음
- `record_parse_performed=false`

실패하면 Prompt 0을 실행하지 말고, 누락된 import·경로·권한을 정확히 기록하라.

## 성공 시 보고 형식

마지막에 다음만 보고하라.

```text
BOOTSTRAP_STATUS=PASS|FAIL
PROJECT_HOST_ROOT=...
WOMD_RESOLVED_HOST_ROOT=...
CONTAINER_WOMD_ROOT=/mnt/womd
TRAINING_TFRECORD_COUNT=...
VALIDATION_TFRECORD_COUNT=...
TESTING_TFRECORD_COUNT=...
WAYMO_SCENARIO_PROTO_IMPORT=PASS|FAIL
WAYMO_PERCEPTION_PROTO_IMPORT=PASS|FAIL
RECORD_PARSE_PERFORMED=false
NEXT_STEP=Run Prompt 0 inside the phase2 container only when PASS
```

`BOOTSTRAP_STATUS=PASS`일 때만 기존 Prompt 0을 실행하라.
