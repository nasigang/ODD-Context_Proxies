FROM python:3.10-slim-bullseye

ARG INSTALL_MODEL=0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/waymo_motion_project

COPY requirements/phase2-core.txt /tmp/phase2-core.txt
COPY requirements/phase2-model.txt /tmp/phase2-model.txt

RUN python -m pip install --upgrade \
    "pip==24.0" \
    "setuptools==69.5.1" \
    "wheel==0.43.0" \
    && python -m pip install --no-cache-dir -r /tmp/phase2-core.txt \
    && if [ "${INSTALL_MODEL}" = "1" ]; then \
         python -m pip install --no-cache-dir -r /tmp/phase2-model.txt; \
       fi \
    && python -m pip check

CMD ["bash"]
