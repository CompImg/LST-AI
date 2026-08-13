# LST-AI, PyTorch inference. Multi-arch (linux/amd64, linux/arm64), CUDA by default.
#
# Every model weight is baked in at build time -- LST-AI's ensemble and atlas, HD-BET's
# parameters, and FastSurfer's if enabled -- so a built image runs without network
# access. Nothing is compiled: torch, greedy and HD-BET all install as wheels.
#
#   # GPU (default)
#   docker buildx build --platform linux/amd64,linux/arm64 -t lst-ai:torch -f docker/Dockerfile .
#
#   # CPU-only: smaller image, no CUDA runtime
#   docker build -f docker/Dockerfile \
#       --build-arg BASE_IMAGE=python:3.11-slim \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t lst-ai:torch-cpu .
#
# TORCH_INDEX must match the CUDA runtime in BASE_IMAGE and the host driver
# (cu121/cu124/cu126/cu128...). Running the GPU image also needs `--gpus all`.
#
# On linux/arm64 there is no picsl-greedy wheel on PyPI (Kitware publishes the VTK
# wheel-SDK for x86_64 only, so greedy must compile VTK from source). An unofficial
# prebuilt wheel matching the image's Python version is used automatically; override
# with --build-arg GREEDY_WHEEL=<url>, or pass an empty one to force PyPI.

ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG WITH_FASTSURFER=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# The CUDA base images ship no Python; python:*-slim already has one. One code path
# covers both.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates \
    && if ! command -v python >/dev/null 2>&1; then \
           apt-get install -y --no-install-recommends python3 python3-pip && \
           ln -sf /usr/bin/python3 /usr/bin/python; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# torch first, so it caches independently of the source tree below.
RUN pip install --index-url ${TORCH_INDEX} torch

# greedy: PyPI on amd64; on arm64 an unofficial prebuilt wheel matching this image's
# Python version, since none is published upstream.
ARG GREEDY_WHEEL=""
ARG GREEDY_WHEEL_BASE="https://github.com/jqmcginnis/greedy_python/releases/download/v1.4.0-aarch64.1"
RUN set -eu; \
    if [ "$(uname -m)" != "aarch64" ]; then \
        pip install picsl-greedy; \
    else \
        if [ -z "$GREEDY_WHEEL" ]; then \
            TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"; \
            GREEDY_WHEEL="$GREEDY_WHEEL_BASE/picsl_greedy-1.4.0-$TAG-$TAG-linux_aarch64.whl"; \
        fi; \
        echo "arm64: installing greedy from $GREEDY_WHEEL"; \
        pip install "$GREEDY_WHEEL"; \
    fi

WORKDIR /opt/lst-ai
COPY . .
RUN pip install .

# --- weights, fetched at build time so the container needs no network -------------

# LST-AI ensemble + atlas. download_data() takes the directory to populate, and the
# result is asserted: letting this fail silently ships an image with no weights that
# only appears to work when a host directory is bind-mounted over the gap.
RUN python -c "import os, LST_AI; from LST_AI.utils import download_data; \
p = os.path.dirname(LST_AI.__file__); download_data(path=p); \
assert os.path.isdir(os.path.join(p, 'model')), 'model directory missing'; \
assert os.path.isdir(os.path.join(p, 'atlas')), 'atlas directory missing'; \
print('LST-AI weights:', sorted(os.listdir(os.path.join(p, 'model'))))"

# HD-BET parameters (one per fold), otherwise fetched on first run.
RUN python -c "import os; \
from brainles_hd_bet.utils import maybe_download_parameters, get_params_fname; \
[maybe_download_parameters(f) for f in range(5)]; \
print('HD-BET params:', [os.path.basename(get_params_fname(f)) for f in range(5)])"

# FastSurfer is needed only for the annotation stage and is a large extra, so it is
# opt-in via --build-arg WITH_FASTSURFER=1. Its shell entry point (run_fastsurfer.sh)
# lives in the repo rather than the wheel, hence the clone as well as the install.
RUN if [ "$WITH_FASTSURFER" = "1" ]; then \
        git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer && \
        pip install /opt/FastSurfer && \
        python -c "import FastSurferCNN; print('FastSurfer installed')"; \
    else \
        echo "FastSurfer not installed; build with --build-arg WITH_FASTSURFER=1 to annotate"; \
    fi
ENV PATH="/opt/FastSurfer:${PATH}"

WORKDIR /data
ENTRYPOINT ["lst"]
CMD ["--help"]
