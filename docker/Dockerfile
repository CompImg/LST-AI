# LST-AI, PyTorch inference. Multi-arch (linux/amd64, linux/arm64).
#
# Nothing is compiled here: PyTorch, HD-BET and greedy all install as wheels, which is
# what keeps this image cheap to build compared to the previous one (which built VTK,
# ITK and greedy from source in-image, ~40 min per rebuild).
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t lst-ai:torch -f docker/Dockerfile .
#
# On linux/arm64 there is no picsl-greedy wheel on PyPI (Kitware publishes the VTK
# wheel-SDK for x86_64 only, so greedy has to compile VTK from source). An unofficial
# aarch64 wheel is used automatically, selected to match the image's Python version
# (cp310-cp313 are published); override with --build-arg GREEDY_WHEEL=<url>.
#
# For GPU, swap the base image for nvidia/cuda:12.3.1-runtime-ubuntu22.04 and install a
# CUDA torch build instead of the CPU index below.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git is needed only if installing LST-AI straight from a repository URL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image an order of magnitude smaller than the default CUDA
# wheels. Installed first so it is cached independently of the source below.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# greedy: PyPI on amd64, an unofficial prebuilt wheel on arm64 (none is published
# upstream yet). Empty GREEDY_WHEEL forces PyPI on any architecture.
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

# Fetch the model bundle and atlas at build time, so the container runs offline and
# every run starts from the same weights.
RUN python -c "from LST_AI.utils import download_data; download_data()" || \
    echo "model download skipped; it will happen on first run"

WORKDIR /data
ENTRYPOINT ["lst"]
CMD ["--help"]
