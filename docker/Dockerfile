# LST-AI, PyTorch inference. Multi-arch (linux/amd64, linux/arm64), CUDA by default.
#
# Every model weight is baked in at build time -- LST-AI's ensemble and atlas, HD-BET's
# parameters, and FastSurfer's VINN checkpoints -- so a built image runs without network
# access. Nothing is compiled: torch, greedy and HD-BET all install as wheels, and
# FastSurfer is a clone plus pure-Python (or prebuilt-wheel) dependencies.
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

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# The CUDA base images ship no Python; python:*-slim already has one. One code path
# covers both. bc is for run_fastsurfer.sh, which uses it to range-check a numeric
# --vox_size (not the path LST-AI takes, but it is a stock part of that script).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates bc \
    && if ! command -v python >/dev/null 2>&1; then \
           apt-get install -y --no-install-recommends python3 python3-pip && \
           ln -sf /usr/bin/python3 /usr/bin/python; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Corporate/TLS-inspecting proxies re-sign HTTPS with a private root the base image
# doesn't trust. Drop any such roots in docker/certs/ (*.crt, PEM format) and they
# get installed here. Empty directory on normal networks -- this is a no-op.
COPY docker/certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

# pip bundles its own certifi store and ignores the system one, hence PIP_CERT.
# The rest cover requests, raw ssl/urllib and git for the weight downloads below.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# torch first, so it caches independently of the source tree below. torchvision is
# resolved in the same command and from the same index, so the pair matches this
# image's CUDA build; FastSurfer's inference module imports it.
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

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

# --- FastSurfer --------------------------------------------------------------------
#
# `lst` annotates lesions by anatomical region in every mode except --segment_only, and
# that annotation *is* a FastSurfer segmentation of the T1 (LST_AI/annotate.py shells
# out to run_fastsurfer.sh). It is therefore part of the standard image on both
# architectures, not an opt-in extra -- without it the default invocation fails halfway
# through, after the segmentation has already been computed.
#
# Laid out the way upstream's own container lays it out: a clone plus FASTSURFER_HOME,
# deliberately not `pip install`ed. run_fastsurfer.sh puts $FASTSURFER_HOME at the front
# of PYTHONPATH itself, so a site-packages copy would never be the one that runs, while
# the checkpoints below live next to the clone -- one root, no split brain.
ARG FASTSURFER_REF=v2.5.4
RUN git clone --depth 1 --branch ${FASTSURFER_REF} \
        https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer
ENV FASTSURFER_HOME=/opt/FastSurfer \
    PATH="/opt/FastSurfer:${PATH}"

# FastSurfer's dependencies for the one code path LST-AI drives
# (--seg_only --no_cereb --no_hypothal --no_cc --no_biasfield, i.e. run_prediction.py
# and reduce_to_aseg.py), at upstream's version floors. Enumerated rather than
# `pip install /opt/FastSurfer`, because its full metadata does two things this image
# cannot accept:
#   * it pins torch==2.7.*, which has no aarch64 CUDA wheel at all, so honouring it
#     would quietly swap the arm64 GPU image's torch for a CPU build;
#   * it pulls meshpy, published as an x86_64-only wheel (an sdist needing a C++
#     toolchain everywhere else), for the corpus-callosum module --no_cc turns off.
# When bumping FASTSURFER_REF, diff this against its pyproject.toml `dependencies`;
# the import check below is what turns a missed addition into a failed build rather
# than a failed run.
RUN pip install \
        'h5py>=3.7' \
        'lapy>=1.5.0' \
        'neuroreg>=0.6.2' \
        'nibabel>=5.4.0' \
        'numpy>=1.25' \
        'pandas>=1.5.3' \
        'pyyaml>=6.0' \
        'requests>=2.31.0' \
        'scikit-image>=0.19.3' \
        'scipy>=1.10.1,!=1.13.0' \
        'torchio>=0.18.83' \
        'tqdm>=4.65' \
        'yacs>=0.1.8' \
    && PYTHONPATH=/opt/FastSurfer python -c "import FastSurferCNN.run_prediction, FastSurferCNN.reduce_to_aseg; \
print('FastSurfer segmentation path imports cleanly')"

# The three VINN checkpoints (axial/coronal/sagittal), which are all this path loads:
# CerebNet, HypVINN and the corpus-callosum weights belong to the modules LST-AI
# switches off. Asserted for the same reason as the LST-AI weights below -- a silent
# miss ships an image that only fails once someone actually annotates.
RUN cd /opt/FastSurfer \
 && PYTHONPATH=/opt/FastSurfer python FastSurferCNN/download_checkpoints.py --vinn \
 && PYTHONPATH=/opt/FastSurfer python -c "\
from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults; \
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT; \
paths = load_checkpoint_config_defaults('checkpoint', get_config_file('FastSurferCNN')).values(); \
ckpts = [FASTSURFER_ROOT / p for p in paths]; \
missing = [str(p) for p in ckpts if not p.is_file()]; \
assert not missing, f'FastSurfer checkpoints missing: {missing}'; \
print('FastSurfer VINN checkpoints:', sorted(p.name for p in ckpts))"

# run_fastsurfer.sh reports its version on every run, shelling out to git and pip to do
# it unless a BUILD.info is there to read instead. Bake one, then the clone's git
# metadata is dead weight.
RUN cd /opt/FastSurfer \
 && PYTHONPATH=/opt/FastSurfer python FastSurferCNN/version.py --sections all -o BUILD.info \
 && rm -rf /opt/FastSurfer/.git

WORKDIR /opt/lst-ai
COPY . .
RUN pip install .

# --- weights, fetched at build time so the container needs no network -------------

# LST-AI ensemble + atlas, into LST_AI.utils.DATA_DIR -- the same constant the CLI reads
# them back from, so the two cannot drift apart.
#
# Run from / and not from WORKDIR: `python -c` puts the cwd first on sys.path, so under
# /opt/lst-ai `import LST_AI` would bind to the copied sources rather than the installed
# package and bake ~400 MB into a directory nothing ever reads. The assert below pins
# that down, since the failure is invisible -- the image looks fine and simply
# re-downloads the bundle on every single run.
RUN cd / && python -c "import os; from LST_AI.utils import DATA_DIR, download_data; \
assert not DATA_DIR.startswith('/opt/lst-ai'), f'baked into the sources, not the install: {DATA_DIR}'; \
download_data(path=DATA_DIR); \
assert os.path.isdir(os.path.join(DATA_DIR, 'model')), 'model directory missing'; \
assert os.path.isdir(os.path.join(DATA_DIR, 'atlas')), 'atlas directory missing'; \
print('LST-AI weights in', DATA_DIR, sorted(os.listdir(os.path.join(DATA_DIR, 'model'))))"

# HD-BET parameters (one per fold), otherwise fetched on first run.
RUN python -c "import os; \
from brainles_hd_bet.utils import maybe_download_parameters, get_params_fname; \
[maybe_download_parameters(f) for f in range(5)]; \
print('HD-BET params:', [os.path.basename(get_params_fname(f)) for f in range(5)])"

WORKDIR /data
ENTRYPOINT ["lst"]
CMD ["--help"]
