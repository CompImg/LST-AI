# LST-AI, PyTorch inference. Multi-arch (linux/amd64, linux/arm64), CUDA by default.
#
# Every model weight is baked in at build time -- LST-AI's ensemble and atlas, HD-BET's
# parameters, and FastSurfer's VINN checkpoints -- so a built image runs without network
# access. Nothing is compiled: torch, greedy and HD-BET all install as wheels, and
# FastSurfer is a source tarball plus pure-Python (or prebuilt-wheel) dependencies.
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
# greedy installs from PyPI on both architectures (picsl-greedy ships linux/aarch64
# wheels since 1.4.0.1), pinned below via GREEDY_VERSION.

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
# The rest cover requests, raw ssl/urllib and git for the downloads below.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# torch first, so it caches independently of the source tree below. torchvision is
# resolved in the same command and from the same index, so the pair matches this
# image's CUDA build; FastSurfer's inference module imports it.
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

# greedy: official PyPI wheels on both architectures since 1.4.0.1, which is the first
# release with linux/aarch64 wheels (pyushkevich/greedy_python#6). Pinned to one version
# for both images -- registration is the pipeline's largest source of run-to-run
# variation, so the two images should at least not differ in which greedy produced the
# result.
ARG GREEDY_VERSION=1.4.0.1
RUN pip install "picsl-greedy==${GREEDY_VERSION}"

# --- FastSurfer --------------------------------------------------------------------
#
# `lst` annotates lesions by anatomical region in every mode except --segment_only, and
# that annotation *is* a FastSurfer segmentation of the T1 (LST_AI/annotate.py shells
# out to run_fastsurfer.sh). It is therefore part of the standard image on both
# architectures, not an opt-in extra -- without it the default invocation fails halfway
# through, after the segmentation has already been computed.
#
# Installed through LST-AI's own installer, LST_AI/fastsurfer.py, rather than cloned to a
# path of this image's choosing: that module is where the version is pinned and where the
# tree's location is decided, and it is the only place LST-AI looks. A hand-placed copy
# plus FASTSURFER_HOME would simply be ignored at run time.
#
# Only that one file is copied in, so this whole stage keeps its layer cache across
# changes to the rest of the sources. It is stdlib-only and runs standalone, which is
# what makes that possible.

# FastSurfer's dependencies for the one code path LST-AI drives
# (--seg_only --no_cereb --no_hypothal --no_cc --no_biasfield, i.e. run_prediction.py
# and reduce_to_aseg.py), at upstream's version floors. Enumerated rather than
# `pip install`ing FastSurfer itself, because its full metadata does two things this
# image cannot accept:
#   * it pins torch==2.7.*, which has no aarch64 CUDA wheel at all, so honouring it
#     would quietly swap the arm64 GPU image's torch for a CPU build;
#   * it pulls meshpy, published as an x86_64-only wheel (an sdist needing a C++
#     toolchain everywhere else), for the corpus-callosum module --no_cc turns off.
# When bumping the pin in LST_AI/fastsurfer.py, diff this against upstream's
# pyproject.toml `dependencies`, and move that module's FASTSURFER_REQUIRES with it --
# that is the same list, for pip installs. It is repeated here rather than imported so
# this layer stays independent of the source tree; the import check below is what turns a
# missed addition into a failed build rather than a failed run.
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
        'yacs>=0.1.8'

# The tree itself, plus the three VINN checkpoints (axial/coronal/sagittal), which are
# all this path loads: CerebNet, HypVINN and the corpus-callosum weights belong to the
# modules LST-AI switches off. Baking them in is what lets the container annotate with no
# network; a pip install fetches them on its first annotation instead.
#
# Everything after the install is assertion: that the segmentation path imports, and that
# the checkpoints are really on disk. Both failures are otherwise invisible at build time
# and only surface once someone actually annotates. BUILD.info last, so run_fastsurfer.sh
# reads its version from a file instead of shelling out to pip on every run. Two traps
# there: `--sections all` demands `+git`, which needs a .git directory a release tarball
# does not carry, and exits 1 without one; and `-o BUILD.info` writes the very file
# version.py reads as its build cache, truncating it before the read and failing to parse
# the empty result -- hence writing elsewhere and moving it into place.
COPY LST_AI/fastsurfer.py /tmp/lst_ai_fastsurfer.py
RUN python /tmp/lst_ai_fastsurfer.py --checkpoints \
 && FASTSURFER_HOME="$(python /tmp/lst_ai_fastsurfer.py | sed -n 's/^FASTSURFER_HOME=//p')" \
 && export PYTHONPATH="$FASTSURFER_HOME" \
 && python -c "import FastSurferCNN.run_prediction, FastSurferCNN.reduce_to_aseg; \
print('FastSurfer segmentation path imports cleanly')" \
 && python -c "\
from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults; \
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT; \
paths = load_checkpoint_config_defaults('checkpoint', get_config_file('FastSurferCNN')).values(); \
ckpts = [FASTSURFER_ROOT / p for p in paths]; \
missing = [str(p) for p in ckpts if not p.is_file()]; \
assert not missing, f'FastSurfer checkpoints missing: {missing}'; \
print('FastSurfer VINN checkpoints:', sorted(p.name for p in ckpts))" \
 && cd "$FASTSURFER_HOME" \
 && python FastSurferCNN/version.py --sections +checkpoints+pip -o /tmp/BUILD.info \
 && mv /tmp/BUILD.info "$FASTSURFER_HOME/BUILD.info" \
 && rm /tmp/lst_ai_fastsurfer.py

WORKDIR /opt/lst-ai
COPY . .
# setup.py installs FastSurfer itself when it is missing (LST_AI/fastsurfer.py), which is
# what makes a plain `pip install` a complete one. Here it is a no-op: the stage above
# already put it where this same module looks for it. Splitting the two is deliberate --
# letting this layer do the work instead would re-download FastSurfer whenever the
# sources change, and would leave the VINN checkpoints out of the image.
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
