# Building the LST-AI Docker images

Most users do not need to build anything — release images are published to Docker Hub as
multi-arch manifests, see the [README](../README.md#dockerfile-and-dockerhub). This page
is for building the images yourself.

## One Dockerfile, two flavours, two architectures

There is a single [Dockerfile](Dockerfile) for the CUDA and the CPU flavour on both
`linux/amd64` and `linux/arm64` — no separate CPU file to drift out of sync. It defaults
to a CUDA base:

```bash
# GPU (default): nvidia/cuda:12.6.3-runtime-ubuntu22.04 + the cu126 torch wheels
docker build -f docker/Dockerfile -t lst-ai:gpu .

# CPU-only: ~4 GB instead of ~17 GB, and no NVIDIA runtime needed to run it
docker build -f docker/Dockerfile -t lst-ai:cpu \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

Run both commands from the repository root — the build context is the whole repository.
`TORCH_INDEX` must match the CUDA runtime in `BASE_IMAGE` and the host driver.

All weights — the LST-AI ensemble, the atlas, HD-BET's five folds and FastSurfer's three
VINN checkpoints — are baked in at build time, so a built container runs with no network
access (`docker run --network none` works). `greedy` installs from PyPI on both
architectures, pinned to one version via `--build-arg GREEDY_VERSION` so the two images
do not differ in which greedy produced the result.

## Building behind a TLS-inspecting proxy

Many hospital and university networks inspect HTTPS traffic, re-signing it with their own
root certificate. Your machine trusts it; a fresh Docker container does not, so the build
fails at the first download with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate`. The certificate has to be added manually:

**1. Copy your organisation's root certificate into `docker/certs/`**

That directory ships with the repository, empty. On Debian/Ubuntu hosts the certificate is
usually already installed locally:

```bash
cp /usr/local/share/ca-certificates/*.crt docker/certs/
```

Otherwise ask your IT department for it. Files must be PEM format
(they start with `-----BEGIN CERTIFICATE-----`) and end in `.crt`, or they are ignored.

**2. Build as usual**

```bash
docker build -f docker/Dockerfile -t lst-ai:gpu .
```

Everything you put in `docker/certs/` is gitignored, so your certificates cannot be
committed by accident. Do not push an image built this way to a public registry — it
trusts your organisation's CA.

## How releases are built

CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) builds all four
{cuda, cpu} × {amd64, arm64} variants on native runners for every push and PR, without
pushing. Releases are pushed by
[.github/workflows/docker.yml](../.github/workflows/docker.yml): each variant builds on
its native runner, pushes by digest, and the digests are stitched into one multi-arch
manifest per flavour. Pre-release tags never move `latest`/`latest-cpu`. The workflow
needs the `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` repository secrets and skips green
without them.
