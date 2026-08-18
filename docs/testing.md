# Testing the PyTorch reimplementation

This is the checklist for validating LST-AI v2.0.0 before it goes to CompImg/LST-AI.
It exists because **the two architectures have been validated very unevenly**, and the
gaps are not obvious from a green CI badge.

If you are testing on **x86_64**, you want [Track B](#track-b--x86_64). On **aarch64**,
[Track A](#track-a--aarch64). Both tracks end with the same
[cross-machine comparison](#4-cross-machine-agreement), which is the interesting result.

## What is already verified, and how

| claim | how it was checked | status |
|---|---|---|
| PyTorch weights equal the released `.h5` | tensor-by-tensor, all three members | verified, exact |
| Shipped `.pt` equal the `.onnx` they came from | 304 state-dict tensors + forward pass, in CI on every push | verified, bitwise |
| Segmentation agrees with TensorFlow | 10 real subjects, full ensemble | Dice **0.99565**, volume **−0.12 %** |
| Package imports and unit-tests pass | CI, Python 3.10 and 3.12 | verified |
| Docker images build | CI, {amd64, arm64} × {cuda, cpu} | verified |
| Full pipeline runs end to end | manual, **aarch64 only** | verified on aarch64 |
| `pip install` → `lst` runs | manual, **aarch64 only** | verified on aarch64 |

## What is not verified — the point of this document

1. **The pipeline has never been run on x86_64.** Images build there; nothing has been
   segmented. This is the single biggest gap, because x86_64 is what almost every user
   is on.
2. **The CUDA image has never been run against a GPU.** It builds, and the code paths are
   exercised on CPU, but no GPU has executed it.
3. **CI does not segment a subject.** It builds and unit-tests only. A regression that
   breaks registration or skull-stripping would pass CI.
4. **FastSurfer annotation (`--annotate`) has had no end-to-end run** on either
   architecture; it is off by default in the image (`WITH_FASTSURFER=1` to include it).

## Before you start: expected variation

**Do not expect two runs to be identical, on the same machine or across machines.**
greedy's affine registration samples internally and is not seeded, so the pipeline has
never been bit-reproducible — this predates the PyTorch work and is not caused by it.

For calibration, measured on the same subject:

| comparison | Dice |
|---|---|
| PyTorch vs TensorFlow, same machine | 0.9957 |
| identical code, container vs virtualenv | 0.883 |

The container-vs-virtualenv number is the important one: **the environment moves the
result more than the framework swap does.** So when comparing x86_64 against aarch64,
treat roughly **Dice > 0.95** as agreement, and investigate anything below ~0.90. A Dice
near 1.0 would be surprising, not reassuring.

## Test data

Any T1 + FLAIR pair works. Use a subject you already have a TensorFlow LST-AI result for
if possible — that turns Track B into a direct regression test. Otherwise any subject
still exercises the pipeline and supports the cross-machine comparison.

```bash
export LST_IN=/absolute/path/to/input     # containing t1.nii.gz and flair.nii.gz
export LST_OUT=/absolute/path/to/output
mkdir -p "$LST_OUT"
```

Paths must be absolute — Docker bind mounts require it.

Two things about the CLI that are easy to get wrong:

- **`--output` is a directory, not a filename.** It is created if missing, and `lst`
  asserts if you point it at a file. The segmentation lands at
  `<output>/space-flair_seg-lst.nii.gz`, with `lesion_stats.csv` beside it (and
  `space-flair_desc-annotated_seg-lst.nii.gz` plus `annotated_lesion_stats.csv` when
  annotating). That filename is what you feed to `compare_segmentations.py`.
- **`--device` defaults to `0`, meaning GPU 0.** On a CPU-only machine you must pass
  `--device cpu` explicitly, or it will try CUDA and fail. It takes a device *index* or
  the literal string `cpu` — not `cuda`.

---

## Track A — aarch64

### A1. Install from source

`picsl-greedy` publishes no linux-aarch64 wheel, so greedy comes from a prebuilt wheel.
Match the tag to your Python version (`cp310`–`cp313`):

```bash
python3 -m venv lst_env && source lst_env/bin/activate
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"

pip install https://github.com/jqmcginnis/greedy_python/releases/download/v1.4.0-aarch64.1/picsl_greedy-1.4.0-cp312-cp312-linux_aarch64.whl
git clone https://github.com/jqmcginnis/LST-AI.git && cd LST-AI
pip install -e .
```

### A2. Run the unit tests

```bash
pip install pytest
pytest tests -q -m "not needs_weights"     # architecture, no download
pytest training/tests -q                   # dataset, losses, trainer
```

Expect `12 passed, 9 deselected` and `78 passed` — 90 in total. The 9 deselected are the
`needs_weights` tests, which additionally need the model bundle; see
[weights parity](#weights-parity-optional-needs-the-bundle) to run those too (99 in all).

### A3. Segment a subject

```bash
lst --t1 "$LST_IN/t1.nii.gz" --flair "$LST_IN/flair.nii.gz" \
    --output "$LST_OUT/arm64" --temp "$LST_OUT/arm64_temp" --device cpu
```

First run downloads ~176 MB of weights plus HD-BET's parameters. Result:
`$LST_OUT/arm64/space-flair_seg-lst.nii.gz`. Then continue to [Docker](#3-docker) and
[cross-machine](#4-cross-machine-agreement).

---

## Track B — x86_64

**This track has never been run. Everything below is untested on this architecture — a
failure here is a real finding, not a mistake on your part.** Please report what happens
either way, including success.

### B1. Install from source

x86_64 needs no wheel workaround; greedy comes straight from PyPI.

```bash
python3 -m venv lst_env && source lst_env/bin/activate
git clone https://github.com/jqmcginnis/LST-AI.git && cd LST-AI
pip install -e .
```

For a GPU box, install the CUDA build of torch first, matching your driver:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

### B2. Run the unit tests

```bash
pip install pytest
pytest tests -q -m "not needs_weights"
pytest training/tests -q
```

### B3. Segment a subject — CPU

```bash
lst --t1 "$LST_IN/t1.nii.gz" --flair "$LST_IN/flair.nii.gz" \
    --output "$LST_OUT/x86_cpu" --temp "$LST_OUT/x86_cpu_temp" --device cpu
```

### B4. Segment a subject — GPU

`--device` takes a CUDA device *index*, not the string `cuda`:

```bash
lst --t1 "$LST_IN/t1.nii.gz" --flair "$LST_IN/flair.nii.gz" \
    --output "$LST_OUT/x86_gpu" --temp "$LST_OUT/x86_gpu_temp" --device 0
```

Worth watching `nvidia-smi` during the run. The move off ONNX Runtime was partly to fix
memory: ORT's CUDA arena transiently grabbed ~40 GB at session init and OOM'd when
sharing a GPU. **Peak usage should now be a few GB.** If you see anything approaching
tens of GB, that is a regression worth reporting.

### B5. Compare CPU against GPU

```bash
python scripts/compare_segmentations.py \
    "$LST_OUT/x86_cpu/space-flair_seg-lst.nii.gz" \
    "$LST_OUT/x86_gpu/space-flair_seg-lst.nii.gz"
```

CPU and GPU will not agree exactly — different kernels, different reduction orders — but
should be well within the tolerances above.

---

## 3. Docker

One Dockerfile covers both flavours and both architectures. All weights are baked in, so
the container needs no network at run time.

```bash
# CPU (~2 GB)
docker build -f docker/Dockerfile -t lst-ai:cpu \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .

# CUDA (~7 GB)
docker build -f docker/Dockerfile -t lst-ai:gpu .
```

Run them:

```bash
docker run --rm -v "$LST_IN":/in -v "$LST_OUT":/out lst-ai:cpu \
  --t1 /in/t1.nii.gz --flair /in/flair.nii.gz --output /out/docker_cpu --device cpu

docker run --rm --gpus all -v "$LST_IN":/in -v "$LST_OUT":/out lst-ai:gpu \
  --t1 /in/t1.nii.gz --flair /in/flair.nii.gz --output /out/docker_gpu --device 0
```

Confirm the image really is offline-capable — this is the claim that was wrong once
already, when `download_data()` was called without its required argument and the error was
swallowed, shipping an image with no weights at all:

```bash
docker run --rm --network none -v "$LST_IN":/in -v "$LST_OUT":/out lst-ai:cpu \
  --t1 /in/t1.nii.gz --flair /in/flair.nii.gz --output /out/offline --device cpu
```

(Everything after the image name is passed to `lst`, so the `-v` mounts have to come
before `lst-ai:cpu`.)

### Weights parity (optional, needs the bundle)

Confirms the shipped `.pt` reproduce the ONNX graphs bit-for-bit — the same check CI runs:

```bash
curl -fsSL -o lst_data.zip https://github.com/jqmcginnis/LST-AI/releases/download/v2.0.0/lst_data_onnx.zip
unzip -q lst_data.zip -d lst_data
pip install 'pytest' 'onnx'
LST_AI_MODEL_DIR=lst_data/model pytest tests -q -m needs_weights
```

---

## 4. Cross-machine agreement

The result worth collecting. Run the *same subject* on x86_64 and aarch64, then:

```bash
python scripts/compare_segmentations.py \
    /path/from/x86_64/space-flair_seg-lst.nii.gz \
    /path/from/aarch64/space-flair_seg-lst.nii.gz
```

Output:

```
Dice                0.9xxxxx
lesion volume       x.xxx mL (reference) vs x.xxx mL
volume difference   +x.xx %
lesion count        N vs M
voxels differing    K
```

Judge against the [table above](#before-you-start-expected-variation): **> 0.95 is
agreement**; below ~0.90 wants investigation.

If you have a TensorFlow LST-AI result for the same subject, compare against that too —
that is the number users actually care about, and the claim to hold us to is Dice ≈ 0.996.

## Reporting

Please open an issue on `jqmcginnis/LST-AI` with:

- architecture, OS, Python version, GPU if any
- which track and which steps, and the `compare_segmentations.py` output
- for failures: the full traceback, plus `pip list` (`torch`, `picsl-greedy`,
  `brainles_hd_bet` versions are the ones that matter)
- for GPU runs: peak memory from `nvidia-smi`

Successful runs are worth reporting too — for x86_64 there is currently no data point at
all, so "it worked" is genuinely new information.
