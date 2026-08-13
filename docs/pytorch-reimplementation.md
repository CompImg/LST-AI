# PyTorch reimplementation — scope and validation

LST-AI's segmentation stage now runs natively in PyTorch. TensorFlow and ONNX Runtime are
both gone; the released weights are unchanged.

## Why

The v1.0.0/v1.1.0 releases ran inference in TensorFlow. Getting off TensorFlow was the
goal, and an ONNX Runtime backend was the first attempt — but measured against the
TensorFlow results users actually have, ONNX Runtime drifts substantially, while PyTorch
does not. PyTorch is also the framework new models are trained in, so one code path now
serves both the released weights and anything trained next.

## What changed

| | before | after |
|---|---|---|
| inference | TensorFlow / ONNX Runtime | native PyTorch (`LST_AI/model.py`) |
| weights | `.h5` / `.onnx` | unchanged — read from the shipped `.onnx`, or `.pt` if present |
| registration | greedy | greedy (unchanged) |
| skull stripping | HD-BET | HD-BET (unchanged, already PyTorch) |
| pre/post-processing | — | unchanged |

`onnx` remains a dependency, but only as a protobuf reader for the shipped weights — no
inference runtime other than PyTorch is involved. `--backend` is gone from the CLI.

## The weights are the released ones

Every convolution kernel, gamma and beta in the PyTorch model equals the corresponding
tensor in the released `.h5` **bit-for-bit** (max |Δ| = 0.000e+00, all three ensemble
members), checked against `lst_data.zip` from CompImg/LST-AI v1.1.0. Nothing was
retrained, refitted or approximated.

Note the `.h5` needs Keras 2 to load (TF ≥ 2.16 / Keras 3 rejects it — `Conv3DTranspose`
carries a `groups: 1` key Keras 3 does not accept). Use `tf-keras` with
`TF_USE_LEGACY_KERAS=1` if you need to reproduce this check.

## Numerics

The released models were trained on TensorFlow with `tfa.layers.InstanceNormalization`,
whose defaults are **not** PyTorch's. These are pinned, and changing any of them breaks
the weights:

| | value | PyTorch default |
|---|---|---|
| LeakyReLU slope | 0.3 | 0.01 |
| instance-norm epsilon | 1e-3 | 1e-5 |
| instance-norm affine | yes | `InstanceNorm3d` → no |
| conv bias | absent | present |

## Results are not bit-identical, and cannot be

No framework swap can promise bit-identical output — kernels and reduction orders differ.
What matters is how far apart they land, and against the TensorFlow baseline PyTorch is
much closer than ONNX Runtime:

| vs released TensorFlow | Dice | lesion volume |
|---|---|---|
| **PyTorch (this branch)** | **0.99565** | **−0.12 %** (worst −2.16 %) |
| ONNX Runtime (previous attempt) | 0.9367 | +14.3 % (worst +39.6 %) |

The whole discrepancy has a single cause. tf2onnx expanded `tfa.InstanceNormalization`
into primitive ops, so each variance became a float32 `ReduceSum` over D·H·W values. At
192³ the running sum reaches ~4e7 while its addends are ~5.6 — the size of the float32
ulp there — so about 1 % of it rounds away. Measured against a float64 reference on the
same conv output:

| | variance error |
|---|---|
| PyTorch | ~0 (matches float64 to 7 digits) |
| TensorFlow (`tf.nn.moments`) | −0.12 % |
| ONNX Runtime | −0.9 % |

Confirmed from the other direction: patching the graph's `ReduceSum` nodes to accumulate
in float64 makes ONNX Runtime agree with PyTorch exactly (Dice 1.000000).

The ONNX error is worst where lesion load is low — +2–3 % on heavily lesioned subjects
but +24.8 % and +39.6 % on the two lightest — which is the population where relative
volume error matters most.

## Validation

Full ensemble, 10 subjects from open_ms_data (MSLUB), preprocessed exactly as
`segment.py` does, PyTorch vs the **real released TensorFlow `.h5` models**:

| subject | GT | TF | PyTorch | Dice | volume Δ |
|---|---|---|---|---|---|
| patient01 | 30620 | 30115 | 30084 | 0.99762 | −0.10 % |
| patient02 | 1381 | 998 | 1005 | 0.99451 | +0.70 % |
| patient03 | 1052 | 1252 | 1225 | 0.98829 | −2.16 % |
| patient04 | 40373 | 37560 | 37557 | 0.99770 | −0.01 % |
| patient05 | 29922 | 16927 | 17053 | 0.99364 | +0.74 % |
| patient06 | 48859 | 45549 | 45548 | 0.99858 | −0.00 % |
| patient07 | 1300 | 1983 | 1972 | 0.99368 | −0.55 % |
| patient08 | 6090 | 7170 | 7196 | 0.99652 | +0.36 % |
| patient09 | 19093 | 18650 | 18631 | 0.99831 | −0.10 % |
| patient10 | 16701 | 19779 | 19768 | 0.99765 | −0.06 % |
| **mean** | | | | **0.99565** | **−0.12 %** |

Lesion Dice against the consensus ground truth is unchanged: **TF 0.6829, PyTorch 0.6835**.
Lesion counts differ by −3..+2 out of 25–330 per subject.

For comparison, the ONNX Runtime backend on the same footing gives Dice 0.9367 and
**+14.3 %** lesion volume against TensorFlow — two orders of magnitude further away.

Where the two disagree, the differences are boundary jitter: ~100 % of differing voxels
lie on lesion surfaces, with median |p − 0.5| ≈ 0.01 — voxels the model was undecided
about. No lesion is gained or lost wholesale.

Both backends are deterministic: bit-identical run to run at fixed thread count, and
thread count changes results by ~2e-6 while flipping zero voxels.

## Guidance for users

- Results shift slightly relative to v1.0.0/v1.1.0. The shift is smaller than
  inter-rater variability (Dice ~0.7–0.8 for MS lesion segmentation) and smaller than
  scan–rescan variation.
- Do not mix versions within a longitudinal study; reprocess baselines with one version.
  That is standard practice and not specific to this change.
- If you previously used the ONNX Runtime backend, expect a **larger** change than if you
  came from TensorFlow, because that backend was the outlier.

## Reproducing

```bash
python tools/validate_pipeline.py --subjects <open_ms_data>/cross_sectional/coregistered \
    --out results/ --limit 10
```

## Reproducibility of the whole pipeline

The segmentation network is deterministic: bit-identical run to run at a fixed thread
count, with thread count changing probabilities by ~2e-6 and flipping zero voxels.

**The pipeline around it is not**, and was not before this change. Running the identical
code and weights on the same subject, once in a virtualenv and once in the Docker image
built from it:

| stage | agreement |
|---|---|
| HD-BET brain mask | Dice 0.993436 |
| MNI FLAIR after greedy | max abs diff 1.93e+02 |
| final lesion mask | Dice 0.883 |

Small float differences in HD-BET's CPU inference (thread counts, BLAS kernels) move the
brain mask by <1 %, which changes the stripped image, which changes the registration, and
the effect compounds. For scale: the whole TensorFlow-to-PyTorch swap perturbs the
segmentation *less* (Dice 0.996) than re-running the same code in a container does
(Dice 0.883).

The practical consequence is the usual one and predates this work: process a study in one
environment, and reprocess baselines rather than mixing. Pinning the container image is
the strongest guarantee available.

### greedy registration is itself nondeterministic

Worth recording, because it bounds what any reproducibility claim can mean. Running the
same affine registration repeatedly, on identical inputs:

| | run-to-run max abs diff in the transform |
|---|---|
| greedy CLI binary, 4 threads | 1.61e-01 |
| picsl_greedy Python API, 4 threads | 9.07e-02 |
| greedy CLI binary, **1 thread** | 2.49e-01 |
| CLI vs Python API | 3.02e-01 |

It is nondeterministic even single-threaded, so this is internal sampling rather than a
thread race. The CLI-versus-API gap is the same order as CLI-versus-CLI, which means
moving from the external binary to `picsl_greedy` introduced nothing new — and that this,
rather than the network, dominates end-to-end variation between runs.
