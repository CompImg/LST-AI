# Training LST-AI models

Trains the same network `LST_AI` runs at inference — `LST_AI.model.NNUNet3D` is imported,
not copied, so what you train is by construction what ships. That matters: the released
ensemble is *heterogeneous* (mdlA 28 filters, mdlB 24 with a single deep-supervision head,
mdlC 32), which is what happens when training and inference drift apart in separate places.

Ported from the original TensorFlow `train_model.py` / `data_loader.py`, and verified
against them rather than by inspection:

| | evidence |
|---|---|
| preprocessing | reproduces `data_loader.py` exactly (max abs diff **0.0**) |
| augmentation | matches under a shared RNG seed, **0.0** on every channel and target |
| losses | match the TF implementations to **1.9e-06** across six regimes |

## Install

```bash
pip install -e .                 # LST_AI itself, from the repository root
pip install -e training          # or just: pip install torch nibabel scipy scikit-image
```

## Data layout

One directory per subject, discovered by globbing for FLAIR:

```
<root>/**/<subject>_flair.nii.gz
<root>/**/<subject>_t1.nii.gz      # only for --in-channels 2
<root>/**/<subject>_seg.nii.gz     # binary lesion mask
```

Volumes must be skull-stripped and in MNI space — background is taken to be exactly zero,
and the brain mask is derived as `flair != 0`. Use `LST_AI`'s own registration and
stripping to prepare a cohort, then train on the MNI-space intermediates it leaves in
`--temp`.

## Train

```bash
# dual-channel FLAIR + T1, the configuration the released models use
python -m lst_training.train --train-data data/train --val-data data/val --in-channels 2

# single-channel FLAIR only
python -m lst_training.train --train-data data/train --in-channels 1
```

Defaults reproduce the original recipe: SGD (momentum 0.9, Nesterov), lr 1e-2 on a cosine
schedule over `--epochs`, deep supervision at 1/2 and 1/4 resolution weighted 4/7 : 2/7 :
1/7, and checkpointing on the best *training* `out_seg` loss (as the original did, not on
validation). `--preset` selects one of the four loss combinations the original swept:
`nnUNet_bce-dice`, `nnUNet_bce-tversky`, `nnUNet_dsTversky`, `nnUNet_dsDice`.

Useful flags: `--filters`, `--conv-blocks`, `--bottleneck-filters` and `--ds-layers` to
reproduce a specific released variant; `--amp` for mixed precision on CUDA; `--no-augment`
to disable augmentation; `--shape` for a different crop.

### Monitoring a run

Every epoch appends to `<out-dir>/UNet3D_MS_final_<name>.json` — loss, per-head loss,
Dice, learning rate and wall time. That file always exists and needs no extra package.

For watching a long run, add `--tensorboard`:

```bash
pip install tensorboard
python -m lst_training.train --train-data data/train --tensorboard
tensorboard --logdir checkpoints/tb
```

Scalars are grouped as `train/…` and `val/…` so both splits of a metric share axes, with
each deep-supervision head logged separately — useful for spotting a head that has stopped
contributing. Pass `--tensorboard DIR` to choose the directory; the default is
`<out-dir>/tb/<name>`. Note this pulls in the standalone `tensorboard` package, which does
not depend on TensorFlow.

Input shape must be divisible by `2 ** conv_blocks`, and must leave more than one voxel in
the bottleneck — instance norm over a single voxel divides by `sqrt(eps)` and collapses the
block to its bias. `check_input_shape` rejects both cases with an explanatory error.

## Numerics: legacy vs modern

The released weights were trained under TensorFlow defaults that are **not** PyTorch's, and
`LST_AI.model` pins them so the shipped weights load correctly: LeakyReLU slope **0.3**
(torch: 0.01), instance-norm epsilon **1e-3** (torch: 1e-5), affine instance norm, no conv
bias, he_uniform init, and `K.epsilon()` = 1e-7 Dice smoothing.

Three of those are not quirks — a bias before an affine norm is redundant either way,
he_uniform is more principled than torch's legacy `kaiming_uniform(a=sqrt(5))`, and at
activation variance ~0.7 the epsilon shifts the denominator by 0.14%. **The slope is the
one real inherited accident**: 0.3 is a Keras default rather than a choice made for this
task, where nnU-Net and most of the literature use 0.01. If you are training fresh models
and do not need to ensemble them with the released ones, it is worth running both.

## Inference on a model you trained

```bash
python -m lst_training.inference --flair f.nii.gz --t1 t1.nii.gz \
    --checkpoints checkpoints/UNet3D_MS_final_<name>.pt --output seg.nii.gz \
    --intensity-range minus1-1 --mask-from flair
```

Those two flags matter. LST-AI's training and inference code disagree on preprocessing:
`data_loader.py` rescales to `[-1, 1]` and masks both modalities with the FLAIR mask, while
`segment.py` stops at `[0, 1]` and derives a mask per modality. A model trained here
follows the loader, so serving it needs the loader's conventions.

## Tests

```bash
python -m pytest training/tests -q
```

The TensorFlow-derived fixtures can be regenerated with `training/tools/make_loss_reference.py`
and `training/tools/make_loader_reference.py` in a throwaway TF environment (TF 2.19 plus
`tf-keras`, since the released `.h5` need Keras 2).
