"""Ensemble segmentation in native PyTorch -- no TensorFlow, onnxruntime or onnx2torch.

Mirrors ``lst_ai/segment.py``: centre-crop to the model's input size, preprocess each
modality, run every ensemble member, average the probabilities, threshold at 0.5, drop
objects below ``--min-lesion-size``, and zero-pad the result back onto the input grid.

    # official v1.3.0 weights, converted once with lst_ai.weights
    python -m lst_training.inference --flair f.nii.gz --t1 t1.nii.gz \\
        --checkpoints checkpoints/UNet3D_MS_final_mdl{A,B,C}.pt --output seg.nii.gz

    # a model trained from scratch by lst_training.train
    python -m lst_training.inference --flair f.nii.gz --checkpoints checkpoints/my.pt \\
        --output seg.nii.gz

Inputs must already be skull-stripped and registered into the MNI space LST-AI uses --
this is the network stage only, not the full pipeline. Background is taken to be
exactly zero, as both the training loader and ``segment.py`` assume.

Two upstream inconsistencies are worth knowing about, since they change results:

* **Intensity range.** ``segment.py`` normalises to ``[0, 1]``, but the training loader
  rescales to ``[-1, 1]``. ``--intensity-range`` selects; the default reproduces
  ``segment.py``, i.e. official inference. Use ``minus1-1`` for a model trained by
  ``lst_training.train``, which follows the training loader.
* **Brain mask.** ``segment.py`` derives one per modality (``t1 != 0`` for the T1);
  the training loader uses the FLAIR mask for both. ``--mask-from`` selects, again
  defaulting to ``segment.py``'s behaviour.

**On agreeing with official LST-AI:** this reproduces ``tw/v200_updates`` (onnx2torch)
to float32 round-off, and so the original TensorFlow model. It does *not* reproduce
``feat/onnx-inference`` exactly, because ONNX Runtime loses about 1% of each
instance-norm variance to float32 accumulation -- see ``tools/compare_backends.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from lst_ai.model import NNUNet3D

__all__ = ["load_checkpoint", "adapt_shape", "preprocess", "segment", "remove_small_objects"]

DEFAULT_THRESHOLD = 0.5
DEFAULT_INPUT_SHAPE = (192, 192, 192)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> NNUNet3D:
    """Rebuild the architecture recorded in a checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not an LST-AI training checkpoint (expected 'config' and 'state_dict'). "
            "Convert a released .onnx with `python -m lst_ai.weights` first."
        )
    cfg = dict(ckpt["config"])
    cfg["ds_layers"] = tuple(cfg.get("ds_layers", ()))
    model = NNUNet3D(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def adapt_shape(volume: np.ndarray, shape=DEFAULT_INPUT_SHAPE):
    """Centre-crop to ``shape``; also return the padding that undoes it.

    Matches ``segment.py``: an odd remainder is taken off the low side.
    """
    slices, padding = [], []
    for size, target in zip(volume.shape, shape):
        diff = size - target
        if diff < 0:
            raise ValueError(
                f"volume of shape {volume.shape} is smaller than the model input {tuple(shape)} "
                f"along an axis ({size} < {target}); resample it into MNI space first"
            )
        left = (diff // 2) + (diff % 2)
        right = diff // 2
        slices.append(slice(left, size - right))
        padding.append((left, right))
    return volume[tuple(slices)].astype(np.float32), padding


def _normalise(vol: np.ndarray, brain: np.ndarray, clipping=(0.5, 99.5)) -> np.ndarray:
    """Clip to the in-brain percentiles, min-max to [0, 1], zero outside the brain."""
    inside = vol[brain != 0]
    vol = np.clip(vol, np.percentile(inside, clipping[0]), np.percentile(inside, clipping[1]))
    vol = vol - vol[brain == 1].min()
    vol = vol / vol[brain == 1].max()
    return (vol * brain).astype(np.float32)


def preprocess(flair: np.ndarray, t1: np.ndarray | None = None,
               intensity_range: str = "0-1", mask_from: str = "modality",
               clipping=(0.5, 99.5)) -> np.ndarray:
    """Apply the intensity pipeline; returns ``(1, C, D, H, W)``.

    ``intensity_range`` is ``"0-1"`` (``segment.py``) or ``"minus1-1"`` (training loader);
    ``mask_from`` is ``"modality"`` (``segment.py``) or ``"flair"`` (training loader).
    """
    if intensity_range not in ("0-1", "minus1-1"):
        raise ValueError(f"intensity_range must be '0-1' or 'minus1-1', got {intensity_range!r}")
    if mask_from not in ("modality", "flair"):
        raise ValueError(f"mask_from must be 'modality' or 'flair', got {mask_from!r}")

    flair_brain = (flair != 0).astype(np.float32)
    if not flair_brain.any():
        raise ValueError("the FLAIR volume is all zeros -- is it skull-stripped correctly?")

    volumes = [flair.astype(np.float32)]
    if t1 is not None:
        if t1.shape != flair.shape:
            raise ValueError(f"T1 shape {t1.shape} does not match FLAIR shape {flair.shape}")
        volumes.append(t1.astype(np.float32))

    channels = []
    for vol in volumes:
        brain = flair_brain if mask_from == "flair" else (vol != 0).astype(np.float32)
        norm = _normalise(vol, brain, clipping)
        if intensity_range == "minus1-1":
            norm = (norm * 2.0 - 1.0) * brain
        channels.append(norm)
    return np.stack(channels, axis=0)[None].astype(np.float32)


def remove_small_objects(mask: np.ndarray, zooms, min_size_mm3: float) -> np.ndarray:
    """Drop connected components below ``min_size_mm3``, as ``segment.py`` does.

    Uses 18-connectivity (``generate_binary_structure(3, 2)``), not skimage's default.
    """
    if min_size_mm3 <= 0:
        return mask
    from scipy.ndimage import generate_binary_structure, label

    labelled, n = label(mask, structure=generate_binary_structure(3, 2))
    min_voxels = np.round(min_size_mm3 / float(np.prod(zooms)))
    out = mask.copy()
    for idx in range(1, n + 1):
        component = labelled == idx
        if np.count_nonzero(component) < min_voxels:
            out[component] = 0
    return out


@torch.no_grad()
def segment(models, image: np.ndarray, device: torch.device | str = "cpu"):
    """Average the ensemble's probabilities. Returns ``(D, H, W)`` float32."""
    x = torch.from_numpy(image).to(device)
    total = None
    for model in models:
        out = model(x)
        prob = (out[0] if isinstance(out, list) else out).float()
        total = prob if total is None else total + prob
    return (total / len(models))[0, 0].cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flair", required=True)
    p.add_argument("--t1", default=None,
                   help="omit for a single-channel model; required by dual-channel ones")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help=".pt files to ensemble (the released models are mdlA, mdlB, mdlC)")
    p.add_argument("--output", required=True)
    p.add_argument("--probability-output", default=None,
                   help="also write the averaged probability map")
    p.add_argument("--input-shape", type=int, nargs=3, default=DEFAULT_INPUT_SHAPE)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--min-lesion-size", type=float, default=0.0,
                   help="mm^3; 0 keeps every component, as official LST-AI defaults to")
    p.add_argument("--intensity-range", default="0-1", choices=("0-1", "minus1-1"),
                   help="0-1 reproduces segment.py; minus1-1 matches the training loader")
    p.add_argument("--mask-from", default="modality", choices=("modality", "flair"),
                   help="modality reproduces segment.py; flair matches the training loader")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    models = [load_checkpoint(c, device) for c in args.checkpoints]

    channels = {m.in_channels for m in models}
    if len(channels) > 1:
        raise SystemExit(f"checkpoints disagree on input channels: {sorted(channels)}")
    n_channels = channels.pop()
    if n_channels == 2 and args.t1 is None:
        raise SystemExit("these checkpoints are dual-channel: pass --t1, or use a "
                         "single-channel model")
    if n_channels == 1 and args.t1 is not None:
        raise SystemExit("these checkpoints are FLAIR-only: drop --t1")

    flair_nib = nib.load(args.flair)
    flair, padding = adapt_shape(flair_nib.get_fdata(), args.input_shape)
    t1 = None
    if args.t1:
        t1, _ = adapt_shape(nib.load(args.t1).get_fdata(), args.input_shape)

    image = preprocess(flair, t1, args.intensity_range, args.mask_from)
    models[0].check_input_shape(tuple(image.shape[2:]))
    prob = segment(models, image, device)

    mask = (prob > args.threshold).astype(np.uint8)
    mask = remove_small_objects(mask, flair_nib.header.get_zooms()[:3], args.min_lesion_size)

    mask = np.pad(mask, padding, "constant", constant_values=0)
    nib.save(nib.Nifti1Image(mask, flair_nib.affine, flair_nib.header), args.output)
    if args.probability_output:
        padded = np.pad(prob, padding, "constant", constant_values=0.0).astype(np.float32)
        nib.save(nib.Nifti1Image(padded, flair_nib.affine, flair_nib.header),
                 args.probability_output)

    print(f"{int(mask.sum())} lesion voxels across {len(models)} model(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
