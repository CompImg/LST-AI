"""Convert the LST-AI TF/Keras UNet3D ensemble (.h5) to ONNX (TF venv only).

The models use GroupNormalization (an instance-norm replacement for the deprecated
tfa InstanceNormalization, see LST_AI/custom_tf.py) — all standard ops, so tf2onnx
converts them cleanly. This script ALSO saves a reference input and each model's TF
out_seg (output index 0) so equivalence can be checked from a separate onnxruntime
venv (avoids numpy/ABI conflicts between TF and onnxruntime).

The reference input defaults to a REAL MNI brain volume (--input-nifti), preprocessed
exactly like LST_AI/segment.py (center crop/pad to 192^3 + intensity standardization),
which is in-distribution for the instance-norm UNets — far fairer than random noise.

Run in the TF venv (tensorflow<2.16 + tf2onnx + onnx + nibabel; NO onnxruntime), then
run scripts/validate_onnx.py in an onnxruntime venv.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import nibabel as nib
import numpy as np
import tensorflow as tf
import tf2onnx

from LST_AI.custom_tf import load_custom_model

_MODELS = ["UNet3D_MS_final_mdlA.h5", "UNet3D_MS_final_mdlB.h5", "UNet3D_MS_final_mdlC.h5"]
_SHAPE = (192, 192, 192)
_OPSET = 18


def _center_fit(img: np.ndarray, shape=_SHAPE) -> np.ndarray:
    """Center crop or zero-pad a volume to `shape` (TF and ONNX get the same tensor)."""
    out = np.zeros(shape, np.float32)
    src, dst = [], []
    for n, m in zip(img.shape, shape):
        if n >= m:
            s = (n - m) // 2
            src.append(slice(s, s + m))
            dst.append(slice(0, m))
        else:
            s = (m - n) // 2
            src.append(slice(0, n))
            dst.append(slice(s, s + n))
    out[tuple(dst)] = img[tuple(src)].astype(np.float32)
    return out


def _standardize(img: np.ndarray, clipping=(0.5, 99.5)) -> np.ndarray:
    """Replicates segment.py preprocess_intensities: clip percentiles, scale to [0,1] over the brain."""
    bm = (img != 0).astype(np.float32)
    lo, hi = np.percentile(img[bm != 0], clipping[0]), np.percentile(img[bm != 0], clipping[1])
    img = np.clip(img, lo, hi)
    img -= img[bm == 1].min()
    img = img / img[bm == 1].max()
    return (img * bm).astype(np.float32)


def _build_input(t1: str | None, flair: str | None, nifti: str | None, seed: int = 0) -> np.ndarray:
    if t1 and flair:  # real LST-AI inputs: MNI-space skull-stripped T1 + FLAIR
        t1v = _standardize(_center_fit(nib.load(t1).get_fdata()))
        flv = _standardize(_center_fit(nib.load(flair).get_fdata()))
        x = np.stack([flv, t1v], axis=-1)  # segment.py order: [flair, t1]
        return np.expand_dims(x, 0).astype(np.float32)
    if nifti:
        vol = _standardize(_center_fit(nib.load(nifti).get_fdata()))
        return np.expand_dims(np.stack([vol, vol], axis=-1), 0).astype(np.float32)
    return np.random.default_rng(seed).random((1, *_SHAPE, 2)).astype(np.float32)


def _tf_out_seg(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    out = model(x)
    return np.squeeze(np.asarray(out[0] if isinstance(out, (list, tuple)) else out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--input-nifti", default=None, help="one real MNI volume (both channels)")
    ap.add_argument("--t1-nifti", default=None, help="real MNI-space skull-stripped T1")
    ap.add_argument("--flair-nifti", default=None, help="real MNI-space skull-stripped FLAIR")
    ap.add_argument("--skip-convert", action="store_true", help="only regenerate TF reference, reuse existing .onnx")
    ap.add_argument("--seed", type=int, default=0, help="random-input seed (when no nifti given)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    x = _build_input(args.t1_nifti, args.flair_nifti, args.input_nifti, args.seed)
    np.save(os.path.join(args.out_dir, "_ref_input.npy"), x)
    src = "t1+flair" if (args.t1_nifti and args.flair_nifti) else (args.input_nifti or "random")
    print(f"reference input: {src} shape={x.shape}")
    spec = (tf.TensorSpec((1, *_SHAPE, 2), tf.float32, name="input"),)

    for name in _MODELS:
        print(f"\n=== {name} ===", flush=True)
        model = load_custom_model(os.path.join(args.model_dir, name), compile=False)
        np.save(os.path.join(args.out_dir, name.replace(".h5", "_tfref.npy")), _tf_out_seg(model, x))
        if not args.skip_convert:
            onnx_path = os.path.join(args.out_dir, name.replace(".h5", ".onnx"))
            tf2onnx.convert.from_keras(model, input_signature=spec, opset=_OPSET, output_path=onnx_path)
            print(f"   wrote {onnx_path}")

    print("\n=== done; run scripts/validate_onnx.py in an onnxruntime venv ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
