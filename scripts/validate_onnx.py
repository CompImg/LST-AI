"""Validate ONNX vs TF equivalence for the LST-AI ensemble (onnxruntime venv only).

Loads the reference input + per-model TF out_seg saved by scripts/tf_to_onnx.py, runs
each ONNX model via onnxruntime, and reports per-model diffs PLUS the ensemble
probability diff and the binary-mask agreement at the 0.5 threshold (Dice + flipped
voxels) — the metric that actually matters for the lesion mask. Run in a clean
onnxruntime venv (no TensorFlow).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort

_MODELS = ["UNet3D_MS_final_mdlA", "UNet3D_MS_final_mdlB", "UNet3D_MS_final_mdlC"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    x = np.load(os.path.join(args.onnx_dir, "_ref_input.npy"))
    tf_segs, onnx_segs = [], []
    for name in _MODELS:
        tf_seg = np.load(os.path.join(args.onnx_dir, f"{name}_tfref.npy"))
        sess = ort.InferenceSession(os.path.join(args.onnx_dir, f"{name}.onnx"), providers=["CPUExecutionProvider"])
        onnx_seg = np.squeeze(sess.run(None, {sess.get_inputs()[0].name: x})[0])
        tf_segs.append(tf_seg)
        onnx_segs.append(onnx_seg)
        d = np.abs(tf_seg - onnx_seg)
        print(f"{name}: max={d.max():.3e} mean={d.mean():.3e}")

    tf_ens = np.mean(tf_segs, axis=0)
    on_ens = np.mean(onnx_segs, axis=0)
    mt, mo = tf_ens > 0.5, on_ens > 0.5
    inter, union = int((mt & mo).sum()), int(mt.sum()) + int(mo.sum())
    dice = 2 * inter / union if union else 1.0
    flips = int((mt != mo).sum())
    print("\n=== ENSEMBLE (mean of 3) ===")
    print(f"prob: max={np.abs(tf_ens-on_ens).max():.3e} mean={np.abs(tf_ens-on_ens).mean():.3e}")
    print(f"mask @0.5: Dice={dice:.5f}  flips={flips}/{mt.size} ({100*flips/mt.size:.5f}%)  "
          f"TF_vox={int(mt.sum())} ONNX_vox={int(mo.sum())}")

    ok = dice >= 0.999 and np.abs(tf_ens - on_ens).mean() <= args.tol
    print(f"\n=== {'PASS' if ok else 'REVIEW'} (Dice>=0.999 & mean prob diff<={args.tol:.0e}) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
