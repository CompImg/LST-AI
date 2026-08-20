#!/usr/bin/env python3
"""Compare two LST-AI segmentations -- Dice, lesion count and lesion volume.

Used to check that a run on one machine agrees with a run on another, and to compare
against a reference produced by the TensorFlow version.

    python scripts/compare_segmentations.py reference.nii.gz new.nii.gz

Do not expect 1.0. greedy's affine registration samples internally and is not
deterministic run to run, so two runs of the *identical* code on the same machine already
differ slightly. See docs/testing.md for the numbers to expect.
"""

from __future__ import annotations

import argparse
import sys

import nibabel as nib
import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Dice on binary masks; 1.0 when both are empty."""
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="binarisation threshold, for comparing probability maps")
    args = ap.parse_args()

    ref_img, cand_img = nib.load(args.reference), nib.load(args.candidate)

    # Squeeze before comparing shapes. A probability map saved as (X, Y, Z, 1) against a
    # (X, Y, Z) mask is a legitimate pairing, and if it is *not* squeezed the two
    # broadcast against each other and yield a silently nonsensical Dice above 1.
    ref, cand = np.squeeze(ref_img.get_fdata()), np.squeeze(cand_img.get_fdata())

    if ref.shape != cand.shape:
        print(f"FAIL: shape mismatch {ref.shape} vs {cand.shape}", file=sys.stderr)
        return 2

    ref, cand = ref > args.threshold, cand > args.threshold

    # Voxel volume from the affine, so lesion load is in mL rather than voxels.
    vox_mm3 = float(np.abs(np.linalg.det(ref_img.affine[:3, :3])))
    ref_ml, cand_ml = ref.sum() * vox_mm3 / 1000.0, cand.sum() * vox_mm3 / 1000.0

    from scipy.ndimage import label
    n_ref = label(ref)[1]
    n_cand = label(cand)[1]

    print(f"Dice                {dice(ref, cand):.6f}")
    print(f"lesion volume       {ref_ml:.3f} mL (reference) vs {cand_ml:.3f} mL")
    if ref_ml > 0:
        print(f"volume difference   {100.0 * (cand_ml - ref_ml) / ref_ml:+.2f} %")
    print(f"lesion count        {n_ref} vs {n_cand}")
    print(f"voxels differing    {int(np.logical_xor(ref, cand).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
