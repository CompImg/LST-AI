"""Deterministic synthetic subjects, shared by the tests and the fixture generator.

``np.random.default_rng`` with a fixed seed is reproducible across NumPy versions, so
this regenerates byte-identical volumes wherever it runs -- which is what lets
``tests/data/tf_loader_reference.npz`` stay small (it holds only the TF loader's output,
not its input).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

SEED = 11
VOLUME_SHAPE = (40, 44, 40)   # deliberately not cubic, to exercise the crop
N_SUBJECTS = 2


def write_subjects(root: str | Path) -> Path:
    """Write ``sub-XX_{flair,t1,seg}.nii.gz`` trees under ``root`` and return it."""
    root = Path(root)
    rng = np.random.default_rng(SEED)
    grid = np.mgrid[:VOLUME_SHAPE[0], :VOLUME_SHAPE[1], :VOLUME_SHAPE[2]].astype(np.float32)
    centre = np.array([20, 22, 20])[:, None, None, None]
    radius = np.sqrt(((grid - centre) ** 2).sum(0))
    brain = (radius < 15).astype(np.float32)   # skull-stripped: background is exactly 0

    for s in range(N_SUBJECTS):
        d = root / f"sub-{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        flair = (rng.uniform(20, 200, VOLUME_SHAPE) * brain).astype(np.float32)
        t1 = (rng.uniform(10, 150, VOLUME_SHAPE) * brain).astype(np.float32)
        seg = ((radius < 5) & (rng.uniform(0, 1, VOLUME_SHAPE) < 0.5)).astype(np.float32)
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(flair, aff), d / f"sub-{s:02d}_flair.nii.gz")
        nib.save(nib.Nifti1Image(t1, aff), d / f"sub-{s:02d}_t1.nii.gz")
        nib.save(nib.Nifti1Image(seg, aff), d / f"sub-{s:02d}_seg.nii.gz")
    return root
