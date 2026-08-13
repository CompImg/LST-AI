"""Dataset for LST-AI training, ported from the TensorFlow ``data_loader.py``.

Faithful to the original pipeline, with one capability added: ``in_channels`` selects
between the dual-channel ``[FLAIR, T1]`` model and a FLAIR-only model. The TF loader
could only do the latter by uncommenting a line, so the T1 file is required only when
``in_channels == 2``.

Layout expected on disk -- one directory per subject, discovered by globbing for FLAIR::

    <root>/**/<subject>_flair.nii.gz
    <root>/**/<subject>_t1.nii.gz      (only needed when in_channels == 2)
    <root>/**/<subject>_seg.nii.gz

Preprocessing, unchanged from the original:

1. centre-crop every volume to ``shape`` using the FLAIR to fix the crop;
2. brain mask := ``flair != 0`` (the inputs are already skull-stripped);
3. per-modality: clip to the [0.5, 99.5] in-brain percentiles, min-max to [0, 1] using
   in-brain extrema, and zero outside the mask;
4. optional augmentation (rotation, two flips, gamma, Gaussian blur);
5. deep-supervision targets by max-pooling the mask -- max, not mean, so small lesions
   survive the coarsening;
6. rescale to [-1, 1] and re-zero outside the brain, as nnU-Net does.

Note that step 5 happens *before* step 6, so the targets stay binary. Tensors come out
channel-first (NCDHW) for PyTorch; the TF loader emitted NDHWC.
"""

from __future__ import annotations

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from skimage.exposure import adjust_gamma
from skimage.measure import block_reduce
from torch.utils.data import Dataset

__all__ = ["MSDataset", "adapt_shape", "normalise_modality"]


def adapt_shape(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Centre-crop to ``shape``, biasing an odd remainder to the low side (as the TF code)."""
    slices = []
    for size, target in zip(volume.shape, shape):
        diff = size - target
        if diff < 0:
            raise ValueError(
                f"volume of shape {volume.shape} is smaller than the target {shape} along "
                f"an axis ({size} < {target}); pad it or lower --shape"
            )
        start = (diff // 2) + (diff % 2)
        slices.append(slice(start, size - (diff // 2)))
    return volume[tuple(slices)].astype(np.float32)


def normalise_modality(vol: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Clip to in-brain [0.5, 99.5] percentiles, min-max to [0, 1], zero outside the brain."""
    inside = vol[brain != 0]
    vol = np.clip(vol, np.percentile(inside, 0.5), np.percentile(inside, 99.5))
    vol = vol - vol[brain == 1].min()
    vol = vol / vol[brain == 1].max()
    return (vol * brain).astype(np.float32)


class MSDataset(Dataset):
    """Paired FLAIR/T1/lesion-mask volumes with deep-supervision targets.

    Parameters
    ----------
    root : str | Path
        Directory searched recursively for ``*_flair.nii.gz``.
    shape : tuple[int, int, int]
        Crop size. Must be divisible by ``2 ** n_conv_blocks`` for the model.
    in_channels : int
        2 for ``[FLAIR, T1]`` (the released models), 1 for FLAIR only.
    augment : bool
        Enable the training augmentations.
    aug_prob : float
        Each augmentation fires when ``random.random() > aug_prob`` -- so the default of
        0.33 applies each one about two-thirds of the time. Kept as-is from the original,
        including the inverted sense of the name.
    n_deep_supervision : int
        How many extra coarsened targets to emit, matching ``len(model.ds_layers)``.

    Returns ``(image, targets)`` where ``image`` is ``(in_channels, *shape)`` and
    ``targets`` is a list of ``(1, ...)`` masks at full, 1/2, 1/4 ... resolution.
    """

    def __init__(
        self,
        root: str | Path,
        shape: tuple[int, int, int] = (192, 192, 192),
        in_channels: int = 2,
        augment: bool = True,
        aug_prob: float = 0.33,
        n_deep_supervision: int = 2,
        max_angle: int = 30,
        max_sigma: float = 1.5,
        gamma_range: tuple[float, float] = (0.5, 1.5),
    ):
        if in_channels not in (1, 2):
            raise ValueError(f"in_channels must be 1 (FLAIR) or 2 (FLAIR+T1), got {in_channels}")
        self.root = Path(root)
        self.flair_paths = sorted(self.root.glob("**/*_flair.nii.gz"))
        if not self.flair_paths:
            raise FileNotFoundError(f"no *_flair.nii.gz found under {self.root}")
        self.shape = tuple(shape)
        self.in_channels = in_channels
        self.augment = augment
        self.aug_prob = aug_prob
        self.n_deep_supervision = n_deep_supervision
        self.max_angle = max_angle
        self.max_sigma = max_sigma
        self.gamma_range = gamma_range

        missing = [p for p in self.flair_paths if not self._sibling(p, "seg").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} FLAIR volumes have no matching _seg.nii.gz, e.g. {missing[0]}"
            )
        if in_channels == 2:
            missing = [p for p in self.flair_paths if not self._sibling(p, "t1").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} FLAIR volumes have no matching _t1.nii.gz, e.g. "
                    f"{missing[0]}. Use in_channels=1 to train a FLAIR-only model."
                )

    @staticmethod
    def _sibling(flair_path: Path, suffix: str) -> Path:
        return Path(str(flair_path).replace("_flair.nii.gz", f"_{suffix}.nii.gz"))

    def __len__(self) -> int:
        return len(self.flair_paths)

    def _augment(self, mods: list[np.ndarray], gt: np.ndarray, brain: np.ndarray):
        """Rotation, two flips, gamma and blur -- geometric ones applied to the mask too."""
        if random.random() > self.aug_prob:
            angle = random.randint(-self.max_angle, self.max_angle)
            axes = random.sample(([1, 0], [2, 1]), 1)[0]
            kw = dict(angle=angle, axes=axes, reshape=False, mode="constant", cval=0.0)
            mods = [ndimage.rotate(m, order=1, **kw) for m in mods]
            gt = ndimage.rotate(gt, order=0, **kw)   # nearest, to keep the mask binary

        # Two independent flip draws, as in the original -- they can pick the same axis
        # and cancel out, which is part of the augmentation distribution it was trained on.
        for _ in range(2):
            if random.random() > self.aug_prob:
                axis = random.sample((0, 1, 2), 1)
                mods = [np.flip(m, axis=axis) for m in mods]
                gt = np.flip(gt, axis=axis)

        # Intensity augmentations touch the images only, never the mask. Each modality
        # draws its own parameter, and the *order* of those draws matters for matching
        # the original: data_loader.py augments T1 first and FLAIR second, so with a
        # shared seed T1 must consume the first draw. Iterating `mods` (FLAIR first)
        # would be distributionally identical but not reproduce the original stream.
        tf_order = list(range(len(mods)))[::-1]
        if random.random() > self.aug_prob:
            for i in tf_order:
                mods[i] = adjust_gamma(mods[i], gamma=random.uniform(*self.gamma_range)) * brain
        if random.random() > self.aug_prob:
            for i in tf_order:
                mods[i] = ndimage.gaussian_filter(
                    mods[i], sigma=random.uniform(0, self.max_sigma)) * brain

        return [np.ascontiguousarray(m, np.float32) for m in mods], np.ascontiguousarray(gt)

    def __getitem__(self, idx: int):
        flair_path = self.flair_paths[idx]

        # The FLAIR fixes the crop for every volume of this subject.
        flair = adapt_shape(nib.load(flair_path).get_fdata(), self.shape)
        gt = adapt_shape(nib.load(self._sibling(flair_path, "seg")).get_fdata(), self.shape)

        brain = np.zeros(flair.shape, np.float32)
        brain[flair != 0] = 1.0
        gt = (gt > 0).astype(np.float32)

        mods = [normalise_modality(flair, brain)]
        if self.in_channels == 2:
            t1 = adapt_shape(nib.load(self._sibling(flair_path, "t1")).get_fdata(), self.shape)
            mods.append(normalise_modality(t1, brain))

        if self.augment:
            mods, gt = self._augment(mods, gt, brain)

        # Coarsen the mask before rescaling, so the targets stay binary. Max-pooling
        # (not mean) keeps small lesions visible at the deep-supervision resolutions.
        targets = [gt]
        for _ in range(self.n_deep_supervision):
            targets.append(block_reduce(targets[-1], (2, 2, 2), np.max))

        mods = [(m * 2.0 - 1.0) * brain for m in mods]

        image = torch.from_numpy(np.stack(mods, axis=0).astype(np.float32))
        targets = [torch.from_numpy(t.astype(np.float32)[None]) for t in targets]
        return image, targets


def collate(batch):
    """Stack ``(image, [targets...])`` pairs; the default collate mishandles the list."""
    images = torch.stack([b[0] for b in batch])
    n = len(batch[0][1])
    targets = [torch.stack([b[1][i] for b in batch]) for i in range(n)]
    return images, targets
