"""Check the ported dataset against the original TensorFlow ``data_loader.py``.

``tests/data/tf_loader_reference.npz`` holds what ``MSSequence_3D`` produced for the
synthetic subjects in ``tests/_synthetic.py``, under TensorFlow 2.19.1 with augmentation
off (see ``tools/make_loader_reference.py``). The preprocessing is deterministic, so the
port is expected to match it exactly, not approximately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synthetic import write_subjects  # noqa: E402
from lst_training.data import MSDataset, adapt_shape, collate  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "tf_loader_reference.npz"
SHAPE = (32, 32, 32)


@pytest.fixture(scope="module")
def subjects(tmp_path_factory):
    return write_subjects(tmp_path_factory.mktemp("subjects"))


@pytest.fixture(scope="module")
def reference():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing")
    ref = np.load(FIXTURE)
    # TF emits NDHWC with a batch axis; the dataset emits NCDHW without one.
    return {k: ref[k][0].transpose(3, 0, 1, 2) for k in ("img", "seg", "ds1", "ds2")}


def test_image_matches_tensorflow_exactly(subjects, reference):
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False, n_deep_supervision=2)
    img, _ = ds[0]
    assert img.shape == reference["img"].shape
    assert np.array_equal(img.numpy(), reference["img"])


@pytest.mark.parametrize("level,key", [(0, "seg"), (1, "ds1"), (2, "ds2")])
def test_targets_match_tensorflow_exactly(subjects, reference, level, key):
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False, n_deep_supervision=2)
    _, targets = ds[0]
    assert np.array_equal(targets[level].numpy(), reference[key])


def test_deep_supervision_targets_halve_and_stay_binary(subjects):
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False, n_deep_supervision=2)
    _, targets = ds[0]
    assert [tuple(t.shape) for t in targets] == [(1, 32, 32, 32), (1, 16, 16, 16), (1, 8, 8, 8)]
    for t in targets:
        assert set(np.unique(t.numpy())).issubset({0.0, 1.0})


def test_max_pooling_preserves_small_lesions(subjects):
    """Coarsening uses max, not mean, so a lesion never vanishes from a coarser target."""
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False, n_deep_supervision=2)
    _, targets = ds[0]
    assert targets[0].sum() > 0
    for coarse in targets[1:]:
        assert coarse.sum() > 0


def test_flair_only_is_channel_zero_of_the_dual_model(subjects):
    dual = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False)[0][0]
    single = MSDataset(subjects, shape=SHAPE, in_channels=1, augment=False)[0][0]
    assert single.shape[0] == 1
    assert torch.equal(single[0], dual[0])


def test_flair_only_does_not_require_t1(subjects, tmp_path):
    """A FLAIR-only dataset must load from a tree that has no T1 at all."""
    import shutil

    root = tmp_path / "no_t1"
    shutil.copytree(subjects, root)
    for p in root.glob("**/*_t1.nii.gz"):
        p.unlink()

    ds = MSDataset(root, shape=SHAPE, in_channels=1, augment=False)
    assert ds[0][0].shape[0] == 1
    with pytest.raises(FileNotFoundError, match="in_channels=1"):
        MSDataset(root, shape=SHAPE, in_channels=2, augment=False)


def test_missing_segmentation_is_reported(subjects, tmp_path):
    import shutil

    root = tmp_path / "no_seg"
    shutil.copytree(subjects, root)
    next(root.glob("**/*_seg.nii.gz")).unlink()
    with pytest.raises(FileNotFoundError, match="_seg.nii.gz"):
        MSDataset(root, shape=SHAPE, in_channels=2, augment=False)


def test_intensities_span_minus_one_to_one_and_background_is_zero(subjects):
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False)
    img, _ = ds[0]
    assert float(img.min()) == pytest.approx(-1.0, abs=1e-5)
    assert float(img.max()) == pytest.approx(1.0, abs=1e-5)
    # Outside the brain the FLAIR is exactly 0, and masking is applied after rescaling.
    background = img[0] == 0
    assert background.any()
    assert torch.all(img[1][background] == 0)


def test_augmentation_perturbs_but_keeps_masks_binary(subjects):
    import random

    random.seed(0)
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=True, aug_prob=0.0)
    plain = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False)[0][0]
    img, targets = ds[0]                       # aug_prob=0 => every augmentation fires
    assert not torch.equal(img, plain)
    for t in targets:
        assert set(np.unique(t.numpy())).issubset({0.0, 1.0})


def test_rejects_bad_channel_count(subjects):
    with pytest.raises(ValueError, match="in_channels"):
        MSDataset(subjects, shape=SHAPE, in_channels=3)


def test_adapt_shape_rejects_undersized_volumes():
    with pytest.raises(ValueError, match="smaller than the target"):
        adapt_shape(np.zeros((10, 10, 10), np.float32), (32, 32, 32))


def test_collate_stacks_images_and_every_target_level(subjects):
    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=False, n_deep_supervision=2)
    images, targets = collate([ds[0], ds[1]])
    assert tuple(images.shape) == (2, 2, 32, 32, 32)
    assert [tuple(t.shape) for t in targets] == [
        (2, 1, 32, 32, 32), (2, 1, 16, 16, 16), (2, 1, 8, 8, 8)]


AUGMENTED_FIXTURE = Path(__file__).parent / "data" / "tf_loader_reference_augmented.npz"


def test_augmentation_matches_tensorflow_under_a_shared_seed(subjects):
    """The augmented sample must match too, not just the deterministic path.

    Both loaders drive their augmentations from the `random` module in the same order,
    so seeding identically makes them directly comparable. This caught a real ordering
    bug: data_loader.py augments T1 before FLAIR, so T1 consumes the first gamma and
    blur draw. Iterating modalities FLAIR-first is distributionally identical but
    yields a different sample, which would silently change what a model trains on.
    """
    import random

    if not AUGMENTED_FIXTURE.exists():
        pytest.skip(f"{AUGMENTED_FIXTURE} missing")
    ref = np.load(AUGMENTED_FIXTURE)

    ds = MSDataset(subjects, shape=SHAPE, in_channels=2, augment=True, aug_prob=0.33,
                   n_deep_supervision=2)
    random.seed(1234)
    img, targets = ds[0]

    assert np.array_equal(img.numpy(), ref["img"][0].transpose(3, 0, 1, 2))
    for level, key in enumerate(("seg", "ds1", "ds2")):
        assert np.array_equal(targets[level].numpy(), ref[key][0].transpose(3, 0, 1, 2))
