"""End-to-end checks: train a small model from scratch, then segment with it.

Deliverable B is "a model that can be trained from scratch on new data", so these run the
real ``train``/``inference`` entry points rather than the pieces in isolation -- at a
reduced size (3 conv blocks, 32^3) so they finish in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synthetic import write_subjects  # noqa: E402
from lst_training.inference import (  # noqa: E402
    adapt_shape, load_checkpoint, preprocess, remove_small_objects, segment,
)
from lst_training.train import build_argparser, cosine_annealing, train  # noqa: E402

SHAPE = (32, 32, 32)


@pytest.fixture(scope="module")
def subjects(tmp_path_factory):
    return write_subjects(tmp_path_factory.mktemp("subjects"))


def _args(subjects, out_dir, **overrides):
    argv = ["--train-data", str(subjects), "--shape", "32", "32", "32",
            "--conv-blocks", "3", "--filters", "4", "--epochs", "2",
            "--batch-size", "1", "--workers", "0", "--device", "cpu",
            "--out-dir", str(out_dir), "--no-augment"]
    for k, v in overrides.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(x) for x in np.atleast_1d(v)])
    return build_argparser().parse_args(argv)


@pytest.mark.parametrize("in_channels", [1, 2])
def test_train_then_segment(subjects, tmp_path, in_channels):
    out_dir = tmp_path / f"ckpt{in_channels}"
    ckpt = train(_args(subjects, out_dir, in_channels=in_channels, name=f"t{in_channels}"))
    assert ckpt.exists()

    model = load_checkpoint(ckpt)
    assert model.in_channels == in_channels

    flair = nib.load(next(Path(subjects).glob("**/*_flair.nii.gz"))).get_fdata()
    t1 = nib.load(next(Path(subjects).glob("**/*_t1.nii.gz"))).get_fdata() if in_channels == 2 else None
    cropped, padding = adapt_shape(flair, SHAPE)
    t1c = adapt_shape(t1, SHAPE)[0] if t1 is not None else None

    image = preprocess(cropped, t1c, intensity_range="minus1-1", mask_from="flair")
    assert image.shape == (1, in_channels, *SHAPE)

    prob = segment([model], image)
    assert prob.shape == SHAPE
    assert 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0

    restored = np.pad((prob > 0.5).astype(np.uint8), padding)
    assert restored.shape == flair.shape


def test_checkpoint_round_trips_the_architecture(subjects, tmp_path):
    ckpt = train(_args(subjects, tmp_path / "rt", in_channels=2, filters=6,
                       ds_layers=[-2], name="rt"))
    model = load_checkpoint(ckpt)
    assert model.n_filters == 6
    assert model.ds_layers == (-2,)
    assert model.n_conv_blocks == 3

    x = torch.randn(1, 2, *SHAPE)
    with torch.no_grad():
        out = model(x)
    assert len(out) == 2   # out_seg + one deep-supervision head


def test_training_reduces_the_loss(subjects, tmp_path):
    """A learnable target must actually be learned -- guards the optimiser wiring."""
    import json

    out_dir = tmp_path / "learn"
    train(_args(subjects, out_dir, in_channels=1, epochs=25, name="learn", loss_out="dice"))
    history = json.loads((out_dir / "UNet3D_MS_final_learn.json").read_text())
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_tensorboard_logging_is_optional_and_writes_scalars(subjects, tmp_path):
    """--tensorboard must produce readable scalars, and its absence must change nothing."""
    pytest.importorskip("tensorboard")

    train(_args(subjects, tmp_path, name="tb", tensorboard=True))

    log_dir = tmp_path / "tb" / "tb"
    events = list(log_dir.glob("events.out.tfevents.*"))
    assert events, f"no event file under {log_dir}"

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    # Split-prefixed so train and val land on shared axes in the UI.
    assert {"train/loss", "train/dice", "misc/lr"} <= tags, tags
    assert [s.step for s in acc.Scalars("train/loss")] == [0, 1]

    # The JSON history is the record that exists with or without tensorboard.
    assert (tmp_path / "UNet3D_MS_final_tb.json").exists()


def test_training_without_tensorboard_still_writes_the_json_history(subjects, tmp_path):
    train(_args(subjects, tmp_path, name="notb"))
    assert (tmp_path / "UNet3D_MS_final_notb.json").exists()
    assert not (tmp_path / "tb").exists()


def test_cosine_schedule_matches_the_tensorflow_scheduler():
    n = 1001
    assert cosine_annealing(0, n) == pytest.approx(1.0)
    assert cosine_annealing(n // 2, n) == pytest.approx(0.5, abs=1e-3)
    assert cosine_annealing(n, n) == pytest.approx(0.0, abs=1e-9)
    assert all(cosine_annealing(e, n) >= cosine_annealing(e + 1, n) for e in range(n))


def test_load_checkpoint_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "plain.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not an LST-AI training checkpoint"):
        load_checkpoint(path)


class TestPreprocess:
    def _volume(self):
        vol = np.zeros((32, 32, 32), np.float32)
        vol[8:24, 8:24, 8:24] = np.linspace(10, 100, 16 ** 3).reshape(16, 16, 16)
        return vol

    def test_segment_py_range_is_zero_to_one(self):
        out = preprocess(self._volume(), intensity_range="0-1")
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_training_range_is_minus_one_to_one(self):
        out = preprocess(self._volume(), intensity_range="minus1-1")
        assert float(out.min()) == pytest.approx(-1.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_background_stays_zero_in_both_conventions(self):
        vol = self._volume()
        for rng in ("0-1", "minus1-1"):
            out = preprocess(vol, intensity_range=rng)[0, 0]
            assert np.all(out[vol == 0] == 0)

    def test_rejects_an_empty_volume(self):
        with pytest.raises(ValueError, match="all zeros"):
            preprocess(np.zeros((8, 8, 8), np.float32))

    def test_rejects_mismatched_t1(self):
        with pytest.raises(ValueError, match="does not match"):
            preprocess(self._volume(), np.zeros((16, 16, 16), np.float32))

    def test_rejects_unknown_options(self):
        with pytest.raises(ValueError, match="intensity_range"):
            preprocess(self._volume(), intensity_range="0-255")
        with pytest.raises(ValueError, match="mask_from"):
            preprocess(self._volume(), mask_from="t1")


class TestAdaptShape:
    def test_crop_and_pad_are_inverse(self):
        vol = np.arange(40 * 44 * 40, dtype=np.float32).reshape(40, 44, 40)
        cropped, padding = adapt_shape(vol, SHAPE)
        assert cropped.shape == SHAPE
        assert np.pad(cropped, padding).shape == vol.shape

    def test_odd_remainder_comes_off_the_low_side(self):
        """Matches segment.py, so masks land on the same voxels as official LST-AI."""
        _, padding = adapt_shape(np.zeros((35, 32, 32), np.float32), SHAPE)
        assert padding[0] == (2, 1)

    def test_rejects_undersized_volumes(self):
        with pytest.raises(ValueError, match="smaller than the model input"):
            adapt_shape(np.zeros((16, 16, 16), np.float32), SHAPE)


class TestRemoveSmallObjects:
    def test_zero_threshold_keeps_everything(self):
        mask = np.zeros((8, 8, 8), np.uint8)
        mask[0, 0, 0] = 1
        assert remove_small_objects(mask, (1, 1, 1), 0).sum() == 1

    def test_drops_components_below_the_threshold(self):
        mask = np.zeros((16, 16, 16), np.uint8)
        mask[0, 0, 0] = 1              # 1 voxel, dropped
        mask[8:12, 8:12, 8:12] = 1     # 64 voxels, kept
        out = remove_small_objects(mask, (1.0, 1.0, 1.0), min_size_mm3=8)
        assert out[0, 0, 0] == 0
        assert out.sum() == 64

    def test_threshold_is_in_cubic_millimetres_not_voxels(self):
        mask = np.zeros((16, 16, 16), np.uint8)
        mask[4:6, 4:6, 4:6] = 1        # 8 voxels
        zooms = (2.0, 2.0, 2.0)        # 8 mm^3 each => 64 mm^3 total
        assert remove_small_objects(mask.copy(), zooms, 32).sum() == 8
        assert remove_small_objects(mask.copy(), zooms, 128).sum() == 0
