"""Regenerate ``tests/data/loss_reference_tf.json`` from the original TensorFlow losses.

The fixture pins what the committed ``unet_models.py`` produces under TensorFlow, so the
ported losses are checked against the real Keras semantics -- the batch-flattened Dice
and the ``K.epsilon()`` = 1e-7 smoothing -- rather than against a reading of them.

TensorFlow is not a dependency of this package, so run this in a throwaway environment::

    uv venv --python 3.11 /tmp/tfvenv
    uv pip install --python /tmp/tfvenv/bin/python "tensorflow==2.19.1"
    /tmp/tfvenv/bin/python tools/make_loss_reference.py

The six cases cover the regimes where the smoothing and clipping actually matter: a
normal mix, a sparse mask (the realistic MS case), empty ground truth, empty prediction,
both empty, and probabilities pushed to 0 and 1.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "data" / "loss_reference_tf.json"
SEED = 7
SHAPE = (2, 1, 8, 8, 8)   # NCDHW, matching what the PyTorch losses take


def build_cases() -> dict[str, dict[str, list]]:
    rng = np.random.default_rng(SEED)
    binary = lambda: rng.integers(0, 2, SHAPE).astype(np.float32)  # noqa: E731
    prob = lambda: rng.uniform(0, 1, SHAPE).astype(np.float32)     # noqa: E731
    raw = {
        "normal": (binary(), prob()),
        "sparse": ((rng.uniform(0, 1, SHAPE) < 0.02).astype(np.float32), prob()),
        "empty_gt": (np.zeros(SHAPE, np.float32), prob()),
        "empty_pred": (binary(), np.zeros(SHAPE, np.float32)),
        "both_empty": (np.zeros(SHAPE, np.float32), np.zeros(SHAPE, np.float32)),
        "saturated": (binary(), np.clip(rng.uniform(-0.1, 1.1, SHAPE), 0, 1).astype(np.float32)),
    }
    return {k: {"y_true": t.tolist(), "y_pred": p.tolist()} for k, (t, p) in raw.items()}


def main() -> int:
    spec = importlib.util.spec_from_file_location("unet_models", REPO / "unet_models.py")
    um = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(um)

    cases = build_cases()
    expected = {}
    for name, case in cases.items():
        # The TF model works in NDHWC; the stored arrays are NCDHW with C=1.
        t = tf.constant(np.moveaxis(np.array(case["y_true"], np.float32), 1, -1))
        p = tf.constant(np.moveaxis(np.array(case["y_pred"], np.float32), 1, -1))
        expected[name] = {
            "dice_binary": float(um.dice_binary(t, p).numpy()),
            "dice_loss_binary": float(um.dice_loss_binary(t, p).numpy()),
            "tversky_loss_binary": float(um.tversky_loss_binary(t, p).numpy()),
            # Keras returns a per-voxel tensor; the trainer reduces it by mean.
            "bce": float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(t, p)).numpy()),
            "bce_dice_loss": float(tf.reduce_mean(um.bce_dice_loss(t, p)).numpy()),
            "bce_tversky_loss": float(tf.reduce_mean(um.bce_tversky_loss(t, p)).numpy()),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"tensorflow_version": tf.__version__, "cases": cases, "expected": expected}))
    print(f"wrote {OUT} from TensorFlow {tf.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
