"""Regenerate ``tests/data/tf_loader_reference.npz`` from the original TF data loader.

The fixture pins what ``MSSequence_3D`` produced for the synthetic subjects defined in
``tests/_synthetic.py``, with augmentation off so the pipeline is deterministic and the
port can be checked for exact equality rather than approximate agreement.

TensorFlow is not a dependency of this package, so run this in a throwaway environment::

    uv venv --python 3.11 /tmp/tfvenv
    uv pip install --python /tmp/tfvenv/bin/python "tensorflow==2.19.1" \\
        nibabel scikit-image scipy
    /tmp/tfvenv/bin/python tools/make_loader_reference.py

Only the loader's *output* is stored -- the input volumes regenerate deterministically
from ``tests/_synthetic.py``, which keeps the fixture small.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "data" / "tf_loader_reference.npz"
SHAPE = (32, 32, 32)


def main() -> int:
    sys.path.insert(0, str(REPO / "tests"))
    from _synthetic import write_subjects

    spec = importlib.util.spec_from_file_location("data_loader", REPO / "data_loader.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    with tempfile.TemporaryDirectory() as tmp:
        root = write_subjects(tmp)
        seq = dl.MSSequence_3D(str(root), batch_size=1, shape=SHAPE, augment=False)
        seq.x = sorted(seq.x)   # the loader shuffles on construction; fix the order
        x, y = seq[0]

        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT, img=x["input_layer"], seg=y["out_seg"],
                 ds1=y["deep_supervision_1"], ds2=y["deep_supervision_2"],
                 subject=np.array([Path(seq.x[0]).name]))
    print(f"wrote {OUT}: image {x['input_layer'].shape}, targets "
          f"{y['out_seg'].shape} / {y['deep_supervision_1'].shape} / "
          f"{y['deep_supervision_2'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def make_augmented_reference():
    """Also pin an *augmented* sample, drawn under a fixed Python RNG seed.

    The augmentations are stochastic, but both loaders drive them from the `random`
    module in the same call order, so seeding identically makes the outputs directly
    comparable. This is what catches ordering bugs -- e.g. data_loader.py augments T1
    before FLAIR, so T1 consumes the first gamma/blur draw; iterating the modalities
    the other way is distributionally identical but produces a different sample.
    """
    import random
    import importlib.util
    import tempfile
    import numpy as np
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("data_loader", REPO / "data_loader.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from _synthetic import write_subjects

    with tempfile.TemporaryDirectory() as tmp:
        root = write_subjects(tmp)
        seq = dl.MSSequence_3D(str(root), batch_size=1, shape=SHAPE, augment=True, aug_prob=0.33)
        seq.x = sorted(seq.x)
        random.seed(1234)
        x, y = seq[0]
        out = REPO / "tests" / "data" / "tf_loader_reference_augmented.npz"
        np.savez(out, img=x["input_layer"], seg=y["out_seg"],
                 ds1=y["deep_supervision_1"], ds2=y["deep_supervision_2"])
    print(f"wrote {out}")
