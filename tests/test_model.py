"""Tests for the PyTorch segmentation network.

Split by cost: everything here except the ``needs_weights`` tests runs without the
~390 MB model bundle, so CI can check the architecture on every push.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lst_ai.model import LEAKY_SLOPE, NORM_EPS, SHIPPED_VARIANTS, NNUNet3D

SMALL = dict(n_conv_blocks=3, n_filters=4)
SIZE = 32

# Set LST_AI_MODEL_DIR to point the weights tests at a bundle.
MODEL_DIR = Path(
    __import__("os").environ.get(
        "LST_AI_MODEL_DIR",
        Path(__file__).resolve().parents[1] / "lst_data" / "model",
    )
)


def _onnx(variant: str) -> Path:
    return MODEL_DIR / f"UNet3D_MS_final_{variant}.onnx"


# --------------------------------------------------------------------- architecture


@pytest.mark.parametrize("in_channels", [1, 2])
def test_forward_shapes(in_channels: int):
    model = NNUNet3D(in_channels=in_channels, ds_layers=(-2, -3), **SMALL)
    out = model(torch.randn(1, in_channels, SIZE, SIZE, SIZE))
    assert [tuple(o.shape) for o in out] == [
        (1, 1, 32, 32, 32), (1, 1, 16, 16, 16), (1, 1, 8, 8, 8)]


def test_outputs_are_probabilities():
    model = NNUNet3D(in_channels=2, ds_layers=(-2,), **SMALL)
    with torch.no_grad():
        for out in model(torch.randn(1, 2, SIZE, SIZE, SIZE) * 10):
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_single_channel_variant_builds():
    """The architecture must also serve a FLAIR-only model for future training."""
    model = NNUNet3D(in_channels=1, **SMALL)
    assert model.stem.block[0].weight.shape[1] == 1


class TestPinnedNumerics:
    """Ancient-TF defaults, not PyTorch's. Changing these breaks the released weights."""

    def test_leaky_relu_slope_is_keras_not_torch(self):
        model = NNUNet3D(in_channels=2, **SMALL)
        slopes = {m.negative_slope for m in model.modules() if isinstance(m, nn.LeakyReLU)}
        assert slopes == {LEAKY_SLOPE} == {0.3}

    def test_instance_norm_epsilon_and_affine(self):
        model = NNUNet3D(in_channels=2, **SMALL)
        norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
        assert norms
        for gn in norms:
            assert gn.eps == NORM_EPS == 1e-3
            assert gn.affine
            assert gn.num_groups == gn.num_channels   # instance, not group, norm

    def test_feature_convolutions_have_no_bias(self):
        model = NNUNet3D(in_channels=2, ds_layers=(-2,), **SMALL)
        heads = {id(model.out_seg), *(id(h) for h in model.ds_heads)}
        for m in model.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) and id(m) not in heads:
                assert m.bias is None


class TestInputShape:
    def test_accepts_the_released_input_size(self):
        NNUNet3D(in_channels=2).check_input_shape((192, 192, 192))

    def test_rejects_sizes_that_go_odd_when_downsampled(self):
        with pytest.raises(ValueError, match="becomes odd"):
            NNUNet3D(in_channels=2, **SMALL).check_input_shape((32, 36, 32))

    def test_rejects_a_single_voxel_bottleneck(self):
        with pytest.raises(ValueError, match="bottleneck"):
            NNUNet3D(in_channels=2, **SMALL).check_input_shape((8, 8, 8))


def test_the_shipped_ensemble_is_heterogeneous():
    """They were trained at different points in the project's history and differ."""
    widths = {v: c["n_filters"] for v, c in SHIPPED_VARIANTS.items()}
    assert len(set(widths.values())) == 3, widths
    assert NNUNet3D.shipped("mdlB").output_names == ["out_seg", "deep_supervision"]
    assert NNUNet3D.shipped("mdlC").output_names == [
        "out_seg", "deep_supervision_1", "deep_supervision_2"]


def test_gradients_reach_every_parameter():
    model = NNUNet3D(in_channels=2, ds_layers=(-2, -3), **SMALL)
    sum(o.mean() for o in model(torch.randn(1, 2, SIZE, SIZE, SIZE))).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


# ------------------------------------------------------------------------- weights


@pytest.mark.needs_weights
@pytest.mark.parametrize("variant", sorted(SHIPPED_VARIANTS))
def test_weights_load_bitwise_from_the_released_graph(variant: str):
    """Every kernel, gamma and beta must equal its ONNX initializer exactly."""
    from lst_ai.weights import convert_variant, extract_onnx_params, ordered_units

    if not _onnx(variant).exists():
        pytest.skip(f"{_onnx(variant)} not present")

    model = convert_variant(variant, MODEL_DIR)
    onnx_units, _ = extract_onnx_params(_onnx(variant))
    for i, (src, (conv, norm)) in enumerate(zip(onnx_units, ordered_units(model))):
        assert np.array_equal(conv.weight.detach().numpy(), src["weight"]), f"unit {i}"
        if norm is not None:
            assert np.array_equal(norm.weight.detach().numpy(), src["gamma"]), f"unit {i}"
            assert np.array_equal(norm.bias.detach().numpy(), src["beta"]), f"unit {i}"


@pytest.mark.needs_weights
@pytest.mark.parametrize("variant", sorted(SHIPPED_VARIANTS))
def test_exported_checkpoint_is_a_bit_exact_substitute_for_the_graph(variant: str, tmp_path):
    """Guards the v2.0.0 bundle: .pt must be indistinguishable from the .onnx it came from.

    The release ships checkpoints exported by ``python -m lst_ai.weights``, so this runs
    that exporter and holds the result to exact equality -- tensors *and* a forward pass.
    A tolerance would be the wrong test here: nothing in the export is allowed to be
    approximate, and if the tensors match then segmentation output matches by
    construction, whatever the subject.
    """
    from lst_ai.weights import convert_variant, main as export_main

    if not _onnx(variant).exists():
        pytest.skip(f"{_onnx(variant)} not present")

    import sys
    from unittest.mock import patch

    argv = ["weights", "--onnx-dir", str(MODEL_DIR), "--out-dir", str(tmp_path),
            "--variants", variant]
    with patch.object(sys, "argv", argv):
        assert export_main() == 0

    # weights_only=True is a property of the artefact we publish, not an implementation
    # detail: a downloaded checkpoint must not be able to execute code at load time.
    ckpt = torch.load(tmp_path / f"UNet3D_MS_final_{variant}.pt",
                      map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    cfg["ds_layers"] = tuple(cfg["ds_layers"])
    restored = NNUNet3D(**cfg)
    restored.load_state_dict(ckpt["state_dict"])
    restored.eval()

    reference = convert_variant(variant, MODEL_DIR)
    a, b = reference.state_dict(), restored.state_dict()
    assert a.keys() == b.keys()
    assert not [k for k in a if not torch.equal(a[k], b[k])]

    torch.manual_seed(0)
    x = torch.randn(1, 2, 64, 64, 64)
    with torch.no_grad():
        ya, yb = reference(x), restored(x)
    for p, q in zip(ya if isinstance(ya, (list, tuple)) else [ya],
                    yb if isinstance(yb, (list, tuple)) else [yb]):
        assert torch.equal(p, q)


@pytest.mark.needs_weights
@pytest.mark.parametrize("variant", sorted(SHIPPED_VARIANTS))
def test_graph_uses_the_numerics_we_pin(variant: str):
    """Read the activation slope and norm epsilon back out of the shipped graph."""
    import onnx
    from onnx import numpy_helper

    if not _onnx(variant).exists():
        pytest.skip(f"{_onnx(variant)} not present")

    graph = onnx.load(str(_onnx(variant))).graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    alphas = {round(float(a.f), 6) for n in graph.node if n.op_type == "LeakyRelu"
              for a in n.attribute if a.name == "alpha"}
    assert alphas == {round(LEAKY_SLOPE, 6)}
    eps = {float(inits[i].reshape(-1)[0]) for n in graph.node if n.op_type == "Add"
           for i in n.input if i in inits and inits[i].size == 1}
    assert all(abs(e - NORM_EPS) < 1e-9 for e in eps)
