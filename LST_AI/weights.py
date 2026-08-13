"""Transfer weights from the shipped LST-AI ONNX ensemble into :class:`NNUNet3D`.

The released ``.onnx`` files were produced by tf2onnx from Keras ``.h5`` models. tf2onnx
wraps the graph in ``Transpose`` nodes (NDHWC->NCDHW at the input, back at the outputs)
but computes in NCDHW throughout, so the convolution kernels are already stored in
PyTorch's layout and transfer verbatim:

    ONNX Conv           weight (out, in,  kd, kh, kw)  == nn.Conv3d.weight
    ONNX ConvTranspose  weight (in,  out, kd, kh, kw)  == nn.ConvTranspose3d.weight

The instance-norm layers are not a single op: tf2onnx expands them into primitive
arithmetic, leaving gamma and beta as folded ``[1,1,1,1,C,1]`` initializers consumed by a
``Mul`` (gamma, against rsqrt(var+eps)) and a ``Sub`` (beta). They are recovered by
walking the graph in file order.

That relies on the convolutions appearing in forward order, which constant folding could
in principle disturb -- it does hoist unrelated nodes into the middle of a block. It was
checked against a longest-path topological sort of every shipped graph and holds for all
of them, with the 1x1 heads the only difference: the file lists ``out_seg`` before the
deep-supervision heads, which is the order ``ordered_units`` builds. ``load_onnx_weights``
validates the count, op type and shape of every unit, so a future graph that did reorder
would fail loudly rather than silently load transposed weights.

Usage::

    python -m lst_ai_torch.convert --onnx-dir lst_data/model --out-dir checkpoints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from LST_AI.model import NNUNet3D, SHIPPED_VARIANTS

__all__ = ["ordered_units", "extract_onnx_params", "load_onnx_weights"]


def ordered_units(model: NNUNet3D) -> list[tuple[nn.Module, nn.Module | None]]:
    """(conv, norm) pairs in forward order; the norm is ``None`` for the 1x1 heads."""
    units: list[tuple[nn.Module, nn.Module | None]] = []

    def add_conv_block(cb):  # ConvBlock -> [conv, norm, act, conv, norm, act]
        units.append((cb.block[0], cb.block[1]))
        units.append((cb.block[3], cb.block[4]))

    def add_single(m):  # DownConv / UpConv -> [conv, norm, act]
        units.append((m.block[0], m.block[1]))

    add_conv_block(model.stem)
    for down, conv in zip(model.enc_down, model.enc_conv):
        add_single(down)
        add_conv_block(conv)

    add_single(model.bottleneck_down)
    add_conv_block(model.bottleneck_conv)

    for up, conv in zip(model.dec_up, model.dec_conv):
        add_single(up)
        add_conv_block(conv)

    units.append((model.out_seg, None))
    for head in model.ds_heads:
        units.append((head, None))
    return units


def extract_onnx_params(onnx_path: str | os.PathLike) -> tuple[list[dict], list[float]]:
    """Walk the graph, returning per-unit {weight, bias, gamma, beta} plus the epsilons.

    A unit is closed by its convolution; norm params seen after it (before the next conv)
    belong to it. The 1x1 heads have no norm, so their gamma/beta stay ``None``.
    """
    graph = onnx.load(str(onnx_path)).graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    units: list[dict] = []
    epsilons: list[float] = []

    def init_of(node, ndim=None, scalar=False):
        for name in node.input:
            arr = inits.get(name)
            if arr is None:
                continue
            if scalar and arr.size == 1:
                return arr
            if ndim is not None and arr.ndim == ndim:
                return arr
        return None

    for node in graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            w = init_of(node, ndim=5)
            if w is None:
                continue
            bias = None
            if len(node.input) > 2 and node.input[2] in inits:
                bias = inits[node.input[2]]
            units.append({"weight": w, "bias": bias, "gamma": None, "beta": None,
                          "op": node.op_type})
        elif node.op_type == "Mul" and units:
            g = init_of(node, ndim=6)
            if g is not None and units[-1]["gamma"] is None:
                units[-1]["gamma"] = g.reshape(-1)
        elif node.op_type == "Sub" and units:
            b = init_of(node, ndim=6)
            if b is not None and units[-1]["beta"] is None:
                units[-1]["beta"] = b.reshape(-1)
        elif node.op_type == "Add":
            e = init_of(node, scalar=True)
            if e is not None:
                epsilons.append(float(np.asarray(e).reshape(-1)[0]))

    return units, epsilons


def load_onnx_weights(model: NNUNet3D, onnx_path: str | os.PathLike,
                      strict: bool = True) -> NNUNet3D:
    """Load ONNX weights into ``model`` in place, validating shapes as it goes."""
    onnx_units, epsilons = extract_onnx_params(onnx_path)
    torch_units = ordered_units(model)

    if len(onnx_units) != len(torch_units):
        raise ValueError(
            f"unit count mismatch: ONNX has {len(onnx_units)} convolutions, the module has "
            f"{len(torch_units)}. The variant config probably does not match this file."
        )

    if strict and epsilons:
        from .model import NORM_EPS
        bad = [e for e in epsilons if abs(e - NORM_EPS) > 1e-9]
        if bad:
            raise ValueError(
                f"ONNX norm epsilon {sorted(set(bad))} != module NORM_EPS {NORM_EPS}; "
                "the released weights expect the TF/tfa default of 1e-3."
            )

    with torch.no_grad():
        for i, (src, (conv, norm)) in enumerate(zip(onnx_units, torch_units)):
            want_transposed = isinstance(conv, nn.ConvTranspose3d)
            if want_transposed != (src["op"] == "ConvTranspose"):
                raise ValueError(
                    f"unit {i}: ONNX op {src['op']} does not match module layer "
                    f"{type(conv).__name__}"
                )
            # np.array copies: ONNX initializers are read-only views into the model file.
            w = torch.from_numpy(np.array(src["weight"]))
            if tuple(w.shape) != tuple(conv.weight.shape):
                raise ValueError(
                    f"unit {i} ({type(conv).__name__}): ONNX weight {tuple(w.shape)} vs "
                    f"module {tuple(conv.weight.shape)}"
                )
            conv.weight.copy_(w)

            if src["bias"] is not None:
                if conv.bias is None:
                    raise ValueError(f"unit {i}: ONNX has a bias but the module layer does not")
                conv.bias.copy_(torch.from_numpy(np.array(src["bias"])))
            elif conv.bias is not None:
                raise ValueError(f"unit {i}: module layer has a bias but ONNX does not")

            if norm is not None:
                if src["gamma"] is None or src["beta"] is None:
                    raise ValueError(f"unit {i}: missing instance-norm gamma/beta in the ONNX graph")
                norm.weight.copy_(torch.from_numpy(np.array(src["gamma"])))
                norm.bias.copy_(torch.from_numpy(np.array(src["beta"])))
            elif src["gamma"] is not None:
                raise ValueError(f"unit {i}: ONNX has norm params but the module layer has no norm")

    return model


def convert_variant(variant: str, onnx_dir: str | os.PathLike) -> NNUNet3D:
    """Build the topology for ``variant`` and fill it from its shipped ONNX file."""
    path = Path(onnx_dir) / f"UNet3D_MS_final_{variant}.onnx"
    model = NNUNet3D.shipped(variant, in_channels=2)
    load_onnx_weights(model, path)
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", required=True, help="directory holding UNet3D_MS_final_mdl*.onnx")
    ap.add_argument("--out-dir", required=True, help="where to write the .pt checkpoints")
    ap.add_argument("--variants", nargs="+", default=list(SHIPPED_VARIANTS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        model = convert_variant(variant, args.onnx_dir)
        dest = out_dir / f"UNet3D_MS_final_{variant}.pt"
        torch.save(
            {"variant": variant, "config": {"in_channels": 2, **SHIPPED_VARIANTS[variant]},
             "state_dict": model.state_dict()},
            dest,
        )
        n = sum(p.numel() for p in model.parameters())
        print(f"{variant}: {n/1e6:.2f} M params -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
