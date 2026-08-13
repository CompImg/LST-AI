"""Native PyTorch nnU-Net-style 3D UNet for LST-AI.

A faithful re-implementation of the TensorFlow ``unet_models.nnunet_3d`` used to train
the LST-AI ensemble. It serves both purposes of this port:

  * loaded with weights transferred from the shipped ``.onnx`` ensemble it reproduces
    official LST-AI inference (see ``convert.py`` / ``tests/test_parity.py``);
  * freshly initialised it trains from scratch on new data, single- or dual-channel.

Numerics: the released models were trained on TensorFlow <2.12 with
``tfa.layers.InstanceNormalization``, whose defaults differ from PyTorch's. These are NOT
PyTorch defaults and must not be "tidied up" -- they are what the released weights expect:

    LeakyReLU slope   0.3     (torch default 0.01)
    norm epsilon      1e-3    (torch default 1e-5)
    norm affine       True    (nn.InstanceNorm3d defaults to False)
    conv bias         absent  (use_bias=False on every conv/downconv/upconv)

``GroupNorm(num_groups=C, num_channels=C)`` is instance norm with per-channel affine
parameters, matching ``tfa.InstanceNormalization`` == ``GroupNormalization(groups=-1)``.

Padding: TF ``padding='same'`` is symmetric for every (kernel, stride) used here as long
as each spatial size is even at the point it is downsampled -- true for the 192^3 input
(192 -> 96 -> 48 -> 24 -> 12 -> 6). ``check_input_shape`` enforces that.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["NNUNet3D", "SHIPPED_VARIANTS", "LEAKY_SLOPE", "NORM_EPS"]

LEAKY_SLOPE = 0.3
NORM_EPS = 1e-3

# Topologies of the three shipped v1.3.0 ensemble members, recovered from their ONNX
# graphs. They were trained at different points in the code's history, hence the drift:
# mdlA/B widen the bottleneck to n_filters*(n_conv_blocks+1); mdlC keeps it at
# n_filters*n_conv_blocks and is the one the committed TF code reproduces. mdlB has a
# single deep-supervision head, and its ONNX names it "deep_supervision" (no suffix).
SHIPPED_VARIANTS = {
    "mdlA": dict(n_filters=28, n_conv_blocks=5, bottleneck_filters=168, ds_layers=(-2, -3)),
    "mdlB": dict(n_filters=24, n_conv_blocks=5, bottleneck_filters=144, ds_layers=(-2,)),
    "mdlC": dict(n_filters=32, n_conv_blocks=5, bottleneck_filters=160, ds_layers=(-2, -3)),
}


def _norm(channels: int) -> nn.GroupNorm:
    """Instance norm with affine params, matching tfa.InstanceNormalization."""
    return nn.GroupNorm(num_groups=channels, num_channels=channels, eps=NORM_EPS, affine=True)


def _act() -> nn.LeakyReLU:
    return nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True)


class ConvBlock(nn.Module):
    """TF ``conv_3d``: two (conv 3x3x3 -> instance norm -> leaky ReLU) stages."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(out_channels),
            _act(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(out_channels),
            _act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownConv(nn.Module):
    """TF ``downconv_3d``: strided 4x4x4 convolution, halving each spatial dim."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            _norm(out_channels),
            _act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConv(nn.Module):
    """TF ``upconv_3d``: strided 4x4x4 transposed convolution, doubling each spatial dim."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2,
                               padding=1, bias=False),
            _norm(out_channels),
            _act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NNUNet3D(nn.Module):
    """nnU-Net-style 3D UNet with optional deep supervision.

    Parameters
    ----------
    in_channels : int
        1 for a FLAIR-only model, 2 for the dual-channel [FLAIR, T1] model. Channel order
        is ``[flair, t1]``, matching ``LST_AI/segment.py``.
    n_conv_blocks : int
        Number of encoder resolution levels (excluding the bottleneck).
    n_filters : int
        Base width; encoder level *i* (1-indexed) has ``n_filters * i`` channels.
    n_classes : int
        1 gives a single sigmoid channel; >1 gives softmax over classes (background
        included, as in the TF code).
    ds_layers : tuple[int, ...]
        Decoder blocks to attach deep-supervision heads to, using the TF code's backwards
        indexing: ``(-2, -3)`` puts heads at 1/2 and 1/4 resolution. Empty disables it.
    bottleneck_filters : int | None
        Bottleneck width. Defaults to ``n_filters * n_conv_blocks`` (mdlC and the
        committed TF code); mdlA/B used ``n_filters * (n_conv_blocks + 1)``.

    Forward returns a list ``[out_seg, *deep_supervision]`` when ``ds_layers`` is
    non-empty, else the single ``out_seg`` tensor -- mirroring the TF model's outputs.
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_conv_blocks: int = 5,
        n_filters: int = 32,
        n_classes: int = 1,
        ds_layers: tuple[int, ...] = (),
        bottleneck_filters: int | None = None,
    ):
        super().__init__()
        if n_conv_blocks < 2:
            raise ValueError("n_conv_blocks must be >= 2")
        ds_layers = tuple(ds_layers)
        for d in ds_layers:
            # decoder_convs holds n_conv_blocks entries; TF indexes it backwards.
            if d == -1:
                raise ValueError(
                    "ds_layers may not contain -1: that is the full-resolution decoder "
                    "output, which out_seg already reads, so a head there would just "
                    "duplicate it."
                )
            if not -n_conv_blocks <= d < 0:
                raise ValueError(
                    f"ds_layers entries must be negative and >= -{n_conv_blocks}, got {d}"
                )

        self.in_channels = in_channels
        self.n_conv_blocks = n_conv_blocks
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.ds_layers = ds_layers
        self.bottleneck_filters = bottleneck_filters or n_filters * n_conv_blocks

        enc_widths = [n_filters * i for i in range(1, n_conv_blocks + 1)]
        self.enc_widths = enc_widths

        # Encoder: level 0 is a plain conv block at full resolution; each subsequent
        # level downsamples then convolves.
        self.stem = ConvBlock(in_channels, enc_widths[0])
        self.enc_down = nn.ModuleList()
        self.enc_conv = nn.ModuleList()
        for i in range(1, n_conv_blocks):
            self.enc_down.append(DownConv(enc_widths[i - 1], enc_widths[i]))
            self.enc_conv.append(ConvBlock(enc_widths[i], enc_widths[i]))

        # Bottleneck
        self.bottleneck_down = DownConv(enc_widths[-1], self.bottleneck_filters)
        self.bottleneck_conv = ConvBlock(self.bottleneck_filters, self.bottleneck_filters)

        # Decoder: mirrors the encoder, concatenating the skip before each conv block.
        self.dec_up = nn.ModuleList()
        self.dec_conv = nn.ModuleList()
        prev = self.bottleneck_filters
        for level in range(n_conv_blocks - 1, -1, -1):
            width = enc_widths[level]
            self.dec_up.append(UpConv(prev, width))
            self.dec_conv.append(ConvBlock(width * 2, width))  # skip is the same width
            prev = width

        # Heads. decoder outputs are ordered coarse -> fine, so decoder[-1] is full
        # resolution (out_seg) and decoder[-2], decoder[-3] are 1/2 and 1/4.
        dec_widths = [enc_widths[level] for level in range(n_conv_blocks - 1, -1, -1)]
        self.out_seg = nn.Conv3d(dec_widths[-1], n_classes, kernel_size=1)
        self.ds_heads = nn.ModuleList(
            [nn.Conv3d(dec_widths[d], n_classes, kernel_size=1) for d in ds_layers]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """he_uniform for feature convs, glorot_uniform for the 1x1 heads (as in TF).

        ``kaiming_uniform_(nonlinearity='relu')`` gives bound ``sqrt(6/fan_in)``, which is
        exactly Keras ``he_uniform``. PyTorch computes fan_in from ``weight.size(1)``,
        which for ``ConvTranspose3d`` is the *output* channel count -- and that matches
        Keras, whose transposed-conv kernel is ``(k, k, k, filters, in_channels)`` with
        ``fan_in = filters * k^3``. So the same call is correct for both conv types.
        """
        heads = {id(self.out_seg), *(id(h) for h in self.ds_heads)}
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if id(m) in heads:
                    nn.init.xavier_uniform_(m.weight)  # Keras glorot_uniform
                else:
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def check_input_shape(self, shape: tuple[int, int, int]) -> None:
        """Raise if this input size is incompatible with the architecture.

        Two constraints. Every spatial dim must stay even through all ``n_conv_blocks``
        downsamplings, otherwise a stride-2 kernel-4 'same' conv pads asymmetrically in
        TF and this module's symmetric ``padding=1`` would not match. And the bottleneck
        must keep more than one voxel, since instance norm over a single voxel divides by
        ``sqrt(eps)`` and collapses the block's output to its bias.
        """
        for axis, size in enumerate(shape):
            s = size
            for level in range(self.n_conv_blocks):
                if s % 2:
                    raise ValueError(
                        f"spatial dim {axis} (size {size}) becomes odd ({s}) at downsampling "
                        f"level {level}; it must be divisible by 2**{self.n_conv_blocks} "
                        f"({2 ** self.n_conv_blocks}) for symmetric padding to match TF."
                    )
                s //= 2

        bottleneck = [s // 2 ** self.n_conv_blocks for s in shape]
        if int(np.prod(bottleneck)) <= 1:
            raise ValueError(
                f"input {tuple(shape)} leaves a {tuple(bottleneck)} bottleneck after "
                f"{self.n_conv_blocks} downsamplings; instance norm needs more than one "
                f"voxel there. Use an input of at least "
                f"{2 ** (self.n_conv_blocks + 1)} along one axis, or lower n_conv_blocks."
            )

    def forward(self, x: torch.Tensor):
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down, conv in zip(self.enc_down, self.enc_conv):
            h = conv(down(h))
            skips.append(h)

        h = self.bottleneck_conv(self.bottleneck_down(h))

        decoder_outs = []
        for i, (up, conv) in enumerate(zip(self.dec_up, self.dec_conv)):
            level = self.n_conv_blocks - 1 - i
            h = up(h)
            h = conv(torch.cat([skips[level], h], dim=1))  # TF concat([skip, deconv])
            decoder_outs.append(h)

        activate = torch.sigmoid if self.n_classes == 1 else lambda t: torch.softmax(t, dim=1)
        out = [activate(self.out_seg(decoder_outs[-1]))]
        for head, d in zip(self.ds_heads, self.ds_layers):
            out.append(activate(head(decoder_outs[d])))

        return out if self.ds_heads else out[0]

    @property
    def output_names(self) -> list[str]:
        """Output names matching the shipped ONNX graphs."""
        if not self.ds_layers:
            return ["out_seg"]
        if len(self.ds_layers) == 1:
            return ["out_seg", "deep_supervision"]  # mdlB's older naming
        return ["out_seg"] + [f"deep_supervision_{i + 1}" for i in range(len(self.ds_layers))]

    @classmethod
    def shipped(cls, variant: str, in_channels: int = 2) -> "NNUNet3D":
        """Build one of the released ensemble topologies (``mdlA`` / ``mdlB`` / ``mdlC``)."""
        if variant not in SHIPPED_VARIANTS:
            raise KeyError(f"unknown variant {variant!r}; expected one of {list(SHIPPED_VARIANTS)}")
        return cls(in_channels=in_channels, **SHIPPED_VARIANTS[variant])
