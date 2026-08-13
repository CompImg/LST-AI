"""Losses and metrics, ported from the TensorFlow ``unet_models.py``.

Two Keras behaviours are reproduced deliberately, because changing either would change
what the network optimises:

* **The Dice/Tversky terms are global, not per-sample.** ``K.flatten`` flattens the batch
  dimension too, so one ratio is computed over the whole batch. Averaging a per-sample
  Dice instead would weight a near-empty volume as heavily as a lesion-rich one.
* **The smoothing constant is ``K.epsilon()`` = 1e-7**, not the usual 1. It sits in both
  numerator and denominator, so an empty prediction on an empty ground truth scores 1.

``tf.keras.losses.binary_crossentropy`` clips probabilities to ``[1e-7, 1-1e-7]`` and
reduces over the channel axis; Keras then means the result down to a scalar. Since the
model emits probabilities (its head has a baked-in sigmoid, as the released graphs do),
these take probabilities rather than logits and clamp the same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "KERAS_EPSILON", "TVERSKY_ALPHA",
    "dice_binary", "dice_multiclass", "dice_loss_binary", "dice_loss_multiclass",
    "tversky_loss_binary", "binary_crossentropy", "categorical_crossentropy",
    "bce_dice_loss", "cce_dice_loss", "bce_tversky_loss",
    "LOSSES", "DeepSupervisionLoss",
]

KERAS_EPSILON = 1e-7
TVERSKY_ALPHA = 0.75  # weights false negatives over false positives


def dice_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Global Dice over the whole batch (Keras ``dice_binary``)."""
    t, p = y_true.reshape(-1), y_pred.reshape(-1)
    intersection = (t * p).sum()
    total = t.sum() + p.sum()
    return (2.0 * intersection + KERAS_EPSILON) / (total + KERAS_EPSILON)


def dice_multiclass(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Global Dice ignoring the background class (channel 0)."""
    t, p = y_true[:, 1:].reshape(-1), y_pred[:, 1:].reshape(-1)
    intersection = (t * p).sum()
    total = t.sum() + p.sum()
    return (2.0 * intersection + KERAS_EPSILON) / (total + KERAS_EPSILON)


def dice_loss_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_binary(y_true, y_pred)


def dice_loss_multiclass(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_multiclass(y_true, y_pred)


def tversky_loss_binary(y_true: torch.Tensor, y_pred: torch.Tensor,
                        alpha: float = TVERSKY_ALPHA) -> torch.Tensor:
    """Tversky loss; ``alpha`` > 0.5 penalises false negatives more (arXiv:1810.07842)."""
    t, p = y_true.reshape(-1), y_pred.reshape(-1)
    true_pos = (t * p).sum()
    false_neg = (t * (1.0 - p)).sum()
    false_pos = ((1.0 - t) * p).sum()
    denom = true_pos + alpha * false_neg + (1.0 - alpha) * false_pos + KERAS_EPSILON
    return 1.0 - (true_pos + KERAS_EPSILON) / denom


def binary_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean BCE over every element, matching Keras (which clips, then means)."""
    p = y_pred.clamp(KERAS_EPSILON, 1.0 - KERAS_EPSILON)
    return -(y_true * p.log() + (1.0 - y_true) * (1.0 - p).log()).mean()


def categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean CCE over the channel axis, matching Keras (renormalise, clip, then mean)."""
    p = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(KERAS_EPSILON)
    p = p.clamp(KERAS_EPSILON, 1.0 - KERAS_EPSILON)
    return -(y_true * p.log()).sum(dim=1).mean()


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return binary_crossentropy(y_true, y_pred) + dice_loss_binary(y_true, y_pred)


def cce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return categorical_crossentropy(y_true, y_pred) + dice_loss_multiclass(y_true, y_pred)


def bce_tversky_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return binary_crossentropy(y_true, y_pred) + tversky_loss_binary(y_true, y_pred)


LOSSES = {
    "bce_dice": bce_dice_loss,
    "bce_tversky": bce_tversky_loss,
    "dice": dice_loss_binary,
    "tversky": tversky_loss_binary,
    "cce_dice": cce_dice_loss,
}

# The four training runs defined in the original train_model.py.
TRAINING_PRESETS = {
    "nnUNet_bce-dice":    {"loss_out": "bce_dice",    "loss_ds": "bce_dice"},
    "nnUNet_bce-tversky": {"loss_out": "bce_tversky", "loss_ds": "bce_tversky"},
    "nnUNet_dsTversky":   {"loss_out": "bce_dice",    "loss_ds": "bce_tversky"},
    "nnUNet_dsDice":      {"loss_out": "bce_tversky", "loss_ds": "bce_dice"},
}

DS_WEIGHTS = (4 / 7, 2 / 7, 1 / 7)


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of the head losses, as Keras' ``loss_weights`` did.

    ``targets`` must be ordered like the model's outputs: full resolution first, then
    each deep-supervision head coarsening by a factor of two.
    """

    def __init__(self, loss_out=bce_dice_loss, loss_ds=bce_dice_loss,
                 weights: tuple[float, ...] = DS_WEIGHTS):
        super().__init__()
        self.loss_out = loss_out
        self.loss_ds = loss_ds
        self.weights = tuple(weights)

    def forward(self, outputs, targets) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if isinstance(targets, torch.Tensor):
            targets = [targets]
        if len(outputs) != len(targets):
            raise ValueError(f"{len(outputs)} outputs but {len(targets)} targets")
        if len(outputs) > len(self.weights):
            raise ValueError(
                f"{len(outputs)} outputs but only {len(self.weights)} loss weights"
            )

        total = outputs[0].new_zeros(())
        parts: dict[str, float] = {}
        for i, (out, tgt) in enumerate(zip(outputs, targets)):
            fn = self.loss_out if i == 0 else self.loss_ds
            term = fn(tgt, out)
            total = total + self.weights[i] * term
            parts["out_seg" if i == 0 else f"deep_supervision_{i}"] = float(term.detach())
        parts["loss"] = float(total.detach())
        return total, parts
