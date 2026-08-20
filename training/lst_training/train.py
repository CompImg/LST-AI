"""Train an LST-AI model from scratch, ported from the TensorFlow ``train_model.py``.

Reproduces the original recipe -- SGD(momentum 0.9, Nesterov), lr 1e-2 on a cosine
schedule, deep supervision at 1/2 and 1/4 resolution weighted 4/7 : 2/7 : 1/7, and
checkpointing on the best *training* ``out_seg`` loss -- while adding the single-channel
option the TF code only had commented out.

    # dual-channel FLAIR + T1, the configuration the released models use
    python -m lst_training.train --train-data DIR --val-data DIR --in-channels 2

    # FLAIR only
    python -m lst_training.train --train-data DIR --in-channels 1

Two deviations from the TF script are deliberate and both are visible in ``--help``:
``--epochs`` defaults to the original 1001 but the cosine schedule is driven by it, so
shortening a run rescales the schedule rather than truncating it; and checkpoints are
saved as ``.pt`` state dicts rather than ``.h5``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import MSDataset, collate
from .losses import DS_WEIGHTS, LOSSES, TRAINING_PRESETS, DeepSupervisionLoss, dice_binary
from lst_ai.model import NNUNet3D

__all__ = ["train", "build_argparser"]


def cosine_annealing(epoch: int, n_epochs: int) -> float:
    """The original schedule, as a multiplier on the initial learning rate."""
    return 0.5 * (1.0 + math.cos((epoch / n_epochs) * math.pi))


class _TensorBoard:
    """Optional TensorBoard logging; a no-op when disabled or unavailable.

    Kept optional on purpose. The JSON history written every epoch is the record that
    always exists and needs no dependency -- TensorBoard is for watching a long run, not
    for reconstructing it afterwards. `tensorboard` is a standalone package and does not
    drag TensorFlow back in, but there is no reason to make training require it.
    """

    def __init__(self, log_dir: Path | None):
        self.writer = None
        if log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise SystemExit(
                f"--tensorboard needs the tensorboard package ({exc}). "
                "pip install tensorboard, or drop the flag: the JSON history in "
                "--out-dir is written either way."
            )
        self.writer = SummaryWriter(str(log_dir))
        print(f"tensorboard: tensorboard --logdir {log_dir}")

    def log(self, row: dict, epoch: int) -> None:
        if self.writer is None:
            return
        for key, value in row.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            # train_dice -> train/dice, so TensorBoard groups the two splits and puts
            # each metric's train and val curves on the same axes.
            tag = key.replace("train_", "train/").replace("val_", "val/")
            self.writer.add_scalar(tag if "/" in tag else f"misc/{tag}", value, epoch)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _run_epoch(model, loader, criterion, device, optimizer=None, amp=False):
    """One pass; trains when ``optimizer`` is given, evaluates otherwise."""
    training = optimizer is not None
    model.train(training)
    scaler = getattr(_run_epoch, "_scaler", None)

    totals: dict[str, float] = {}
    dice_sum, n = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images)
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
                loss, parts = criterion(outputs, targets[: len(outputs)])

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        dice_sum += float(dice_binary(targets[0], outputs[0].detach().float()))
        n += 1

    if n == 0:
        raise RuntimeError("the data loader produced no batches -- is the dataset empty?")
    metrics = {k: v / n for k, v in totals.items()}
    metrics["dice"] = dice_sum / n
    return metrics


def train(args) -> Path:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ds_layers = tuple(args.ds_layers)
    n_ds = len(ds_layers)

    train_ds = MSDataset(args.train_data, shape=tuple(args.shape),
                         in_channels=args.in_channels, augment=not args.no_augment,
                         aug_prob=args.aug_prob, n_deep_supervision=n_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              drop_last=True, pin_memory=device.type == "cuda")
    val_loader = None
    if args.val_data:
        val_ds = MSDataset(args.val_data, shape=tuple(args.shape),
                           in_channels=args.in_channels, augment=False,
                           n_deep_supervision=n_ds)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, collate_fn=collate,
                                pin_memory=device.type == "cuda")

    model = NNUNet3D(in_channels=args.in_channels, n_conv_blocks=args.conv_blocks,
                     n_filters=args.filters, ds_layers=ds_layers,
                     bottleneck_filters=args.bottleneck_filters).to(device)
    model.check_input_shape(tuple(args.shape))

    preset = TRAINING_PRESETS.get(args.preset)
    loss_out = LOSSES[preset["loss_out"] if preset else args.loss_out]
    loss_ds = LOSSES[preset["loss_ds"] if preset else args.loss_ds]
    criterion = DeepSupervisionLoss(loss_out=loss_out, loss_ds=loss_ds,
                                    weights=DS_WEIGHTS[: n_ds + 1])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: cosine_annealing(e, args.epochs))
    _run_epoch._scaler = torch.amp.GradScaler(device.type) if args.amp else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.preset or f"nnUNet_{args.in_channels}ch"
    config = {"in_channels": args.in_channels, "n_conv_blocks": args.conv_blocks,
              "n_filters": args.filters, "ds_layers": list(ds_layers),
              "bottleneck_filters": model.bottleneck_filters}

    print(f"{name}: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params, "
          f"{len(train_ds)} training volumes"
          + (f", {len(val_ds)} validation volumes" if val_loader else "")
          + f", {n_ds} deep-supervision head(s), device {device}")

    tb_dir = None
    if args.tensorboard:
        tb_dir = Path(args.tensorboard) if args.tensorboard is not True else out_dir / "tb" / name
    board = _TensorBoard(tb_dir)

    history, best = [], float("inf")
    best_path = out_dir / f"UNet3D_MS_lowestTrainLoss_{name}.pt"
    for epoch in range(args.epochs):
        t0 = time.time()
        stats = _run_epoch(model, train_loader, criterion, device, optimizer, args.amp)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               **{f"train_{k}": v for k, v in stats.items()}}

        if val_loader is not None:
            val = _run_epoch(model, val_loader, criterion, device, None, args.amp)
            row.update({f"val_{k}": v for k, v in val.items()})

        scheduler.step()
        row["seconds"] = time.time() - t0
        history.append(row)

        # The original saved on the best *training* out_seg loss, not validation.
        if stats["out_seg"] < best:
            best = stats["out_seg"]
            torch.save({"variant": name, "config": config, "epoch": epoch,
                        "train_out_seg_loss": best, "state_dict": model.state_dict()},
                       best_path)

        msg = (f"epoch {epoch:4d}/{args.epochs}  lr {row['lr']:.2e}  "
               f"loss {stats['loss']:.4f}  dice {stats['dice']:.4f}")
        if val_loader is not None:
            msg += f"  |  val loss {row['val_loss']:.4f}  val dice {row['val_dice']:.4f}"
        print(msg + f"  ({row['seconds']:.1f}s)")

        board.log(row, epoch)
        (out_dir / f"UNet3D_MS_final_{name}.json").write_text(json.dumps(history, indent=1))

    board.close()
    final_path = out_dir / f"UNet3D_MS_final_{name}.pt"
    torch.save({"variant": name, "config": config, "epoch": args.epochs - 1,
                "state_dict": model.state_dict()}, final_path)
    print(f"wrote {final_path} (best training out_seg loss {best:.4f} -> {best_path})")
    return final_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--train-data", required=True, help="directory of *_flair.nii.gz subjects")
    d.add_argument("--val-data", default=None)
    d.add_argument("--shape", type=int, nargs=3, default=(192, 192, 192))
    d.add_argument("--in-channels", type=int, default=2, choices=(1, 2),
                   help="1 = FLAIR only, 2 = FLAIR + T1 (the released configuration)")
    d.add_argument("--no-augment", action="store_true")
    d.add_argument("--aug-prob", type=float, default=0.33,
                   help="each augmentation fires when random() > this (original semantics)")

    m = p.add_argument_group("model")
    m.add_argument("--filters", type=int, default=32)
    m.add_argument("--conv-blocks", type=int, default=5)
    m.add_argument("--bottleneck-filters", type=int, default=None,
                   help="default n_filters*n_conv_blocks (mdlC); mdlA/B used one level wider")
    m.add_argument("--ds-layers", type=int, nargs="*", default=(-2, -3),
                   help="decoder blocks carrying deep-supervision heads; empty disables it")

    o = p.add_argument_group("optimisation")
    o.add_argument("--epochs", type=int, default=1001)
    o.add_argument("--batch-size", type=int, default=2)
    o.add_argument("--lr", type=float, default=1e-2)
    o.add_argument("--preset", default=None, choices=sorted(TRAINING_PRESETS),
                   help="one of the four original runs; overrides --loss-out/--loss-ds")
    o.add_argument("--loss-out", default="bce_dice", choices=sorted(LOSSES))
    o.add_argument("--loss-ds", default="bce_dice", choices=sorted(LOSSES))

    r = p.add_argument_group("runtime")
    r.add_argument("--out-dir", default="checkpoints")
    r.add_argument("--name", default=None)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tensorboard", nargs="?", const=True, default=None, metavar="DIR",
                   help="log scalars to TensorBoard; defaults to <out-dir>/tb/<name>. "
                        "Needs `pip install tensorboard`. The JSON history is written "
                        "regardless.")
    return p


def main() -> int:
    train(build_argparser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
