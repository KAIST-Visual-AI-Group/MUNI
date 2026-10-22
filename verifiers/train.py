"""Train the PolyMNIST visual verifier with fixed final-epoch deployment."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from itertools import chain
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from verifiers import build_verifier_dataset
from verifiers.config import load_verifier_config
from verifiers.model import build_verifier_model


def cosine_warmup_lr(
    step: int,
    total: int,
    warmup: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class MetricAverager:
    def __init__(self):
        self.totals: dict[str, list[float]] = {}

    def update(self, values: Mapping[str, float], batch_size: int) -> None:
        for name, value in values.items():
            total = self.totals.setdefault(name, [0.0, 0.0])
            total[0] += float(value) * batch_size
            total[1] += batch_size

    def compute(self, accelerator: Accelerator) -> dict[str, float]:
        names = sorted(self.totals)
        values = torch.tensor(
            [self.totals[name] for name in names],
            dtype=torch.float64,
            device=accelerator.device,
        )
        values = accelerator.reduce(values, reduction="sum")
        return {
            name: float(numerator / denominator)
            for name, (numerator, denominator) in zip(names, values.tolist())
        }


@contextmanager
def _without_even_batch_padding(data_loader) -> Iterator[None]:
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    if batch_sampler is None or not hasattr(batch_sampler, "even_batches"):
        yield
        return
    even_batches = batch_sampler.even_batches
    batch_sampler.even_batches = False
    try:
        yield
    finally:
        batch_sampler.even_batches = even_batches


class EMA:
    """Exponential moving average of trainable parameters and float buffers."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in chain(model.named_parameters(), model.named_buffers())
            if (not isinstance(tensor, torch.nn.Parameter) or tensor.requires_grad)
            and tensor.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        tensors = dict(chain(model.named_parameters(), model.named_buffers()))
        for name, average in self.shadow.items():
            average.mul_(self.decay).add_(
                tensors[name].detach(), alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        tensors = dict(chain(model.named_parameters(), model.named_buffers()))
        backup = {name: tensors[name].detach().clone() for name in self.shadow}
        with torch.no_grad():
            for name, average in self.shadow.items():
                tensors[name].copy_(average)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    tensors[name].copy_(value)


def _polymnist_step(model, batch, cfg, *, train: bool):
    smoothing = float(cfg["train"].get("label_smoothing", 0.0)) if train else 0.0
    outputs = model(batch["img"])
    losses = {
        name: F.cross_entropy(outputs[name], batch[name], label_smoothing=smoothing)
        for name in ("digit", "quadrant")
    }
    weights = cfg["train"]["task_weights"]
    loss = sum(float(weights[name]) * losses[name] for name in losses)
    metrics = {
        "loss": loss.item(),
        **{f"{name}_ce": value.item() for name, value in losses.items()},
    }
    if not train:
        metrics.update(
            {
                f"{name}_acc": (
                    (outputs[name].argmax(-1) == batch[name]).float().mean().item()
                )
                for name in losses
            }
        )
    return loss, metrics


def _make_loader(dataset, cfg: dict, *, train: bool) -> DataLoader:
    workers = int(cfg["train"].get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def _evaluate(model, loader, step_fn, cfg, accelerator) -> dict[str, float]:
    metrics = MetricAverager()
    with _without_even_batch_padding(loader):
        for batch in loader:
            _, values = step_fn(model, batch, cfg, train=False)
            metrics.update(values, len(batch["img"]))
    return metrics.compute(accelerator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()

    cfg = load_verifier_config(args.config)
    set_seed(int(cfg["train"].get("seed", 0)))
    dataset_type = cfg.get("data", {}).get("type", "polymnist")
    step_fn = _polymnist_step
    log_keys = ("digit_ce", "quadrant_ce")

    accelerator = Accelerator(mixed_precision="bf16")
    train_dataset = build_verifier_dataset(cfg, "train")
    val_dataset = build_verifier_dataset(cfg, "val")
    train_loader = _make_loader(train_dataset, cfg, train=True)
    val_loader = _make_loader(val_dataset, cfg, train=False)

    model = build_verifier_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.999),
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    unwrapped = accelerator.unwrap_model(model)
    ema = EMA(unwrapped, float(cfg["train"].get("ema_decay", 0.9999)))
    start_epoch = 0

    if args.resume and Path(args.resume).is_file():
        checkpoint = torch.load(args.resume, map_location="cpu")
        unwrapped.load_state_dict(checkpoint["model"])
        ema.shadow = {
            name: value.to(accelerator.device)
            for name, value in checkpoint["ema"].items()
        }
        start_epoch = int(checkpoint.get("epoch", 0))

    output_dir = Path(cfg["log"]["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[verifier] {cfg['name']} data.type={dataset_type} "
            f"train={len(train_dataset)} val={len(val_dataset)}",
            flush=True,
        )

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["train"].get("warmup_epochs", 0)) * steps_per_epoch
    base_lr = float(cfg["train"]["lr"])
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    log_freq = int(cfg["log"].get("log_freq", 50))
    eval_every = int(cfg["log"].get("eval_every", 5))
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        train_metrics = MetricAverager()
        for step, batch in enumerate(train_loader):
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, base_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, values = step_fn(model, batch, cfg, train=True)
            accelerator.backward(loss)
            if grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(unwrapped)
            train_metrics.update(values, len(batch["img"]))

            if accelerator.is_main_process and global_step % log_freq == 0:
                details = "  ".join(f"{name}={values[name]:.3f}" for name in log_keys)
                print(
                    f"[verifier] ep {epoch:3d} step {step:4d}/{steps_per_epoch} "
                    f"lr={lr:.2e} loss={values['loss']:.4f}  {details}",
                    flush=True,
                )
            global_step += 1

        train_values = train_metrics.compute(accelerator)
        if accelerator.is_main_process:
            summary = "  ".join(
                f"{name}={value:.4f}" for name, value in train_values.items()
            )
            print(f"[verifier] ep {epoch:3d} TRAIN  {summary}", flush=True)

        if (epoch + 1) % eval_every and epoch + 1 != epochs:
            continue

        model.eval()
        with ema.average_parameters(unwrapped):
            val_values = _evaluate(unwrapped, val_loader, step_fn, cfg, accelerator)

        if accelerator.is_main_process:
            summary = "  ".join(
                f"{name}={value:.4f}"
                for name, value in val_values.items()
                if name.endswith("_acc")
            )
            print(
                f"[verifier] ep {epoch:3d} OPERATIONAL(EMA)  {summary}",
                flush=True,
            )
            checkpoint = {
                "model": unwrapped.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,
                "val_metrics": val_values,
            }
            torch.save(checkpoint, output_dir / "last.pth")


if __name__ == "__main__":
    main()
