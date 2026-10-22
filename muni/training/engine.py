"""Training loop for config-driven stochastic routing."""

from collections import defaultdict
import math

import torch

import muni.util.misc as misc
from muni.util.misc import move_to_device, print_on_main as print

from muni.training.routing import (
    RoutingPolicy,
    _prefix_training_metrics,
)


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    lr_scheduler,
    device,
    epoch,
    log_writer=None,
    *,
    cfg,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = int(cfg.checkpoint.get("print_freq", 50))

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    log_freq = int(cfg.checkpoint.log_freq)
    if log_freq <= 0:
        raise ValueError(f"checkpoint.log_freq must be positive, got {log_freq}.")
    routing = RoutingPolicy.from_config(
        cfg.model.routing,
        seed=int(cfg.train.seed) + epoch * 10007,
    )

    # Accumulate sparse route metrics so every metric observed in an interval
    # is logged, even when it is absent on the final step.
    metric_accum: dict[str, list[float]] = defaultdict(list)
    loss_accum: list[float] = []

    max_steps = cfg.train.get("max_steps", None)
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        dict_data = misc.extract_dict_data(batch)

        dict_data = {
            key: move_to_device(value, device, non_blocking=True)
            for key, value in dict_data.items()
        }

        available = [
            modality for modality in routing.modalities if modality in dict_data
        ]
        train_path_type, enc_modalities, tgt_modality, stop_grad_modalities = (
            routing.sample(
                available,
                device,
            )
        )

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            loss, raw_metrics = model(
                dict_data,
                enc_modalities,
                tgt_modality,
                stop_grad_modalities=stop_grad_modalities,
            )

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                "Non-finite loss encountered before optimizer step: "
                f"loss={loss_value}, epoch={epoch}, step={data_iter_step}, "
                f"path={train_path_type}, enc={enc_modalities}, tgt={tgt_modality}"
            )

        loss.backward()
        grad_clip = cfg.train.get("grad_clip", None)
        if grad_clip is not None and grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(
                model_without_ddp.parameters(), max_norm=float(grad_clip), foreach=True
            )
        lr_current = optimizer.param_groups[0]["lr"]
        optimizer.step()
        lr_scheduler.step()

        model_without_ddp.ema.update(model_without_ddp)

        prefixed = _prefix_training_metrics(
            raw_metrics,
            tgt_modality=tgt_modality,
            loss_value=loss_value,
        )
        loss_accum.append(loss_value)
        for k, v in prefixed.items():
            metric_accum[k].append(float(v))

        do_print = data_iter_step % print_freq == 0
        if do_print:
            metric_logger.update(loss=loss_value, **prefixed)

        metric_logger.update(lr=lr_current)

        is_last_step = (data_iter_step + 1) == len(data_loader)
        is_max_steps = max_steps is not None and data_iter_step + 1 >= max_steps
        is_interval_end = (data_iter_step % log_freq) == (log_freq - 1)
        do_log = is_interval_end or is_last_step or is_max_steps
        if do_log and loss_accum:
            n = len(loss_accum)
            loss_mean = sum(loss_accum) / n
            metric_means = {
                k: sum(vs) / len(vs) for k, vs in metric_accum.items() if vs
            }
            loss_mean_reduce = misc.all_reduce_mean(loss_mean)
            metric_means_reduce = {
                k: misc.all_reduce_mean(v) for k, v in metric_means.items()
            }
            if log_writer is not None:
                global_step = epoch * len(data_loader) + data_iter_step
                log_writer.add_scalar("train_loss", loss_mean_reduce, global_step)
                for k, v_red in metric_means_reduce.items():
                    log_writer.add_scalar(k, v_red, global_step)
                log_writer.add_scalar("lr", lr_current, global_step)
            loss_accum.clear()
            metric_accum.clear()

        if is_max_steps:
            break
