"""Training loop for config-driven stochastic routing."""

from collections.abc import Mapping
from contextlib import nullcontext
from collections import defaultdict
import math

import torch

import muni.util.misc as misc
from muni.util.misc import move_to_device, print_on_main as print
from muni.training.batching import resolve_batch_sizes

from muni.training.routing import (
    RoutingPolicy,
    _prefix_training_metrics,
)


def _batch_size(data):
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    if isinstance(data, Mapping):
        for value in data.values():
            size = _batch_size(value)
            if size is not None:
                return size
    return None


def _slice_batch(data, start, end):
    if isinstance(data, torch.Tensor):
        return data[start:end]
    if isinstance(data, Mapping):
        return {key: _slice_batch(value, start, end) for key, value in data.items()}
    return data


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
    _, per_rank_batch, micro_batch = resolve_batch_sizes(
        cfg.train, misc.get_world_size()
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

        batch_size = _batch_size(dict_data)
        if batch_size != per_rank_batch:
            raise ValueError(
                "Training loader must yield a full per-rank optimizer batch: "
                f"expected {per_rank_batch} samples, got {batch_size}."
            )

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

        loss_value = 0.0
        split_batch = micro_batch < batch_size
        if split_batch:
            metric_sums = defaultdict(float)
            metric_weights = defaultdict(float)
        for start in range(0, batch_size, micro_batch):
            end = min(start + micro_batch, batch_size)
            weight = (end - start) / batch_size
            # Keep the optimizer batch on CPU and transfer only the current
            # chunk. Do not slice or multiply the loss on the no-split path.
            # Stochastic draws, forward-local subsampling, and batch-statistic
            # regularizers/diagnostics remain microbatch-local.
            chunk = _slice_batch(dict_data, start, end) if split_batch else dict_data
            chunk = move_to_device(chunk, device, non_blocking=True)
            sync_context = (
                model.no_sync()
                if end < batch_size and isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else nullcontext()
            )
            # DDP requires both forward and backward inside no_sync.
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    loss, raw_metrics = model(
                        chunk,
                        enc_modalities,
                        tgt_modality,
                        stop_grad_modalities=stop_grad_modalities,
                    )

                chunk_loss = float(loss.detach().item())
                if not math.isfinite(chunk_loss):
                    raise FloatingPointError(
                        "Non-finite loss encountered before optimizer step: "
                        f"loss={chunk_loss}, epoch={epoch}, step={data_iter_step}, "
                        f"path={train_path_type}, enc={enc_modalities}, tgt={tgt_modality}"
                    )
                (loss if weight == 1.0 else loss * weight).backward()

            if split_batch:
                loss_value += chunk_loss * weight
                for key, value in raw_metrics.items():
                    metric_sums[key] += float(value) * weight
                    metric_weights[key] += weight
                del raw_metrics
            else:
                loss_value = chunk_loss
            del chunk, loss

        if split_batch:
            raw_metrics = {
                key: value / metric_weights[key] for key, value in metric_sums.items()
            }
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
