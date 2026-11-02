"""MUNI training entrypoint."""

import argparse
import datetime
import os
from pathlib import Path
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

from muni.config_utils import (
    ConfigOverrideError,
    MissingConfigEnvironmentError,
    load_config,
    print_config,
)
from muni.data import build_dataset, build_configured_eval_loader, build_train_loader
from muni.evaluation.runner import run_evaluation_suite
from muni.framework.model import EMACheckpointError, build_model
from muni.training.batching import resolve_batch_sizes
from muni.training.engine import train_one_epoch
import muni.util.misc as misc
from muni.util.misc import print_on_main as print


def _persist_run_config(cfg, *, resume: bool) -> None:
    """Persist the active config, backing up a differing resume config."""
    output_dir = cfg.checkpoint.output_dir
    config_path = os.path.join(output_dir, "config.yaml")
    current = OmegaConf.to_yaml(cfg)

    if resume and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if existing == current:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(output_dir, f"config_{ts}.yaml.bak")
        shutil.copyfile(config_path, backup)
        print(
            f"[train] resume config differs; saving prior config "
            f"to {backup} before writing current config."
        )

    OmegaConf.save(cfg, config_path)


def _print_checkpoint_load_info(prefix: str, load_info: dict[str, object]) -> None:
    translated = int(load_info.get("translated_keys", 0))
    if translated:
        print(
            f"{prefix}: adapted {translated} checkpoint key(s) for compile-on/off compatibility."
        )

    unused = list(load_info.get("unused", []))
    if unused:
        preview = ", ".join(unused[:5])
        print(
            f"{prefix}: ignored {len(unused)} unused checkpoint key(s). First few: {preview}"
        )


def save_model(
    cfg,
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    epoch_name: str | None = None,
):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(cfg.checkpoint.output_dir)
    checkpoint_path = output_dir / ("checkpoint-%s.pth" % epoch_name)
    rng_state = misc.capture_rng_state(torch.device(cfg.distributed.device))
    if misc.get_world_size() > 1:
        rng_states = [None] * misc.get_world_size() if misc.is_main_process() else None
        torch.distributed.gather_object(rng_state, rng_states, dst=0)
    else:
        rng_states = [rng_state]
    if not misc.is_main_process():
        return

    # Frozen foundation weights are reconstructed when the model is built.
    model_state = model_without_ddp.trainable_state_dict()

    to_save = {
        "model": model_state,
        # Shared parameters require the owner's deduplicated EMA format.
        "model_ema1": model_without_ddp.ema.export(),
        "ema_steps": int(model_without_ddp.ema.steps),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "rng_states": rng_states,
    }
    # Atomic replacement preserves the previous checkpoint on write failure.
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(to_save, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(cfg, resume: str = "", init_from: str = ""):
    misc.init_distributed_mode(cfg)
    print_config(cfg, title="MUNI Multimodal Configuration")

    device = torch.device(cfg.distributed.device)
    misc.seed_everything(cfg.train.seed + misc.get_rank())
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    global_rank = misc.get_rank()
    num_tasks = misc.get_world_size()
    global_batch, per_rank_batch, micro_batch = resolve_batch_sizes(cfg.train, num_tasks)
    output_dir = cfg.checkpoint.output_dir

    dataset_train = build_dataset(cfg.dataset)
    data_loader, sampler_train = build_train_loader(
        dataset_train,
        batch_size=per_rank_batch,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_mem,
        num_replicas=num_tasks,
        rank=global_rank,
        seed=cfg.train.seed,
    )

    data_loader_eval = build_configured_eval_loader(
        cfg, num_replicas=num_tasks, rank=global_rank
    )

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.optimize_ddp = False

    model = build_model(cfg)
    print("MUNI Model Built")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable: {n / 1e6:.3f}M params")

    model.to(device)

    lr = cfg.train.lr if cfg.train.lr is not None else (cfg.train.blr * global_batch / 256)
    print(
        f"lr={lr:.2e}, global_batch={global_batch}, "
        f"per_rank_batch={per_rank_batch}, micro_batch={micro_batch}"
    )

    if cfg.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.distributed.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    param_groups = misc.add_weight_decay(model_without_ddp, cfg.train.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    warmup_steps = int(float(cfg.train.warmup_epochs) * len(data_loader))
    if warmup_steps < 2:
        raise ValueError("train.warmup_epochs must span at least two optimizer steps")
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps - 1,
    )

    start_epoch = cfg.train.start_epoch

    if resume and init_from:
        raise ValueError("Cannot specify both --resume and --init_from")

    # CLI checkpoint choices take precedence over the config bootstrap.
    if not resume and not init_from:
        init_from = cfg.train.get("init_from", "") or ""
        if init_from:
            print(f"[init_from] using config-default checkpoint: {init_from}")

    if resume:
        ckpt_path = os.path.join(resume, "checkpoint-last.pth") if os.path.isdir(resume) else resume
        ckpt = torch.load(ckpt_path, map_location="cpu")
        required_state = {"model", "model_ema1", "optimizer", "lr_scheduler", "epoch"}
        missing_state = required_state.difference(ckpt)
        if missing_state:
            raise ValueError(
                f"--resume requires a training checkpoint, missing {sorted(missing_state)}. "
                "Use --init_from to start a new run from model weights."
            )
        _, _, load_info = model_without_ddp.load_trainable_state_dict(ckpt["model"])
        _print_checkpoint_load_info("Resume", load_info)
        model_without_ddp.ema.load(
            model_without_ddp, ckpt["model_ema1"], device, history=ckpt,
        )
        if "ema_steps" not in ckpt:
            print(
                f"[resume] checkpoint predates step-aware EMA; "
                f"setting ema_steps to {model_without_ddp.ema.steps} (≈ 2 EMA time constants)"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print("Loaded optimizer and learning-rate scheduler state!")
        rng_states = ckpt.get("rng_states")
        if rng_states is None or len(rng_states) != num_tasks:
            warnings.warn(
                "Checkpoint RNG state is absent or its world size differs; "
                "weights and optimizer resume, but the stochastic trajectory is not replayed.",
                stacklevel=2,
            )
        else:
            misc.restore_rng_state(rng_states[global_rank], device)
            if device.type == "cuda" and rng_states[global_rank]["cuda"] is None:
                warnings.warn("Checkpoint has no CUDA RNG state; device-changed resume is not exact.", stacklevel=2)
        del ckpt
        print("Resumed from", ckpt_path)
    elif init_from:
        ckpt = torch.load(init_from, map_location="cpu")
        missing, unexpected, load_info = model_without_ddp.load_trainable_state_dict(
            ckpt["model"], strict=False
        )
        print(f"Initialized weights from {init_from}")
        _print_checkpoint_load_info("Init", load_info)
        if missing:
            print(f"Missing keys during initialization: {missing}")
        if unexpected:
            print(f"Unexpected keys during initialization: {unexpected}")
        if "model_ema1" in ckpt:
            try:
                model_without_ddp.ema.load(
                    model_without_ddp, ckpt["model_ema1"], device, history=ckpt,
                )
            except EMACheckpointError as e:
                print(
                    "Init: source EMA does not cover this model's trainable parameters. "
                    f"Falling back to fresh EMA initialisation. Reason: {e}"
                )
                model_without_ddp.ema.initialize(model_without_ddp)
            else:
                print("Initialized EMA weights.")
        else:
            model_without_ddp.ema.initialize(model_without_ddp)
            print("No EMA weights found, initialized EMA from current model state.")
        del ckpt
        print("Starting training from scratch with pre-trained weights.")
    else:
        model_without_ddp.ema.initialize(model_without_ddp)
        print("Training from scratch")

    # Do not change run records until checkpoint restoration has succeeded.
    if global_rank == 0 and output_dir:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(output_dir, exist_ok=True)
        _persist_run_config(cfg, resume=bool(resume))
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        log_writer = None

    print(f"Training for {cfg.train.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        sampler_train.set_epoch(epoch)

        train_one_epoch(
            model,
            model_without_ddp,
            data_loader,
            optimizer,
            lr_scheduler,
            device,
            epoch,
            log_writer=log_writer,
            cfg=cfg,
        )

        if epoch % cfg.checkpoint.save_last_freq == 0 or epoch + 1 == cfg.train.epochs:
            save_model(
                cfg=cfg,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                epoch_name="last",
            )
        if epoch % cfg.checkpoint.save_ckpt_freq == 0 and epoch > 0:
            save_model(
                cfg=cfg,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
            )

        eval_cfg = cfg.eval
        eval_freq = eval_cfg.eval_freq
        if eval_freq > 0 and epoch % eval_freq == 0:
            with model_without_ddp.ema.evaluate(model_without_ddp):
                run_evaluation_suite(
                    model_without_ddp, data_loader_eval, cfg, log_writer, epoch, device
                )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training time:", elapsed)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "MUNI Training",
        epilog="Config overrides: key=value for existing keys; +key=value for new keys.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default="configs/image_text_audio/muni_poe.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Path to checkpoint for weight initialization (no optimizer state)",
    )
    parser.add_argument("config_overrides", nargs="*", metavar="key=value")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, cli_overrides=args.config_overrides)
    except (ConfigOverrideError, MissingConfigEnvironmentError) as exc:
        parser.error(str(exc))

    main(cfg, resume=args.resume, init_from=args.init_from)
