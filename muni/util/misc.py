import builtins
import datetime
import os
import random
import time
from collections.abc import Mapping
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_print_master = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(device: torch.device) -> dict:
    """Capture process RNGs using checkpoint-safe primitives and tensors."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


@contextmanager
def preserve_rng_state(device: torch.device):
    """Isolate evaluation from the caller's CPU and selected-device RNGs."""
    state = capture_rng_state(device)
    try:
        yield
    finally:
        restore_rng_state(state, device)


class SmoothedValue:
    """Track windowed and global statistics for a scalar series."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize global statistics, but not the local window."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        total_iterations = len(iterable)
        if print_freq <= 0:
            raise ValueError(f"print_freq must be positive, got {print_freq}.")
        if total_iterations == 0:
            print_on_main(f"{header} Total time: 0:00:00 (0 iterations)")
            return
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(total_iterations))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_iterations - 1:
                eta_seconds = iter_time.global_avg * (total_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_on_main(
                        log_msg.format(
                            i,
                            total_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print_on_main(
                        log_msg.format(
                            i,
                            total_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_on_main(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / total_iterations
            )
        )


def print_on_main(*args, **kwargs):
    """Print MUNI output with its configured rank gate and timestamp."""
    if _print_master is None:
        return builtins.print(*args, **kwargs)
    force = kwargs.pop("force", False) or get_world_size() > 8
    if _print_master or force:
        builtins.print("[{}] ".format(datetime.datetime.now().time()), end="")
        builtins.print(*args, **kwargs)


def setup_for_distributed(is_master):
    """Configure MUNI output without replacing interpreter-wide print."""
    global _print_master
    _print_master = is_master


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(cfg):
    """Initialize distributed training. Mutates cfg.distributed in-place."""
    from omegaconf import open_dict

    with open_dict(cfg):
        d = cfg.distributed
        if d.dist_on_itp:
            d.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            d.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            d.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
            d.dist_url = "tcp://%s:%s" % (
                os.environ["MASTER_ADDR"],
                os.environ["MASTER_PORT"],
            )
            os.environ["LOCAL_RANK"] = str(d.gpu)
            os.environ["RANK"] = str(d.rank)
            os.environ["WORLD_SIZE"] = str(d.world_size)
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            d.rank = int(os.environ["RANK"])
            d.world_size = int(os.environ["WORLD_SIZE"])
            d.gpu = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            d.rank = int(os.environ["SLURM_PROCID"])
            d.gpu = d.rank % torch.cuda.device_count()
        else:
            print_on_main("Not using distributed mode")
            setup_for_distributed(is_master=True)
            d.distributed = False
            return

        d.distributed = True

        torch.cuda.set_device(d.gpu)
        d.dist_backend = "nccl"
        print_on_main(
            "| distributed init (rank {}): {}, gpu {}".format(
                d.rank, d.dist_url, d.gpu
            ),
            flush=True,
        )
        # Long rank-zero evaluations can exceed NCCL's default timeout.
        torch.distributed.init_process_group(
            backend=d.dist_backend,
            init_method=d.dist_url,
            world_size=d.world_size,
            rank=d.rank,
            timeout=datetime.timedelta(hours=2),
        )
        torch.distributed.barrier()
        setup_for_distributed(d.rank == 0)


def unwrap_modality_feature(x):
    """Extract ``feature`` from a structured modality payload when present."""
    if isinstance(x, Mapping) and "feature" in x:
        return x["feature"]
    return x


def move_to_device(obj, device, *, non_blocking: bool = False):
    """Move a tensor or nested mapping of tensors to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {
            k: move_to_device(v, device, non_blocking=non_blocking)
            for k, v in obj.items()
        }
    return obj


@contextmanager
def evaluation_mode(model):
    """Temporarily evaluate a model, then restore its previous training mode."""
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(was_training)


def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


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
    rng_state = capture_rng_state(torch.device(cfg.distributed.device))
    if get_world_size() > 1:
        rng_states = [None] * get_world_size() if is_main_process() else None
        dist.gather_object(rng_state, rng_states, dst=0)
    else:
        rng_states = [rng_state]
    if not is_main_process():
        return

    from omegaconf import OmegaConf

    # Frozen foundation weights are reconstructed when the model is built.
    if hasattr(model_without_ddp, "trainable_state_dict"):
        model_state = model_without_ddp.trainable_state_dict()
    else:
        model_state = model_without_ddp.state_dict()

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


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Force float32 so int/bool inputs don't trip the in-place division below.
        x_reduce = torch.tensor(x, device=device, dtype=torch.float32)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def extract_dict_data(batch):
    """Extract the modality mapping from a loader batch."""
    if isinstance(batch, Mapping):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], Mapping):
        return batch[0]
    raise TypeError(
        f"Expected a batch payload shaped as dict[str, Tensor] or (dict[str, Tensor], ...), got {type(batch)!r}."
    )


def next_eval_batch(data_loader, device):
    """Fetch and transfer one evaluation batch, or return ``None``."""
    if not data_loader:
        return None
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        return None
    dict_data = extract_dict_data(batch)
    return {key: move_to_device(value, device) for key, value in dict_data.items()}
