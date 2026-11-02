"""Resolve a fixed optimizer batch and a per-device memory cap."""


def resolve_batch_sizes(train_cfg, world_size: int) -> tuple[int, int, int]:
    """Return global, per-rank optimizer, and per-rank microbatch sizes."""
    if "batch_size" in train_cfg and "global_batch_size" in train_cfg:
        raise ValueError("Cannot set both train.batch_size and train.global_batch_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    global_batch = train_cfg.get("global_batch_size")
    if "global_batch_size" not in train_cfg and "batch_size" in train_cfg:
        # train.batch_size specifies the per-rank optimizer batch.
        batch_size = train_cfg["batch_size"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("train.batch_size must be a positive integer")
        global_batch = batch_size * world_size
    if isinstance(global_batch, bool) or not isinstance(global_batch, int) or global_batch <= 0:
        raise ValueError("train.global_batch_size must be a positive integer")
    if global_batch % world_size:
        raise ValueError(
            f"train.global_batch_size={global_batch} must be divisible by world_size={world_size}"
        )
    per_rank_batch = global_batch // world_size
    micro_batch = train_cfg.get("micro_batch_size")
    if micro_batch is None:
        return global_batch, per_rank_batch, per_rank_batch
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch <= 0:
        raise ValueError("train.micro_batch_size must be a positive integer or null")
    return global_batch, per_rank_batch, min(micro_batch, per_rank_batch)


def resolve_eval_batch_size(train_cfg, eval_cfg, world_size: int) -> int:
    """Resolve a per-device evaluation batch without optimizer divisibility."""
    if "batch_size" in train_cfg and "global_batch_size" in train_cfg:
        raise ValueError("Cannot set both train.batch_size and train.global_batch_size")
    batch_size = eval_cfg.get("batch_size")
    if batch_size is None:
        batch_size = train_cfg.get("micro_batch_size")
    if batch_size is None and "global_batch_size" not in train_cfg:
        batch_size = train_cfg.get("batch_size")
    if batch_size is None:
        global_batch = train_cfg.get("global_batch_size")
        if isinstance(global_batch, bool) or not isinstance(global_batch, int) or global_batch <= 0:
            raise ValueError("train.global_batch_size must be a positive integer")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        batch_size = (global_batch + world_size - 1) // world_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("evaluation batch size must be a positive integer")
    return batch_size
