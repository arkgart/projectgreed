"""Multi-GPU dispatch for fastgreeks.

The double-barrier kernel is embarrassingly parallel across options:
each option is one independent thread on one GPU.  Multi-GPU is
therefore pure data parallelism:

    rank r processes options [r * shard, (r+1) * shard)
    rank r calls the local Triton kernel
    rank r writes to its local output slice; we all-gather at the end.

We use torchrun + NCCL when available, falling back to single-device
when not.  See ``fastgreeks.launch.bench_a100`` for the Modal launcher
that provisions the cluster and invokes this module.
"""

from .runner import (
    init_distributed,
    shutdown_distributed,
    price_options_distributed,
    DistInfo,
)

__all__ = [
    "init_distributed",
    "shutdown_distributed",
    "price_options_distributed",
    "DistInfo",
]
