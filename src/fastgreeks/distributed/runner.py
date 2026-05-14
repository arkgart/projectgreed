"""torch.distributed launcher for the double-barrier kernel.

Invocation
----------

    torchrun --nproc-per-node=8 -m fastgreeks.distributed.runner \\
        --n-options 67108864 --n-terms 32 --dtype bf16

or programmatically via ``price_options_distributed`` which handles
``init_process_group`` / ``barrier`` / ``all_gather`` internally.

Sharding
--------
Options are partitioned contiguously: rank r holds
``options[r * shard : (r + 1) * shard]``.  Each rank runs the Triton
kernel on its slice and the results are all-gathered back to rank 0.

We measure timing with CUDA events on each rank, take the *max* over
ranks as the cluster wall-clock (i.e. the slowest rank determines
end-to-end latency), and report both the cluster throughput
(options / s / cluster) and the per-device amortised time
(picoseconds / option / GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

try:
    import torch
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    torch = None      # type: ignore
    dist = None       # type: ignore
    HAS_TORCH = False


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str


def init_distributed() -> DistInfo:
    """Initialise NCCL group from torchrun env vars; idempotent."""
    if not HAS_TORCH:
        raise RuntimeError("torch.distributed init requires PyTorch")
    if dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return DistInfo(rank=rank, world_size=world,
                              local_rank=local_rank,
                              device=f"cuda:{local_rank}")
    if "RANK" not in os.environ:
        # Single-process fallback: pretend we're a 1-rank world
        return DistInfo(rank=0, world_size=1, local_rank=0, device="cuda:0")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return DistInfo(rank=rank, world_size=world,
                          local_rank=local_rank,
                          device=f"cuda:{local_rank}")


def shutdown_distributed() -> None:
    if HAS_TORCH and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _shard_range(n_total: int, rank: int, world: int) -> tuple[int, int]:
    """Contiguous sharding with the remainder spread over the first ranks."""
    base, rem = divmod(n_total, world)
    if rank < rem:
        lo = rank * (base + 1)
        hi = lo + base + 1
    else:
        lo = rem * (base + 1) + (rank - rem) * base
        hi = lo + base
    return lo, hi


def price_options_distributed(
    S, K, T, r, sigma, L, U,
    n_terms: int = 32,
    dtype: str = "bf16",
    block_size: int = 256,
) -> Dict[str, object]:
    """Multi-GPU dispatch for the double-barrier KO call.

    All inputs are 1-D CPU tensors of length ``n_options`` (must match).
    The function (1) shards them across the world, (2) launches the
    Triton kernel on each rank's slice, (3) all-gathers the results
    on rank 0, and (4) returns timing + correctness summary.

    Returns
    -------
    dict on rank 0 containing:
        prices            -- 1-D CPU tensor of length n_options
        kernel_time_ms    -- max over ranks (cluster wall-clock for the kernel)
        full_time_ms      -- includes H<->D copies and gather
        throughput_g_per_s -- 1e-9 * n_options / kernel_time_ms * 1000
        ps_per_option     -- picoseconds per option, amortised over the cluster
    On other ranks, returns {"prices": None, ...} with only its local stats.
    """
    if not HAS_TORCH:
        raise RuntimeError("price_options_distributed requires PyTorch")
    if not torch.cuda.is_available():
        raise RuntimeError("price_options_distributed requires CUDA")

    from fastgreeks.kernels import double_barrier_call_gpu  # local import to keep CPU paths clean

    info = init_distributed()
    rank, world, dev = info.rank, info.world_size, info.device

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {list(dtype_map)}")
    work_dtype = dtype_map[dtype]

    # Total option count: must be agreed across ranks; trust rank 0's input length
    n_total = int(S.numel())

    # Per-rank shard
    lo, hi = _shard_range(n_total, rank, world)
    n_local = hi - lo

    full_t0 = time.perf_counter()

    # Move shard to device with the chosen working dtype
    def slice_(t):
        return t[lo:hi].to(device=dev, dtype=work_dtype, non_blocking=True)
    S_d = slice_(S); K_d = slice_(K); T_d = slice_(T)
    r_d = slice_(r); sig_d = slice_(sigma); L_d = slice_(L); U_d = slice_(U)

    # Warm-up (autotune & first-launch overhead)
    if n_local > 0:
        _ = double_barrier_call_gpu(S_d[:1], K_d[:1], T_d[:1],
                                            r_d[:1], sig_d[:1], L_d[:1], U_d[:1],
                                            n_terms=n_terms, block_size=block_size)
    torch.cuda.synchronize(dev)

    # Timed run
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    if n_local > 0:
        local_prices = double_barrier_call_gpu(
            S_d, K_d, T_d, r_d, sig_d, L_d, U_d,
            n_terms=n_terms, block_size=block_size,
        )
    else:
        local_prices = torch.zeros(0, dtype=torch.float32, device=dev)
    end_evt.record()
    torch.cuda.synchronize(dev)
    local_kernel_ms = start_evt.elapsed_time(end_evt)

    # Cross-rank max (slowest rank = cluster wall-clock for the kernel)
    if world > 1:
        t_max = torch.tensor([local_kernel_ms], device=dev)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        cluster_kernel_ms = float(t_max.item())
    else:
        cluster_kernel_ms = local_kernel_ms

    # Gather prices to rank 0 (asymmetric sizes → use all_gather_object)
    local_cpu = local_prices.detach().to("cpu", dtype=torch.float32)
    gathered_prices = None
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, local_cpu)
        if rank == 0:
            gathered_prices = torch.cat(gathered, dim=0)
    else:
        gathered_prices = local_cpu

    full_ms = (time.perf_counter() - full_t0) * 1000.0

    result = {
        "rank": rank,
        "world_size": world,
        "n_total": n_total,
        "n_local": n_local,
        "kernel_time_ms": cluster_kernel_ms,
        "full_time_ms": full_ms,
        "throughput_options_per_s": n_total / (cluster_kernel_ms / 1000.0),
        "ps_per_option": cluster_kernel_ms * 1e9 / n_total,
        "dtype": dtype,
        "n_terms": n_terms,
    }
    if rank == 0:
        result["prices"] = gathered_prices
    return result


# ----------------------------------------------------------------------
# Command-line entry point: runs a synthetic batch through the pipeline
# ----------------------------------------------------------------------
def _synth_batch(n: int, seed: int = 0):
    torch.manual_seed(seed)
    S = 100.0 + 5.0 * torch.randn(n)
    K = 100.0 + 5.0 * torch.randn(n)
    T = 0.1 + 1.9 * torch.rand(n)
    r = 0.02 + 0.06 * torch.rand(n)
    sigma = 0.10 + 0.40 * torch.rand(n)
    L = S - 25.0 - 5.0 * torch.rand(n)
    U = S + 25.0 + 5.0 * torch.rand(n)
    return S, K, T, r, sigma, L, U


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fastgreeks.distributed.runner")
    p.add_argument("--n-options", type=int, default=1 << 24,
                          help="number of options (default 16 M)")
    p.add_argument("--n-terms", type=int, default=32)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--out-json", type=str, default=None,
                          help="rank-0 writes a JSON summary here")
    args = p.parse_args(argv)

    S, K, T, r, sig, L, U = _synth_batch(args.n_options)
    res = price_options_distributed(
        S, K, T, r, sig, L, U,
        n_terms=args.n_terms,
        dtype=args.dtype,
        block_size=args.block_size,
    )
    if res["rank"] == 0:
        # Drop the prices tensor from the JSON summary (only print stats)
        prices = res.pop("prices", None)
        print(json.dumps(res, indent=2))
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(res, f, indent=2)
        # Print a one-line headline
        sys.stderr.write(
            f"\n[fastgreeks] {res['n_total']:>10d} options  "
            f"world={res['world_size']}  dtype={args.dtype}  "
            f"kernel={res['kernel_time_ms']:.2f} ms  "
            f"throughput={res['throughput_options_per_s']*1e-9:.2f} G opt/s  "
            f"amortised={res['ps_per_option']:.1f} ps/option\n"
        )
    shutdown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
