"""fastgreeks: fused-CUDA double-barrier option pricing for 8x A100.

Quickstart (CPU reference)
--------------------------

    >>> from fastgreeks import double_barrier_call
    >>> v = double_barrier_call(S=100, K=100, T=1.0, r=0.05, sigma=0.25,
    ...                              L=80, U=120)
    >>> float(v)
    0.541...

Quickstart (GPU, single device)
-------------------------------

    >>> import torch
    >>> from fastgreeks.kernels import double_barrier_call_gpu
    >>> S = torch.full((1_000_000,), 100.0, device='cuda', dtype=torch.bfloat16)
    >>> ...
    >>> prices = double_barrier_call_gpu(S, K, T, r, sigma, L, U)

Quickstart (8x A100 cluster, Modal)
-----------------------------------

    $ modal run -m fastgreeks.launch.bench_a100 --n-options 67108864
"""

from .reference import (
    BarrierResult,
    double_barrier_call_fourier as double_barrier_call,
    double_barrier_call_fourier,
    double_barrier_call_pde,
    double_barrier_call_mc,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BarrierResult",
    "double_barrier_call",
    "double_barrier_call_fourier",
    "double_barrier_call_pde",
    "double_barrier_call_mc",
]
