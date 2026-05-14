"""GPU kernels for fastgreeks.

Currently ships one kernel:

* :func:`double_barrier_call_gpu` -- fused Triton kernel mirroring the
  CPU Fourier-series reference in ``fastgreeks.reference``.

The kernel module raises a useful error when imported without CUDA;
on CPU-only hosts use ``fastgreeks.reference`` directly.
"""

from .triton_double_barrier import HAS_TRITON, double_barrier_call_gpu

__all__ = ["double_barrier_call_gpu", "HAS_TRITON"]
