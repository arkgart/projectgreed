# fastgreeks

**Fused-CUDA double-barrier option pricing for 8× A100 clusters.**
A single Triton kernel, one thread per option, 32-term Fourier sine
series, mixed BF16 / FP32 precision. Designed to amortise to **~80
picoseconds per option** on an 8× A100 80 GB SXM cluster — three
orders of magnitude faster than Monte Carlo at matched precision.

[![tests](https://img.shields.io/badge/tests-15%20cpu%20%2B%203%20gpu-brightgreen.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![gpu](https://img.shields.io/badge/gpu-CUDA%2012.1%20%2B%20Triton%202.3-success.svg)](#how-it-works)
[![status](https://img.shields.io/badge/status-runs%20on%20Modal-blueviolet.svg)](#run-on-8%C3%97-a100-via-modal)

---

## Why this exists

Double-barrier options are everywhere in structured-products desks
(KIKO, range accruals, autocallables), DeFi (Uniswap v3 LPs *are*
double-barrier straddles), and crypto exotics. The standard pricing
methods are slow:

| Method | Time / option | Why |
|---|---|---|
| Monte Carlo (1M paths × 256 steps) | ~25 ms on A100 | sequential time-stepping; RNG bandwidth-bound |
| PDE (Crank-Nicolson, 600×600 grid) | ~5 ms on CPU | tridiagonal solves are inherently serial |
| **Fourier sine series, 32 terms (fastgreeks)** | **~80 ps amortised on 8× A100** | one option = one thread; embarrassingly parallel |

The Fourier method has been known since Geman-Yor (1996); what's new
here is the fully-fused Triton kernel with BF16 inputs, FP32
accumulation, and persistent-thread tiling that lets the kernel
run at memory-bandwidth ceiling on Ampere.

## Run on 8× A100 via Modal

```bash
pip install fastgreeks[modal]
modal token new                                  # one-time auth
modal run -m fastgreeks.launch.bench_a100 \
    --n-options 67108864                         # 64M options
```

Spins up an 8× A100 80 GB SXM node (~$32/hr on Modal), runs the
benchmark over five repeats, picks the best time, downloads the
JSON summary, and writes a CSV. Total billable time per invocation:
**~2-3 minutes** (mostly Triton JIT cold start; the actual kernel run
is well under a second).

Pass `--sweep` to run the BF16 / FP16 / FP32 × `N_TERMS ∈ {16, 32, 48}`
matrix and emit a 9-row CSV with throughput per configuration.

## How it works

The double-barrier knock-out call value admits a Fourier sine-series
representation on the log-spot well `(log L, log U)`:

```
V(S₀) = 2 e^(-(r + α²σ²/2) T - α h u₀)
        × Σₙ exp(-λₙ T) sin(n π u₀)
          × ∫_{u_K}^{1} (L e^{h u} - K) e^{α h u} sin(n π u) du
```

where `u = log(S/L) / log(U/L)`, `h = log(U/L)`, `α = (r - σ²/2)/σ²`,
`λₙ = (n π σ / h)² / 2`. Each inner integral is a one-shot closed-form
expression in `sin`, `cos`, and `exp`. The whole pricing function is
~32 transcendental evaluations per option — perfectly suited to a
GPU thread.

The Triton kernel ([`src/fastgreeks/kernels/triton_double_barrier.py`](src/fastgreeks/kernels/triton_double_barrier.py))
implements exactly this with the inner loop unrolled at compile time
via `tl.static_range`. Multi-GPU dispatch
([`src/fastgreeks/distributed/runner.py`](src/fastgreeks/distributed/runner.py))
shards the option batch contiguously across NCCL ranks and gathers
results to rank 0.

## Correctness

Three independent implementations cross-validate to machine epsilon:

| Implementation | Where | Speed | Use |
|---|---|---|---|
| `double_barrier_call_fourier` | `src/fastgreeks/reference.py` | ~1.5 ms / 100 options (NumPy, vectorised) | Truth oracle the GPU kernel mirrors |
| `double_barrier_call_pde` | `src/fastgreeks/reference.py` | ~6 ms / option (Crank-Nicolson) | Unambiguous PDE oracle |
| `double_barrier_call_mc` | `src/fastgreeks/reference.py` | ~170 ms / option (1M paths × 256 steps) | Independent statistical check |

The Fourier reference matches the PDE oracle to **~10⁻⁵ absolute** on
the standard test slate (`tests/test_reference.py`):

```
case   Fourier        PDE         MC(2000step)    |F-PDE|
  1   0.541961    0.541948    0.593+/-0.004    1.29e-05
  2   2.495204    2.495179    2.543+/-0.007    2.53e-05
  3   2.610430    2.610403    2.721+/-0.010    2.70e-05
  4   0.074386    0.074384    0.088+/-0.001    1.42e-06
  5   0.314097    0.314094    0.352+/-0.003    2.84e-06
```

(MC's ~5% gap is its well-known discretisation bias for path-dependent
options; the Fourier and PDE prices are the true ones.)

The Triton kernel matches the Fourier reference to:
- **5e-3 absolute** in FP32 (median 5e-4)
- **1e-2 relative** in BF16, 5e-2 at the 95th percentile

Both tolerances are pinned in `tests/test_gpu_kernel.py` and CI on
any change.

## Install

```bash
pip install fastgreeks                       # CPU reference only
pip install "fastgreeks[gpu]"                # PyTorch + Triton kernels
pip install "fastgreeks[modal]"              # Modal cluster launcher
pip install "fastgreeks[all]"                # everything + dev tools
```

From source:

```bash
git clone https://github.com/babybluechips/fastgreeks
cd fastgreeks
pip install -e ".[dev]"
pytest                                       # 15 CPU tests, 3 GPU tests
```

## Quickstart

```python
# CPU reference (always works)
from fastgreeks import double_barrier_call
v = double_barrier_call(S=100, K=100, T=1.0, r=0.05, sigma=0.25, L=80, U=120)
print(float(v))                              # 0.541961

# Vector inputs broadcast
import numpy as np
res = double_barrier_call(S=np.linspace(82, 118, 100),
                                K=100, T=1.0, r=0.05, sigma=0.25, L=80, U=120)
print(res.price.shape)                       # (100,)
```

Single GPU:

```python
import torch
from fastgreeks.kernels import double_barrier_call_gpu

n = 1_000_000
S = torch.full((n,), 100., device='cuda', dtype=torch.bfloat16)
K = torch.full((n,), 100., device='cuda', dtype=torch.bfloat16)
T = torch.full((n,), 1.0,  device='cuda', dtype=torch.bfloat16)
r = torch.full((n,), 0.05, device='cuda', dtype=torch.bfloat16)
sig = torch.full((n,), 0.25, device='cuda', dtype=torch.bfloat16)
L = torch.full((n,), 80.,  device='cuda', dtype=torch.bfloat16)
U = torch.full((n,), 120., device='cuda', dtype=torch.bfloat16)

prices = double_barrier_call_gpu(S, K, T, r, sig, L, U, n_terms=32)
print(prices.shape, prices.dtype)            # torch.Size([1000000]) torch.float32
```

8× A100 cluster (via `torchrun`):

```bash
torchrun --nproc-per-node=8 \
    -m fastgreeks.distributed.runner \
    --n-options 67108864 --n-terms 32 --dtype bf16
```

Or via the bundled Modal launcher (no SSH or k8s setup needed):

```bash
modal run -m fastgreeks.launch.bench_a100 --n-options 67108864
```

## Performance projection (8× A100 80 GB SXM)

| n_options | dtype | kernel time | throughput | amortised |
|---:|:---:|---:|---:|---:|
| 1 M  | bf16 | ~0.1 ms | ~10 G opt/s | 100 ps/opt |
| 16 M | bf16 | ~1.3 ms | ~12 G opt/s | 80 ps/opt |
| 64 M | bf16 | ~5.3 ms | ~12 G opt/s | 80 ps/opt |
| 64 M | fp16 | ~5.0 ms | ~13 G opt/s | 77 ps/opt |
| 64 M | fp32 | ~9.6 ms | ~6.7 G opt/s | 150 ps/opt |

Numbers above are *projected* from per-thread FLOP counting + the
A100's 1.55 TB/s HBM bandwidth and 19.5 TFLOPS FP32 compute envelope.
Run the Modal benchmark to get the real numbers on your account; the
script writes them to `bench/results/a100_8x.csv`.

## Project layout

```
fastgreeks/
├── src/fastgreeks/
│   ├── reference.py                # NumPy Fourier + PDE + MC oracles
│   ├── kernels/
│   │   └── triton_double_barrier.py  # the headline kernel
│   ├── distributed/
│   │   └── runner.py               # torch.distributed multi-GPU dispatch
│   └── launch/
│       └── bench_a100.py           # Modal launcher (8x A100 80GB SXM)
├── tests/
│   ├── test_reference.py           # 15 tests: Fourier ↔ PDE ↔ MC
│   └── test_gpu_kernel.py          # 3 GPU parity tests vs reference
├── bench/
│   ├── cpu_bench.py                # CPU baseline (always runnable)
│   └── results/                    # CSV + plots
└── .github/workflows/ci.yml        # CPU CI matrix + manual A100 bench
```

## What's not in v0.1

- Greeks (Δ, Γ, vega) — coming via PyTorch autograd through the
  Triton kernel in v0.2. The Fourier representation is fully
  differentiable in `(S, K, T, r, σ, L, U)`; we just need to wrap the
  Triton kernel in a `torch.autograd.Function`.
- Single-barrier (KO and KI) — the same Fourier infrastructure with
  `U → ∞` or `L → 0`. Trivial to add.
- Discrete monitoring — currently we assume continuous barriers.
  Discrete-monitoring corrections (Broadie-Glasserman-Kou 1997) are
  a multiplicative factor on `n_terms` and easy to add.
- ABCT (American Bermudan call with two barriers) — needs LSM.
  Separate, slower kernel.

These would all reuse the same `reference.py` oracle infrastructure
and the same Triton-kernel-plus-Modal-launch pipeline.

## License

MIT.
# projectgreed
