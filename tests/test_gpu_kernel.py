"""GPU kernel correctness: Triton output must match the CPU Fourier reference.

Skipped automatically when CUDA / Triton is unavailable (e.g. on CI's
CPU runners).  The Modal launcher in ``launch/bench_a100.py`` exercises
the same kernel against a much larger batch on real A100s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from fastgreeks.kernels import double_barrier_call_gpu
from fastgreeks.reference import double_barrier_call_fourier


def _rand_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    S = 100.0 + 5.0 * rng.standard_normal(n)
    K = 100.0 + 5.0 * rng.standard_normal(n)
    T = 0.1 + 1.9 * rng.random(n)
    r = 0.02 + 0.06 * rng.random(n)
    sigma = 0.10 + 0.40 * rng.random(n)
    L = S - 25.0 - 5.0 * rng.random(n)
    U = S + 25.0 + 5.0 * rng.random(n)
    return [np.asarray(x, dtype=np.float64) for x in (S, K, T, r, sigma, L, U)]


@pytest.mark.gpu
def test_triton_matches_reference_fp32():
    S, K, T, r, sig, L, U = _rand_batch(4096, seed=42)
    ref = double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sig,
                                              L=L, U=U, n_terms=32).price
    Sg, Kg, Tg, rg, sg, Lg, Ug = [
        torch.tensor(x, dtype=torch.float32, device="cuda")
        for x in (S, K, T, r, sig, L, U)
    ]
    got = double_barrier_call_gpu(Sg, Kg, Tg, rg, sg, Lg, Ug,
                                          n_terms=32).cpu().numpy()
    err = np.abs(ref - got)
    assert err.max() < 5e-3, (
        f"max GPU-vs-ref error {err.max():.3e}"
    )
    # Median error should be much tighter (the max picks up boundary cases)
    assert np.median(err) < 5e-4


@pytest.mark.gpu
def test_triton_matches_reference_bf16():
    """BF16 should match to ~1e-2 absolute on price (BF16 has 7-bit mantissa)."""
    S, K, T, r, sig, L, U = _rand_batch(4096, seed=43)
    ref = double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sig,
                                              L=L, U=U, n_terms=32).price
    Sg, Kg, Tg, rg, sg, Lg, Ug = [
        torch.tensor(x, dtype=torch.bfloat16, device="cuda")
        for x in (S, K, T, r, sig, L, U)
    ]
    got = double_barrier_call_gpu(Sg, Kg, Tg, rg, sg, Lg, Ug,
                                          n_terms=32).cpu().to(torch.float32).numpy()
    err = np.abs(ref - got)
    # BF16 tolerance: ~1% of price for the worst case
    rel_err = err / np.maximum(np.abs(ref), 1e-6)
    assert np.median(rel_err) < 1e-2
    assert np.percentile(rel_err, 95) < 5e-2


@pytest.mark.gpu
def test_triton_one_million():
    """Smoke-test 1M options: just check the kernel doesn't crash and prices are sane."""
    S, K, T, r, sig, L, U = _rand_batch(1_000_000, seed=7)
    Sg, Kg, Tg, rg, sg, Lg, Ug = [
        torch.tensor(x, dtype=torch.bfloat16, device="cuda")
        for x in (S, K, T, r, sig, L, U)
    ]
    out = double_barrier_call_gpu(Sg, Kg, Tg, rg, sg, Lg, Ug, n_terms=32)
    assert out.shape == Sg.shape
    out_np = out.cpu().to(torch.float32).numpy()
    assert (out_np >= 0).all()
    assert (out_np < 1000).all()   # sanity bound for these params
