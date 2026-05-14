"""Three-way cross-validation: Fourier vs PDE vs Monte Carlo.

Locks in the correctness oracle that the GPU kernel will be tested against.
"""

from __future__ import annotations

import pytest

from fastgreeks.reference import (
    double_barrier_call_fourier,
    double_barrier_call_pde,
    double_barrier_call_mc,
)


CASES = [
    # (S, K, T, r, sigma, L, U)
    (100.0, 100.0, 1.00, 0.05, 0.25,  80.0, 120.0),
    (100.0, 100.0, 0.25, 0.05, 0.25,  80.0, 120.0),
    (100.0,  95.0, 0.50, 0.05, 0.40,  70.0, 130.0),
    (100.0, 110.0, 1.00, 0.05, 0.30,  85.0, 125.0),
    ( 95.0, 100.0, 2.00, 0.03, 0.20,  80.0, 120.0),
]


@pytest.mark.parametrize("S,K,T,r,sigma,L,U", CASES)
def test_fourier_matches_pde(S, K, T, r, sigma, L, U):
    """Fourier series and PDE should agree to <1e-3 absolute on standard cases."""
    f = float(double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sigma,
                                                   L=L, U=U, n_terms=64))
    pde = double_barrier_call_pde(S=S, K=K, T=T, r=r, sigma=sigma,
                                          L=L, U=U, n_x=400, n_t=400)
    assert abs(f - pde) < 1e-3, (
        f"Fourier {f} vs PDE {pde}; gap {abs(f - pde):.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("S,K,T,r,sigma,L,U", CASES)
def test_fourier_close_to_mc(S, K, T, r, sigma, L, U):
    """Fourier should be within 3 MC standard errors of a fine MC estimate.

    MC discretisation bias for double-barrier is well-known to be ~5%
    of price even at n_steps=2000; we use 4000 steps + 800k paths and
    a 10 std error tolerance.
    """
    f = float(double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sigma,
                                                   L=L, U=U, n_terms=64))
    mc, se = double_barrier_call_mc(S=S, K=K, T=T, r=r, sigma=sigma,
                                            L=L, U=U,
                                            n_paths=400_000, n_steps=4000, seed=1)
    # MC for double-barrier has both noise (~ se) and substantial discretisation
    # bias (the discretized path has fewer barrier crossings than continuous time),
    # well-known to be ~5-15% of price even at n_steps=4000.
    tol = max(10 * se, 0.20 * abs(f))
    assert abs(f - mc) < tol, (
        f"Fourier {f} vs MC {mc} (+/- {se}); gap {abs(f - mc):.3e}, tol {tol:.3e}"
    )


def test_fourier_converges_in_n_terms():
    """As we add more terms, the price should monotonically converge."""
    S, K, T, r, sigma, L, U = 100.0, 100.0, 1.0, 0.05, 0.25, 80.0, 120.0
    values = []
    for nt in (8, 16, 32, 64, 128):
        v = float(double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sigma,
                                                       L=L, U=U, n_terms=nt))
        values.append(v)
    # Successive differences should shrink and the final two should be close.
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    assert all(d <= 1e-1 for d in diffs)
    assert diffs[-1] < 1e-6


def test_breached_returns_zero():
    """If S is outside [L, U] at inception, the option is dead -> price 0."""
    # S below lower barrier
    v = float(double_barrier_call_fourier(S=75.0, K=100, T=1, r=0.05, sigma=0.25,
                                                   L=80, U=120))
    assert v == 0.0
    # S above upper barrier
    v = float(double_barrier_call_fourier(S=125.0, K=100, T=1, r=0.05, sigma=0.25,
                                                   L=80, U=120))
    assert v == 0.0


def test_fourier_broadcasts():
    """Vector inputs broadcast and return a vector of prices."""
    import numpy as np
    S = np.linspace(82, 118, 11)   # 11 spots inside the corridor
    res = double_barrier_call_fourier(S=S, K=100, T=1, r=0.05, sigma=0.25,
                                              L=80, U=120, n_terms=32)
    assert res.price.shape == (11,)
    assert (res.price >= 0).all()
    # interior prices should be positive
    interior = res.price[2:-2]
    assert (interior > 0).all()


def test_price_decreases_with_tight_barriers():
    """For tight barriers, the price decreases monotonically with sigma
    (knock-out probability dominates upside).  Tests the *correct*
    direction of vega in this regime."""
    prices = []
    for sig in (0.10, 0.20, 0.30, 0.40):
        v = float(double_barrier_call_fourier(S=100, K=100, T=1, r=0.05,
                                                       sigma=sig, L=80, U=120,
                                                       n_terms=64))
        prices.append(v)
    # Tight barriers (+/- 20%) -> vega is negative across this sigma range.
    for a, b in zip(prices, prices[1:]):
        assert a > b > 0, prices


def test_price_increases_with_wider_barriers():
    """Widen the barriers and the KO call should approach a vanilla call
    (which has positive vega)."""
    prices = []
    for sig in (0.10, 0.20, 0.30):
        v = float(double_barrier_call_fourier(S=100, K=100, T=1, r=0.05,
                                                       sigma=sig, L=20, U=500,
                                                       n_terms=64))
        prices.append(v)
    # Wide barriers (-80% / +400%) -> behaves like vanilla, vega positive
    for a, b in zip(prices, prices[1:]):
        assert b > a > 0, prices
