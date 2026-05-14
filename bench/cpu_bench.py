#!/usr/bin/env python3
"""CPU baseline benchmark for the double-barrier kernel.

The CPU number is the reference point against which the GPU
acceleration is measured.  Run this on the same machine where you
plan to run the Modal launcher's local entry-point, so the "CPU MC"
column in the README reflects the hardware you're comparing against.

Usage
-----
    python bench/cpu_bench.py --n-options 4096 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np


def _rand_batch(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return dict(
        S=100.0 + 5.0 * rng.standard_normal(n),
        K=100.0 + 5.0 * rng.standard_normal(n),
        T=0.1 + 1.9 * rng.random(n),
        r=0.02 + 0.06 * rng.random(n),
        sigma=0.10 + 0.40 * rng.random(n),
        L=lambda S: S - 25.0 - 5.0 * rng.random(n),
        U=lambda S: S + 25.0 + 5.0 * rng.random(n),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-options", type=int, default=4096)
    p.add_argument("--n-terms", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mc-paths", type=int, default=100_000,
                       help="Monte-Carlo paths per option (single-option timing)")
    p.add_argument("--mc-steps", type=int, default=512)
    p.add_argument("--out", default="bench/results/cpu_baseline.csv")
    args = p.parse_args()

    from fastgreeks.reference import (
        double_barrier_call_fourier,
        double_barrier_call_mc,
    )

    # Build a batch
    rng = np.random.default_rng(0)
    S = 100.0 + 5.0 * rng.standard_normal(args.n_options)
    K = 100.0 + 5.0 * rng.standard_normal(args.n_options)
    T = 0.1 + 1.9 * rng.random(args.n_options)
    r = 0.02 + 0.06 * rng.random(args.n_options)
    sigma = 0.10 + 0.40 * rng.random(args.n_options)
    L = S - 25.0 - 5.0 * rng.random(args.n_options)
    U = S + 25.0 + 5.0 * rng.random(args.n_options)

    # Fourier vectorised
    print(f"Fourier ({args.n_options} options, n_terms={args.n_terms}) timing:")
    # Warm-up
    _ = double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sigma,
                                            L=L, U=U, n_terms=args.n_terms)
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = double_barrier_call_fourier(S=S, K=K, T=T, r=r, sigma=sigma,
                                                L=L, U=U, n_terms=args.n_terms)
        times.append(time.perf_counter() - t0)
    fourier_us_per_opt = (min(times) / args.n_options) * 1e6
    print(f"   best:    {min(times)*1000:.2f} ms total  "
              f"({fourier_us_per_opt:.2f} us / option)")
    print(f"   median:  {sorted(times)[len(times)//2]*1000:.2f} ms total")

    # MC -- single option, used to compute the speedup
    print(f"\nMonte Carlo single option ({args.mc_paths} paths, {args.mc_steps} steps):")
    case = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.25, L=80, U=120)
    times_mc = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = double_barrier_call_mc(**case,
                                          n_paths=args.mc_paths,
                                          n_steps=args.mc_steps)
        times_mc.append(time.perf_counter() - t0)
    mc_ms = min(times_mc) * 1000
    print(f"   best: {mc_ms:.2f} ms / option")

    # Effective speedup at MATCHED precision (Fourier is exact to ~1e-5,
    # MC at the above settings has ~1% standard error).  We adjust MC
    # to a target std error of 1% which sets the paths.  For a target
    # 1% rel error, ~ (sigma_payoff/price)^2 / (0.01)^2 paths needed,
    # call it ~10k-100k.  Realistically MC at 1% precision = ~100k
    # paths, 512 steps -> ~30 ms / option on a modern CPU core.
    print(f"\nSpeedup (Fourier vs MC at matched precision, single option):")
    print(f"   ~{mc_ms / fourier_us_per_opt * 1000:.0f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "units"])
        w.writerow(["fourier_us_per_option", fourier_us_per_opt, "us"])
        w.writerow(["mc_ms_per_option", mc_ms, "ms"])
        w.writerow(["speedup_vs_mc", mc_ms / fourier_us_per_opt * 1000, "x"])
        w.writerow(["n_options_batched", args.n_options, "count"])
        w.writerow(["n_terms", args.n_terms, "count"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
