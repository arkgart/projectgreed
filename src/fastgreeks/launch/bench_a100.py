"""Modal launcher: 8x A100 80 GB SXM cluster, double-barrier benchmark.

Usage
-----

    # one-time
    pip install modal
    modal token new

    # provision the cluster and run the benchmark (~$15-25 of compute)
    modal run -m fastgreeks.launch.bench_a100 --n-options=67108864

The launcher:
    1. Builds an A100-ready container image with CUDA 12.1, PyTorch 2.3,
       Triton 2.3, and the fastgreeks package.
    2. Provisions an 8x A100 80 GB SXM node (Modal billing kicks in here).
    3. Runs `torchrun --nproc-per-node=8 -m fastgreeks.distributed.runner`
       with the benchmark batch size.
    4. Downloads the JSON summary and prints it locally.
    5. Optionally re-runs the benchmark across dtype (bf16/fp16/fp32)
       and N_TERMS sweeps and emits a CSV.

The total billable time per invocation is ~2-3 minutes (most of it
container cold-start + Triton JIT compile); the actual kernel run is
sub-second.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Modal is an optional dependency
try:
    import modal
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "modal is required to launch on cloud GPUs.  "
        "Install with `pip install modal` and authenticate with `modal token new`.\n"
    )
    raise


# ----------------------------------------------------------------------
# Image definition
# ----------------------------------------------------------------------
CUDA_TAG = "12.1.0-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.3.0",
        "triton==2.3.0",
        "numpy>=1.24",
        "scipy>=1.10",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install("pandas")
    # Install fastgreeks itself from the local repo (mounted as /repo)
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[3]),
        remote_path="/repo",
    )
    .run_commands("pip install -e /repo")
)


app = modal.App("fastgreeks-bench", image=image)


# ----------------------------------------------------------------------
# Remote function: runs torchrun on 8x A100 inside the container
# ----------------------------------------------------------------------
GPU_CONFIG = modal.gpu.A100(count=8, size="80GB")


@app.function(
    gpu=GPU_CONFIG,
    timeout=900,                # 15 min hard cap; the actual run is ~3 min
    retries=0,
)
def run_benchmark(
    n_options: int = 1 << 24,
    n_terms: int = 32,
    dtype: str = "bf16",
    block_size: int = 256,
    repeats: int = 5,
) -> dict:
    """Runs the distributed runner on the 8 A100s and returns the summary."""
    import subprocess
    import tempfile
    import time

    out_path = Path("/tmp/fastgreeks_bench.json")
    if out_path.exists():
        out_path.unlink()

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node=8",
        "-m", "fastgreeks.distributed.runner",
        "--n-options", str(n_options),
        "--n-terms", str(n_terms),
        "--dtype", dtype,
        "--block-size", str(block_size),
        "--out-json", str(out_path),
    ]
    print(f"[host] launching: {' '.join(cmd)}", flush=True)

    summaries = []
    for k in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            return {
                "ok": False,
                "stderr": proc.stderr[-4000:],
                "stdout": proc.stdout[-1000:],
            }
        with out_path.open() as f:
            r = json.load(f)
        r["wall_clock_s"] = wall
        r["repeat"] = k
        summaries.append(r)
        print(f"[host] repeat {k}: kernel={r['kernel_time_ms']:.2f} ms  "
                  f"throughput={r['throughput_options_per_s']*1e-9:.2f} G/s  "
                  f"ps/opt={r['ps_per_option']:.1f}", flush=True)

    # Take the best (fastest) kernel-time run as the headline
    best = min(summaries, key=lambda x: x["kernel_time_ms"])
    median = sorted(s["kernel_time_ms"] for s in summaries)[len(summaries) // 2]
    return {
        "ok": True,
        "n_repeats": repeats,
        "best_kernel_ms": best["kernel_time_ms"],
        "median_kernel_ms": median,
        "best_throughput_g_per_s": best["throughput_options_per_s"] * 1e-9,
        "best_ps_per_option": best["ps_per_option"],
        "config": {
            "n_options": n_options, "n_terms": n_terms,
            "dtype": dtype, "block_size": block_size,
        },
        "runs": summaries,
    }


# ----------------------------------------------------------------------
# Local entry: run a sweep and write CSV locally
# ----------------------------------------------------------------------
@app.local_entrypoint()
def main(
    n_options: int = 1 << 24,
    n_terms: int = 32,
    dtype: str = "bf16",
    sweep: bool = False,
    out_csv: str = "bench/results/a100_8x.csv",
):
    """Local CLI entry.

    With ``--sweep`` set, runs over {bf16, fp16, fp32} x {16, 32, 48}
    terms and writes a CSV.  Without it, runs a single config.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    if sweep:
        configs = []
        for dt in ("bf16", "fp16", "fp32"):
            for nt in (16, 32, 48):
                configs.append((dt, nt))
    else:
        configs = [(dtype, n_terms)]

    rows = []
    for dt, nt in configs:
        print(f"\n=== dtype={dt}  n_terms={nt}  n_opts={n_options} ===", flush=True)
        with app.run():
            res = run_benchmark.remote(
                n_options=n_options, n_terms=nt, dtype=dt, repeats=5,
            )
        if not res["ok"]:
            print("FAILED:", res, file=sys.stderr)
            continue
        print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))
        rows.append({
            "dtype": dt,
            "n_terms": nt,
            "n_options": n_options,
            "best_kernel_ms": res["best_kernel_ms"],
            "median_kernel_ms": res["median_kernel_ms"],
            "best_g_per_s": res["best_throughput_g_per_s"],
            "best_ps_per_option": res["best_ps_per_option"],
        })

    # CSV
    import csv
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
