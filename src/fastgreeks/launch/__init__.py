"""Cloud launchers for fastgreeks GPU benchmarks.

* ``bench_a100.py`` -- Modal launcher targeting 8x A100 80 GB SXM.
  Run with:  ``modal run -m fastgreeks.launch.bench_a100``.

Modal-specific dependencies live entirely in these files; the rest of
the library does not import modal.
"""
