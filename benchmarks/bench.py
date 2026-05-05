"""Throughput benchmark for the C++ MOID kernel and the Python reference.

Run with ``python benchmarks/bench.py [N]``.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import gronchi_moid as gm
from gronchi_moid.reference import Orbit, moid_gronchi


def _random_orbits(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.5, 3.0, n),
        rng.uniform(0.05, 0.45, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("N", nargs="?", type=int, default=10_000,
                   help="number of orbit pairs (default 10000)")
    p.add_argument("--py-pairs", type=int, default=200,
                   help="number of pairs to time the pure-Python reference on")
    p.add_argument("--threads", type=str, default="1,2,4,8",
                   help="comma-separated thread counts to try")
    args = p.parse_args(argv)

    earth = np.array([1.0, 0.0167, 0.0, 0.0, 1.796767421])
    asts = _random_orbits(args.N, seed=0)

    # warm-up
    gm.moid(earth, asts[:16])

    print(f"OpenMP available: {gm.has_openmp()}")
    print(f"N = {args.N} pairs")
    print()
    print(f"{'threads':>8}  {'time':>10}  {'pairs/sec':>12}  {'us/pair':>9}")
    for nt_str in args.threads.split(","):
        nt = int(nt_str)
        t0 = time.perf_counter()
        gm.moid(earth, asts, n_threads=nt)
        dt = time.perf_counter() - t0
        rate = args.N / dt
        per = dt * 1e6 / args.N
        print(f"{nt:>8}  {dt*1e3:>8.1f} ms  {rate:>12.0f}  {per:>7.2f}")

    if args.py_pairs > 0:
        n = min(args.py_pairs, args.N)
        o1 = Orbit(*earth, mu=gm.GM_SUN)
        t0 = time.perf_counter()
        for i in range(n):
            o2 = Orbit(*asts[i], mu=gm.GM_SUN)
            try:
                moid_gronchi(o1, o2)
            except RuntimeError:
                pass  # algorithm singularity on ~rare configurations
        dt = time.perf_counter() - t0
        per = dt * 1e6 / n
        print(f"\nPython reference (n={n}): {dt*1e3:.1f} ms  ({n/dt:.0f} pairs/sec, {per:.0f} us/pair)")


if __name__ == "__main__":
    sys.exit(main())
