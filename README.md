# gronchi-moid

Fast vectorized **Minimum Orbit Intersection Distance (MOID)** between
Keplerian elliptical orbits, following the ordinary-polynomial / eccentric-
anomaly method of Gronchi, Baù & Grassi (2023):

> Gronchi, G.F., Baù, G. and Grassi, C., 2023. *Revisiting the computation of
> the critical points of the Keplerian distance.* Celestial Mechanics and
> Dynamical Astronomy, 135(5), p.48.

The hot path is implemented in C++ on top of [Eigen](https://eigen.tuxfamily.org)
and bound to Python with [pybind11](https://pybind11.readthedocs.io). The API
takes NumPy arrays in and returns NumPy arrays out, with optional OpenMP
parallelization over orbit pairs.

A 1:1 Python translation of the algorithm is preserved as
``gronchi_moid.reference`` for cross-checking and pedagogy.

## Installation

```bash
git clone --recurse-submodules https://github.com/mjuric/gronchi_moid.git
cd gronchi_moid
pip install .
```

If you cloned without ``--recurse-submodules``, fetch Eigen first:

```bash
git submodule update --init --recursive
```

OpenMP is auto-detected at build time. To force a serial build, set
``GRONCHI_NO_OPENMP=1`` before ``pip install``.

## Quick start

```python
import numpy as np
import gronchi_moid as gm

# Single pair: orbits are 1-D arrays of [a, e, inc, Omega, omega] (radians).
earth = np.array([1.0,  0.0167, 0.0,  0.0,  1.796767421])
ast   = np.array([1.4,  0.23,   0.15, 0.4,  1.1])
print(gm.moid(earth, ast))               # -> array([0.14487695])

# Vectorized: (N, 5) arrays. A scalar orbit broadcasts against many.
N = 1_000_000
asts = np.column_stack([
    np.random.uniform(0.7, 3.0, N),
    np.random.uniform(0.0, 0.5, N),
    np.random.uniform(0.0, 0.5, N),
    np.random.uniform(0.0, 2*np.pi, N),
    np.random.uniform(0.0, 2*np.pi, N),
])
moids = gm.moid(earth, asts, n_threads=8)
```

For full output (eccentric and true anomalies at the MOID, position and
velocity vectors, and a per-pair status code), use ``moid_full``:

```python
res = gm.moid_full(earth, asts)
res["moid"]    # (N,)  MOID distance
res["u1"]      # (N,)  eccentric anomaly on orbit 1 at the MOID
res["u2"]      # (N,)  eccentric anomaly on orbit 2
res["f1"]      # (N,)  true anomaly on orbit 1
res["f2"]      # (N,)  true anomaly on orbit 2
res["state1"]  # (N, 6) [x, y, z, vx, vy, vz] for orbit 1 at the MOID point
res["state2"]  # (N, 6) same for orbit 2
res["status"]  # (N,) int8: 0=ok, 1=used_shift fallback,
               #            2=accepted with elevated gradient (delicate),
               #            3=algorithm singular for this pair (NaN row)
```

State vectors require a gravitational parameter; ``mu1``/``mu2`` keyword
arguments override the default heliocentric value (``gm.GM_SUN`` in
AU³/day²).

## Performance

On a 4-core / 8-thread machine, with random Earth-vs-asteroid pairs:

| threads | μs / pair | pairs / sec |
| ------: | --------: | ----------: |
|       1 |        55 |      18 000 |
|       2 |        26 |      38 000 |
|       4 |        17 |      60 000 |
|       8 |        15 |      68 000 |

The pure-Python reference runs at ~2.5 ms / pair, so the C++ kernel is
~45× faster single-threaded and ~170× faster with 8 threads.

Re-run on your machine with ``python benchmarks/bench.py [N]``.

## Numerical caveats

The algorithm is singular on configurations where the critical-point set is
not isolated — most notably **two perfectly circular coplanar orbits** (any
common-symmetry case). For these, ``moid_full`` returns ``status == 3`` and
``moid == nan`` rather than a misleading number; ``moid`` returns ``nan``.

The C++ kernel matches the Python reference to within floating-point noise
(typically ``< 1e-15`` absolute difference) on non-degenerate inputs; this is
verified by ``tests/test_reference_parity.py``.

Only **elliptical orbits** are supported (``0 ≤ e < 1``). Hyperbolic and
parabolic orbits are rejected with ``ValueError``.

## Layout

```
src/                  C++ core
  orbit.hpp           frame vectors, position/state from eccentric anomaly
  moid.hpp            algorithm: A-coeffs, polynomial setup, root recovery
  linalg.hpp          Eigen wrappers (6x6 det, 17x17 solve, real roots)
  _core.cpp           pybind11 bindings
python/gronchi_moid/  Python package
  __init__.py         public re-exports
  _api.py             broadcasting, validation, dtype handling
  reference.py        verbatim Python implementation
third_party/eigen/    vendored Eigen (git submodule)
tests/                pytest suite
benchmarks/bench.py   throughput benchmark
```

## License

MIT, see [LICENSE](LICENSE).

Original Python implementation by Siegfried Eggl (2026).
