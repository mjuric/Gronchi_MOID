"""Closed-form sanity checks: the algorithm should reproduce trivially-known
MOIDs to high precision.

The Gronchi-Bau-Grassi method has known singular configurations (e.g.
perfectly circular coplanar orbits, where a continuous family of critical
points exists). Tests use mildly non-degenerate inputs unless they are
explicitly checking the degenerate-case status reporting.
"""
import numpy as np
import pytest

import gronchi_moid as gm


def test_identical_orbit_returns_zero():
    o = np.array([1.5, 0.2, 0.3, 0.4, 0.5])
    d = gm.moid(o, o)[0]
    assert d == pytest.approx(0.0, abs=1e-9)


def test_identical_periapsis_aligned_coplanar():
    """Two coplanar ellipses sharing the same periapsis point should give
    MOID approximately zero. (Both orbits have a*(1-e) = 0.8 along the
    +x direction.)"""
    o1 = np.array([1.0, 0.2, 0.0, 0.0, 0.0])
    o2 = np.array([2.0, 0.6, 0.0, 0.0, 0.0])
    d = gm.moid(o1, o2)[0]
    assert d == pytest.approx(0.0, abs=1e-6)


def test_circular_coplanar_marked_failed():
    """The algorithm is singular for perfectly circular coplanar orbits
    (continuous family of critical points). The kernel must surface this as
    status=3 with NaN MOID rather than silently returning a wrong number."""
    o1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    o2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    res = gm.moid_full(o1, o2)
    assert res["status"][0] == 3
    assert np.isnan(res["moid"][0])


def test_status_codes_within_known_range_on_random_sample():
    rng = np.random.default_rng(0)
    n = 20
    asts = np.column_stack([
        rng.uniform(0.7, 2.5, n),
        rng.uniform(0.05, 0.4, n),
        rng.uniform(0.05, 0.4, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])
    earth = np.array([1.0, 0.0167, 0.0, 0.0, 1.79])
    res = gm.moid_full(earth, asts)
    assert set(res["status"].tolist()).issubset({0, 1, 2, 3})
    assert (res["status"] != 3).all(), "no failures expected on benign random sample"
