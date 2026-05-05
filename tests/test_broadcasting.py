import numpy as np
import pytest

import gronchi_moid as gm


def _earth():
    return np.array([1.0, 0.0167, 0.0, 0.0, 1.796767421])


def _random_orbits(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.7, 2.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])


def test_scalar_vs_scalar_returns_length_one():
    d = gm.moid(_earth(), _random_orbits(1)[0])
    assert d.shape == (1,)


def test_scalar_vs_array_broadcasts():
    asts = _random_orbits(7)
    d = gm.moid(_earth(), asts)
    assert d.shape == (7,)


def test_array_vs_scalar_broadcasts():
    asts = _random_orbits(7)
    d = gm.moid(asts, _earth())
    assert d.shape == (7,)


def test_array_vs_array_same_length():
    a = _random_orbits(5, seed=1)
    b = _random_orbits(5, seed=2)
    d = gm.moid(a, b)
    assert d.shape == (5,)


def test_incompatible_lengths_raises():
    with pytest.raises(ValueError):
        gm.moid(_random_orbits(3), _random_orbits(4))


def test_invalid_eccentricity_raises():
    bad = _earth().copy()
    bad[1] = 1.5
    with pytest.raises(ValueError):
        gm.moid(bad, _earth())


def test_threading_gives_same_answer():
    asts = _random_orbits(50)
    d1 = gm.moid(_earth(), asts, n_threads=1)
    d4 = gm.moid(_earth(), asts, n_threads=4)
    np.testing.assert_allclose(d1, d4, atol=0, rtol=0)


def test_full_result_shapes():
    n = 6
    asts = _random_orbits(n)
    res = gm.moid_full(_earth(), asts)
    assert res["moid"].shape == (n,)
    assert res["u1"].shape == (n,)
    assert res["u2"].shape == (n,)
    assert res["f1"].shape == (n,)
    assert res["f2"].shape == (n,)
    assert res["state1"].shape == (n, 6)
    assert res["state2"].shape == (n, 6)
    assert res["status"].shape == (n,)
    # u, f in [0, 2pi)
    assert ((0 <= res["u1"]) & (res["u1"] < 2*np.pi + 1e-12)).all()
    assert ((0 <= res["f1"]) & (res["f1"] < 2*np.pi + 1e-12)).all()


def test_per_orbit_mu():
    n = 4
    asts = _random_orbits(n)
    mu = np.full(n, gm.GM_SUN)
    res = gm.moid_full(_earth(), asts, mu1=mu, mu2=mu)
    assert res["state1"].shape == (n, 6)
