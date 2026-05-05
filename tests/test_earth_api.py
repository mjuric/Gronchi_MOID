import numpy as np
import pytest

import gronchi_moid as gm
from gronchi_moid.reference import earth_elements_standish


J2000_MJD = 51544.5


def _random_orbits(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.7, 2.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 2 * np.pi, n),
        rng.uniform(0.0, 2 * np.pi, n),
    ])


def _random_mjds(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(J2000_MJD - 50 * 365.25, J2000_MJD + 50 * 365.25, n)


def test_standish_at_j2000_matches_hand_set_earth():
    a, e, inc, Omega, omega = earth_elements_standish(J2000_MJD)
    assert abs(a - 1.0) < 1e-4
    assert abs(e - 0.0167) < 1e-4
    assert abs(inc) < 1e-3
    assert abs(Omega) < 1e-12
    assert abs(omega - 1.796767421) < 1e-3


def test_moid_earth_matches_generic_moid():
    n = 64
    asts = _random_orbits(n)
    mjds = _random_mjds(n)
    earth_rows = earth_elements_standish(mjds)
    expected = gm.moid(earth_rows, asts)
    got = gm.moid_earth(asts, mjds)
    # Earth-element evaluation differs only by FP reassociation under
    # -ffast-math; allow tight ULP-level tolerance.
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_moid_earth_scalar_mjd_broadcasts():
    asts = _random_orbits(7)
    d = gm.moid_earth(asts, J2000_MJD)
    assert d.shape == (7,)
    expected = gm.moid(np.asarray(earth_elements_standish(J2000_MJD)), asts)
    np.testing.assert_allclose(d, expected, rtol=1e-12, atol=1e-14)


def test_moid_earth_scalar_elements_array_mjd():
    ast = _random_orbits(1)[0]
    mjds = _random_mjds(5)
    d = gm.moid_earth(ast, mjds)
    assert d.shape == (5,)
    earth_rows = earth_elements_standish(mjds)
    expected = gm.moid(earth_rows, np.broadcast_to(ast, (5, 5)).copy())
    np.testing.assert_allclose(d, expected, rtol=1e-12, atol=1e-14)


def test_moid_earth_scalar_scalar_returns_length_one():
    ast = _random_orbits(1)[0]
    d = gm.moid_earth(ast, J2000_MJD)
    assert d.shape == (1,)


def test_moid_earth_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        gm.moid_earth(_random_orbits(3), _random_mjds(4))


def test_moid_earth_full_shapes_and_echo():
    n = 6
    asts = _random_orbits(n)
    mjds = _random_mjds(n)
    res = gm.moid_earth_full(asts, mjds)
    assert res["moid"].shape == (n,)
    assert res["u1"].shape == (n,)
    assert res["u2"].shape == (n,)
    assert res["f1"].shape == (n,)
    assert res["f2"].shape == (n,)
    assert res["state1"].shape == (n, 6)
    assert res["state2"].shape == (n, 6)
    assert res["status"].shape == (n,)
    assert res["earth_elements"].shape == (n, 5)
    np.testing.assert_allclose(res["earth_elements"],
                               earth_elements_standish(mjds),
                               rtol=1e-12, atol=1e-14)


def test_moid_earth_threading_consistency():
    asts = _random_orbits(50)
    mjds = _random_mjds(50)
    d1 = gm.moid_earth(asts, mjds, n_threads=1)
    d4 = gm.moid_earth(asts, mjds, n_threads=4)
    np.testing.assert_array_equal(d1, d4)  # threading is a parallel reduction over independent rows; bit-identical


def test_earth_eccentricity_decreases_over_centuries():
    e_now = earth_elements_standish(J2000_MJD)[1]
    e_future = earth_elements_standish(J2000_MJD + 200 * 36525.0)[1]
    assert e_future < e_now


def test_moid_earth_invalid_eccentricity_raises():
    bad = np.array([1.5, 1.2, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gm.moid_earth(bad, J2000_MJD)
