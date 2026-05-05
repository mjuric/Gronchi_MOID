"""C++ kernel must agree with the Python reference to ~ floating-point noise
on a sweep of random orbit pairs.
"""
import numpy as np
import pytest

import gronchi_moid as gm
from gronchi_moid.reference import Orbit, moid_gronchi


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_random_pairs_match_reference(seed):
    rng = np.random.default_rng(seed)
    n = 40
    e1 = np.column_stack([
        rng.uniform(0.5, 2.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])
    e2 = np.column_stack([
        rng.uniform(0.5, 3.0, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 0.6, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])

    fast = gm.moid(e1, e2)

    ref = np.empty(n)
    for i in range(n):
        o1 = Orbit(*e1[i], mu=gm.GM_SUN)
        o2 = Orbit(*e2[i], mu=gm.GM_SUN)
        ref[i] = moid_gronchi(o1, o2).moid

    # MOID itself must match to ~ floating-point precision (the C++ port is a
    # 1:1 translation of the reference).
    np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-12)


def test_full_state_matches_reference():
    rng = np.random.default_rng(11)
    n = 8
    e1 = np.column_stack([
        rng.uniform(0.7, 2.0, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])
    e2 = np.column_stack([
        rng.uniform(0.7, 2.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 2*np.pi, n),
        rng.uniform(0.0, 2*np.pi, n),
    ])
    res = gm.moid_full(e1, e2)
    for i in range(n):
        o1 = Orbit(*e1[i], mu=gm.GM_SUN)
        o2 = Orbit(*e2[i], mu=gm.GM_SUN)
        ref = moid_gronchi(o1, o2)
        assert res["moid"][i] == pytest.approx(ref.moid, rel=1e-10, abs=1e-12)
        # Angles can differ by 2π between equivalent representations, so
        # compare via sin/cos rather than raw values.
        for got, want in [(res["u1"][i], ref.u1), (res["u2"][i], ref.u2),
                          (res["f1"][i], ref.f1), (res["f2"][i], ref.f2)]:
            assert np.cos(got) == pytest.approx(np.cos(want), abs=1e-7)
            assert np.sin(got) == pytest.approx(np.sin(want), abs=1e-7)
        np.testing.assert_allclose(res["state1"][i], ref.state1, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(res["state2"][i], ref.state2, rtol=1e-7, atol=1e-9)
