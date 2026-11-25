"""Reproduce Tables 2 and 3 of Gronchi, Baù & Grassi (2023), CMDA 135:48.

Table 2 prints the elements to only 5-6 significant digits, so the exact
critical points of the printed elements differ from Table 3 by ~1e-3 deg in
angle and up to a few 1e-5 in distance; tolerances are set accordingly.
"""
import numpy as np
import pytest

import gronchi_moid as gm
from gronchi_moid.reference import Orbit, moid_gronchi

# Table 2: cometary elements (q, e, i, Omega, omega[deg])
Q1, E1, W1 = 0.16582, 0.84577, 9.09466
Q2, E2, W2 = 1.0, 0.2, 10.0
EL1 = np.array([Q1 / (1 - E1), E1, 0.0, 0.0, np.deg2rad(W1)])
EL2 = np.array([Q2 / (1 - E2), E2, 0.0, 0.0, np.deg2rad(W2)])

# Table 3: (u1 [deg], u2 [deg], d, type)
TABLE3 = [
    (116.0625325, 153.9899286, 0.0000000, "minimum"),
    (243.6382848, 203.6865581, 0.0000000, "minimum"),
    (179.8948964, 178.9198966, 0.4845432, "saddle"),
    (1.6247542, 2.0946456, 0.8341185, "minimum"),
    (24.0090191, 38.3799855, 0.8401907, "saddle"),
    (334.2162041, 317.5202237, 0.8445898, "saddle"),
    (324.5270438, 126.4762243, 1.6264123, "saddle"),
    (34.8254033, 231.0377067, 1.6334795, "saddle"),
    (0.9077692, 180.7796090, 1.6658557, "maximum"),
    (179.9346562, 358.9929507, 2.9845260, "maximum"),
]
ANG_TOL_DEG = 5e-3
DIST_TOL = 1e-4


def _angdiff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@pytest.mark.parametrize("swap", [False, True], ids=["paper-order", "swapped-order"])
def test_reference_finds_all_table3_critical_points(swap):
    o1 = Orbit(*EL1, mu=gm.GM_SUN)
    o2 = Orbit(*EL2, mu=gm.GM_SUN)
    res = moid_gronchi(o2, o1) if swap else moid_gronchi(o1, o2)
    cps = res.critical_points
    assert len(cps) == 10

    found = []
    for c in cps:
        u1, u2 = np.rad2deg(c.u1), np.rad2deg(c.u2)
        found.append((u2, u1, c) if swap else (u1, u2, c))

    for u1_ref, u2_ref, d_ref, kind_ref in TABLE3:
        err, (_, _, c) = min(
            ((max(_angdiff(f[0], u1_ref), _angdiff(f[1], u2_ref)), f) for f in found),
            key=lambda x: x[0],
        )
        assert err < ANG_TOL_DEG, f"Table 3 point ({u1_ref}, {u2_ref}) not found"
        assert c.distance == pytest.approx(d_ref, abs=DIST_TOL)
        assert c.kind.split("/")[0] == kind_ref


def test_reference_morse_and_weierstrass_checks():
    cps = moid_gronchi(Orbit(*EL1, mu=gm.GM_SUN), Orbit(*EL2, mu=gm.GM_SUN)).critical_points
    kinds = [c.kind.split("/")[0] for c in cps]
    n_min, n_max = kinds.count("minimum"), kinds.count("maximum")
    assert n_min >= 1 and n_max >= 1
    assert len(cps) == 2 * (n_min + n_max)


def test_u2_near_pi_recovered():
    """Regression: points with u2 ~ 180 deg were lost via s = tan(u2/2)."""
    for a, b in [(EL1, EL2), (EL2, EL1)]:
        cps = moid_gronchi(Orbit(*a, mu=gm.GM_SUN), Orbit(*b, mu=gm.GM_SUN)).critical_points
        assert any(abs(np.rad2deg(c.u2) - 180.0) < 2.0 for c in cps)


@pytest.mark.parametrize("swap", [False, True])
def test_fast_path_table2_moid(swap):
    res = gm.moid_full(EL2, EL1) if swap else gm.moid_full(EL1, EL2)
    assert res["moid"][0] < 1e-10
    assert res["status"][0] in (0, 1)
