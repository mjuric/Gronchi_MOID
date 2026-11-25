"""Compare the fast path against an independent brute-force MOID.

The sample deliberately over-represents the configurations that used to fail:
near-coplanar orbits, (near-)circular orbit 1 and highly eccentric orbit 2.
"""
import numpy as np
import pytest

import gronchi_moid as gm

AU_KM = 1.495978707e8


def _frame(el):
    _, _, i, O, w = el
    co, so, cO, sO, ci, si = np.cos(w), np.sin(w), np.cos(O), np.sin(O), np.cos(i), np.sin(i)
    P = np.array([co * cO - ci * so * sO, co * sO + ci * so * cO, so * si])
    Q = np.array([-so * cO - ci * co * sO, -so * sO + ci * co * cO, co * si])
    return P, Q


def _pos_and_derivs(el, u):
    a, e = el[0], el[1]
    P, Q = _frame(el)
    b = a * np.sqrt(1 - e * e)
    X = a * (np.cos(u) - e) * P + b * np.sin(u) * Q
    dX = -a * np.sin(u) * P + b * np.cos(u) * Q
    ddX = -a * np.cos(u) * P - b * np.sin(u) * Q
    return X, dX, ddX


def brute_force_moid(el1, el2, n=720, seeds=12, iters=50):
    """Grid search over (u1, u2) followed by damped Newton on grad(d^2)."""
    u = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    a1, e1 = el1[0], el1[1]
    a2, e2 = el2[0], el2[1]
    P1, Q1 = _frame(el1)
    P2, Q2 = _frame(el2)
    X1 = np.outer(a1 * (np.cos(u) - e1), P1) + np.outer(a1 * np.sqrt(1 - e1 ** 2) * np.sin(u), Q1)
    X2 = np.outer(a2 * (np.cos(u) - e2), P2) + np.outer(a2 * np.sqrt(1 - e2 ** 2) * np.sin(u), Q2)
    D = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    best = D.min()
    for idx in np.argsort(D, axis=None)[: seeds * 8 : 8]:
        i, j = np.unravel_index(idx, D.shape)
        v = np.array([u[i], u[j]])
        for _ in range(iters):
            x1, d1, dd1 = _pos_and_derivs(el1, v[0])
            x2, d2, dd2 = _pos_and_derivs(el2, v[1])
            r = x1 - x2
            g = 2 * np.array([d1 @ r, -d2 @ r])
            H = 2 * np.array([[d1 @ d1 + dd1 @ r, -d1 @ d2], [-d1 @ d2, d2 @ d2 - dd2 @ r]])
            # Levenberg-style damping keeps the iteration descending
            lam = max(0.0, -np.linalg.eigvalsh(H)[0]) + 1e-12 * a1 * a2
            step = np.linalg.solve(H + lam * np.eye(2), g)
            v = v - step
            if np.max(np.abs(step)) < 1e-14:
                break
        x1, _, _ = _pos_and_derivs(el1, v[0])
        x2, _, _ = _pos_and_derivs(el2, v[1])
        best = min(best, float((x1 - x2) @ (x1 - x2)))
    return float(np.sqrt(max(best, 0.0)))


def _sample(n, seed):
    rng = np.random.default_rng(seed)
    r1, r2 = [], []
    while len(r1) < n:
        k = len(r1)
        q = rng.uniform(0.1, 1.3)
        e = rng.uniform(0.0, 0.95)
        a = q / (1 - e)
        if a > 6:
            continue
        r1.append([1.0, [0.0167, 0.0, rng.uniform(0, 0.5)][k % 3],
                   [0.0, 0.0, rng.uniform(0, 0.3)][k % 3], 0.0, rng.uniform(0, 2 * np.pi)])
        r2.append([a, e, [0.0, 1e-4, rng.uniform(0, 1.0)][k % 3],
                   rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)])
    return np.array(r1), np.array(r2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_moid_matches_brute_force(seed):
    E1, E2 = _sample(60, seed)
    res = gm.moid_full(E1, E2)
    ref = np.array([brute_force_moid(a, b) for a, b in zip(E1, E2)])
    assert np.all(np.isfinite(res["moid"]))
    np.testing.assert_allclose(res["moid"], ref, atol=1e-6, rtol=0)
    assert (res["status"] != 2).all()


@pytest.mark.parametrize("seed", [3, 4])
def test_reference_matches_brute_force(seed):
    from gronchi_moid.reference import Orbit, moid_gronchi
    E1, E2 = _sample(20, seed)
    for a, b in zip(E1, E2):
        got = moid_gronchi(Orbit(*a, mu=gm.GM_SUN), Orbit(*b, mu=gm.GM_SUN)).moid
        assert got == pytest.approx(brute_force_moid(a, b), abs=1e-6)


def test_unit_independence():
    E1, E2 = _sample(60, 5)
    m_au = gm.moid_full(E1, E2)
    K1, K2 = E1.copy(), E2.copy()
    K1[:, 0] *= AU_KM
    K2[:, 0] *= AU_KM
    m_km = gm.moid_full(K1, K2, mu1=1.32712440018e11, mu2=1.32712440018e11)
    np.testing.assert_allclose(m_km["moid"] / AU_KM, m_au["moid"], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(m_km["status"], m_au["status"])


def test_swap_symmetry():
    E1, E2 = _sample(60, 6)
    np.testing.assert_allclose(gm.moid(E1, E2), gm.moid(E2, E1), atol=1e-9, rtol=0)


@pytest.mark.parametrize("q,e,r", [(0.5, 0.3, 1.0), (1.5, 0.1, 1.0), (0.2, 0.9, 3.0)])
@pytest.mark.parametrize("omega", [0.0, 0.6, 3.5])
def test_coplanar_ellipse_vs_circle_closed_form(q, e, r, omega):
    """Coplanar ellipse vs circle of radius r: MOID = max(0, q - r, r - Q)."""
    ell = np.array([q / (1 - e), e, 0.0, 0.0, omega])
    circ = np.array([r, 0.0, 0.0, 0.0, 0.0])
    Q = q * (1 + e) / (1 - e)
    assert gm.moid(ell, circ)[0] == pytest.approx(max(0.0, q - r, r - Q), abs=1e-9)
