# -------------------------------------------------------------------------------
# Minimum Orbit Intersection Distance (MOID) calculation for two Keplerian Orbits
#
# following Gronchi, G.F., Baù, G. and Grassi, C., 2023. 
# Revisiting the computation of the critical points of the Keplerian distance. 
# Celestial Mechanics and Dynamical Astronomy, 135(5), p.48.
#
# Implementation by Siegfried Eggl 2026/04/06 (with assistance from AI: ChatGPT Version 1.2026.104)
# -------------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

pi = np.pi
TWOPI = 2.0 * pi

dot = np.dot
sin = np.sin
cos = np.cos
norm = np.linalg.norm
sqrt = np.sqrt
atan = np.atan
atan2 = np.atan2
hypot = np.hypot


@dataclass(frozen=True)
class Orbit:
    """Keplerian ellipse with angles in radians."""

    a: float
    e: float
    inc: float
    Omega: float
    omega: float
    mu: float
    
    def __post_init__(self):
        if not (self.a > 0.0):
            raise ValueError("Semi-major axis must be positive.")
        if not (0.0 <= self.e < 1.0):
            raise ValueError("This implementation supports only elliptic orbits: 0 <= e < 1.")


@dataclass
class CriticalPoint:
    u1: float
    u2: float
    distance: float
    distance2: float
    grad_norm: float
    kind: str = "unknown"


@dataclass
class MOIDResult:
    moid: float
    u1: float
    u2: float
    f1: float
    f2: float
    state1: np.ndarray
    state2: np.ndarray
    critical_points: List[CriticalPoint]
    method: str
    used_shift: bool


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def _frame_vectors(orb: Orbit) -> Tuple[np.ndarray, np.ndarray]:
    co = cos(orb.omega)
    so = sin(orb.omega)
    cO = cos(orb.Omega)
    sO = sin(orb.Omega)
    ci = cos(orb.inc)
    si = sin(orb.inc)

    P = np.array([
        co * cO - ci * so * sO,
        co * sO + ci * so * cO,
        so * si,
    ], dtype=float)
    Q = np.array([
        -so * cO - ci * co * sO,
        -so * sO + ci * co * cO,
        co * si,
    ], dtype=float)
    return P, Q


def _invariants(o1: Orbit, o2: Orbit) -> Tuple[float, float, float, float]:
    P, Q = _frame_vectors(o1)
    p, q = _frame_vectors(o2)
    K = float(dot(P, p))
    L = float(dot(Q, p))
    M = float(dot(P, q))
    N = float(dot(Q, q))
    return K, L, M, N


def position_from_eccentric_anomaly(orb: Orbit, u: float) -> np.ndarray:
    P, Q = _frame_vectors(orb)
    x = orb.a * (cos(u) - orb.e)
    y = orb.a * sqrt(1.0 - orb.e * orb.e) * sin(u)
    return x * P + y * Q

def state_from_eccentric_anomaly(orb: Orbit, u: float) -> np.ndarray:
    P, Q = _frame_vectors(orb)
    x = orb.a * (cos(u) - orb.e)
    y = orb.a * sqrt(1.0 - orb.e * orb.e) * sin(u)
    r = x * P + y * Q
    rn = norm(r)
    vx = -sqrt(orb.mu*orb.a)/rn*sin(u)
    vy = sqrt(orb.mu*orb.a*(1.0 - orb.e * orb.e))/rn*np.cos(u)
    v = vx * P + vy * Q
    return np.array([r, v]).flatten() 


def distance_squared(o1: Orbit, o2: Orbit, u1: float, u2: float) -> float:
    r = position_from_eccentric_anomaly(o1, u1) - position_from_eccentric_anomaly(o2, u2)
    return float(dot(r, r))


def true_anomaly_from_eccentric_anomaly(e: float, u: float) -> float:
    f = 2.0 * atan2(
        sqrt(1.0 + e) * sin(0.5 * u),
        sqrt(1.0 - e) * cos(0.5 * u),
    )
    return (f + TWOPI) % TWOPI

# -----------------------------------------------------------------------------
# Section 3 coefficients (ordinary polynomials, eccentric anomalies)
# -----------------------------------------------------------------------------

def _A_coeffs(o1: Orbit, o2: Orbit) -> Tuple[float, ...]:
    a1, e1 = o1.a, o1.e
    a2, e2 = o2.a, o2.e
    K, L, M, N = _invariants(o1, o2)
    se1 = sqrt(1.0 - e1 * e1)
    se2 = sqrt(1.0 - e2 * e2)

    A1 = a1 * a1 * (1.0 - e1 * e1)
    A2 = 0.0
    A3 = a1 * a1
    A4 = a2 * a2 * (1.0 - e2 * e2)
    A5 = 0.0
    A6 = a2 * a2
    A7 = -2.0 * a1 * a2 * se1 * se2 * N
    A8 = -2.0 * a1 * a2 * se1 * L
    A9 = -2.0 * a1 * a2 * se2 * M
    A10 = -2.0 * a1 * a2 * K
    A11 = 2.0 * a1 * a2 * e2 * se1 * L
    A12 = 2.0 * a1 * (a2 * e2 * K - a1 * e1)
    A13 = 2.0 * a1 * a2 * e1 * se2 * M
    A14 = 2.0 * a2 * (a1 * e1 * K - a2 * e2)
    return (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)


def _poly_coeffs_unshifted(o1: Orbit, o2: Orbit):
    A1, _, A3, A4, _, A6, A7, A8, A9, A10, A11, A12, A13, A14 = _A_coeffs(o1, o2)
    # ascending powers in t
    alpha = np.array([
        (A11 - A8),
        (4*A1 - 4*A3 + 2*A10 - 2*A12),
        0.0,
        (-4*A1 + 4*A3 + 2*A10 - 2*A12),
        (A8 - A11),
    ], dtype=float)
    beta = np.array([2*A7, -4*A9, 0.0, -4*A9, -2*A7], dtype=float)
    gamma = np.array([
        (A11 + A8),
        (4*A1 - 4*A3 - 2*A10 - 2*A12),
        0.0,
        (-4*A1 + 4*A3 - 2*A10 - 2*A12),
        -(A8 + A11),
    ], dtype=float)
    A = np.array([-(A9 + A13), -2*A7, (A9 - A13)], dtype=float)
    B = np.array([(-4*A4 + 4*A6 - 2*A10 - 2*A14), -4*A8, (-4*A4 + 4*A6 + 2*A10 - 2*A14)], dtype=float)
    D = np.array([(4*A4 - 4*A6 - 2*A10 - 2*A14), -4*A8, (4*A4 - 4*A6 + 2*A10 - 2*A14)], dtype=float)
    return alpha, beta, gamma, A, B, D


def _poly_coeffs_shifted(o1: Orbit, o2: Orbit, s1: float):
    A1, _, A3, A4, _, A6, A7, A8, A9, A10, A11, A12, A13, A14 = _A_coeffs(o1, o2)
    cs = cos(s1)
    ss = sin(s1)
    d13 = (A1 - A3)

    at = np.array([
        2*d13*cs*ss - (A8 - A11)*cs + (A10 - A12)*ss,
        8*d13*cs*cs + 2*(A10 - A12)*cs + 2*(A8 - A11)*ss - 4*d13,
        -12*d13*cs*ss,
        -8*d13*cs*cs + 2*(A10 - A12)*cs + 2*(A8 - A11)*ss + 4*d13,
        2*d13*cs*ss + (A8 - A11)*cs - (A10 - A12)*ss,
    ], dtype=float)

    bt = np.array([
        2*A7*cs - 2*A9*ss,
        4*(-A7*ss - A9*cs),
        0.0,
        4*(-A7*ss - A9*cs),
        2*(-A7*cs + A9*ss),
    ], dtype=float)

    gt = np.array([
        2*d13*cs*ss + (A8 + A11)*cs - (A10 + A12)*ss,
        8*d13*cs*cs - 2*(A10 + A12)*cs - 2*(A8 + A11)*ss - 4*d13,
        -12*d13*cs*ss,
        -8*d13*cs*cs - 2*(A10 + A12)*cs - 2*(A8 + A11)*ss + 4*d13,
        2*d13*cs*ss - (A8 + A11)*cs + (A10 + A12)*ss,
    ], dtype=float)

    At = np.array([
        -A7*ss - A9*cs - A13,
        2*(-A7*cs + A9*ss),
        A7*ss + A9*cs - A13,
    ], dtype=float)

    Bt = np.array([
        -2*A10*cs - 2*A8*ss - 2*A14 - 4*(A4 - A6),
        4*(-A8*cs + A10*ss),
        2*(A10*cs + A8*ss - A14 - 2*(A4 - A6)),
    ], dtype=float)

    Dt = np.array([
        -2*A10*cs - 2*A8*ss - 2*A14 + 4*(A4 - A6),
        4*(-A8*cs + A10*ss),
        2*(A10*cs + A8*ss - A14 + 2*(A4 - A6)),
    ], dtype=float)

    return at, bt, gt, At, Bt, Dt


# -----------------------------------------------------------------------------
# Polynomial helpers
# -----------------------------------------------------------------------------

def _peval(c: np.ndarray, x: complex) -> complex:
    return np.polynomial.polynomial.polyval(x, c)


def _det_shat_eval(t: complex, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                   A: np.ndarray, B: np.ndarray, D: np.ndarray) -> complex:
    a = _peval(alpha, t)
    b = _peval(beta, t)
    g = _peval(gamma, t)
    Ap = _peval(A, t)
    Bp = _peval(B, t)
    Dp = _peval(D, t)

    sig1 = a - g
    sig2 = b
    sig3 = Bp - Dp
    den = 1.0 + t*t
    hs1 = sig1 / den
    hs2 = sig2 / den
    hs3 = sig3 / den
    sig4 = g
    sig5 = Dp
    sig6 = Ap

    M = np.array([
        [hs1, -hs2, -hs1, hs2, 0.0, -hs3],
        [hs2, hs1, -hs2, -hs1, hs3, 0.0],
        [sig4, sig2, sig1, -sig2, sig6, sig3],
        [0.0, sig4, sig2, sig1, sig5, sig6],
        [0.0, 0.0, sig4, sig2, -sig6, sig5],
        [0.0, 0.0, 0.0, sig4, 0.0, -sig6],
    ], dtype=complex)
    return complex(np.linalg.det(M))


def _interpolate_degree_16(eval_fn) -> np.ndarray:
    """Return ascending monomial coefficients c[0] + c[1] t + ... + c[16] t^16."""
    n = 17
    rho = 1.2  # slightly away from singular points at t=±i
    ts = np.array([rho * np.exp(2j * np.pi * k / n) for k in range(n)], dtype=complex)
    ys = np.array([eval_fn(t) for t in ts], dtype=complex)
    V = np.vander(ts, N=n, increasing=True)
    c = np.linalg.solve(V, ys)
    c = np.real_if_close(c, tol=1e5)
    return np.asarray(c, dtype=complex)


def _roots_real_from_poly(coeffs_asc: np.ndarray, tol_imag: float = 1e-8) -> np.ndarray:
    coeffs_desc = np.array(coeffs_asc[::-1], dtype=complex)
    # trim leading tiny coefficients
    idx = 0
    while idx < len(coeffs_desc) - 1 and abs(coeffs_desc[idx]) < 1e-14:
        idx += 1
    coeffs_desc = coeffs_desc[idx:]
    roots = np.roots(coeffs_desc)
    real_roots = []
    for r in roots:
        if abs(r.imag) < tol_imag * max(1.0, abs(r.real)):
            real_roots.append(float(r.real))
    if not real_roots:
        return np.array([], dtype=float)
    real_roots = np.array(sorted(real_roots), dtype=float)
    # deduplicate
    dedup = [real_roots[0]]
    for x in real_roots[1:]:
        if abs(x - dedup[-1]) > 1e-7 * max(1.0, abs(x), abs(dedup[-1])):
            dedup.append(x)
    return np.array(dedup, dtype=float)


# -----------------------------------------------------------------------------
# Recover (u1,u2) from polynomial roots
# -----------------------------------------------------------------------------

def _solve_s_from_pq(t: float, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                     A: np.ndarray, B: np.ndarray, D: np.ndarray) -> Optional[float]:
    a = _peval(alpha, t)
    b = _peval(beta, t)
    g = _peval(gamma, t)
    Ap = _peval(A, t)
    Bp = _peval(B, t)
    Dp = _peval(D, t)

    coeffs = np.array([a, b, g], dtype=complex)
    # Handle degeneracy gracefully.
    if abs(a) < 1e-14 and abs(b) < 1e-14:
        return None
    if abs(a) < 1e-14:
        roots = np.array([-g / b], dtype=complex)
    else:
        roots = np.roots(np.array([a, b, g], dtype=complex))

    best_s = None
    best_res = float('inf')
    for s in roots:
        qv = Ap * s**4 + Bp * s**3 + Dp * s - Ap
        res = abs(qv)
        if res < best_res:
            best_res = res
            best_s = s
    if best_s is None:
        return None
    if abs(best_s.imag) > 1e-7 * max(1.0, abs(best_s.real)):
        return None
    return float(best_s.real)


def _gradients_eq4(o1: Orbit, o2: Orbit, u1: float, u2: float) -> Tuple[float, float]:
    A1, _, A3, A4, _, A6, A7, A8, A9, A10, A11, A12, A13, A14 = _A_coeffs(o1, o2)
    su1, cu1 = sin(u1), cos(u1)
    su2, cu2 = sin(u2), cos(u2)
    g1 = 2*(A1 - A3)*su1*cu1 + A7*cu1*su2 + A8*cu1*cu2 - A9*su1*su2 - A10*su1*cu2 + A11*cu1 - A12*su1
    g2 = 2*(A4 - A6)*su2*cu2 + A7*su1*cu2 - A8*su1*su2 + A9*cu1*cu2 - A10*cu1*su2 + A13*cu2 - A14*su2
    return g1, g2


def _classify_point(points: List[Tuple[float, float, float]]) -> List[str]:
    # Hessian via finite differences for type classification.
    out = []
    for _, _, lam in points:
        if lam > 1e-10:
            out.append("minimum")
        elif lam < -1e-10:
            out.append("maximum")
        else:
            out.append("saddle/degenerate")
    return out


def _hessian_signature(o1: Orbit, o2: Orbit, u1: float, u2: float) -> float:
    h = 1e-6
    f00 = distance_squared(o1, o2, u1, u2)
    fpp = distance_squared(o1, o2, u1 + h, u2)
    fmm = distance_squared(o1, o2, u1 - h, u2)
    gpp = distance_squared(o1, o2, u1, u2 + h)
    gmm = distance_squared(o1, o2, u1, u2 - h)
    fxy1 = distance_squared(o1, o2, u1 + h, u2 + h)
    fxy2 = distance_squared(o1, o2, u1 + h, u2 - h)
    fxy3 = distance_squared(o1, o2, u1 - h, u2 + h)
    fxy4 = distance_squared(o1, o2, u1 - h, u2 - h)
    fuu = (fpp - 2*f00 + fmm) / (h*h)
    fvv = (gpp - 2*f00 + gmm) / (h*h)
    fuv = (fxy1 - fxy2 - fxy3 + fxy4) / (4*h*h)
    H = np.array([[fuu, fuv], [fuv, fvv]])
    eig = np.linalg.eigvalsh(H)
    if eig[0] > 0 and eig[1] > 0:
        return 1.0
    if eig[0] < 0 and eig[1] < 0:
        return -1.0
    return 0.0


def _recover_candidates(o1: Orbit, o2: Orbit, coeffs, shift: float = 0.0) -> List[CriticalPoint]:
    alpha, beta, gamma, A, B, D = coeffs
    poly = _interpolate_degree_16(lambda t: _det_shat_eval(t, alpha, beta, gamma, A, B, D))
    poly = np.real_if_close(poly, tol=1e5)
    roots_t = _roots_real_from_poly(poly)

    cand: List[CriticalPoint] = []
    for t in roots_t:
        s = _solve_s_from_pq(t, alpha, beta, gamma, A, B, D)
        if s is None:
            continue
        u1 = 2.0 * atan(t) + shift
        u2 = 2.0 * atan(s)
        u1 = float((u1 + TWOPI) % TWOPI)
        u2 = float((u2 + TWOPI) % TWOPI)
        g1, g2 = _gradients_eq4(o1, o2, u1, u2)
        d2 = distance_squared(o1, o2, u1, u2)
        cand.append(CriticalPoint(
            u1=u1,
            u2=u2,
            distance=sqrt(max(d2, 0.0)),
            distance2=d2,
            grad_norm=hypot(g1, g2),
        ))
    return _deduplicate_candidates(o1, o2, cand)


def _deduplicate_candidates(o1: Orbit, o2: Orbit, pts: List[CriticalPoint]) -> List[CriticalPoint]:
    out: List[CriticalPoint] = []
    for p in sorted(pts, key=lambda x: (x.distance2, x.grad_norm, x.u1, x.u2)):
        keep = True
        for q in out:
            du1 = min(abs(p.u1 - q.u1), TWOPI - abs(p.u1 - q.u1))
            du2 = min(abs(p.u2 - q.u2), TWOPI - abs(p.u2 - q.u2))
            if du1 < 1e-7 and du2 < 1e-7:
                keep = False
                break
        if keep:
            out.append(p)
    # classify
    enriched = []
    for p in out:
        lam = _hessian_signature(o1, o2, p.u1, p.u2)
        p.kind = "minimum" if lam > 0 else ("maximum" if lam < 0 else "saddle/degenerate")
        enriched.append(p)
    return enriched


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def moid_gronchi(o1: Orbit, o2: Orbit, *,
                 use_shift: bool = True,
                 shift_angle: Optional[float] = None,
                 max_candidate_grad_norm: float = 1e-5) -> MOIDResult:
    """
    Compute the MOID for two elliptic confocal orbits using the ordinary-polynomial,
    eccentric-anomaly approach of Section 3 in Gronchi, Baù & Grassi (2023).

    Parameters
    ----------
    o1, o2
        Elliptic orbits with angles in radians.
    use_shift
        If True, retry once with an angular shift for improved numerical stability.
    shift_angle
        Optional manual shift angle s1 (radians). If None and shift is used, pi/3 is applied.
    max_candidate_grad_norm
        Acceptance threshold for the residual of the critical-point equations.
    """
    coeffs = _poly_coeffs_unshifted(o1, o2)
    cand = _recover_candidates(o1, o2, coeffs, shift=0.0)
    good = [c for c in cand if c.grad_norm <= max_candidate_grad_norm]

    used_shift = False
    if (not good) and use_shift:
        used_shift = True
        s1 = float(pi / 3 if shift_angle is None else shift_angle)
        coeffs2 = _poly_coeffs_shifted(o1, o2, s1)
        cand = _recover_candidates(o1, o2, coeffs2, shift=s1)
        good = [c for c in cand if c.grad_norm <= max_candidate_grad_norm]

    if not good:
        # keep best residual if no point passed the threshold
        if not cand:
            raise RuntimeError("No critical-point candidates were recovered. The configuration may be numerically ill-conditioned.")
        good = sorted(cand, key=lambda c: (c.grad_norm, c.distance2))

    mins = [c for c in good if c.kind == "minimum"] or good
    best = min(mins, key=lambda c: c.distance2)
   
    #p1 = position_from_eccentric_anomaly(o1, best.u1)
    #p2 = position_from_eccentric_anomaly(o2, best.u2)

    s1 = state_from_eccentric_anomaly(o1, best.u1)
    s2 = state_from_eccentric_anomaly(o2, best.u2)

    true_anomaly1 = true_anomaly_from_eccentric_anomaly(o1.e, best.u1)
    true_anomaly2 = true_anomaly_from_eccentric_anomaly(o2.e, best.u2)
    
    return MOIDResult(
        moid = best.distance,
        u1 = best.u1,
        u2 = best.u2,
        f1 = true_anomaly1,
        f2 = true_anomaly2,
        state1 = s1,
        state2 = s2,
        critical_points=sorted(good, key=lambda c: (c.distance2, c.u1, c.u2)),
        method="Section 3 ordinary-polynomial / eccentric-anomaly method",
        used_shift=used_shift,
    )


# -----------------------------------------------------------------------------
# Earth elements via the Standish 1800-2050 secular fit
# -----------------------------------------------------------------------------
#
# Linear approximation to the Earth-Moon barycenter osculating elements in the
# J2000 mean ecliptic frame. Source: E. M. Standish, "Keplerian Elements for
# Approximate Positions of the Major Planets" (JPL technical memo).
#
# Only the four MOID-relevant elements are returned (a, e, inc, Omega, omega);
# mean anomaly is irrelevant to MOID.

_STANDISH_J2000_MJD = 51544.5      # JD 2451545.0 - 2400000.5
_STANDISH_CENTURY = 36525.0
_EARTH_A0 = 1.00000261;     _EARTH_AD = 0.00000562
_EARTH_E0 = 0.01671123;     _EARTH_ED = -0.00004392
_EARTH_I0_DEG = -0.00001531;     _EARTH_ID_DEG = -0.01294668
_EARTH_VARPI0_DEG = 102.93768193; _EARTH_VARPID_DEG = 0.32327364


def earth_elements_standish(mjd):
    """Earth-Moon-barycenter Keplerian elements at MJD (TDB).

    Uses the Standish 1800-2050 linear secular fit. Returns
    ``(a, e, inc, Omega, omega)`` with ``a`` in AU and angles in radians.
    Accepts a scalar or numpy-array ``mjd``; returns matching shape (the
    last axis has length 5 for array input).
    """
    mjd_arr = np.asarray(mjd, dtype=np.float64)
    T = (mjd_arr - _STANDISH_J2000_MJD) / _STANDISH_CENTURY
    a = _EARTH_A0 + _EARTH_AD * T
    e = _EARTH_E0 + _EARTH_ED * T
    inc = np.deg2rad(_EARTH_I0_DEG + _EARTH_ID_DEG * T)
    Omega = np.zeros_like(T)
    omega = np.mod(np.deg2rad(_EARTH_VARPI0_DEG + _EARTH_VARPID_DEG * T), TWOPI)
    if mjd_arr.ndim == 0:
        return float(a), float(e), float(inc), float(Omega), float(omega)
    return np.stack([a, e, inc, Omega, omega], axis=-1)


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple smoke test on two mildly inclined, non-circular ellipses.
    earth_like = Orbit(a=1.0, e=0.0167, inc=0.0, Omega=0.0, omega=1.796767421)  # rad
    test_ast = Orbit(a=1.4, e=0.23, inc=0.15, Omega=0.4, omega=1.1)
    res = moid_gronchi(earth_like, test_ast)
    print(f"MOID = {res.moid:.12f}")
    print(f"u1 = {res.u1:.12f}, u2 = {res.u2:.12f}")
    print(f"used_shift = {res.used_shift}")
    print("Critical points:")
    for cp in res.critical_points:
        print(f"  kind={cp.kind:17s} d={cp.distance:.12f} f1={cp.f1:.12f} f2={cp.f2:.12f} |grad|={cp.grad_norm:.3e}")
