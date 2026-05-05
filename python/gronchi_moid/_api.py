"""High-level numpy-array MOID API.

Generic two-orbit entry points:

* :func:`moid` — fast path, returns a 1-D ndarray of MOID distances.
* :func:`moid_full` — returns a dict of arrays (moid, u1, u2, f1, f2, state1,
  state2, status) for callers that need the rich result.

Both accept ``elements1`` / ``elements2`` shaped ``(N, 5)`` or ``(5,)`` holding
``[a, e, inc, Omega, omega]``. A shape-``(5,)`` orbit is broadcast against the
other operand.

Specialized Earth-MOID entry points (Earth's elements at the asteroid's epoch
are evaluated inside the C++ kernel via the Standish 1800-2050 linear fit):

* :func:`moid_earth` — returns a 1-D ndarray of MOID distances.
* :func:`moid_earth_full` — also returns the per-row Earth elements used.
"""

from __future__ import annotations

import numpy as np

from . import _core

GM_SUN = _core.GM_SUN_AUDAY2  # AU^3 / day^2

_STATUS_LABELS = {
    0: "ok",
    1: "used_shift",
    2: "fallback",
    3: "failed",
}


def _normalize_elements(arr, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] != 5:
            raise ValueError(f"{name} 1-D must have length 5; got shape {a.shape}")
        a = a.reshape(1, 5)
    elif a.ndim == 2:
        if a.shape[1] != 5:
            raise ValueError(f"{name} 2-D must have shape (N, 5); got {a.shape}")
    else:
        raise ValueError(f"{name} must be 1-D (5,) or 2-D (N, 5); got ndim={a.ndim}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    if (a[:, 0] <= 0).any():
        raise ValueError(f"{name}: semi-major axis 'a' must be positive")
    if ((a[:, 1] < 0) | (a[:, 1] >= 1.0)).any():
        raise ValueError(f"{name}: eccentricity must satisfy 0 <= e < 1 (elliptic only)")
    return a


def _broadcast_pair(elements1, elements2):
    e1 = _normalize_elements(elements1, "elements1")
    e2 = _normalize_elements(elements2, "elements2")
    n1, n2 = e1.shape[0], e2.shape[0]
    if n1 == n2:
        pass
    elif n1 == 1:
        e1 = np.broadcast_to(e1, (n2, 5))
    elif n2 == 1:
        e2 = np.broadcast_to(e2, (n1, 5))
    else:
        raise ValueError(
            f"elements1 and elements2 lengths are incompatible: {n1} vs {n2}"
        )
    return np.ascontiguousarray(e1), np.ascontiguousarray(e2)


def _broadcast_mu(mu, n: int, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    if a.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array")
    if a.shape[0] == 1 or a.shape[0] == n:
        return np.ascontiguousarray(a)
    raise ValueError(f"{name} length {a.shape[0]} incompatible with N={n}")


def moid(elements1, elements2, *, n_threads: int = 1) -> np.ndarray:
    """Compute MOID for one or many pairs of Keplerian orbits.

    Parameters
    ----------
    elements1, elements2 : array_like
        Shape ``(5,)`` or ``(N, 5)`` of ``[a, e, inc, Omega, omega]`` with
        angles in radians and ``a`` in any consistent length unit. A scalar
        orbit (shape ``(5,)``) is broadcast against the other operand.
    n_threads : int, optional
        Number of OpenMP threads. Default 1. Pass 0 to use the maximum.
        If the package was built without OpenMP, ``n_threads > 1`` issues a
        ``RuntimeWarning`` and runs serially.

    Returns
    -------
    moid : ndarray, shape (N,)
        MOID distance in the same length unit as the input ``a``.
    """
    e1, e2 = _broadcast_pair(elements1, elements2)
    return _core._moid_batch(e1, e2, int(n_threads))


def moid_full(elements1, elements2, *, mu1=GM_SUN, mu2=GM_SUN,
              n_threads: int = 1) -> dict:
    """Compute MOID and auxiliary quantities for one or many orbit pairs.

    Parameters
    ----------
    elements1, elements2 : array_like
        See :func:`moid`.
    mu1, mu2 : float or array_like, optional
        Gravitational parameter ``GM`` for each system, used only to recover
        velocity vectors. Scalar (default ``GM_SUN`` in AU³/day²) or
        a 1-D array of length N.
    n_threads : int, optional
        See :func:`moid`.

    Returns
    -------
    result : dict
        Keys:
          * ``moid`` (N,): minimum distance.
          * ``u1``, ``u2`` (N,): eccentric anomalies at the MOID, in [0, 2π).
          * ``f1``, ``f2`` (N,): true anomalies at the MOID, in [0, 2π).
          * ``state1``, ``state2`` (N, 6): position and velocity at the MOID
            in the inertial frame, ``[x, y, z, vx, vy, vz]``.
          * ``status`` (N,): 0=ok, 1=used_shift fallback, 2=accepted with
            elevated gradient (numerically delicate), 3=failed (NaN row).
    """
    e1, e2 = _broadcast_pair(elements1, elements2)
    n = e1.shape[0]
    m1 = _broadcast_mu(mu1, n, "mu1")
    m2 = _broadcast_mu(mu2, n, "mu2")
    return _core._moid_batch_full(e1, e2, m1, m2, int(n_threads))


def _broadcast_elements_mjd(elements, mjd):
    e = _normalize_elements(elements, "elements")
    m = np.atleast_1d(np.asarray(mjd, dtype=np.float64))
    if m.ndim != 1:
        raise ValueError("mjd must be a scalar or 1-D array")
    if not np.isfinite(m).all():
        raise ValueError("mjd contains non-finite values")
    ne, nm = e.shape[0], m.shape[0]
    if ne == nm:
        n = ne
    elif ne == 1:
        e = np.broadcast_to(e, (nm, 5))
        n = nm
    elif nm == 1:
        m = np.broadcast_to(m, (ne,))
        n = ne
    else:
        raise ValueError(
            f"elements and mjd lengths are incompatible: {ne} vs {nm}"
        )
    return np.ascontiguousarray(e), np.ascontiguousarray(m), n


def moid_earth(elements, mjd, *, n_threads: int = 1) -> np.ndarray:
    """Compute MOID against Earth at one or many asteroid epochs.

    Earth's osculating elements at each ``mjd`` are evaluated in the C++
    kernel via the Standish 1800-2050 linear fit (Earth-Moon barycenter,
    J2000 mean ecliptic frame).

    Parameters
    ----------
    elements : array_like
        Asteroid Keplerian elements. Shape ``(5,)`` or ``(N, 5)`` of
        ``[a, e, inc, Omega, omega]`` (AU + radians, J2000 ecliptic).
    mjd : float or array_like
        Modified Julian Date (TDB) of each asteroid orbit. Scalar or
        shape ``(N,)``. Scalar broadcasts against ``elements``; a single
        ``elements`` row broadcasts against an array ``mjd``.
    n_threads : int, optional
        OpenMP thread count. See :func:`moid`.

    Returns
    -------
    moid : ndarray, shape (N,)
        MOID in AU.
    """
    e, m, _ = _broadcast_elements_mjd(elements, mjd)
    return _core._moid_batch_earth(e, m, int(n_threads))


def moid_earth_full(elements, mjd, *, mu_ast=GM_SUN,
                    n_threads: int = 1) -> dict:
    """MOID against Earth plus auxiliary quantities.

    Parameters
    ----------
    elements, mjd
        See :func:`moid_earth`.
    mu_ast : float or array_like, optional
        Asteroid gravitational parameter (only affects velocity recovery).
        Scalar or 1-D array of length N. Earth's ``mu`` is hard-wired to
        :data:`GM_SUN`.
    n_threads : int, optional
        See :func:`moid`.

    Returns
    -------
    result : dict
        Same keys as :func:`moid_full` (with ``state1``/``f1``/``u1``
        referring to Earth and ``state2``/``f2``/``u2`` to the asteroid),
        plus ``earth_elements`` (N, 5) with the Standish-evaluated Earth
        row used for each pair.
    """
    e, m, n = _broadcast_elements_mjd(elements, mjd)
    mu_a = _broadcast_mu(mu_ast, n, "mu_ast")
    return _core._moid_batch_earth_full(e, m, mu_a, int(n_threads))


def has_openmp() -> bool:
    """True if the C++ extension was compiled with OpenMP support."""
    return _core._has_openmp()


def status_label(code: int) -> str:
    """Map a numeric status code (returned by :func:`moid_full`) to a string."""
    return _STATUS_LABELS.get(int(code), "unknown")
