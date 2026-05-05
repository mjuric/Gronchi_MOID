"""Vectorized Minimum Orbit Intersection Distance (MOID) for Keplerian orbits.

Algorithm: Gronchi, Bau & Grassi (2023), Section 3 (ordinary-polynomial /
eccentric-anomaly approach). The fast path is implemented in C++; a verbatim
Python reference is available as ``gronchi_moid.reference`` for cross-checking.

Quick usage:

    >>> import numpy as np
    >>> import gronchi_moid as gm
    >>> earth = np.array([1.0, 0.0167, 0.0, 0.0, 1.7967])
    >>> ast = np.array([1.4, 0.23, 0.15, 0.4, 1.1])
    >>> gm.moid(earth, ast)
    array([0.30...])
"""

from ._api import GM_SUN, has_openmp, moid, moid_full, status_label
from . import reference

__all__ = [
    "moid",
    "moid_full",
    "GM_SUN",
    "has_openmp",
    "status_label",
    "reference",
]

__version__ = "0.1.0"
