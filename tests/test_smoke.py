import numpy as np

import gronchi_moid as gm


def test_module_metadata():
    assert hasattr(gm, "moid")
    assert hasattr(gm, "moid_full")
    assert hasattr(gm, "GM_SUN")
    assert isinstance(gm.has_openmp(), bool)


def test_readme_example_runs():
    earth = np.array([1.0, 0.0167, 0.0, 0.0, 1.796767421])
    ast = np.array([1.4, 0.23, 0.15, 0.4, 1.1])
    d = gm.moid(earth, ast)
    assert d.shape == (1,)
    assert np.isfinite(d[0])
    assert 0.0 <= d[0] < 5.0
