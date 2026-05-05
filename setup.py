"""Build glue for the C++ extension.

Most metadata lives in pyproject.toml; this file only describes the native
extension because pyproject can't drive Pybind11Extension directly.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).parent.resolve()
EIGEN_DIR = ROOT / "third_party" / "eigen"

if not EIGEN_DIR.exists():
    raise RuntimeError(
        f"Vendored Eigen not found at {EIGEN_DIR}. "
        "Run `git submodule update --init --recursive` first."
    )


def _compiler_supports_flag(flag: str) -> bool:
    """Probe whether the user's C++ compiler accepts a flag."""
    cc = os.environ.get("CXX") or ("cl" if platform.system() == "Windows" else "c++")
    src = "int main(){return 0;}"
    with tempfile.TemporaryDirectory() as td:
        srcfile = Path(td) / "probe.cpp"
        outfile = Path(td) / ("probe.exe" if platform.system() == "Windows" else "probe")
        srcfile.write_text(src)
        try:
            subprocess.run(
                [cc, str(srcfile), flag, "-o", str(outfile)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False
        return True


def _detect_openmp() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Return (extra_compile_args, extra_link_args, define_macros) for OpenMP.

    Returns empty lists if no OpenMP toolchain is available; the build then
    proceeds serially.
    """
    if os.environ.get("GRONCHI_NO_OPENMP"):
        return [], [], []
    system = platform.system()
    if system == "Windows":
        # MSVC: /openmp; clang-cl: -openmp:experimental
        return ["/openmp"], [], [("GRONCHI_USE_OPENMP", "1")]
    if system == "Darwin":
        # AppleClang ships without libomp; require explicit -lomp.
        flags = ["-Xpreprocessor", "-fopenmp"]
        if _compiler_supports_flag("-Xpreprocessor"):
            return flags, ["-lomp"], [("GRONCHI_USE_OPENMP", "1")]
        return [], [], []
    if _compiler_supports_flag("-fopenmp"):
        return ["-fopenmp"], ["-fopenmp"], [("GRONCHI_USE_OPENMP", "1")]
    return [], [], []


omp_compile, omp_link, omp_defines = _detect_openmp()

extra_compile_args = []
extra_link_args = []
if platform.system() == "Windows":
    extra_compile_args += ["/O2", "/fp:fast", "/EHsc"]
else:
    extra_compile_args += ["-O3", "-ffast-math", "-fno-finite-math-only"]
extra_compile_args += omp_compile
extra_link_args += omp_link

ext = Pybind11Extension(
    "gronchi_moid._core",
    sources=["src/_core.cpp"],
    include_dirs=["src", str(EIGEN_DIR)],
    cxx_std=17,
    define_macros=[("EIGEN_MPL2_ONLY", "1"), *omp_defines],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
