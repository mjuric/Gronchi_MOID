// pybind11 bindings: vectorized MOID over (N, 5) numpy arrays.
#include "moid.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#ifdef GRONCHI_USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using gronchi::Orbit;

namespace {

inline Orbit row_to_orbit(const double* row, double mu) {
    return Orbit{row[0], row[1], row[2], row[3], row[4], mu};
}

inline void check_pair_shapes(const py::array_t<double>& e1,
                              const py::array_t<double>& e2) {
    if (e1.ndim() != 2 || e1.shape(1) != 5)
        throw std::invalid_argument("elements1 must have shape (N, 5)");
    if (e2.ndim() != 2 || e2.shape(1) != 5)
        throw std::invalid_argument("elements2 must have shape (N, 5)");
    if (e1.shape(0) != e2.shape(0))
        throw std::invalid_argument("elements1 and elements2 must have the same length after broadcasting");
}

#ifdef GRONCHI_USE_OPENMP
inline int resolve_threads(int n_threads) {
    if (n_threads <= 0) return omp_get_max_threads();
    return n_threads;
}
#else
inline int resolve_threads(int n_threads) {
    if (n_threads > 1) {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "gronchi_moid was built without OpenMP; running serially.",
                     1);
    }
    return 1;
}
#endif

py::array_t<double> moid_batch(py::array_t<double, py::array::c_style | py::array::forcecast> e1,
                               py::array_t<double, py::array::c_style | py::array::forcecast> e2,
                               int n_threads) {
    check_pair_shapes(e1, e2);
    const py::ssize_t N = e1.shape(0);
    py::array_t<double> out(N);

    const double* p1 = e1.data();
    const double* p2 = e2.data();
    double* po = out.mutable_data();

    const int threads = resolve_threads(n_threads);
    {
        py::gil_scoped_release gil;
#ifdef GRONCHI_USE_OPENMP
        #pragma omp parallel for if(threads > 1) num_threads(threads) schedule(static)
#endif
        for (py::ssize_t i = 0; i < N; ++i) {
            gronchi::Orbit o1 = row_to_orbit(p1 + 5*i, gronchi::GM_SUN_AUDAY2);
            gronchi::Orbit o2 = row_to_orbit(p2 + 5*i, gronchi::GM_SUN_AUDAY2);
            gronchi::MoidResult r;
            gronchi::moid_pair(o1, o2, r);
            po[i] = r.moid;
        }
    }
    (void)threads;
    return out;
}

py::dict moid_batch_full(py::array_t<double, py::array::c_style | py::array::forcecast> e1,
                         py::array_t<double, py::array::c_style | py::array::forcecast> e2,
                         py::array_t<double, py::array::c_style | py::array::forcecast> mu1,
                         py::array_t<double, py::array::c_style | py::array::forcecast> mu2,
                         int n_threads) {
    check_pair_shapes(e1, e2);
    const py::ssize_t N = e1.shape(0);
    if (mu1.ndim() != 1 || (mu1.shape(0) != N && mu1.shape(0) != 1))
        throw std::invalid_argument("mu1 must be 1-D of length N or 1");
    if (mu2.ndim() != 1 || (mu2.shape(0) != N && mu2.shape(0) != 1))
        throw std::invalid_argument("mu2 must be 1-D of length N or 1");

    py::array_t<double> moid_arr(N);
    py::array_t<double> u1_arr(N);
    py::array_t<double> u2_arr(N);
    py::array_t<double> f1_arr(N);
    py::array_t<double> f2_arr(N);
    py::array_t<double> state1_arr({N, py::ssize_t(6)});
    py::array_t<double> state2_arr({N, py::ssize_t(6)});
    py::array_t<int8_t> status_arr(N);

    const double* p1 = e1.data();
    const double* p2 = e2.data();
    const double* pmu1 = mu1.data();
    const double* pmu2 = mu2.data();
    const bool mu1_scalar = (mu1.shape(0) == 1);
    const bool mu2_scalar = (mu2.shape(0) == 1);
    double* pmoid = moid_arr.mutable_data();
    double* pu1 = u1_arr.mutable_data();
    double* pu2 = u2_arr.mutable_data();
    double* pf1 = f1_arr.mutable_data();
    double* pf2 = f2_arr.mutable_data();
    double* ps1 = state1_arr.mutable_data();
    double* ps2 = state2_arr.mutable_data();
    int8_t* pst = status_arr.mutable_data();

    const int threads = resolve_threads(n_threads);
    {
        py::gil_scoped_release gil;
#ifdef GRONCHI_USE_OPENMP
        #pragma omp parallel for if(threads > 1) num_threads(threads) schedule(static)
#endif
        for (py::ssize_t i = 0; i < N; ++i) {
            const double m1 = pmu1[mu1_scalar ? 0 : i];
            const double m2 = pmu2[mu2_scalar ? 0 : i];
            gronchi::Orbit o1 = row_to_orbit(p1 + 5*i, m1);
            gronchi::Orbit o2 = row_to_orbit(p2 + 5*i, m2);
            gronchi::MoidResult r;
            gronchi::moid_pair(o1, o2, r);
            pmoid[i] = r.moid;
            pu1[i] = r.u1;
            pu2[i] = r.u2;
            pf1[i] = r.f1;
            pf2[i] = r.f2;
            for (int k = 0; k < 3; ++k) {
                ps1[6*i + k]     = r.r1[k];
                ps1[6*i + 3 + k] = r.v1[k];
                ps2[6*i + k]     = r.r2[k];
                ps2[6*i + 3 + k] = r.v2[k];
            }
            pst[i] = r.status;
        }
    }
    (void)threads;

    py::dict out;
    out["moid"]   = std::move(moid_arr);
    out["u1"]     = std::move(u1_arr);
    out["u2"]     = std::move(u2_arr);
    out["f1"]     = std::move(f1_arr);
    out["f2"]     = std::move(f2_arr);
    out["state1"] = std::move(state1_arr);
    out["state2"] = std::move(state2_arr);
    out["status"] = std::move(status_arr);
    return out;
}

bool has_openmp() {
#ifdef GRONCHI_USE_OPENMP
    return true;
#else
    return false;
#endif
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native MOID kernel (Gronchi, Bau & Grassi 2023, Section 3).";
    m.def("_moid_batch", &moid_batch,
          py::arg("elements1"), py::arg("elements2"), py::arg("n_threads") = 1);
    m.def("_moid_batch_full", &moid_batch_full,
          py::arg("elements1"), py::arg("elements2"),
          py::arg("mu1"), py::arg("mu2"),
          py::arg("n_threads") = 1);
    m.def("_has_openmp", &has_openmp);
    m.attr("GM_SUN_AUDAY2") = gronchi::GM_SUN_AUDAY2;
}
