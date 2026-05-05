// Thin Eigen-backed helpers used by the MOID core:
//   * 6x6 complex determinant
//   * 17x17 complex Vandermonde linear solve (interpolation)
//   * Real roots of a real polynomial of degree <= 16 via companion matrix.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace gronchi {

using cd = std::complex<double>;

// Determinant of a 6x6 complex matrix.
// M is a flat row-major buffer of length 36.
inline cd det6_complex(const cd* M) {
    Eigen::Matrix<cd, 6, 6> A;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            A(i, j) = M[i * 6 + j];
    return A.determinant();
}

// Solve V * c = y where V is the 17x17 ascending-power Vandermonde matrix
// V[i,j] = ts[i]^j. ts[] and ys[] are length 17; out[] receives 17 ascending
// coefficients (c[0] + c[1]*t + ... + c[16]*t^16).
inline void solve_vandermonde17(const cd* ts, const cd* ys, cd* out) {
    constexpr int N = 17;
    Eigen::Matrix<cd, N, N> V;
    Eigen::Matrix<cd, N, 1> y;
    for (int i = 0; i < N; ++i) {
        y(i) = ys[i];
        cd p = 1.0;
        for (int j = 0; j < N; ++j) {
            V(i, j) = p;
            p *= ts[i];
        }
    }
    Eigen::Matrix<cd, N, 1> c = V.partialPivLu().solve(y);
    for (int i = 0; i < N; ++i) out[i] = c(i);
}

// Real roots of a real polynomial p(t) = c[0] + c[1]*t + ... + c[deg]*t^deg.
// Companion matrix in real arithmetic; eigenvalues whose imag part is small
// relative to the magnitude are taken as real roots. After collecting, the
// roots are sorted and near-duplicates are removed.
inline std::vector<double> real_roots_real_poly(const double* c_asc, int deg,
                                                double tol_imag = 1e-8,
                                                double tol_dedup = 1e-7) {
    // trim leading near-zero coefficients in descending order
    while (deg > 0 && std::abs(c_asc[deg]) < 1e-14) --deg;
    if (deg <= 0) return {};

    if (deg == 1) {
        return { -c_asc[0] / c_asc[1] };
    }
    if (deg == 2) {
        const double a = c_asc[2], b = c_asc[1], cc = c_asc[0];
        const double disc = b * b - 4 * a * cc;
        if (disc < 0) return {};
        const double s = std::sqrt(disc);
        return { (-b - s) / (2 * a), (-b + s) / (2 * a) };
    }

    Eigen::MatrixXd C(deg, deg);
    C.setZero();
    const double an = c_asc[deg];
    for (int i = 0; i < deg; ++i) C(i, deg - 1) = -c_asc[i] / an;
    for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;

    Eigen::EigenSolver<Eigen::MatrixXd> es(C, /*computeEigenvectors=*/false);
    if (es.info() != Eigen::Success) return {};

    std::vector<double> roots;
    roots.reserve(deg);
    auto eigs = es.eigenvalues();
    for (int i = 0; i < deg; ++i) {
        const double re = eigs(i).real();
        const double im = eigs(i).imag();
        if (std::abs(im) < tol_imag * std::max(1.0, std::abs(re)))
            roots.push_back(re);
    }
    if (roots.empty()) return roots;
    std::sort(roots.begin(), roots.end());
    std::vector<double> dedup;
    dedup.reserve(roots.size());
    dedup.push_back(roots.front());
    for (size_t i = 1; i < roots.size(); ++i) {
        const double scale = std::max({1.0, std::abs(roots[i]), std::abs(dedup.back())});
        if (std::abs(roots[i] - dedup.back()) > tol_dedup * scale)
            dedup.push_back(roots[i]);
    }
    return dedup;
}

}  // namespace gronchi
