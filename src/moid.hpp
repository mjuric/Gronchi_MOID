// MOID computation following Gronchi, Bau & Grassi (2023), Section 3
// (ordinary-polynomial / eccentric-anomaly approach). Direct port of
// python/gronchi_moid/reference.py.
#pragma once

#include "linalg.hpp"
#include "orbit.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace gronchi {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWOPI = 2.0 * PI;

// Default heliocentric gravitational parameter, AU^3 / day^2.
constexpr double GM_SUN_AUDAY2 = 2.9591220828559115e-04;

// Tunables (kept in lockstep with the reference).
constexpr double INTERP_RADIUS = 1.2;
constexpr double GRAD_CUTOFF = 1e-5;
constexpr double DEDUP_ANGLE_TOL = 1e-7;
constexpr double FALLBACK_SHIFT = PI / 3.0;
constexpr double HESSIAN_FD_STEP = 1e-6;
constexpr double SOLVE_S_REAL_TOL = 1e-7;

struct CritPoint {
    double u1;
    double u2;
    double dist2;
    double grad_norm;
    int kind;  // -1 max, 0 saddle/degenerate, +1 min
};

// status codes for batch results
enum Status : int8_t {
    STATUS_OK = 0,
    STATUS_USED_SHIFT = 1,
    STATUS_FALLBACK = 2,  // no candidate cleared the gradient cutoff
    STATUS_FAILED = 3,
};

struct MoidResult {
    double moid;
    double u1, u2;
    double f1, f2;
    double r1[3], v1[3];
    double r2[3], v2[3];
    int8_t status;
};

// ---------------------------------------------------------------------------
// A-coefficients
// ---------------------------------------------------------------------------
struct ACoeffs {
    double A1, A3, A4, A6, A7, A8, A9, A10, A11, A12, A13, A14;
};

inline ACoeffs compute_A(const Orbit& o1, const Orbit& o2) {
    Vec3 P, Q, p, q;
    frame_vectors(o1, P, Q);
    frame_vectors(o2, p, q);
    const double K = P.x * p.x + P.y * p.y + P.z * p.z;
    const double L = Q.x * p.x + Q.y * p.y + Q.z * p.z;
    const double M = P.x * q.x + P.y * q.y + P.z * q.z;
    const double N = Q.x * q.x + Q.y * q.y + Q.z * q.z;

    const double a1 = o1.a, e1 = o1.e, a2 = o2.a, e2 = o2.e;
    const double se1 = std::sqrt(1.0 - e1 * e1);
    const double se2 = std::sqrt(1.0 - e2 * e2);

    ACoeffs A;
    A.A1  = a1 * a1 * (1.0 - e1 * e1);
    A.A3  = a1 * a1;
    A.A4  = a2 * a2 * (1.0 - e2 * e2);
    A.A6  = a2 * a2;
    A.A7  = -2.0 * a1 * a2 * se1 * se2 * N;
    A.A8  = -2.0 * a1 * a2 * se1 * L;
    A.A9  = -2.0 * a1 * a2 * se2 * M;
    A.A10 = -2.0 * a1 * a2 * K;
    A.A11 = 2.0 * a1 * a2 * e2 * se1 * L;
    A.A12 = 2.0 * a1 * (a2 * e2 * K - a1 * e1);
    A.A13 = 2.0 * a1 * a2 * e1 * se2 * M;
    A.A14 = 2.0 * a2 * (a1 * e1 * K - a2 * e2);
    return A;
}

// ---------------------------------------------------------------------------
// Polynomial coefficient blocks. alpha/beta/gamma are degree-4 in t,
// A,B,D are degree-2 in t, all stored ascending (c[0] + c[1]*t + ...).
// ---------------------------------------------------------------------------
struct PolyCoeffs {
    double alpha[5];
    double beta[5];
    double gamma[5];
    double A[3];
    double B[3];
    double D[3];
};

inline PolyCoeffs poly_coeffs_unshifted(const ACoeffs& a) {
    PolyCoeffs P;
    P.alpha[0] = (a.A11 - a.A8);
    P.alpha[1] = (4*a.A1 - 4*a.A3 + 2*a.A10 - 2*a.A12);
    P.alpha[2] = 0.0;
    P.alpha[3] = (-4*a.A1 + 4*a.A3 + 2*a.A10 - 2*a.A12);
    P.alpha[4] = (a.A8 - a.A11);

    P.beta[0] = 2*a.A7;
    P.beta[1] = -4*a.A9;
    P.beta[2] = 0.0;
    P.beta[3] = -4*a.A9;
    P.beta[4] = -2*a.A7;

    P.gamma[0] = (a.A11 + a.A8);
    P.gamma[1] = (4*a.A1 - 4*a.A3 - 2*a.A10 - 2*a.A12);
    P.gamma[2] = 0.0;
    P.gamma[3] = (-4*a.A1 + 4*a.A3 - 2*a.A10 - 2*a.A12);
    P.gamma[4] = -(a.A8 + a.A11);

    P.A[0] = -(a.A9 + a.A13);
    P.A[1] = -2*a.A7;
    P.A[2] = (a.A9 - a.A13);

    P.B[0] = (-4*a.A4 + 4*a.A6 - 2*a.A10 - 2*a.A14);
    P.B[1] = -4*a.A8;
    P.B[2] = (-4*a.A4 + 4*a.A6 + 2*a.A10 - 2*a.A14);

    P.D[0] = (4*a.A4 - 4*a.A6 - 2*a.A10 - 2*a.A14);
    P.D[1] = -4*a.A8;
    P.D[2] = (4*a.A4 - 4*a.A6 + 2*a.A10 - 2*a.A14);
    return P;
}

inline PolyCoeffs poly_coeffs_shifted(const ACoeffs& a, double s1) {
    PolyCoeffs P;
    const double cs = std::cos(s1);
    const double ss = std::sin(s1);
    const double d13 = a.A1 - a.A3;

    P.alpha[0] = 2*d13*cs*ss - (a.A8 - a.A11)*cs + (a.A10 - a.A12)*ss;
    P.alpha[1] = 8*d13*cs*cs + 2*(a.A10 - a.A12)*cs + 2*(a.A8 - a.A11)*ss - 4*d13;
    P.alpha[2] = -12*d13*cs*ss;
    P.alpha[3] = -8*d13*cs*cs + 2*(a.A10 - a.A12)*cs + 2*(a.A8 - a.A11)*ss + 4*d13;
    P.alpha[4] = 2*d13*cs*ss + (a.A8 - a.A11)*cs - (a.A10 - a.A12)*ss;

    P.beta[0] = 2*a.A7*cs - 2*a.A9*ss;
    P.beta[1] = 4*(-a.A7*ss - a.A9*cs);
    P.beta[2] = 0.0;
    P.beta[3] = 4*(-a.A7*ss - a.A9*cs);
    P.beta[4] = 2*(-a.A7*cs + a.A9*ss);

    P.gamma[0] = 2*d13*cs*ss + (a.A8 + a.A11)*cs - (a.A10 + a.A12)*ss;
    P.gamma[1] = 8*d13*cs*cs - 2*(a.A10 + a.A12)*cs - 2*(a.A8 + a.A11)*ss - 4*d13;
    P.gamma[2] = -12*d13*cs*ss;
    P.gamma[3] = -8*d13*cs*cs - 2*(a.A10 + a.A12)*cs - 2*(a.A8 + a.A11)*ss + 4*d13;
    P.gamma[4] = 2*d13*cs*ss - (a.A8 + a.A11)*cs + (a.A10 + a.A12)*ss;

    P.A[0] = -a.A7*ss - a.A9*cs - a.A13;
    P.A[1] = 2*(-a.A7*cs + a.A9*ss);
    P.A[2] = a.A7*ss + a.A9*cs - a.A13;

    P.B[0] = -2*a.A10*cs - 2*a.A8*ss - 2*a.A14 - 4*(a.A4 - a.A6);
    P.B[1] = 4*(-a.A8*cs + a.A10*ss);
    P.B[2] = 2*(a.A10*cs + a.A8*ss - a.A14 - 2*(a.A4 - a.A6));

    P.D[0] = -2*a.A10*cs - 2*a.A8*ss - 2*a.A14 + 4*(a.A4 - a.A6);
    P.D[1] = 4*(-a.A8*cs + a.A10*ss);
    P.D[2] = 2*(a.A10*cs + a.A8*ss - a.A14 + 2*(a.A4 - a.A6));
    return P;
}

// ---------------------------------------------------------------------------
// Polynomial evaluation. Coefficients ascending: c[0] + c[1]*x + c[2]*x^2 ...
// ---------------------------------------------------------------------------
template <typename T, int N>
inline T peval(const double (&c)[N], T x) {
    T y = T(c[N - 1]);
    for (int i = N - 2; i >= 0; --i) y = y * x + T(c[i]);
    return y;
}

// 6x6 determinant of the matrix from eq. (3.something) in the paper at a
// given complex t. Returns det * (1 + t^2)^3 implicitly via the rationals
// hs1,hs2,hs3 == sigma_i / (1 + t^2). This matches the Python reference.
inline cd det_shat_eval(cd t, const PolyCoeffs& P) {
    const cd a  = peval(P.alpha, t);
    const cd b  = peval(P.beta,  t);
    const cd g  = peval(P.gamma, t);
    const cd Ap = peval(P.A, t);
    const cd Bp = peval(P.B, t);
    const cd Dp = peval(P.D, t);

    const cd sig1 = a - g;
    const cd sig2 = b;
    const cd sig3 = Bp - Dp;
    const cd den  = cd(1.0, 0.0) + t * t;
    const cd hs1 = sig1 / den;
    const cd hs2 = sig2 / den;
    const cd hs3 = sig3 / den;
    const cd sig4 = g;
    const cd sig5 = Dp;
    const cd sig6 = Ap;

    cd M[36] = {
        hs1, -hs2, -hs1,  hs2,  cd(0.0,0.0), -hs3,
        hs2,  hs1, -hs2, -hs1,  hs3,  cd(0.0,0.0),
        sig4, sig2, sig1, -sig2, sig6, sig3,
        cd(0.0,0.0), sig4, sig2, sig1, sig5, sig6,
        cd(0.0,0.0), cd(0.0,0.0), sig4, sig2, -sig6, sig5,
        cd(0.0,0.0), cd(0.0,0.0), cd(0.0,0.0), sig4, cd(0.0,0.0), -sig6,
    };
    return det6_complex(M);
}

// Sample det_shat at 17 points on a circle of radius rho around the origin
// and recover ascending coefficients via a 17x17 Vandermonde solve. The
// resulting polynomial has real coefficients up to floating-point noise; we
// take real parts.
inline void interpolate_degree16(const PolyCoeffs& P, double rho,
                                 double out_real[17]) {
    cd ts[17], ys[17], coeffs[17];
    for (int k = 0; k < 17; ++k) {
        const double ang = 2.0 * PI * static_cast<double>(k) / 17.0;
        ts[k] = cd(rho * std::cos(ang), rho * std::sin(ang));
        ys[k] = det_shat_eval(ts[k], P);
    }
    solve_vandermonde17(ts, ys, coeffs);
    for (int i = 0; i < 17; ++i) out_real[i] = coeffs[i].real();
}

// Given a real root t, solve the quadratic alpha(t) s^2 + beta(t) s + gamma(t)
// for s, picking the root that minimizes the residual on the quartic
// A(t) s^4 + B(t) s^3 + D(t) s - A(t). Returns false if no real s is found.
inline bool solve_s_from_pq(double t, const PolyCoeffs& P, double& s_out) {
    const double a = peval(P.alpha, t);
    const double b = peval(P.beta,  t);
    const double g = peval(P.gamma, t);
    const double Ap = peval(P.A, t);
    const double Bp = peval(P.B, t);
    const double Dp = peval(P.D, t);

    cd roots[2];
    int nroots = 0;
    if (std::abs(a) < 1e-14 && std::abs(b) < 1e-14) return false;
    if (std::abs(a) < 1e-14) {
        roots[0] = cd(-g / b, 0.0);
        nroots = 1;
    } else {
        const cd disc = cd(b * b - 4.0 * a * g, 0.0);
        const cd sq = std::sqrt(disc);
        roots[0] = (-cd(b, 0.0) - sq) / (2.0 * a);
        roots[1] = (-cd(b, 0.0) + sq) / (2.0 * a);
        nroots = 2;
    }

    double best_res = std::numeric_limits<double>::infinity();
    cd best_s(0.0, 0.0);
    bool found = false;
    for (int i = 0; i < nroots; ++i) {
        const cd s = roots[i];
        const cd s2 = s * s;
        const cd s3 = s2 * s;
        const cd s4 = s2 * s2;
        const cd qv = Ap * s4 + Bp * s3 + Dp * s - Ap;
        const double res = std::abs(qv);
        if (res < best_res) {
            best_res = res;
            best_s = s;
            found = true;
        }
    }
    if (!found) return false;
    if (std::abs(best_s.imag()) > SOLVE_S_REAL_TOL * std::max(1.0, std::abs(best_s.real())))
        return false;
    s_out = best_s.real();
    return true;
}

inline double distance_squared(const Orbit& o1, const Orbit& o2,
                               double u1, double u2) {
    Vec3 r1, r2;
    position_from_eccentric(o1, u1, r1);
    position_from_eccentric(o2, u2, r2);
    const double dx = r1.x - r2.x, dy = r1.y - r2.y, dz = r1.z - r2.z;
    return dx*dx + dy*dy + dz*dz;
}

inline void gradients_eq4(const ACoeffs& a, double u1, double u2,
                          double& g1, double& g2) {
    const double su1 = std::sin(u1), cu1 = std::cos(u1);
    const double su2 = std::sin(u2), cu2 = std::cos(u2);
    g1 = 2*(a.A1 - a.A3)*su1*cu1 + a.A7*cu1*su2 + a.A8*cu1*cu2
       - a.A9*su1*su2 - a.A10*su1*cu2 + a.A11*cu1 - a.A12*su1;
    g2 = 2*(a.A4 - a.A6)*su2*cu2 + a.A7*su1*cu2 - a.A8*su1*su2
       + a.A9*cu1*cu2 - a.A10*cu1*su2 + a.A13*cu2 - a.A14*su2;
}

// Hessian classification by 5-point finite difference of distance_squared.
// Returns +1 if positive-definite (minimum), -1 if negative-definite, 0
// otherwise.
inline int hessian_signature(const Orbit& o1, const Orbit& o2,
                             double u1, double u2) {
    const double h = HESSIAN_FD_STEP;
    const double f00 = distance_squared(o1, o2, u1, u2);
    const double fpp = distance_squared(o1, o2, u1 + h, u2);
    const double fmm = distance_squared(o1, o2, u1 - h, u2);
    const double gpp = distance_squared(o1, o2, u1, u2 + h);
    const double gmm = distance_squared(o1, o2, u1, u2 - h);
    const double f1 = distance_squared(o1, o2, u1 + h, u2 + h);
    const double f2 = distance_squared(o1, o2, u1 + h, u2 - h);
    const double f3 = distance_squared(o1, o2, u1 - h, u2 + h);
    const double f4 = distance_squared(o1, o2, u1 - h, u2 - h);
    const double fuu = (fpp - 2*f00 + fmm) / (h*h);
    const double fvv = (gpp - 2*f00 + gmm) / (h*h);
    const double fuv = (f1 - f2 - f3 + f4) / (4*h*h);
    // 2x2 eigenvalues
    const double tr = fuu + fvv;
    const double det = fuu * fvv - fuv * fuv;
    const double disc = std::sqrt(std::max(0.0, tr*tr - 4*det));
    const double lam1 = 0.5 * (tr - disc);
    const double lam2 = 0.5 * (tr + disc);
    if (lam1 > 1e-10 && lam2 > 1e-10) return +1;
    if (lam1 < -1e-10 && lam2 < -1e-10) return -1;
    return 0;
}

// Build candidate critical points from polynomial roots, dedup, classify.
inline void recover_candidates(const Orbit& o1, const Orbit& o2,
                               const ACoeffs& a, const PolyCoeffs& P,
                               double shift, std::vector<CritPoint>& out) {
    out.clear();
    double poly[17];
    interpolate_degree16(P, INTERP_RADIUS, poly);
    auto roots = real_roots_real_poly(poly, 16);
    out.reserve(roots.size());
    for (double t : roots) {
        double s;
        if (!solve_s_from_pq(t, P, s)) continue;
        double u1 = 2.0 * std::atan(t) + shift;
        double u2 = 2.0 * std::atan(s);
        u1 = std::fmod(u1, TWOPI); if (u1 < 0) u1 += TWOPI;
        u2 = std::fmod(u2, TWOPI); if (u2 < 0) u2 += TWOPI;
        double g1, g2;
        gradients_eq4(a, u1, u2, g1, g2);
        const double d2 = distance_squared(o1, o2, u1, u2);
        out.push_back({u1, u2, d2, std::hypot(g1, g2), 0});
    }
    // dedup: sort by (dist2, grad_norm, u1, u2), drop near-duplicates in (u1,u2)
    std::sort(out.begin(), out.end(), [](const CritPoint& a, const CritPoint& b){
        if (a.dist2 != b.dist2) return a.dist2 < b.dist2;
        if (a.grad_norm != b.grad_norm) return a.grad_norm < b.grad_norm;
        if (a.u1 != b.u1) return a.u1 < b.u1;
        return a.u2 < b.u2;
    });
    std::vector<CritPoint> kept;
    kept.reserve(out.size());
    for (const auto& p : out) {
        bool dup = false;
        for (const auto& q : kept) {
            const double du1 = std::min(std::abs(p.u1 - q.u1), TWOPI - std::abs(p.u1 - q.u1));
            const double du2 = std::min(std::abs(p.u2 - q.u2), TWOPI - std::abs(p.u2 - q.u2));
            if (du1 < DEDUP_ANGLE_TOL && du2 < DEDUP_ANGLE_TOL) { dup = true; break; }
        }
        if (!dup) kept.push_back(p);
    }
    for (auto& p : kept) p.kind = hessian_signature(o1, o2, p.u1, p.u2);
    out.swap(kept);
}

// Full pipeline for one orbit pair. Always writes a result; failure modes:
//  - status == STATUS_FAILED if no candidates were recovered at all (NaN out)
inline void moid_pair(const Orbit& o1, const Orbit& o2, MoidResult& out) {
    const ACoeffs a = compute_A(o1, o2);

    std::vector<CritPoint> cand;
    cand.reserve(16);

    PolyCoeffs P0 = poly_coeffs_unshifted(a);
    recover_candidates(o1, o2, a, P0, 0.0, cand);
    bool used_shift = false;

    auto count_good = [](const std::vector<CritPoint>& v) {
        int n = 0;
        for (const auto& p : v) if (p.grad_norm <= GRAD_CUTOFF) ++n;
        return n;
    };

    if (count_good(cand) == 0) {
        used_shift = true;
        PolyCoeffs P1 = poly_coeffs_shifted(a, FALLBACK_SHIFT);
        recover_candidates(o1, o2, a, P1, FALLBACK_SHIFT, cand);
    }

    if (cand.empty()) {
        out.status = STATUS_FAILED;
        out.moid = std::numeric_limits<double>::quiet_NaN();
        out.u1 = out.u2 = out.f1 = out.f2 = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < 3; ++i) out.r1[i] = out.v1[i] = out.r2[i] = out.v2[i]
            = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Build the "good" pool: those clearing the gradient cutoff; if none,
    // fall back to all candidates sorted by gradient norm.
    std::vector<const CritPoint*> good;
    good.reserve(cand.size());
    bool fallback = false;
    for (const auto& p : cand) if (p.grad_norm <= GRAD_CUTOFF) good.push_back(&p);
    if (good.empty()) {
        fallback = true;
        std::vector<size_t> idx(cand.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            if (cand[a].grad_norm != cand[b].grad_norm) return cand[a].grad_norm < cand[b].grad_norm;
            return cand[a].dist2 < cand[b].dist2;
        });
        for (size_t i : idx) good.push_back(&cand[i]);
    }

    // Among "good", prefer minima; otherwise just pick lowest dist2.
    const CritPoint* best = nullptr;
    for (const auto* p : good) {
        if (p->kind == +1) {
            if (!best || p->dist2 < best->dist2) best = p;
        }
    }
    if (!best) {
        for (const auto* p : good) {
            if (!best || p->dist2 < best->dist2) best = p;
        }
    }

    out.moid = std::sqrt(std::max(0.0, best->dist2));
    out.u1 = best->u1;
    out.u2 = best->u2;
    out.f1 = true_from_eccentric(o1.e, best->u1);
    out.f2 = true_from_eccentric(o2.e, best->u2);
    state_from_eccentric(o1, best->u1, out.r1, out.v1);
    state_from_eccentric(o2, best->u2, out.r2, out.v2);
    out.status = fallback ? STATUS_FALLBACK : (used_shift ? STATUS_USED_SHIFT : STATUS_OK);
}

}  // namespace gronchi
