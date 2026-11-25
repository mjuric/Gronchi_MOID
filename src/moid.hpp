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
// Residual cutoff for |grad d^2| / max(a1, a2)^2 (dimensionless, so results
// do not depend on the length unit of `a`).
constexpr double GRAD_CUTOFF = 1e-5;
constexpr double DEDUP_ANGLE_TOL = 1e-7;
constexpr double FALLBACK_SHIFT = PI / 3.0;
// Relative eigenvalue threshold for the analytic Hessian classification.
constexpr double HESSIAN_REL_TOL = 1e-10;
// Max dimensionless residual of the 2nd critical-point equation for a
// recovered u2 branch to be kept (the better branch is always kept).
constexpr double U2_BRANCH_TOL = 1e-3;
// Newton polishing of each recovered critical point.
constexpr int NEWTON_ITERS = 3;
constexpr double NEWTON_MAX_STEP = 1e-2;
// det(S^) is treated as identically zero (degenerate configuration, e.g.
// concentric coplanar circles) if all samples are below this fraction of the
// scale of the matrix entries^6.
constexpr double DEGENERATE_DET_TOL = 1e-12;

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
// Returns false if det(S^) vanishes identically (to rounding), i.e. the
// configuration is degenerate and has a continuum of critical points.
inline bool interpolate_degree16(const PolyCoeffs& P, double rho,
                                 double out_real[17]) {
    cd ts[17], ys[17], coeffs[17];
    double ymax = 0.0;
    for (int k = 0; k < 17; ++k) {
        const double ang = 2.0 * PI * static_cast<double>(k) / 17.0;
        ts[k] = cd(rho * std::cos(ang), rho * std::sin(ang));
        ys[k] = det_shat_eval(ts[k], P);
        ymax = std::max(ymax, std::abs(ys[k]));
    }
    // Scale of a 6x6 determinant built from the polynomial blocks.
    double cmax = 0.0;
    for (int i = 0; i < 5; ++i)
        cmax = std::max({cmax, std::abs(P.alpha[i]), std::abs(P.beta[i]), std::abs(P.gamma[i])});
    for (int i = 0; i < 3; ++i)
        cmax = std::max({cmax, std::abs(P.A[i]), std::abs(P.B[i]), std::abs(P.D[i])});
    const double scale = std::pow(cmax * std::pow(std::max(1.0, rho), 4), 6);
    if (!(ymax > DEGENERATE_DET_TOL * scale)) return false;
    solve_vandermonde17(ts, ys, coeffs);
    for (int i = 0; i < 17; ++i) out_real[i] = coeffs[i].real();
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

// Jacobian of (g1, g2) from gradients_eq4 w.r.t. (u1, u2). Symmetric.
inline void jacobian_eq4(const ACoeffs& a, double u1, double u2,
                         double& j11, double& j12, double& j22) {
    const double s1 = std::sin(u1), c1 = std::cos(u1);
    const double s2 = std::sin(u2), c2 = std::cos(u2);
    j11 = 2*(a.A1 - a.A3)*(c1*c1 - s1*s1) - a.A7*s1*s2 - a.A8*s1*c2
        - a.A9*c1*s2 - a.A10*c1*c2 - a.A11*s1 - a.A12*c1;
    j12 = a.A7*c1*c2 - a.A8*c1*s2 - a.A9*s1*c2 + a.A10*s1*s2;
    j22 = 2*(a.A4 - a.A6)*(c2*c2 - s2*s2) - a.A7*s1*s2 - a.A8*s1*c2
        - a.A9*c1*s2 - a.A10*c1*c2 - a.A13*s2 - a.A14*c2;
}

inline double grad_scale(const Orbit& o1, const Orbit& o2) {
    const double m = std::max(o1.a, o2.a);
    return m * m;
}

// Hessian classification from the analytic Hessian of d^2(u1, u2).
// Returns +1 if positive-definite (minimum), -1 if negative-definite, 0
// otherwise. Unlike finite differences this is independent of the length
// unit (no round-off amplification for large `a`).
inline int hessian_signature(const Orbit& o1, const Orbit& o2,
                             double u1, double u2) {
    Vec3 P1, Q1, P2, Q2;
    frame_vectors(o1, P1, Q1);
    frame_vectors(o2, P2, Q2);
    const double b1 = o1.a * std::sqrt(1.0 - o1.e * o1.e);
    const double b2 = o2.a * std::sqrt(1.0 - o2.e * o2.e);
    const double c1 = std::cos(u1), s1 = std::sin(u1);
    const double c2 = std::cos(u2), s2 = std::sin(u2);
    double X1[3], d1[3], dd1[3], X2[3], d2[3], dd2[3];
    const double P1v[3] = {P1.x, P1.y, P1.z}, Q1v[3] = {Q1.x, Q1.y, Q1.z};
    const double P2v[3] = {P2.x, P2.y, P2.z}, Q2v[3] = {Q2.x, Q2.y, Q2.z};
    for (int k = 0; k < 3; ++k) {
        X1[k]  = o1.a * (c1 - o1.e) * P1v[k] + b1 * s1 * Q1v[k];
        d1[k]  = -o1.a * s1 * P1v[k] + b1 * c1 * Q1v[k];
        dd1[k] = -o1.a * c1 * P1v[k] - b1 * s1 * Q1v[k];
        X2[k]  = o2.a * (c2 - o2.e) * P2v[k] + b2 * s2 * Q2v[k];
        d2[k]  = -o2.a * s2 * P2v[k] + b2 * c2 * Q2v[k];
        dd2[k] = -o2.a * c2 * P2v[k] - b2 * s2 * Q2v[k];
    }
    double h11 = 0, h12 = 0, h22 = 0;
    for (int k = 0; k < 3; ++k) {
        const double r = X1[k] - X2[k];
        h11 += d1[k] * d1[k] + dd1[k] * r;
        h12 -= d1[k] * d2[k];
        h22 += d2[k] * d2[k] - dd2[k] * r;
    }
    h11 *= 2; h12 *= 2; h22 *= 2;
    const double tr = h11 + h22;
    const double det = h11 * h22 - h12 * h12;
    const double disc = std::sqrt(std::max(0.0, tr*tr - 4*det));
    const double lam1 = 0.5 * (tr - disc);
    const double lam2 = 0.5 * (tr + disc);
    const double tol = HESSIAN_REL_TOL * grad_scale(o1, o2);
    if (lam1 > tol && lam2 > tol) return +1;
    if (lam1 < -tol && lam2 < -tol) return -1;
    return 0;
}

// Given u1, solve the first critical-point equation of (4) for u2. It is
// linear in (cos u2, sin u2):  Ac cos u2 + Bc sin u2 + Cc = 0  (cf. Sect. 3.1
// of the paper). Solving for the angle directly avoids s = tan(u2/2), which
// diverges at u2 = pi and made those critical points unrecoverable. Returns
// the number of branches written to u2_out (0, 1 or 2), best branch first.
inline int solve_u2_given_u1(const Orbit& o1, const Orbit& o2, const ACoeffs& a,
                             double u1, double u2_out[2]) {
    const double su = std::sin(u1), cu = std::cos(u1);
    const double Ac = a.A8 * cu - a.A10 * su;
    const double Bc = a.A7 * cu - a.A9 * su;
    const double Cc = 2 * (a.A1 - a.A3) * su * cu + a.A11 * cu - a.A12 * su;
    const double R = std::hypot(Ac, Bc);
    if (!(R > 0.0)) return 0;
    double x = -Cc / R;
    if (std::abs(x) > 1.0 + 1e-8) return 0;
    x = std::min(1.0, std::max(-1.0, x));
    const double phi = std::atan2(Bc, Ac);
    const double dphi = std::acos(x);
    double cand[2] = {phi + dphi, phi - dphi};
    double res[2];
    const double sc = grad_scale(o1, o2);
    for (int i = 0; i < 2; ++i) {
        cand[i] = std::fmod(cand[i], TWOPI);
        if (cand[i] < 0) cand[i] += TWOPI;
        double g1, g2;
        gradients_eq4(a, u1, cand[i], g1, g2);
        res[i] = std::abs(g2) / sc;
    }
    const int ib = (res[0] <= res[1]) ? 0 : 1;
    int n = 0;
    u2_out[n++] = cand[ib];
    if (res[1 - ib] < U2_BRANCH_TOL) u2_out[n++] = cand[1 - ib];
    return n;
}

// A few Newton steps on (g1, g2) = 0 with the analytic Jacobian; steps larger
// than NEWTON_MAX_STEP are rejected (keeps the point in its basin).
inline void newton_polish(const ACoeffs& a, double& u1, double& u2) {
    for (int it = 0; it < NEWTON_ITERS; ++it) {
        double g1, g2, j11, j12, j22;
        gradients_eq4(a, u1, u2, g1, g2);
        jacobian_eq4(a, u1, u2, j11, j12, j22);
        const double det = j11 * j22 - j12 * j12;
        if (!(std::abs(det) > 0.0)) break;
        const double du1 = -( j22 * g1 - j12 * g2) / det;
        const double du2 = -(-j12 * g1 + j11 * g2) / det;
        if (!std::isfinite(du1) || !std::isfinite(du2) ||
            std::max(std::abs(du1), std::abs(du2)) > NEWTON_MAX_STEP) break;
        u1 += du1;
        u2 += du2;
    }
    u1 = std::fmod(u1, TWOPI); if (u1 < 0) u1 += TWOPI;
    u2 = std::fmod(u2, TWOPI); if (u2 < 0) u2 += TWOPI;
}

// Sort by (dist2, grad_norm, u1, u2), drop near-duplicates, classify.
inline void dedup_and_classify(const Orbit& o1, const Orbit& o2,
                               std::vector<CritPoint>& out) {
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

// Build candidate critical points from polynomial roots, dedup, classify.
// Returns false if the configuration is degenerate (det(S^) == 0).
inline bool recover_candidates(const Orbit& o1, const Orbit& o2,
                               const ACoeffs& a, const PolyCoeffs& P,
                               double shift, std::vector<CritPoint>& out) {
    out.clear();
    double poly[17];
    if (!interpolate_degree16(P, INTERP_RADIUS, poly)) return false;
    auto roots = real_roots_real_poly(poly, 16);
    out.reserve(2 * roots.size());
    const double sc = grad_scale(o1, o2);
    for (double t : roots) {
        double u1 = 2.0 * std::atan(t) + shift;
        u1 = std::fmod(u1, TWOPI); if (u1 < 0) u1 += TWOPI;
        double u2b[2];
        const int nb = solve_u2_given_u1(o1, o2, a, u1, u2b);
        for (int k = 0; k < nb; ++k) {
            double v1 = u1, v2 = u2b[k];
            newton_polish(a, v1, v2);
            double g1, g2;
            gradients_eq4(a, v1, v2, g1, g2);
            const double d2 = distance_squared(o1, o2, v1, v2);
            out.push_back({v1, v2, d2, std::hypot(g1, g2) / sc, 0});
        }
    }
    dedup_and_classify(o1, o2, out);
    return true;
}

// Full pipeline for one orbit pair. Always writes a result; failure modes:
//  - status == STATUS_FAILED if no candidates were recovered at all (NaN out)
inline void moid_pair(const Orbit& o1, const Orbit& o2, MoidResult& out) {
    const ACoeffs a = compute_A(o1, o2);

    std::vector<CritPoint> cand;
    cand.reserve(16);

    PolyCoeffs P0 = poly_coeffs_unshifted(a);
    const bool ok0 = recover_candidates(o1, o2, a, P0, 0.0, cand);
    bool used_shift = false;

    // Weierstrass check (Sect. 6 of the paper): d^2 on the torus must have at
    // least one minimum and one maximum among the accepted points.
    auto checks_pass = [](const std::vector<CritPoint>& v) {
        bool has_min = false, has_max = false;
        for (const auto& p : v) {
            if (p.grad_norm > GRAD_CUTOFF) continue;
            has_min |= (p.kind == +1);
            has_max |= (p.kind == -1);
        }
        return has_min && has_max;
    };

    if (ok0 && !checks_pass(cand)) {
        used_shift = true;
        std::vector<CritPoint> cand1;
        PolyCoeffs P1 = poly_coeffs_shifted(a, FALLBACK_SHIFT);
        if (recover_candidates(o1, o2, a, P1, FALLBACK_SHIFT, cand1)) {
            cand.insert(cand.end(), cand1.begin(), cand1.end());
            dedup_and_classify(o1, o2, cand);
        }
    }

    // Degenerate configuration (det(S^) == 0): the only closed-form case we
    // handle is two coincident orbits (MOID = 0 everywhere on the diagonal).
    if (!ok0) {
        Vec3 P1, Q1, P2, Q2;
        frame_vectors(o1, P1, Q1);
        frame_vectors(o2, P2, Q2);
        const double tol = 1e-12;
        const bool same =
            std::abs(o1.a - o2.a) <= tol * std::max(o1.a, o2.a) &&
            std::abs(o1.e - o2.e) <= tol &&
            std::abs(P1.x - P2.x) + std::abs(P1.y - P2.y) + std::abs(P1.z - P2.z) <= tol &&
            std::abs(Q1.x - Q2.x) + std::abs(Q1.y - Q2.y) + std::abs(Q1.z - Q2.z) <= tol;
        if (same) cand.push_back({0.0, 0.0, distance_squared(o1, o2, 0.0, 0.0), 0.0, +1});
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
