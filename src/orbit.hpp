// Keplerian orbit primitives: frame vectors, position/state from eccentric
// anomaly, true anomaly conversion. Mirrors python/gronchi_moid/reference.py.
#pragma once

#include <array>
#include <cmath>

namespace gronchi {

struct Orbit {
    double a;
    double e;
    double inc;
    double Omega;
    double omega;
    double mu;
};

struct Vec3 {
    double x, y, z;
};

inline void frame_vectors(const Orbit& o, Vec3& P, Vec3& Q) {
    const double co = std::cos(o.omega);
    const double so = std::sin(o.omega);
    const double cO = std::cos(o.Omega);
    const double sO = std::sin(o.Omega);
    const double ci = std::cos(o.inc);
    const double si = std::sin(o.inc);

    P.x = co * cO - ci * so * sO;
    P.y = co * sO + ci * so * cO;
    P.z = so * si;

    Q.x = -so * cO - ci * co * sO;
    Q.y = -so * sO + ci * co * cO;
    Q.z = co * si;
}

inline void position_from_eccentric(const Orbit& o, double u, Vec3& r) {
    Vec3 P, Q;
    frame_vectors(o, P, Q);
    const double x = o.a * (std::cos(u) - o.e);
    const double y = o.a * std::sqrt(1.0 - o.e * o.e) * std::sin(u);
    r.x = x * P.x + y * Q.x;
    r.y = x * P.y + y * Q.y;
    r.z = x * P.z + y * Q.z;
}

inline void state_from_eccentric(const Orbit& o, double u, double r[3], double v[3]) {
    Vec3 P, Q;
    frame_vectors(o, P, Q);
    const double cu = std::cos(u);
    const double su = std::sin(u);
    const double sse = std::sqrt(1.0 - o.e * o.e);
    const double x = o.a * (cu - o.e);
    const double y = o.a * sse * su;
    r[0] = x * P.x + y * Q.x;
    r[1] = x * P.y + y * Q.y;
    r[2] = x * P.z + y * Q.z;
    const double rn = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    const double vx = -std::sqrt(o.mu * o.a) / rn * su;
    const double vy =  std::sqrt(o.mu * o.a * (1.0 - o.e * o.e)) / rn * cu;
    v[0] = vx * P.x + vy * Q.x;
    v[1] = vx * P.y + vy * Q.y;
    v[2] = vx * P.z + vy * Q.z;
}

inline double true_from_eccentric(double e, double u) {
    constexpr double TWOPI = 6.283185307179586476925286766559;
    double f = 2.0 * std::atan2(std::sqrt(1.0 + e) * std::sin(0.5 * u),
                                std::sqrt(1.0 - e) * std::cos(0.5 * u));
    f = std::fmod(f + TWOPI, TWOPI);
    if (f < 0.0) f += TWOPI;
    return f;
}

}  // namespace gronchi
