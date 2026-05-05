// Earth (Earth-Moon barycenter) Keplerian elements vs. epoch via the linear
// Standish secular fit (1800-2050 set, J2000 ecliptic frame).
//
// Reference: E. M. Standish, "Keplerian Elements for Approximate Positions of
// the Major Planets" (JPL technical memo). The 1800-2050 table gives, for
// each planet/barycenter, six elements as constant + linear-rate-per-Julian-
// century since J2000.0 TDB. We only carry the ones MOID needs: a, e, I,
// Omega and omega = varpi - Omega. Mean longitude L is unused (MOID is
// independent of position along the orbit).
//
// Inputs are MJD (TDB; TT close enough at this accuracy):
//     T = (mjd - 51544.5) / 36525     (JD 2451545.0 == MJD 51544.5)
#pragma once

#include <cmath>

namespace gronchi {

struct EarthElements {
    double a;       // AU
    double e;
    double inc;     // rad
    double Omega;   // rad
    double omega;   // rad, in [0, 2*pi)
};

constexpr double STANDISH_J2000_MJD = 51544.5;
constexpr double STANDISH_CENTURY   = 36525.0;

// 1800-2050 table, Earth-Moon barycenter row.
constexpr double EARTH_A0          = 1.00000261;     // AU
constexpr double EARTH_AD          = 0.00000562;     // AU / century
constexpr double EARTH_E0          = 0.01671123;
constexpr double EARTH_ED          = -0.00004392;    // / century
constexpr double EARTH_I0_DEG      = -0.00001531;
constexpr double EARTH_ID_DEG      = -0.01294668;    // deg / century
constexpr double EARTH_VARPI0_DEG  = 102.93768193;
constexpr double EARTH_VARPID_DEG  = 0.32327364;     // deg / century
// Longitude of node Omega for Earth is set to zero in the 1800-2050 table
// (the orbit is essentially the J2000 ecliptic plane, so node is degenerate).

inline void earth_elements_at_mjd(double mjd, EarthElements& o) {
    constexpr double DEG2RAD = 0.017453292519943295769;
    constexpr double TWOPI   = 6.283185307179586476925;
    const double T = (mjd - STANDISH_J2000_MJD) / STANDISH_CENTURY;

    o.a     = EARTH_A0 + EARTH_AD * T;
    o.e     = EARTH_E0 + EARTH_ED * T;
    o.inc   = (EARTH_I0_DEG + EARTH_ID_DEG * T) * DEG2RAD;
    o.Omega = 0.0;

    // omega = varpi - Omega = varpi (since Omega == 0). Wrap into [0, 2*pi).
    double w = (EARTH_VARPI0_DEG + EARTH_VARPID_DEG * T) * DEG2RAD;
    w = std::fmod(w, TWOPI);
    if (w < 0.0) w += TWOPI;
    o.omega = w;
}

}  // namespace gronchi
