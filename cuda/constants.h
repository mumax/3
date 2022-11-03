#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#define PI     3.1415926535897932384626433 
#define MU0    (4*PI*1e-7)        // Permeability of vacuum in Tm/A
#define QE     1.60217646E-19     // Electron charge in C
#define MUB    9.2740091523E-24   // Bohr magneton in J/T
// GAMMA0 should NOT be used. It is a user definable parameter, not constant!
// Anyway, now we implement the region-wise g. It was only used in zhangli.cu
//#define GAMMA0 1.7595e11          // Gyromagnetic ratio of electron, in rad/Ts
#define HBAR   1.05457173E-34

#endif
