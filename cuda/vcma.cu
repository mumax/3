#include <stdint.h>
#include "stencil.h"
#include "amul.h"

// Voltage-controlled magnetic anisotropy (VCMA).
//
// An electric field E across the oxide modulates the interfacial perpendicular
// anisotropy. The areal anisotropy change is dK = xi * E, where xi is the areal
// VCMA coefficient (J/(V m), typically ~50-100 fJ/(V m)) and E is in V/m. This
// gives a perpendicular (z) anisotropy field
//
//     B_z = 2 * (xi * E) * m_z / (Msat * t)
//
// with t the ferromagnet/interface thickness. Uses the 1/Msat convention (no
// explicit mu0), consistent with the uniaxial anisotropy field in mumax.
// Perpendicular (z) easy axis. See vcma.go.
extern "C" __global__ void
addvcma(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
        float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
        float* __restrict__ Ms_, float Ms_mul,
        float* __restrict__ E_, float E_mul,
        float* __restrict__ xi_, float xi_mul,
        float* __restrict__ thickness_, float thickness_mul,
        int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int i = idx(ix, iy, iz);

    float ms = amul(Ms_, Ms_mul, i);
    float e  = amul(E_, E_mul, i);
    float t  = amul(thickness_, thickness_mul, i);
    if (ms == 0.0f || e == 0.0f || t <= 0.0f) {
        return;
    }
    float xi = amul(xi_, xi_mul, i);

    float dK = xi * e;                          // areal anisotropy change, J/m^2
    Bz[i] += 2.0f * dK * mz[i] / (ms * t);
}
