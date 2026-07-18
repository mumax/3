#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Energy density of the biquadratic RKKY interlayer coupling (see biquad.cu).
// At each interface cell, half of the pair energy is deposited, for each side:
//
//     edens += -0.5 * ( J1*(m.mp) + J2*(m.mp)^2 ) / dz     [J/m^3]
//
// so that summing over the two interface cells and multiplying by the cell
// volume recovers  E = -J1 A (m1.m2) - J2 A (m1.m2)^2  per interface. A
// dedicated kernel is used because the generic -1/2 (M.B) energy density would
// double-count the quartic (biquadratic) term. The interface-cell selection
// (nearest partner on each side, same-region stop) matches addbiquadrkky
// exactly. See biquad.go for the host-side wrapper.
extern "C" __global__ void
addbiquadenergy(float* __restrict__ edens,
                float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                uint8_t* __restrict__ regions, float J1, float J2, int region1, int region2,
                float dz, int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);
    int r = regions[I];
    int partner;
    if (r == region1) {
        partner = region2;
    } else if (r == region2) {
        partner = region1;
    } else {
        return;
    }

    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    if (is0(m0)) {
        return;
    }

    for (int dir = -1; dir <= 1; dir += 2) {
        for (int j = iz + dir; j >= 0 && j < Nz; j += dir) {
            int rj = regions[idx(ix, iy, j)];
            if (rj == r) {
                break;  // buried on this side
            }
            if (rj == partner) {
                int P = idx(ix, iy, j);
                float3 mp = make_float3(mx[P], my[P], mz[P]);
                float dot = m0.x * mp.x + m0.y * mp.y + m0.z * mp.z;
                edens[I] += -0.5f * (J1 * dot + J2 * dot * dot) / dz;
                break;  // nearest partner on this side
            }
        }
    }
}
