#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native biquadratic interlayer exchange (RKKY) coupling.
//
// For every cell belonging to region1 or region2, the bilinear + biquadratic
// RKKY field of its nearest partner-region cell in the same (x,y) column is
// added, from the areal energy
//
//     E = -J1 (m.mp) - J2 (m.mp)^2 ,
//
// giving the exact variational-derivative field
//
//     B += ( J1 + 2*J2*(m.mp) ) / (Msat * dz) * mp ,
//
// where mp is the partner magnetisation, J1 the bilinear and J2 the
// biquadratic areal coupling (J/m^2), and dz the cell size along z. J1<0 is
// antiferromagnetic; J2<0 favours the 90-degree (perpendicular) state that is
// characteristic of synthetic antiferromagnets. With J2=0 this reduces exactly
// to the bilinear RKKY field. The partner cell is located by scanning the
// z-column, so the coupling spans a nonmagnetic spacer gap between the layers.
//
// See biquad.go for the host-side wrapper.
extern "C" __global__ void
addbiquadrkky(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
              float* __restrict__ Ms_, float Ms_mul,
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

    // locate the nearest partner-region cell in the same (x,y) column along z,
    // scanning outward from iz and stopping at the first hit (checking the
    // lower side first for a deterministic tie-break).
    int P = -1;
    int dir = 0;
    for (int d = 1; d < Nz; d++) {
        int lo = iz - d;
        if (lo >= 0) {
            int Q = idx(ix, iy, lo);
            if (regions[Q] == partner) {
                P = Q;
                dir = -1;
                break;
            }
        }
        int hi = iz + d;
        if (hi < Nz) {
            int Q = idx(ix, iy, hi);
            if (regions[Q] == partner) {
                P = Q;
                dir = 1;
                break;
            }
        }
    }
    if (P < 0) {
        return;
    }

    // Apply the areal coupling only at the interface cell (see rationale in
    // rkky.cu), so the coupling stays independent of layer thickness.
    int inb = iz + dir;
    if (inb >= 0 && inb < Nz && regions[idx(ix, iy, inb)] == r) {
        return;
    }

    float3 mp = make_float3(mx[P], my[P], mz[P]);
    float dot = m0.x * mp.x + m0.y * mp.y + m0.z * mp.z;
    float pref = (J1 + 2.0f * J2 * dot) * inv_Msat(Ms_, Ms_mul, I) / dz;
    Bx[I] += pref * mp.x;
    By[I] += pref * mp.y;
    Bz[I] += pref * mp.z;
}
