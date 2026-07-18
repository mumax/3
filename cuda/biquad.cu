#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native biquadratic interlayer exchange (RKKY) coupling.
//
// For every cell belonging to region1 or region2, the bilinear + biquadratic
// RKKY field of its nearest partner-region cell on EACH side along z is added,
// from the areal energy
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
// to the bilinear RKKY field. The partner cell is located by walking outward
// along z, so the coupling spans a nonmagnetic spacer gap between the layers.
//
// Interface selection matches rkky.cu: in each z-direction the walk stops at
// the first partner cell (couple) or the first same-region cell (buried on
// that side, no coupling); coupling on both sides supports superlattices.
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

    float invMs = inv_Msat(Ms_, Ms_mul, I);

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
                float pref = (J1 + 2.0f * J2 * dot) * invMs / dz;
                Bx[I] += pref * mp.x;
                By[I] += pref * mp.y;
                Bz[I] += pref * mp.z;
                break;  // nearest partner on this side
            }
        }
    }
}
