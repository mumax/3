#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native interlayer Dzyaloshinskii-Moriya interaction (chiral interlayer coupling).
//
// For every cell belonging to region1 or region2, the interlayer-DMI field of
// its nearest partner-region cell on EACH side along z is added:
//
//     B += s * ( D / (Msat * dz) ) * ( zhat x m_partner )
//
// with D the areal interlayer-DMI strength (J/m^2), dz the cell size along z,
// zhat the interface normal, and s = +1 when the partner sits above the cell
// (higher z) and s = -1 when it sits below. This antisymmetric sign makes the
// pair of fields the exact variational derivative of the interlayer-DMI energy
//
//     E = D * zhat . (m1 x m2) * A ,
//
// i.e. B1 = +(D/(Msat*dz)) (zhat x m2) and B2 = -(D/(Msat*dz)) (zhat x m1),
// which is the interlayer analogue of interfacial DMI. The partner cell is
// located by walking outward along z, so the coupling spans a (nonmagnetic)
// spacer gap between the layers, unlike the nearest-neighbour exchange field.
// Note  zhat x m = (-m.y, m.x, 0).
//
// Interface selection matches rkky.cu: in each z-direction the walk stops at
// the first partner cell (couple, with its own sign s = dir) or the first
// same-region cell (buried on that side, no coupling). Coupling on both sides
// lets a layer between two partners get both chiral contributions.
//
// See idmi.go for the host-side wrapper.
extern "C" __global__ void
addidmi(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
        float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
        float* __restrict__ Ms_, float Ms_mul,
        uint8_t* __restrict__ regions, float D, int region1, int region2,
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

    float base = D * inv_Msat(Ms_, Ms_mul, I) / dz;

    for (int dir = -1; dir <= 1; dir += 2) {
        for (int j = iz + dir; j >= 0 && j < Nz; j += dir) {
            int rj = regions[idx(ix, iy, j)];
            if (rj == r) {
                break;  // buried on this side
            }
            if (rj == partner) {
                int P = idx(ix, iy, j);
                float pref = (float)dir * base;  // s = dir (+1 above, -1 below)
                // zhat x m_partner = (-m_partner.y, m_partner.x, 0)
                Bx[I] += pref * (-my[P]);
                By[I] += pref * (mx[P]);
                // Bz unchanged: (zhat x m)_z = 0
                break;  // nearest partner on this side
            }
        }
    }
}
