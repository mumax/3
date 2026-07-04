#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native interlayer Dzyaloshinskii-Moriya interaction (chiral interlayer coupling).
//
// For every cell belonging to region1 or region2, the interlayer-DMI field of
// its nearest partner-region cell in the same (x,y) column is added:
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
// located by scanning the z-column, so the coupling spans a (nonmagnetic)
// spacer gap between the layers, unlike the nearest-neighbour exchange field.
// Note  zhat x m = (-m.y, m.x, 0).
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

    // locate the nearest partner-region cell in the same (x,y) column along z,
    // scanning outward from iz and stopping at the first hit (checking the
    // lower side first for a deterministic tie-break). s records whether that
    // partner sits below (-1) or above (+1) this cell, giving the required
    // antisymmetry of the interlayer-DMI field.
    int P = -1;
    float s = 0.0f;
    for (int d = 1; d < Nz; d++) {
        int lo = iz - d;
        if (lo >= 0) {
            int Q = idx(ix, iy, lo);
            if (regions[Q] == partner) {
                P = Q;
                s = -1.0f;
                break;
            }
        }
        int hi = iz + d;
        if (hi < Nz) {
            int Q = idx(ix, iy, hi);
            if (regions[Q] == partner) {
                P = Q;
                s = 1.0f;
                break;
            }
        }
    }
    if (P < 0) {
        return;
    }

    // Apply the areal coupling only at the interface cell (see rationale in
    // rkky.cu): skip if the neighbour toward the partner is the same region,
    // so the coupling stays independent of layer thickness (D is areal, J/m^2).
    int inb = iz + (int)s;
    if (inb >= 0 && inb < Nz && regions[idx(ix, iy, inb)] == r) {
        return;
    }

    float pref = s * D * inv_Msat(Ms_, Ms_mul, I) / dz;
    // zhat x m_partner = (-m_partner.y, m_partner.x, 0)
    Bx[I] += pref * (-my[P]);
    By[I] += pref * (mx[P]);
    // Bz unchanged: (zhat x m)_z = 0
}
