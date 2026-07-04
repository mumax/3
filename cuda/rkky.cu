#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native RKKY interlayer exchange coupling.
//
// For every cell belonging to region1 or region2, the bilinear RKKY field of
// its nearest partner-region cell in the same (x,y) column is added:
//
//     B += ( J / (Msat * dz) ) * m_partner
//
// with J the areal coupling strength (J/m^2) and dz the cell size along z.
// The partner cell is located by scanning the z-column, so the coupling spans
// a (nonmagnetic) spacer gap between the layers, unlike the nearest-neighbour
// exchange field. J < 0 yields antiferromagnetic (synthetic-antiferromagnet,
// SAF) coupling.
//
// See rkky.go for the host-side wrapper.
extern "C" __global__ void
addrkky(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
        float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
        float* __restrict__ Ms_, float Ms_mul,
        uint8_t* __restrict__ regions, float J, int region1, int region2,
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
    // scanning outward from iz and stopping at the first hit (O(distance),
    // checking the lower side first to keep a deterministic tie-break). dir
    // records whether the partner is below (-1) or above (+1).
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

    // Apply the areal coupling only at the interface cell. If the neighbour
    // toward the partner belongs to the same region, a cell closer to the
    // partner exists and this one is not on the interface; skipping it keeps
    // the coupling independent of layer thickness (J is areal, J/m^2).
    int inb = iz + dir;
    if (inb >= 0 && inb < Nz && regions[idx(ix, iy, inb)] == r) {
        return;
    }

    float pref = J * inv_Msat(Ms_, Ms_mul, I) / dz;
    Bx[I] += pref * mx[P];
    By[I] += pref * my[P];
    Bz[I] += pref * mz[P];
}
