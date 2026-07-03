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

    // locate the nearest partner-region cell in the same (x,y) column along z
    int P = -1;
    int best = Nz + 1;
    for (int jz = 0; jz < Nz; jz++) {
        if (jz == iz) {
            continue;
        }
        int Q = idx(ix, iy, jz);
        if (regions[Q] == partner) {
            int d = (jz > iz) ? (jz - iz) : (iz - jz);
            if (d < best) {
                best = d;
                P = Q;
            }
        }
    }
    if (P < 0) {
        return;
    }

    float pref = J * inv_Msat(Ms_, Ms_mul, I) / dz;
    Bx[I] += pref * mx[P];
    By[I] += pref * my[P];
    Bz[I] += pref * mz[P];
}
