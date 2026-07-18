#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Native RKKY interlayer exchange coupling.
//
// For every cell belonging to region1 or region2, the bilinear RKKY field of
// its nearest partner-region cell on EACH side along z is added:
//
//     B += ( J / (Msat * dz) ) * m_partner
//
// with J the areal coupling strength (J/m^2) and dz the cell size along z.
// The partner is located by walking outward along z, so the coupling spans a
// (nonmagnetic) spacer gap between the layers, unlike the nearest-neighbour
// exchange field. J < 0 yields antiferromagnetic (synthetic-antiferromagnet,
// SAF) coupling.
//
// Interface selection: in each z-direction the walk stops at the first cell
// that is either (a) the partner region -> couple to it (this cell is the
// interface on that side), or (b) the same region as this cell -> stop without
// coupling (a closer same-region cell is the interface, so this cell is buried
// on that side). Coupling to the nearest partner on BOTH sides lets a layer
// sandwiched between two partner layers (a superlattice) couple to both, while
// the same-region stop keeps the areal coupling independent of layer thickness.
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

    float pref = J * inv_Msat(Ms_, Ms_mul, I) / dz;

    for (int dir = -1; dir <= 1; dir += 2) {
        for (int j = iz + dir; j >= 0 && j < Nz; j += dir) {
            int rj = regions[idx(ix, iy, j)];
            if (rj == r) {
                break;  // buried on this side: a closer same-region cell exists
            }
            if (rj == partner) {
                int P = idx(ix, iy, j);
                Bx[I] += pref * mx[P];
                By[I] += pref * my[P];
                Bz[I] += pref * mz[P];
                break;  // nearest partner on this side
            }
        }
    }
}
