#include "stencil.h"
#include "amul.h"

// Populate 'top' with the values from 'src' at the highest cell in the geometry.
// src is assumed to be at XY-offset w.r.t. the global mesh.
// If a column is completely empty, the bottom cell is used.
extern "C" __global__ void
attop(float* __restrict__ topx, float* __restrict__ topy, float* __restrict__ topz,
       float* __restrict__ srcx, float* __restrict__ srcy, float* __restrict__ srcz,
       float* __restrict__ Ms_, float Ms_mul, float* __restrict__ vol,
       float offsetx, float offsety,
       float cx, float cy,
       int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) {
        return;
    }

    // Snap to global mesh cell center
    int ixglobal = ix + int(offsetx/cx + 0.5f);
    int iyglobal = iy + int(offsety/cy + 0.5f);
    ixglobal = PBCx? MOD(ixglobal, Nx) : ixglobal; // TODO: use finite PBC amount
    iyglobal = PBCy? MOD(iyglobal, Ny) : iyglobal;

    int Itop = idx(ix, iy, 0); // Default: use value of bottom layer
    // If still in global mesh given offset & PBC, search topmost cell within geometry
    if (ixglobal >= 0 && ixglobal < Nx && iyglobal >= 0 && iyglobal < Ny) {
        for (int iz = Nz-1; iz >=0; iz--) { // Top to bottom
            int I = idx(ixglobal, iyglobal, iz);
            float v = amul(vol, 1.0f, I);
            float ms = amul(Ms_, Ms_mul, I);
            if (ms != 0.0f && v != 0.0f || iz == 0) { // Non-empty cell with a magnetic moment
                Itop = idx(ix, iy, iz);
                break;
            }
        }
    }

    int Iflat = idx(ix, iy, 0);
    topx[Iflat] = srcx[Itop];
    if (topy == nullptr || topz == nullptr || srcy == nullptr || srcz == nullptr) {
        return;
    }
    topy[Iflat] = srcy[Itop];
    topz[Iflat] = srcz[Itop];
}