#include <stdint.h>
#include "stencil.h"
#include "float3.h"
#include "exchange.h"

// see exchange.go
extern "C" __global__ void
exchangedecode(float* __restrict__ dst, float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
               float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    uint8_t r0 = regions[I];

    int i_;    // neighbor index
    float avg = 0.0f;

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    avg += aLUT2d[symidx(r0, regions[i_])];

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        avg += aLUT2d[symidx(r0, regions[i_])];

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        avg += aLUT2d[symidx(r0, regions[i_])];
        
        avg /= 6;
    } else {
        avg /= 4;
    }

    dst[I] = avg;
}

