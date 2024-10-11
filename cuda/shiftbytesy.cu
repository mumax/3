#include <stdint.h>
#include "stencil.h"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clamp.
extern "C" __global__ void
shiftbytesy(uint8_t* __restrict__  dst, uint8_t* __restrict__  src,
            int Nx,  int Ny,  int Nz, int shy, uint8_t clamp) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        uint8_t newval;
        if (iy2 < 0 || iy2 >= Ny) {
            newval = clamp;
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

