#include <stdint.h>
#include "stencil.h"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clamp.
extern "C" __global__ void
shiftbytes(uint8_t* __restrict__  dst, uint8_t* __restrict__  src,
           int Nx,  int Ny,  int Nz, int shx, uint8_t clamp) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int ix2 = ix-shx;
        uint8_t newval;
        if (ix2 < 0 || ix2 >= Nx) {
            newval = clamp;
        } else {
            newval = src[idx(ix2, iy, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

