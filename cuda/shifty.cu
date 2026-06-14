#include "stencil.h"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clampD at bottom edge (-Y) or clampU at top edge (+Y).
extern "C" __global__ void
shifty(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz, int shy, float clampD, float clampU) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        float newval;
        if (iy2 < 0) {
            newval = clampD;
        } else if (iy2 >= Ny) {
            newval = clampU;
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

