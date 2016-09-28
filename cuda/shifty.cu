#include "stencil.h"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clampL at left edge or clampR at right edge.
extern "C" __global__ void
shifty(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz, int shy, float clampL, float clampR) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        float newval;
        if (iy2 < 0) {
            newval = clampL;
        } else if (iy2 >= Ny) {
            newval = clampR;
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

