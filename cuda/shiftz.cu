#include "stencil.h"

// shift dst by shy cells (positive or negative) along Z-axis.
// new edge value is clampL at left edge or clampR at right edge.
extern "C" __global__ void
shiftz(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz, int shz, float clampL, float clampR) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iz2 = iz-shz;
        float newval;
        if (iz2 < 0) {
            newval = clampL;
        } else if (iz2 >= Nz) {
            newval = clampR;
        } else {
            newval = src[idx(ix, iy, iz2)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

