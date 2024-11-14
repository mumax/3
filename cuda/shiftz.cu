#include "stencil.h"

// shift dst by shz cells (positive or negative) along Z-axis.
// new edge value is clampB at back edge (-Z) or clampF at front edge (+Z).
extern "C" __global__ void
shiftz(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz, int shz, float clampB, float clampF) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iz2 = iz-shz;
        float newval;
        if (iz2 < 0) {
            newval = clampB;
        } else if (iz2 >= Nz) {
            newval = clampF;
        } else {
            newval = src[idx(ix, iy, iz2)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

