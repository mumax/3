#include "stencil.h"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is the current edge value.
extern "C" __global__ void
shiftfudgey(float* __restrict__  dst, float* __restrict__  src,
    int Nx,  int Ny,  int Nz, int shy) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        float newval;
        if (iy2 < 0) {
            newval = src[idx(ix, 0, iz)];
        } else if (iy2 >= Ny) {
            newval = src[idx(ix, Ny-1, iz)];
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}
