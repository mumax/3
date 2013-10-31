#include "stencil.h"

// Copies src to dst, shifting elements by (u, v, w).
// Clamps at the boundaries.
extern "C" __global__ void
shift(float* __restrict__  dst, float* __restrict__  src,
      int Nx,  int Ny,  int Nz, int shx, int shy, int shz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        dst[idx(ix, iy, iz)] = src[idxclamp(ix-shx, iy-shy, iz-shz)];
    }
}

