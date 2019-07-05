#include "stencil.h"

extern "C" __global__ void
setPhi(float* __restrict__ phi, float* __restrict__ mx, float* __restrict__ my, int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    phi[I] = atan2f(my[I], mx[I]);
}