#include "stencil.h"

extern "C" __global__ void
setrxyphitheta(float* __restrict__ rxy, float* __restrict__ phi, float* __restrict__ theta,
       float* __restrict__ mx, float* __restrict__ my,float* __restrict__ mz,
       int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    rxy[I] = sqrtf(mx[I]*mx[I] + my[I]*my[I]);
    phi[I] = atan2f(my[I], mx[I]);
    theta[I] = acosf(mz[I]);
}