#include "float3.h"

// set dst such that its elements are uniform from -Sd/2 to Sd/2 along each axis
extern "C" __global__ void
posscaled(
        float* __restrict__  dstx,
        float* __restrict__  dsty,
        float* __restrict__  dstz, 
        float Nx, float Ny, float Nz,
        float Sx, float Sy, float Sz,
        int N
    ) 
{
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    float offsetx = (Nx - 1) / 2;
    float offsety = (Ny - 1) / 2;
    float offsetz = (Nz - 1) / 2;

    if (i < N) {
        float idx_i = fmodf(i, Nx) - offsetx;
        float idx_j = floorf(fmodf(i, Nx*Ny) / Nx) - offsety;
        float idx_k = floorf(fmodf(i, N) / (Nx*Ny)) - offsetz;

        dstx[i] = idx_i * Sx;
        dsty[i] = idx_j * Sy;
        dstz[i] = idx_k * Sz;
    }
}

