#include "float3.h"

// set dst such that its elements are uniform from -Sd/2 to Sd/2 along each axis
extern "C" __global__ void
spatialfield(
        float* __restrict__  dstx,
        float* __restrict__  dsty,
        float* __restrict__  dstz, 
        float nx, float ny, float nz,
        float sx, float sy, float sz,
        int N
    ) 
{
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    float offsetx = (nx - 1) / 2;
    float offsety = (ny - 1) / 2;
    float offsetz = (nz - 1) / 2;

    if (i < N) {
        float idx_i = fmodf(i, nx) - offsetx;
        float idx_j = floorf(fmodf(i, nx*ny) / nx) - offsety;
        float idx_k = floorf(fmodf(i, N) / (nx*ny)) - offsetz;

        dstx[i] = idx_i * sx;
        dsty[i] = idx_j * sy;
        dstz[i] = idx_k * sz;
    }
}

