#include "float3.h"

extern "C" __global__ void
crossproduct(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
           float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
           int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        float3 AxB = cross(A, B);
        dstx[i] = AxB.x;
        dsty[i] = AxB.y;
        dstz[i] = AxB.z;
    }
}

