
#include "float3.h"

// dst = v * dot(a,b)
extern "C" __global__ void
dotproduct(float* __restrict__ dst,
           float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
           float* __restrict__ vol, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = (vol == NULL? 1.0f: vol[i]);
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        dst[i] = v * dot(A, B);
    }
}

