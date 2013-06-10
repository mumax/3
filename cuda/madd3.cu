#include "mask.h"

// dst[i] = fac1 * src1[i] + fac2 * src2[i] + fac3 * src3[i]
extern "C" __global__ void
madd3(float* __restrict__ dst,
      float* __restrict__ src1, float fac1,
      float* __restrict__ src2, float fac2,
      float* __restrict__ src3, float fac3, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        float s1 = loadmask(src1, i);
        float s2 = loadmask(src2, i);
        float s3 = loadmask(src3, i);
        dst[i] = (fac1 * s1) + (fac2 * s2 + fac3 * s3);
        // parens for better accuracy heun solver.
    }
}

