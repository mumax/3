#include <stdint.h>

// add region-based vector to dst:
// dst[i] += LUT[region[i]]
extern "C" __global__ void
regionaddv(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
           float* __restrict__ LUTx, float* __restrict__ LUTy, float* __restrict__ LUTz,
           uint16_t* regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        uint16_t r = regions[i];
        dstx[i] += LUTx[r];
        dsty[i] += LUTy[r];
        dstz[i] += LUTz[r];
    }
}

