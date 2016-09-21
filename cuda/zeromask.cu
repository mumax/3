#include <stdint.h>
#include "float3.h"

// set dst to zero in cells where mask != 0
extern "C" __global__ void
zeromask(float* __restrict__  dst, float* __restrict__ mask, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        if (mask[i] != 0.0f) {
            dst[i] = 0.0f;
        }
    }
}
