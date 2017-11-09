#include <stdint.h>
#include "amul.h"

extern "C"

 __global__ void
InittemperatureJH(float* __restrict__  tempJH,
                float* __restrict__ TSubs_, float TSubs_mul,
                int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float TSubs = amul(TSubs_, TSubs_mul, i);
        tempJH[i] = TSubs;
    }
}


