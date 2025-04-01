#include "amul.h"
#include "float3.h"
#include <stdint.h>

// scale torque by scalar parameter GammaFactor
extern "C" __global__ void
scalegamma(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  scalegamma_, float scalegamma_mul, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float gammaf = amul(scalegamma_, scalegamma_mul,i);
        tx[i] = tx[i]*gammaf;
        ty[i] = ty[i]*gammaf;
        tz[i] = tz[i]*gammaf;
    }
}
