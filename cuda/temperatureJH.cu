#include <stdint.h>
#include "amul.h"

extern "C"

 __global__ void
settemperatureJH(float* __restrict__  B,      float* __restrict__ noise, float kB2_VgammaDt,
                float* __restrict__ Ms_, float Ms_mul,
                float* __restrict__ tempJH,
                float* __restrict__ alpha_, float alpha_mul,
                int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        
        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float temp = tempJH[i];
        float alpha = amul(alpha_, alpha_mul, i);
        B[i] = noise[i] * sqrtf((kB2_VgammaDt * alpha * temp * invMs ));
    }
}


