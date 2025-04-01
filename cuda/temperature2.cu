#include <stdint.h>
#include "amul.h"

// TODO: this could act on x,y,z, so that we need to call it only once.
extern "C" __global__ void
settemperature2(float* __restrict__  B,      float* __restrict__ noise, float kB2_VgammaDt,
                float* __restrict__ Ms_, float Ms_mul,
                float* __restrict__ temp_, float temp_mul,
                float* __restrict__ alpha_, float alpha_mul,
                float* __restrict__ g_, float g_mul,
                int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float temp = amul(temp_, temp_mul, i);
        float alpha = amul(alpha_, alpha_mul, i);
        float g = amul(g_, g_mul, i);
        B[i] = noise[i] * sqrtf((kB2_VgammaDt * alpha * temp * invMs / g));
    }
}
