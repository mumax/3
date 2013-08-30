#include "float3.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
extern "C" __global__ void
adduniaxialanisotropy(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                      float* __restrict__ K1LUT,
                      float* __restrict__ uxLUT, float* __restrict__ uyLUT, float* __restrict__ uzLUT,
                      int8_t* __restrict__ regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        int8_t reg = regions[i];
        float  ux  = uxLUT[reg];
        float  uy  = uyLUT[reg];
        float  uz  = uzLUT[reg];
        float3 u   = normalized(make_float3(ux, uy, uz));
        float  K1  = K1LUT[reg];
        float3 m   = {mx[i], my[i], mz[i]};
        float3 Ba  = 2.0f * K1 * dot(m, u) * u;

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}

