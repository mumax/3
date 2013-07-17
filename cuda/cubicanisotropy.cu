#include "float3.h"

// add cubic anisotropy field to B.
// Author: Arne Vansteenkiste, based on Kelvin Fong's mumax2 implementation
extern "C" __global__ void
addcubicanisotropy(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
                   float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                   float* __restrict__ K1LUT,
                   float* __restrict__ C1xLUT, float* __restrict__ C1yLUT, float* __restrict__ C1zLUT,
                   float* __restrict__ C2xLUT, float* __restrict__ C2yLUT, float* __restrict__ C2zLUT,
                   int8_t* __restrict__ regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        int8_t r  = regions[i];
        float  k1 = K1LUT[r];
        float3 c1 = normalized(make_float3(C1xLUT[r], C1yLUT[r], C1zLUT[r]));
        float3 c2 = normalized(make_float3(C1xLUT[r], C1yLUT[r], C1zLUT[r]));
        float3 c3 = cross(c1, c2);
        float3 m  = make_float3(mx[i], my[i], mz[i]);

        float a1 = dot(c1, m);
        float a2 = dot(c2, m);
        float a3 = dot(c3, m);

        float3 A = k1 * make_float3(a1*(a2*a2+a3*a3), a2*(a1*a1+a3*a3), a3*(a1*a1+a2*a2));

        Bx[i] += A.x*c1.x + A.y*c2.x + A.z*c3.x;
        By[i] += A.x*c1.y + A.y*c2.y + A.z*c3.y;
        Bz[i] += A.x*c1.z + A.y*c2.z + A.z*c3.z;
    }
}
