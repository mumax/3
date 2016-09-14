#include <stdint.h>
#include "float3.h"
#include "amul.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
extern "C" __global__ void
adduniaxialanisotropy2(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                       float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                       float* __restrict__ Ms_, float Ms_mul,
                       float* __restrict__ K1_, float K1_mul,
                       float* __restrict__ K2_, float K2_mul,
                       float* __restrict__ ux_, float ux_mul,
                       float* __restrict__ uy_, float uy_mul,
                       float* __restrict__ uz_, float uz_mul,
                       int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 u   = normalized(vmul(ux_, uy_, uz_, ux_mul, uy_mul, uz_mul, i));
        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float  K1  = amul(K1_, K1_mul, i) * invMs;
        float  K2  = amul(K2_, K2_mul, i) * invMs;
        float3 m   = {mx[i], my[i], mz[i]};
        float  mu  = dot(m, u);
        float3 Ba  = 2.0f*K1*    (mu)*u+
                     4.0f*K2*pow3(mu)*u;

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}

