#include "float3.h"
#include "stencil.h"
#include "constants.h"

#define PREFACTOR ((MUB * MU0) / (2 * QE * GAMMA0))

// (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)
extern "C" __global__ void
addzhanglitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                 float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                 float* __restrict__ jx, float* __restrict__ jy, float* __restrict__ jz,
                 float cx, float cy, float cz,
                 float* __restrict__ bsatLUT, float* __restrict__ alphaLUT, float* __restrict__ xiLUT, int8_t* __restrict__ regions,
                 int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N0 || j >= N1 || k >= N2) {
        return;
    }

    int I = idx(i, j, k);

    int8_t r = regions[I];
    float alpha = alphaLUT[r];
    float xi    = xiLUT[r];
    float bsat  = bsatLUT[r];
    float b = PREFACTOR / (bsat * (1 + xi*xi));
    float Jx = jx[I];
    float Jy = jy[I];
    float Jz = jz[I];

    float3 hspin = make_float3(0, 0, 0); // (u·∇)m
    if (Jx != 0.) {
        hspin += (b/cx)*Jx * make_float3(delta(mx, 1,0,0), delta(my, 1,0,0), delta(mz, 1,0,0));
    }
    if (Jy != 0.) {
        hspin += (b/cy)*Jy * make_float3(delta(mx, 0,1,0), delta(my, 0,1,0), delta(mz, 0,1,0));
    }
    if (Jz != 0.) {
        hspin += (b/cz)*Jz * make_float3(delta(mx, 0,0,1), delta(my, 0,0,1), delta(mz, 0,0,1));
    }

    float3 m      = make_float3(mx[I], my[I], mz[I]);
    float3 torque = (-1./(1. + alpha*alpha)) * (
                        (1+xi*alpha) * cross(m, cross(m, hspin))
                        +(  xi-alpha) * cross(m, hspin)           );

    // write back, adding to torque
    tx[I] += torque.x;
    ty[I] += torque.y;
    tz[I] += torque.z;
}

