#include "amul.h"
#include "constants.h"
#include "float3.h"
#include "stencil.h"
#include <stdint.h>

// #define PREFACTOR ((MUB) / (2 * QE * GAMMA0))
#define PREFACTOR ((HBAR) / (2 * QE))

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

extern "C" __global__ void
addzhanglitorque2(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                  float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                  float* __restrict__ Ms_, float Ms_mul,
                  float* __restrict__ jx_, float jx_mul,
                  float* __restrict__ jy_, float jy_mul,
                  float* __restrict__ jz_, float jz_mul,
                  float* __restrict__ alpha_, float alpha_mul,
                  float* __restrict__ xi_, float xi_mul,
                  float* __restrict__ pol_, float pol_mul,
                  float* __restrict__ g_, float g_mul,
                  float cx, float cy, float cz,
                  int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i = idx(ix, iy, iz);

    float alpha = amul(alpha_, alpha_mul, i);
    float xi    = amul(xi_, xi_mul, i);
    float pol   = amul(pol_, pol_mul, i);
    float g     = amul(g_, g_mul, i);
    float invMs = inv_Msat(Ms_, Ms_mul, i);
    float b = invMs * PREFACTOR / (g*(1.0f + xi*xi));
    float3 J = pol*vmul(jx_, jy_, jz_, jx_mul, jy_mul, jz_mul, i);

    float3 hspin = make_float3(0.0f, 0.0f, 0.0f); // (u·∇)m
    if (J.x != 0.0f) {
        hspin += (b/cx)*J.x * make_float3(deltax(mx), deltax(my), deltax(mz));
    }
    if (J.y != 0.0f) {
        hspin += (b/cy)*J.y * make_float3(deltay(mx), deltay(my), deltay(mz));
    }
    if (J.z != 0.0f) {
        hspin += (b/cz)*J.z * make_float3(deltaz(mx), deltaz(my), deltaz(mz));
    }

    float3 m      = make_float3(mx[i], my[i], mz[i]);
    float3 torque = (-1.0f/(1.0f + alpha*alpha)) * (
                        (1.0f+xi*alpha) * cross(m, cross(m, hspin))
                        +(  xi-alpha) * cross(m, hspin)           );

    // write back, adding to torque
    tx[i] += torque.x;
    ty[i] += torque.y;
    tz[i] += torque.z;
}
