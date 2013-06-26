#include "stencil.h"

// Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength / Msat, in Tesla*m
extern "C" __global__ void
adddmi(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float cx, float cy, float cz,
       float* __restrict__ DLUT, int8_t* __restrict__ regions, int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N0 || j >= N1 || k >= N2) {
        return;
    }

    int I = idx(i, j, k);
    float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H
    float D = DLUT[regions[I]];

    // TODO: proper boundary conditions
    h.x += D * delta(mz, 0, 0, 1) / cz;
    h.x += D * delta(my, 0, 1, 0) / cy;
    h.y -= D * delta(mx, 0, 1, 0) / cy;
    h.z -= D * delta(mx, 0, 0, 1) / cz;
    // note: actually 2*D * delta / (2*c)

    // write back, result is H + Hdmi
    Hx[I] = h.x;
    Hy[I] = h.y;
    Hz[I] = h.z;
}

