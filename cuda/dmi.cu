#include "stencil.h"

// Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength / Msat, in Tesla*m
// A: Aex/Msat
extern "C" __global__ void
adddmi(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float cx, float cy, float cz, float D, float A, int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N0 || j >= N1 || k >= N2) {
        return;
    }

    int I = idx(i, j, k);
    float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H
    float m1, m2; // neighbors
    float D_2A = (D/(2.*A));

    m1 = (k+1<N2)? mz[idx(i, j, k+1)] : (mz[I] - (cz * D_2A * mx[I]));
    m2 = (k-1>=0)? mz[idx(i, j, k-1)] : (mz[I] + (cz * D_2A * mx[I]));
    h.x += D*(m1-m2)/cz;

    m1 = (k+1<N2)? mx[idx(i, j, k+1)] : (mx[I] + (cz * D_2A * mz[I]));
    m2 = (k-1>=0)? mx[idx(i, j, k-1)] : (mx[I] - (cz * D_2A * mz[I]));
    h.z -= D*(m1-m2)/cz;

    m1 = (j+1<N1)? my[idx(i, j+1, k)] : (my[I] - (cy * D_2A * mx[I]));
    m2 = (j-1>=0)? my[idx(i, j-1, k)] : (my[I] + (cy * D_2A * mx[I]));
    h.x += D*(m1-m2)/cy;

    m1 = (j+1<N1)? mx[idx(i, j+1, k)] : (mx[I] + (cy * D_2A * my[I]));
    m2 = (j-1>=0)? mx[idx(i, j-1, k)] : (mx[I] - (cy * D_2A * my[I]));
    h.y -= D*(m1-m2)/cy;

    // note: actually 2*D * delta / (2*c)

    // write back, result is H + Hdmi
    Hx[I] = h.x;
    Hy[I] = h.y;
    Hz[I] = h.z;
}

