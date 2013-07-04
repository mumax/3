#include "stencil.h"
#include "float3.h"

// Exchange + Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// Taking into account proper boundary conditions.
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

    int I = idx(i, j, k);                        // central cell index
    float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H
    float3 m = make_float3(mx[I], my[I], mz[I]); // central m
    float D_2A = (D/(2.*A));

    // z derivatives (along length)
    {
        int I1 = idx(i, j, hclamp(k+1, N2));  // right index, clamped
        int I2 = idx(i, j, lclamp(k-1));      // left index, clamped

        // DMI
        float mz1 = (k+1<N2)? mz[I1] : (m.z - (cz * D_2A * m.x));
        float mz2 = (k-1>=0)? mz[I2] : (m.z + (cz * D_2A * m.x));
        h.x += D*(mz1-mz2)/cz;
        // note: actually 2*D * delta / (2*c)

        float mx1 = (k+1<N2)? mx[I1] : (m.x + (cz * D_2A * m.z));
        float mx2 = (k-1>=0)? mx[I2] : (m.x - (cz * D_2A * m.z));
        h.z -= D*(mx1-mx2)/cz;

        // Exchange
        float3 m1 = make_float3(mx1, my[I1], mz1);
        float3 m2 = make_float3(mx2, my[I2], mz2);
        h +=  (2*A/(cz*cz)) * ((m1 - m) + (m2 - m));
    }

    // y derivatives (along height)
    {
        int I1 = idx(i, hclamp(j+1, N1), k);
        int I2 = idx(i, lclamp(j-1), k);

        // DMI
        float my1 = (j+1<N1)? my[I1] : (m.y - (cy * D_2A * m.x));
        float my2 = (j-1>=0)? my[I2] : (m.y + (cy * D_2A * m.x));
        h.x += D*(my1-my2)/cy;

        float mx1 = (j+1<N1)? mx[I1] : (m.x + (cy * D_2A * m.y));
        float mx2 = (j-1>=0)? mx[I2] : (m.x - (cy * D_2A * m.y));
        h.y -= D*(mx1-mx2)/cy;

        // Exchange
        float3 m1 = make_float3(mx1, my1, mz[I1]);
        float3 m2 = make_float3(mx2, my2, mz[I2]);
        h +=  (2*A/(cy*cy)) * ((m1 - m) + (m2 - m));
    }

    // write back, result is H + Hdmi + Hex
    Hx[I] = h.x;
    Hy[I] = h.y;
    Hz[I] = h.z;
}

