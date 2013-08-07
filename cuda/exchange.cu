#include "stencil.h"
#include "float3.h"

// indexing in symmetric matrix
#define symidx(i, j) ( (j<=i)? ( (((i)*((i)+1)) /2 )+(j) )  :  ( (((j)*((j)+1)) /2 )+(i) ) )

// m is normalized.
// See exchange.go for more details.
extern "C" __global__ void
addexchange(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* aLUT2d, int8_t* regions,
            float wx, float wy, float wz, int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>=N0 || j >= N1 || k >= N2) {
        return;
    }

    // central cell
    int I = idx(i, j, k);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    int8_t r0 = regions[I];
    float3 B  = make_float3(Bx[I], By[I], Bz[I]);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(i, j, lclamp(k-1));
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wz * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(i, j, hclamp(k+1, N2));
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wz * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(i, lclamp(j-1), k);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(i, hclamp(j+1, N1), k);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (N0 != 1) {
        // bottom neighbor
        i_  = idx(lclamp(i-1), j, k);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wx * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(hclamp(i+1, N0), j, k);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wx * a__ *(m_ - m0);
    }

    Bx[I] = B.x;
    By[I] = B.y;
    Bz[I] = B.z;
}

