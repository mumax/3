#include "stencil.h"
#include "float3.h"

// indexing in symmetric matrix
#define symidx(i, j) ( (j<=i)? ( (((i)*((i)+1)) /2 )+(j) )  :  ( (((j)*((j)+1)) /2 )+(i) ) )

// m is normalized.
// See exchange.go for more details.
extern "C" __global__ void
addexchange(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ aLUT2d, int8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, int8_t PBC) {

    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iz>=Nz || iy >= Ny || ix >= Nx) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    int8_t r0 = regions[I];
    float3 B  = make_float3(Bx[I], By[I], Bz[I]);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(i, j, lclamp2(k-1));
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wz * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(i, j, hclamp2(k+1));
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wz * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(i, lclamp1(j-1), k);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(i, hclamp1(j+1), k);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (N0 != 1) {
        // bottom neighbor
        i_  = idx(lclamp0(i-1), j, k);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wx * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(hclamp0(i+1), j, k);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wx * a__ *(m_ - m0);
    }

    Bx[I] = B.x;
    By[I] = B.y;
    Bz[I] = B.z;
}

