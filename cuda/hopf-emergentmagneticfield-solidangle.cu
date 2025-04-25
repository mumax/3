#include <stdint.h>
#include <math.h>
#include "float3.h"
#include "stencil.h"

// Returns the topological charge contribution on an elementary triangle ijk
// Order of arguments is important here to preserve the same measure of chirality
// Note: the result is zero if an argument is zero, or when two arguments are the same
__device__ inline float triangleCharge(float3 mi, float3 mj, float3 mk) {
    float numer   = dot(mi, cross(mj, mk));
    float denom   = 1.0f + dot(mi, mj) + dot(mi, mk) + dot(mj, mk);
    return 2.0f * atan2(numer, denom);
}

// Set the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k) based on the solid angle
// subtended by triangle associated with three spins: a,b,c
//
// 	  q_{a,b,c} = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)]
//
//    F_i = (1/16) (q_{0,1,2} + q_{0,2,3} + q_{0,3,4} + q_{0,4,1})
//
// analogous to the method for calculating the topological charge density in topologicalchargelattice.cu
extern "C" __global__ void
setemergentmagneticfieldsolidangle(float* __restrict__ Fx, float* __restrict__ Fy, float* __restrict__ Fz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float prefactor, float icycz, float iczcx, float icxcy,
                     int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i0 = idx(ix, iy, iz);                        // central cell index
    float3 m0 = make_float3(mx[i0], my[i0], mz[i0]); // central cell magnetization

    if(is0(m0)) {
        Fx[i0] = 0.0f;
        Fy[i0] = 0.0f;
        Fz[i0] = 0.0f;
        return;
    }


    ////////
    // Fx //
    ////////

    // accumulator for Fx
    float fx = 0.0; 

    // indices of the 4 neighbors (counter clockwise)
    int i1 = idx(ix, hclampy(iy+1), iz); // (i+1,j)
    int i2 = idx(ix, iy, hclampz(iz+1)); // (i,j+1)
    int i3 = idx(ix, lclampy(iy-1), iz); // (i-1,j)
    int i4 = idx(ix, iy, lclampz(iz-1)); // (i,j-1)

    // magnetization of the 4 neighbors
    float3 m1 = make_float3(mx[i1], my[i1], mz[i1]);
    float3 m2 = make_float3(mx[i2], my[i2], mz[i2]);
    float3 m3 = make_float3(mx[i3], my[i3], mz[i3]);
    float3 m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((iy+1<Ny || PBCy) && (iz+1<Nz || PBCz)) { 
        int i_ = idx(ix, hclampy(iy+1), hclampz(iz+1)); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge(m0, m1, m2);
    }

    // upper left
    if ((iy-1>=0 || PBCy) && (iz+1<Nz || PBCz)) { 
        int i_ = idx(ix, lclampy(iy-1), hclampz(iz+1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge(m0, m2, m3);
    }

    // bottom left
    if ((iy-1>=0 || PBCy) && (iz-1>=0 || PBCz)) { 
        int i_ = idx(ix, lclampy(iy-1), lclampz(iz-1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge(m0, m3, m4);
    }

    // bottom right
    if ((iy+1<Ny || PBCy) && (iz-1>=0 || PBCz)) { 
        int i_ = idx(ix, hclampy(iy+1), lclampz(iz-1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge(m0, m4, m1);
    }


    ////////
    // Fy //
    ////////

    // accumulator for Fy
    float fy = 0.0; 

    // indices of the 4 neighbors (counter clockwise)
    i1 = idx(ix, iy, hclampz(iz+1)); // (i+1,j)
    i2 = idx(hclampx(ix+1), iy, iz); // (i,j+1)
    i3 = idx(ix, iy, lclampz(iz-1)); // (i-1,j)
    i4 = idx(lclampx(ix-1), iy, iz); // (i,j-1)

    // magnetization of the 4 neighbors
    m1 = make_float3(mx[i1], my[i1], mz[i1]);
    m2 = make_float3(mx[i2], my[i2], mz[i2]);
    m3 = make_float3(mx[i3], my[i3], mz[i3]);
    m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((iz+1<Nz || PBCz) && (ix+1<Nx || PBCx)) { 
        int i_ = idx(hclampx(ix+1), iy, hclampz(iz+1)); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge(m0, m1, m2);
    }

    // upper left
    if ((iz-1>=0 || PBCz) && (ix+1<Nx || PBCx)) { 
        int i_ = idx(hclampx(ix+1), iy, lclampz(iz-1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge(m0, m2, m3);
    }

    // bottom left
    if ((ix-1>=0 || PBCx) && (iy-1>=0 || PBCy)) { 
        int i_ = idx(lclampx(ix-1), iy, lclampz(iz-1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge(m0, m3, m4);
    }

    // bottom right
    if ((ix+1<Nx || PBCx) && (iy-1>=0 || PBCy)) { 
        int i_ = idx(lclampx(ix-1), iy, hclampz(iz+1)); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge(m0, m4, m1);
    }


    ////////
    // Fz //
    ////////

    // accumulator for Fz
    float fz = 0.0; 

    // indices of the 4 neighbors (counter clockwise)
    i1 = idx(hclampx(ix+1), iy, iz); // (i+1,j)
    i2 = idx(ix, hclampy(iy+1), iz); // (i,j+1)
    i3 = idx(lclampx(ix-1), iy, iz); // (i-1,j)
    i4 = idx(ix, lclampy(iy-1), iz); // (i,j-1)

    // magnetization of the 4 neighbors
    m1 = make_float3(mx[i1], my[i1], mz[i1]);
    m2 = make_float3(mx[i2], my[i2], mz[i2]);
    m3 = make_float3(mx[i3], my[i3], mz[i3]);
    m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((ix+1<Nx || PBCx) && (iy+1<Ny || PBCy)) { 
        int i_ = idx(hclampx(ix+1), hclampy(iy+1), iz); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge(m0, m1, m2);
    }

    // upper left
    if ((ix-1>=0 || PBCx) && (iy+1<Ny || PBCy)) { 
        int i_ = idx(lclampx(ix-1), hclampy(iy+1), iz); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge(m0, m2, m3);
    }

    // bottom left
    if ((ix-1>=0 || PBCx) && (iy-1>=0 || PBCy)) { 
        int i_ = idx(lclampx(ix-1), lclampy(iy-1), iz); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge(m0, m3, m4);
    }

    // bottom right
    if ((ix+1<Nx || PBCx) && (iy-1>=0 || PBCy)) { 
        int i_ = idx(hclampx(ix+1), lclampy(iy-1), iz); 
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge(m0, m4, m1);
    }

    Fx[i0] = 2 * prefactor * icycz * fx;
    Fy[i0] = 2 * prefactor * iczcx * fy;
    Fz[i0] = 2 * prefactor * icxcy * fz;
}
