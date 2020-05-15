#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Set s to the toplogogical charge density for lattices
// 	  s = 2 atan(a . b x c /(1 + a.b + a.c + b.c) )
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981).
// See topologicalchargelattice.go.
extern "C" __global__ void
settopologicalchargelattice(float* __restrict__ s,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]); // +0
    float3 axb = make_float3(0.0f, 0.0f, 0.0f);   // a x b
    int i_;                                       // neighbour index

    if(is0(m0))
    {
        s[I] = 0.0f;
        return;
    }

    // Assign neigbouring spins with the convention:
    // 0: (i,j)
    // 1: (i+1,j)
    // 2: (i,j+1)
    // 3: (i-1,j)
    // 4: (i,j-1)
    // The four triangles are 012, 023, 034, 041
    float trig012, trig023, trig034, trig041;
    float numer, denom;

    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);      // load neighbour m if inside grid, keep 0 otherwise
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-1), iy, iz);
        if (ix-1 >= 0 || PBCx)
        {
            m3 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m4 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m4 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        // We don't care whether the neighbours exist or not, since the dot and cross products will be zero
        // 012
        axb     = cross(m0, m1);
        numer   = dot(m2, axb);
        denom   = 1.0f + dot(m0, m1) + dot(m0, m2) + dot(m1, m2);
        trig012 = 2.0f * atan2(numer, denom);

        // 023
        axb     = cross(m0, m2);
        numer   = dot(m3, axb);
        denom   = 1.0f + dot(m0, m2) + dot(m0, m3) + dot(m2, m3);
        trig023 = 2.0f * atan2(numer, denom);

        // 034
        axb     = cross(m0, m3);
        numer   = dot(m4, axb);
        denom   = 1.0f + dot(m0, m3) + dot(m0, m4) + dot(m3, m4);
        trig034 = 2.0f * atan2(numer, denom);

        // 041
        axb     = cross(m0, m4);
        numer   = dot(m1, axb);
        denom   = 1.0f + dot(m0, m1) + dot(m0, m4) + dot(m1, m4);
        trig041 = 2.0f * atan2(numer, denom);
    }

    // The on-site value of s is the sum of these 4 triangles divided by 2
    s[I] = 0.5f * ( trig012 + trig023 + trig034 + trig041 );
}
